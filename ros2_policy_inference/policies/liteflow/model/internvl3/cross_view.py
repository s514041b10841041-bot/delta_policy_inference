"""Trainable cross-view fusion and predictive supervision.

The main path always receives clean projected visual features.  The auxiliary
CVP path reuses the exact same directional attention modules, but consumes the
raw (ungated) attention delta so that CVP can train the inference-time adapter
even while the residual gates are initialized to zero.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


def _first_parameter_zero(module: nn.Module) -> torch.Tensor:
    """Return a zero scalar connected to every parameter in ``module``.

    Missing camera pairs can otherwise leave a newly-added module unused on one
    distributed rank.  Referencing one element of each parameter produces zero
    gradients without changing values or paying for a full parameter reduction
    in the forward graph.
    """

    zero = None
    for parameter in module.parameters():
        if parameter.numel() == 0:
            continue
        term = parameter.reshape(-1)[0] * 0.0
        zero = term if zero is None else zero + term
    if zero is None:
        raise RuntimeError("Expected module to contain at least one parameter")
    return zero


def _make_prediction_head(embed_dim: int, output_norm: bool = False) -> nn.Module:
    """LayerNorm -> Linear -> GELU -> Linear [-> LayerNorm], shared by whoever consumes a raw delta.

    The LayerNorm at the front means this head is invariant to the raw delta's own
    magnitude and can apply an arbitrary rotation before the first Linear -- exactly the
    freedom that let the first CVP design train `head(delta) ~ target` while `delta` itself
    (what the main path actually adds, unheaded) pointed somewhere unrelated. Sharing one
    head instance between CVP and the main path (see CrossViewAdapter.use_head) closes
    that gap: whatever transformation CVP teaches is the same transformation the residual
    add uses.

    The front LayerNorm only bounds the *input*; nothing bounds the final Linear's output,
    so `gamma * head(raw_delta)`'s magnitude on content the shared attention never trained
    on (e.g. an occluded view) is whatever that Linear happens to produce -- unconstrained,
    not just "small because gamma is small". `output_norm=True` appends a trailing
    LayerNorm so the quantity gamma actually scales has a fixed, input-independent norm:
    a hard bound on the injected perturbation instead of a statistical tendency observed
    only on the training distribution.
    """

    layers = [
        nn.LayerNorm(embed_dim),
        nn.Linear(embed_dim, embed_dim),
        nn.GELU(),
        nn.Linear(embed_dim, embed_dim),
    ]
    if output_norm:
        layers.append(nn.LayerNorm(embed_dim))
    return nn.Sequential(*layers)


class DirectionalCrossAttention(nn.Module):
    """One directional pre-normalized cross-attention block."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_dim = int(embed_dim)
        self.q_norm = nn.LayerNorm(embed_dim)
        self.kv_norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def attend(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        kv_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the raw attention delta, without a residual or gamma gate."""

        if q.ndim != 3 or kv.ndim != 3:
            raise ValueError(
                f"q and kv must be [B, N, D], got q={tuple(q.shape)}, kv={tuple(kv.shape)}"
            )
        if q.shape[0] != kv.shape[0] or q.shape[-1] != self.embed_dim or kv.shape[-1] != self.embed_dim:
            raise ValueError(
                f"Incompatible q/kv shapes for embed_dim={self.embed_dim}: "
                f"q={tuple(q.shape)}, kv={tuple(kv.shape)}"
            )
        if kv.shape[1] == 0:
            raise ValueError("Cross-attention source cannot have zero tokens")
        if kv_padding_mask is not None:
            expected = kv.shape[:2]
            if tuple(kv_padding_mask.shape) != tuple(expected):
                raise ValueError(
                    f"kv_padding_mask must have shape {tuple(expected)}, "
                    f"got {tuple(kv_padding_mask.shape)}"
                )
            kv_padding_mask = kv_padding_mask.to(device=kv.device, dtype=torch.bool)
            if kv_padding_mask.all(dim=1).any():
                raise ValueError("Cross-attention received a sample with every source token padded")

        normalized_kv = self.kv_norm(kv)
        delta, _ = self.attention(
            self.q_norm(q),
            normalized_kv,
            normalized_kv,
            key_padding_mask=kv_padding_mask,
            need_weights=False,
        )
        return delta


class CrossViewAdapter(nn.Module):
    """Bidirectional TP/Wrist attention with gated residuals.

    On the gate initialization -- measured, not assumed. The first run of this module
    (libero_spatial_crossview, 40k steps) ended with alpha_tp = -0.0045 and
    alpha_wrist = +0.0025, i.e. gamma ~ 5e-4: the adapter never turned on, and the
    clean-suite "tie" with baseline was the module doing nothing rather than doing no harm.

    That displacement is exactly what a pure random walk predicts. AdamW normalizes away
    gradient magnitude, so each step moves alpha by about lr regardless of how large the
    gradient is; over N steps with no consistent direction the net displacement is
    ~sqrt(N) * lr = sqrt(40000) * 2e-5 = 0.004, against 0.0045 observed. So the action loss
    carried no consistent signal about whether to open the gate -- raising the learning rate
    alone would only random-walk faster.

    The dead loop is: gamma ~ 0 scales the action gradient reaching the attention to ~0, so
    the attention never learns a delta worth adding, so adding it never helps, so gamma has
    no reason to grow. `gate_init` > 0 breaks it by giving the attention a real share of the
    action gradient from step 0, and `gate_per_channel` lets useful channels open while
    harmful ones stay shut instead of forcing one scalar to be right for all of embed_dim.

    On `use_head`: CVP trains `head(raw_delta) ~ target`, not `raw_delta ~ target`. A
    cosine loss on the head's output does not constrain what raw_delta itself looks like --
    the head's LayerNorm strips its scale, and the two Linears can apply an arbitrary
    rotation before comparing to target. The main path historically added raw_delta
    directly, unheaded, so CVP could perfectly satisfy its own loss while raw_delta pointed
    somewhere the main path's residual add never benefits from. `use_head=True` makes the
    main path add `head(raw_delta)` instead, so both consumers of the shared attention are
    optimizing the same quantity.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        gamma_scale: float = 0.1,
        gate_init: float = 0.0,
        gate_per_channel: bool = False,
        use_head: bool = False,
        head_output_norm: bool = False,
        orthogonal_residual: bool = False,
    ):
        super().__init__()
        if gamma_scale <= 0.0:
            raise ValueError(f"gamma_scale must be positive, got {gamma_scale}")
        self.embed_dim = int(embed_dim)
        self.gamma_scale = float(gamma_scale)
        self.gate_per_channel = bool(gate_per_channel)
        self.use_head = bool(use_head)
        self.head_output_norm = bool(head_output_norm)
        self.orthogonal_residual = bool(orthogonal_residual)
        self.tp_from_wrist = DirectionalCrossAttention(embed_dim, num_heads, dropout)
        self.wrist_from_tp = DirectionalCrossAttention(embed_dim, num_heads, dropout)
        gate_shape = (self.embed_dim,) if self.gate_per_channel else ()
        self.alpha_tp = nn.Parameter(torch.full(gate_shape, float(gate_init)))
        self.alpha_wrist = nn.Parameter(torch.full(gate_shape, float(gate_init)))
        # Owned here, not by CrossViewPredictiveBranch: remove_training_only_modules()
        # deletes the CVP branch for inference, and the main path needs this same
        # transformation to still be there afterward (see class docstring, use_head).
        self.head_tp = (
            _make_prediction_head(embed_dim, output_norm=self.head_output_norm) if self.use_head else None
        )
        self.head_wrist = (
            _make_prediction_head(embed_dim, output_norm=self.head_output_norm) if self.use_head else None
        )
        # Set by the training loop on log steps only; see residual_summary().
        self.collect_stats = False
        self._last_stats = {}

    def gamma_tp(self) -> torch.Tensor:
        return self.gamma_scale * torch.tanh(self.alpha_tp)

    def gamma_wrist(self) -> torch.Tensor:
        return self.gamma_scale * torch.tanh(self.alpha_wrist)

    @staticmethod
    def _reject_parallel(residual: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Remove the component of ``residual`` parallel to ``x``, per token.

        CVP's target is the token itself, so its pull drives the residual toward a rescale
        of x -- redundant by construction, since x already carries that information. What the
        opposite view can genuinely contribute is only the component x does NOT have, and
        this projection keeps exactly that: <fused, x>/<x, x> stays exactly 1 (the original
        feature is untouched along its own direction), and the added vector is orthogonal --
        purely new directions. With this in place CVP may reconstruct as perfectly as it
        likes; the redundant part of its success is discarded at fusion instead of being
        added as a useless gain, so the cos->1 degeneration mode is removed structurally.

        The trade-off, stated plainly: the adapter loses the ability to *suppress* x along
        its own direction (e.g. damping an occluder's feature). That is deliberate -- the
        priority is that clean-scene features are provably not disturbed.

        Computed in fp32: a 896-dim dot product in bf16 loses ~2 decimal digits, and the
        orthogonality guarantee is the whole point. Zeroed (padded) residual rows stay
        exactly zero; a zero x row is guarded by the clamp.
        """

        residual32 = residual.float()
        x32 = x.float()
        coefficient = (residual32 * x32).sum(dim=-1, keepdim=True) / (
            x32.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-6)
        )
        return (residual32 - coefficient * x32).to(residual.dtype)

    @staticmethod
    @torch.no_grad()
    def _mean_cos(residual: torch.Tensor, x: torch.Tensor, padding: torch.Tensor) -> float:
        keep = ~padding
        if not bool(keep.any()):
            return 0.0
        r = residual[keep].float()
        t = x[keep].float()
        denominator = (r.norm(dim=-1) * t.norm(dim=-1)).clamp_min(1e-6)
        return float(((r * t).sum(dim=-1) / denominator).mean())

    @torch.no_grad()
    def _residual_stats(self, *directions) -> dict:
        """Cosine(residual, token) and |residual|/|token|, averaged over unpadded tokens."""

        stats = {}
        for name, (original, updated, padding) in zip(("tp", "wrist"), directions):
            keep = ~padding
            if not bool(keep.any()):
                continue
            x = original[keep].float()
            residual = (updated[keep] - original[keep]).float()
            x_norm = x.norm(dim=-1)
            residual_norm = residual.norm(dim=-1)
            denominator = (x_norm * residual_norm).clamp_min(1e-6)
            stats[f"residual_cos_{name}"] = float(
                ((x * residual).sum(dim=-1) / denominator).mean()
            )
            stats[f"residual_ratio_{name}"] = float(
                (residual_norm / x_norm.clamp_min(1e-6)).mean()
            )
        return stats

    @torch.no_grad()
    def residual_summary(self) -> Optional[dict]:
        """How much of the residual is new information rather than a rescale of the token.

        CVP's target is the token itself (`head(attend(...)) ~ x_i`), so a CVP that succeeded
        perfectly would make the main path compute `x_i + gamma*x_i` -- a pure rescale that
        Qwen's pre-norm largely absorbs, carrying no cross-view information at all. The
        module's value therefore lives entirely in the reconstruction *error*: the part of the
        delta that points somewhere other than the token.

        `residual_cos` is the cosine between the added residual and the token it is added to.
        Near 0 means the residual is mostly new evidence from the opposite view; approaching
        1 (or -1) means the adapter has degenerated into a scalar gain and is adding nothing,
        no matter how low cvp_loss_raw has gone. Nothing else in the logs distinguishes those
        two states -- the same lesson as the dead gate, which cost a 13-hour run because it
        was never logged.

        Populated only on steps where ``collect_stats`` was set before forward().
        """

        return dict(self._last_stats) if self._last_stats else None

    @torch.no_grad()
    def gate_summary(self) -> dict:
        """Current gamma values, for the training log.

        The first run shipped without this and the dead gate only surfaced by reading the
        finished checkpoint 13 hours later. A scalar gate logs one number per direction; a
        per-channel gate logs mean and max-magnitude.
        """

        summary = {}
        for name, gamma in (("gamma_tp", self.gamma_tp()), ("gamma_wrist", self.gamma_wrist())):
            gamma = gamma.detach().float()
            if gamma.ndim == 0:
                summary[name] = float(gamma)
            else:
                summary[f"{name}_mean"] = float(gamma.mean())
                summary[f"{name}_absmax"] = float(gamma.abs().max())
        return summary

    def raw_tp_from_wrist(
        self,
        q: torch.Tensor,
        wrist_features: torch.Tensor,
        wrist_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.tp_from_wrist.attend(q, wrist_features, wrist_padding_mask)

    def raw_wrist_from_tp(
        self,
        q: torch.Tensor,
        tp_features: torch.Tensor,
        tp_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.wrist_from_tp.attend(q, tp_features, tp_padding_mask)

    def forward(
        self,
        x_tp: torch.Tensor,
        x_wrist: torch.Tensor,
        tp_padding_mask: Optional[torch.Tensor] = None,
        wrist_padding_mask: Optional[torch.Tensor] = None,
        pair_valid: Optional[torch.Tensor] = None,
        attn_tp: Optional[torch.Tensor] = None,
        attn_wrist: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``attn_tp``/``attn_wrist``: what the cross-attention reads as Q and K/V, when it
        should differ from what the gated residual adds onto -- e.g. a language-grounded
        version of the token (see ``TextGroundingAttention``). Default to ``x_tp``/
        ``x_wrist``, which reproduces every existing call site exactly: the residual base
        (and the orthogonal-projection reference) is always ``x_tp``/``x_wrist``, never
        these, so language (or anything else routed through here) only ever reaches the
        fused feature through the one existing gate, applied once, at the end.
        """

        if x_tp.ndim != 3 or x_wrist.ndim != 3:
            raise ValueError(
                f"x_tp/x_wrist must be [B, N, D], got {tuple(x_tp.shape)} and {tuple(x_wrist.shape)}"
            )
        if x_tp.shape[0] != x_wrist.shape[0] or x_tp.shape[-1] != self.embed_dim or x_wrist.shape[-1] != self.embed_dim:
            raise ValueError(
                f"Incompatible TP/Wrist shapes for embed_dim={self.embed_dim}: "
                f"{tuple(x_tp.shape)} and {tuple(x_wrist.shape)}"
            )
        if attn_tp is None:
            attn_tp = x_tp
        elif attn_tp.shape != x_tp.shape:
            raise ValueError(
                f"attn_tp must match x_tp's shape {tuple(x_tp.shape)}, got {tuple(attn_tp.shape)}"
            )
        if attn_wrist is None:
            attn_wrist = x_wrist
        elif attn_wrist.shape != x_wrist.shape:
            raise ValueError(
                f"attn_wrist must match x_wrist's shape {tuple(x_wrist.shape)}, got {tuple(attn_wrist.shape)}"
            )

        batch_size = x_tp.shape[0]
        device = x_tp.device
        if tp_padding_mask is None:
            tp_padding_mask = torch.zeros(x_tp.shape[:2], dtype=torch.bool, device=device)
        else:
            tp_padding_mask = tp_padding_mask.to(device=device, dtype=torch.bool)
        if wrist_padding_mask is None:
            wrist_padding_mask = torch.zeros(x_wrist.shape[:2], dtype=torch.bool, device=device)
        else:
            wrist_padding_mask = wrist_padding_mask.to(device=device, dtype=torch.bool)
        if pair_valid is None:
            pair_valid = torch.ones(batch_size, dtype=torch.bool, device=device)
        else:
            pair_valid = pair_valid.to(device=device, dtype=torch.bool)
        pair_valid = pair_valid & ~tp_padding_mask.all(dim=1) & ~wrist_padding_mask.all(dim=1)

        if self.collect_stats:
            # A collecting step whose batch has no valid pair must report nothing, not an
            # earlier step's numbers.
            self._last_stats = {}

        fused_tp = x_tp.clone()
        fused_wrist = x_wrist.clone()
        valid_indices = pair_valid.nonzero(as_tuple=False).flatten()
        if valid_indices.numel() > 0:
            tp_valid = x_tp.index_select(0, valid_indices)
            wrist_valid = x_wrist.index_select(0, valid_indices)
            attn_tp_valid = attn_tp.index_select(0, valid_indices)
            attn_wrist_valid = attn_wrist.index_select(0, valid_indices)
            tp_pad_valid = tp_padding_mask.index_select(0, valid_indices)
            wrist_pad_valid = wrist_padding_mask.index_select(0, valid_indices)

            delta_tp = self.raw_tp_from_wrist(attn_tp_valid, attn_wrist_valid, wrist_pad_valid)
            delta_wrist = self.raw_wrist_from_tp(attn_wrist_valid, attn_tp_valid, tp_pad_valid)
            if self.use_head:
                # .to(dtype) is load-bearing, not defensive: under torch.amp.autocast --
                # which Evo1_server.py enables and DeepSpeed training does NOT -- nn.LayerNorm
                # is an fp32-list op and RETURNS fp32, so head_output_norm=True makes the head
                # emit fp32 while the visual features stay bf16. The residual add then
                # produces fp32 and index_copy() below raises. Training never saw it (accelerate's
                # autocast is a no-op under DeepSpeed bf16), so it only surfaced when a
                # checkpoint was first served.
                delta_tp = self.head_tp(delta_tp).to(tp_valid.dtype)
                delta_wrist = self.head_wrist(delta_wrist).to(wrist_valid.dtype)
            # Re-zero after head: head's own bias terms can make head(0) != 0, and a
            # padded position must still contribute exactly nothing to the residual.
            delta_tp = delta_tp.masked_fill(tp_pad_valid.unsqueeze(-1), 0.0)
            delta_wrist = delta_wrist.masked_fill(wrist_pad_valid.unsqueeze(-1), 0.0)

            residual_tp = self.gamma_tp().to(delta_tp.dtype) * delta_tp
            residual_wrist = self.gamma_wrist().to(delta_wrist.dtype) * delta_wrist
            raw_pull_stats = {}
            if self.orthogonal_residual:
                if self.collect_stats:
                    # The CVP pull toward a rescale, measured before it is projected away.
                    # Post-projection residual_cos is ~0 by construction and says nothing.
                    raw_pull_stats = {
                        "residual_cos_raw_tp": self._mean_cos(residual_tp, tp_valid, tp_pad_valid),
                        "residual_cos_raw_wrist": self._mean_cos(residual_wrist, wrist_valid, wrist_pad_valid),
                    }
                residual_tp = self._reject_parallel(residual_tp, tp_valid)
                residual_wrist = self._reject_parallel(residual_wrist, wrist_valid)
            # Second guard on the same hazard: _reject_parallel computes in fp32 by design and
            # any future autocast-promoting op in this block would resurface it here.
            tp_updated = tp_valid + residual_tp.to(tp_valid.dtype)
            wrist_updated = wrist_valid + residual_wrist.to(wrist_valid.dtype)
            fused_tp = fused_tp.index_copy(0, valid_indices, tp_updated)
            fused_wrist = fused_wrist.index_copy(0, valid_indices, wrist_updated)

            if self.collect_stats:
                self._last_stats = self._residual_stats(
                    (tp_valid, tp_updated, tp_pad_valid),
                    (wrist_valid, wrist_updated, wrist_pad_valid),
                )
                self._last_stats.update(raw_pull_stats)

        # Keep every adapter parameter in the distributed autograd graph even on a
        # rank whose current batch has no valid camera pair.  Numerically this is +0.
        graph_zero = _first_parameter_zero(self).to(dtype=fused_tp.dtype)
        return fused_tp + graph_zero, fused_wrist + graph_zero


class TextGroundingAttention(nn.Module):
    """Each view attends to the instruction before cross-view fusion: "does what the
    instruction asks for appear in my patches, and where."

    Shared weights across TP and Wrist -- both views run the same operation (their own
    patches as queries, the instruction as keys/values), so one instance trains from both
    views' gradient instead of splitting it across two separate sets of weights.

    The residual here is a plain, ungated pre-norm addition (coefficient 1), not
    CrossViewAdapter's learned gamma gate: this only preserves each token's own appearance
    while mixing in language relevance, the way an ordinary transformer sub-layer would.
    How much this grounded pathway is allowed to change the feature going into Qwen is
    still decided exactly once, by CrossViewAdapter's existing gate, applied at the very
    end onto the untouched original token -- see ``CrossViewAdapter.forward``'s
    ``attn_tp``/``attn_wrist`` parameters, which is what this feeds.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attend_text = DirectionalCrossAttention(embed_dim, num_heads, dropout)

    def forward(
        self,
        x: torch.Tensor,
        text_embeds: torch.Tensor,
        text_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return x + self.attend_text.attend(x, text_embeds, text_padding_mask)


@dataclass
class CVPResult:
    loss: torch.Tensor
    num_predictions: int
    num_tp_targets: int
    num_wrist_targets: int


class CrossViewPredictiveBranch(nn.Module):
    """Training-only masking/prediction scaffolding that reuses ``CrossViewAdapter``
    attention -- and, when ``owns_heads=False``, its heads as well (see
    ``CrossViewAdapter.use_head``: the head must live on the adapter for the main path to
    keep using it after this branch is deleted for inference).

    On ``content_query_ratio``: mask queries (mask_token + position + view embedding) and
    the main path's queries (the token's own projected feature) are disjoint input
    distributions for the shared attention's query projection. CVP with mask queries
    therefore trains the K/V/value/output/head pathway the main path uses, but never
    trains q_norm/q_proj on the content inputs inference actually feeds them -- that side
    is left to the action gradient through the gate, the same weak channel whose failure
    (the dead gate) motivated CVP in the first place. For the given fraction of sampled
    positions the query is instead the token's own detached clean feature, bare, exactly
    as the main path presents it; the target stays the same token's clean feature and the
    source stays the opposite view, so the trained computation head(attend(x_i, opposite))
    ~ x_i is literally the main path's inference computation. 0.0 preserves the original
    mask-query-only behaviour bit-for-bit, including the RNG stream."""

    VALID_SOURCE_MODES = {"opposite_only", "target_visible_plus_opposite"}

    def __init__(
        self,
        embed_dim: int,
        max_position_tokens: int,
        mask_ratio: float = 0.5,
        source_mode: str = "opposite_only",
        detach_source: bool = True,
        content_query_ratio: float = 0.0,
        owns_heads: bool = True,
    ):
        super().__init__()
        if max_position_tokens <= 0:
            raise ValueError(f"max_position_tokens must be positive, got {max_position_tokens}")
        if not 0.0 < mask_ratio <= 1.0:
            raise ValueError(f"mask_ratio must be in (0, 1], got {mask_ratio}")
        if source_mode not in self.VALID_SOURCE_MODES:
            raise ValueError(
                f"source_mode must be one of {sorted(self.VALID_SOURCE_MODES)}, got {source_mode!r}"
            )
        if not 0.0 <= content_query_ratio <= 1.0:
            raise ValueError(
                f"content_query_ratio must be in [0, 1], got {content_query_ratio}"
            )
        self.embed_dim = int(embed_dim)
        self.max_position_tokens = int(max_position_tokens)
        self.mask_ratio = float(mask_ratio)
        self.source_mode = source_mode
        self.detach_source = bool(detach_source)
        self.content_query_ratio = float(content_query_ratio)
        self.owns_heads = bool(owns_heads)

        self.mask_token = nn.Parameter(torch.empty(1, embed_dim))
        self.position_embedding = nn.Parameter(torch.empty(max_position_tokens, embed_dim))
        self.view_embedding = nn.Parameter(torch.empty(2, embed_dim))
        # Legacy path (owns_heads=True): this branch has its own heads, never seen by the
        # main path -- the mismatch use_head=True/owns_heads=False exists to fix. Kept for
        # existing checkpoints and for an explicit A/B against the shared-head design.
        self.cvp_head_tp = _make_prediction_head(embed_dim) if self.owns_heads else None
        self.cvp_head_wrist = _make_prediction_head(embed_dim) if self.owns_heads else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.view_embedding, std=0.02)

    @staticmethod
    def _lengths_from_padding(padding_mask: torch.Tensor) -> torch.Tensor:
        return (~padding_mask).sum(dim=1)

    def _predict_direction(
        self,
        adapter_direction: DirectionalCrossAttention,
        head: nn.Module,
        target_features: torch.Tensor,
        opposite_features: torch.Tensor,
        target_padding_mask: torch.Tensor,
        opposite_padding_mask: torch.Tensor,
        sample_indices: torch.Tensor,
        target_view_index: int,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
        if sample_indices.numel() == 0:
            return None, None, 0

        target_lengths = self._lengths_from_padding(target_padding_mask)
        opposite_lengths = self._lengths_from_padding(opposite_padding_mask)
        query_list = []
        target_list = []
        source_list = []

        for batch_index_tensor in sample_indices:
            batch_index = int(batch_index_tensor.item())
            target_length = int(target_lengths[batch_index].item())
            opposite_length = int(opposite_lengths[batch_index].item())
            if target_length <= 0 or opposite_length <= 0:
                continue
            if target_length > self.max_position_tokens:
                raise ValueError(
                    f"CVP received {target_length} target tokens, but only "
                    f"{self.max_position_tokens} positional embeddings are configured. "
                    "Dynamic tiling requires tile-aware positional embeddings."
                )

            num_masked = max(1, min(target_length, round(target_length * self.mask_ratio)))
            masked_indices = torch.randperm(target_length, device=target_features.device)[:num_masked]
            masked_indices = masked_indices.sort().values
            query = (
                self.mask_token
                + self.position_embedding.index_select(0, masked_indices)
                + self.view_embedding[target_view_index].unsqueeze(0)
            )
            clean_target = target_features[batch_index].index_select(0, masked_indices).detach()
            if self.content_query_ratio > 0.0:
                # Bare clean feature, no mask/position/view embedding added: the main
                # path's query is the token alone, and matching that regime is the whole
                # point (see the class docstring). Guarded so ratio 0.0 does not consume
                # RNG and stays bit-identical to the mask-query-only behaviour.
                content_rows = (
                    torch.rand(num_masked, device=target_features.device)
                    < self.content_query_ratio
                )
                if content_rows.any():
                    query = torch.where(
                        content_rows.unsqueeze(-1), clean_target.to(query.dtype), query
                    )
            opposite = opposite_features[batch_index, :opposite_length]

            if self.source_mode == "target_visible_plus_opposite":
                visible_mask = torch.ones(target_length, dtype=torch.bool, device=target_features.device)
                visible_mask[masked_indices] = False
                visible_target = target_features[batch_index, :target_length][visible_mask]
                source = torch.cat([visible_target, opposite], dim=0)
            else:
                source = opposite
            if self.detach_source:
                source = source.detach()

            query_list.append(query)
            target_list.append(clean_target)
            source_list.append(source)

        if not query_list:
            return None, None, 0

        queries = pad_sequence(query_list, batch_first=True)
        targets = pad_sequence(target_list, batch_first=True)
        sources = pad_sequence(source_list, batch_first=True)
        query_lengths = torch.tensor(
            [query.shape[0] for query in query_list], device=queries.device, dtype=torch.long
        )
        source_lengths = torch.tensor(
            [source.shape[0] for source in source_list], device=sources.device, dtype=torch.long
        )
        query_padding_mask = (
            torch.arange(queries.shape[1], device=queries.device).unsqueeze(0)
            >= query_lengths.unsqueeze(1)
        )
        source_padding_mask = (
            torch.arange(sources.shape[1], device=sources.device).unsqueeze(0)
            >= source_lengths.unsqueeze(1)
        )

        raw_delta = adapter_direction.attend(queries, sources, source_padding_mask)
        predictions = head(raw_delta)
        valid_queries = ~query_padding_mask
        return predictions[valid_queries], targets[valid_queries], int(valid_queries.sum().item())

    def forward(
        self,
        adapter: CrossViewAdapter,
        x_tp: torch.Tensor,
        x_wrist: torch.Tensor,
        tp_padding_mask: torch.Tensor,
        wrist_padding_mask: torch.Tensor,
        pair_valid: torch.Tensor,
        target_is_tp: Optional[torch.Tensor] = None,
    ) -> CVPResult:
        if x_tp.shape[0] != x_wrist.shape[0]:
            raise ValueError("TP and Wrist batches must have the same batch size")
        batch_size = x_tp.shape[0]
        pair_valid = pair_valid.to(device=x_tp.device, dtype=torch.bool)
        pair_valid = pair_valid & ~tp_padding_mask.all(dim=1) & ~wrist_padding_mask.all(dim=1)
        valid_indices = pair_valid.nonzero(as_tuple=False).flatten()

        if target_is_tp is not None:
            if tuple(target_is_tp.shape) != (batch_size,):
                raise ValueError(
                    f"target_is_tp must have shape ({batch_size},), got {tuple(target_is_tp.shape)}"
                )
            assignments = target_is_tp.to(device=x_tp.device, dtype=torch.bool)
        elif valid_indices.numel() > 0:
            assignments = torch.zeros(batch_size, dtype=torch.bool, device=x_tp.device)
            num_valid = valid_indices.numel()
            num_tp = num_valid // 2
            if num_valid % 2:
                num_tp += int(torch.randint(0, 2, (), device=x_tp.device).item())
            permutation = torch.randperm(num_valid, device=x_tp.device)
            assignments[valid_indices[permutation[:num_tp]]] = True
        else:
            assignments = torch.zeros(batch_size, dtype=torch.bool, device=x_tp.device)

        tp_target_indices = (pair_valid & assignments).nonzero(as_tuple=False).flatten()
        wrist_target_indices = (pair_valid & ~assignments).nonzero(as_tuple=False).flatten()

        # Same head the main path adds through (use_head=True) when this branch doesn't
        # own its own -- see the class docstring for why the head must live on `adapter`.
        head_tp = self.cvp_head_tp if self.owns_heads else adapter.head_tp
        head_wrist = self.cvp_head_wrist if self.owns_heads else adapter.head_wrist
        if head_tp is None or head_wrist is None:
            raise RuntimeError(
                "CVP has no prediction head to use: owns_heads=False requires "
                "adapter.use_head=True (adapter.head_tp/head_wrist are None otherwise)."
            )

        pred_tp, target_tp, count_tp = self._predict_direction(
            adapter.tp_from_wrist,
            head_tp,
            x_tp,
            x_wrist,
            tp_padding_mask,
            wrist_padding_mask,
            tp_target_indices,
            target_view_index=0,
        )
        pred_wrist, target_wrist, count_wrist = self._predict_direction(
            adapter.wrist_from_tp,
            head_wrist,
            x_wrist,
            x_tp,
            wrist_padding_mask,
            tp_padding_mask,
            wrist_target_indices,
            target_view_index=1,
        )

        predictions = [tensor for tensor in (pred_tp, pred_wrist) if tensor is not None]
        targets = [tensor for tensor in (target_tp, target_wrist) if tensor is not None]
        graph_zero = (_first_parameter_zero(self) + _first_parameter_zero(adapter)).float()
        if predictions:
            prediction = torch.cat(predictions, dim=0).float()
            target = torch.cat(targets, dim=0).float()
            loss = 1.0 - F.cosine_similarity(
                prediction,
                target,
                dim=-1,
                eps=1e-6,
            ).mean()
            loss = loss + graph_zero
        else:
            loss = graph_zero

        return CVPResult(
            loss=loss,
            num_predictions=count_tp + count_wrist,
            num_tp_targets=int(tp_target_indices.numel()),
            num_wrist_targets=int(wrist_target_indices.numel()),
        )
