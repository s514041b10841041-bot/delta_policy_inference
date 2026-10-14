"""Attention-bottleneck fusion of the two camera views.

The sparse counterpart to ``CrossViewAdapter``. That module lets all 256 TP tokens
cross-attend directly to all 256 Wrist tokens; this one forbids the two views from ever
looking at each other and forces every cross-view fact through K learned bottleneck tokens
(Nagrani et al. 2021, *Attention Bottlenecks for Multimodal Fusion*). Both sit at the same
insertion point -- after InternViT + pixel-shuffle + ``mlp1``, before the features are
spliced into the ``<IMG_CONTEXT>`` positions -- and both add their result through the same
``gamma = gamma_scale * tanh(alpha)`` per-channel gate, so a run of one against the other
differs in the fusion topology and nothing else.

Two steps, in this order, no weight sharing between them:

  1. ``z`` reads both views at once. Query is ``z``; keys/values are
     ``[x_tp + e_tp ; x_wrist + e_wrist ; z]``. This is the only place the two views meet.
  2. Every patch reads the updated ``z``. Query is the patch, keys/values are ``z`` alone,
     so a TP patch's own update is a function of the bottleneck summary and never of an
     individual Wrist patch. Patches do not attend to each other here either -- InternViT
     already did that, and adding it back would mix a second effect into the measurement.

``z`` is discarded afterwards. Token count, order and sequence length into the LLM are
exactly the input's, so the only difference from a baseline run is the value of 2x256
visual features.

View partition (``view_partition=True``, the default): in step 1 the first K//2 bottleneck
tokens may only read TP keys and the remaining K - K//2 may only read Wrist keys (every
token still sees all K ``z`` keys). This is a fixed attention mask, no parameters. It makes
"z learns to listen to one camera only" structurally impossible, which matters here because
the two cameras are not symmetric -- one is fixed, one rides on the wrist and its frames
change ~2.7x more per step -- so an unconstrained z has an easy degenerate solution. The
question then moves to step 2, where a patch chooses which memos to read; that is what
``tp_reads_wrist`` / ``wrist_reads_tp`` in ``residual_summary`` measure. This is also what
MBT does (per-modality bottleneck copies), minus the averaging.
"""

from typing import Optional

import torch
import torch.nn as nn

from .cross_view import CrossViewAdapter, _first_parameter_zero

# One implementation of residual_cos_*/residual_ratio_* for both fusion families: the two
# modules write the same `cross_view/*` log keys and the results table puts their numbers in
# adjacent columns, so they must be computed the same way. CrossViewAdapter._residual_stats
# never touches `self` (it only zips its positional arguments), so calling it as a plain
# function with this module as the first argument is exactly what the adapter does.
_residual_stats = CrossViewAdapter._residual_stats


def _make_mlp(embed_dim: int, mlp_ratio: int) -> nn.Sequential:
    """D -> mlp_ratio*D -> D with GELU, PyTorch's default initialization."""

    hidden_dim = int(embed_dim) * int(mlp_ratio)
    return nn.Sequential(
        nn.Linear(embed_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, embed_dim),
    )


class BottleneckFusion(nn.Module):
    """Bottleneck cross-view fusion with the adapter's gated residual.

    On the initialization of ``z`` and the view embeddings: both are drawn from
    ``randn * 0.02`` rather than zeroed. K zero-initialized bottleneck tokens are literally
    identical vectors, so attention treats them as one token and they only separate through
    accumulated gradient noise -- the same slow start that ``cross_view_gate_init`` exists to
    avoid on the gate (cross-view.md 4.3). The view embeddings are what lets ``z`` tell a TP
    key from a Wrist key at all, since step 1 concatenates the two views into one key
    sequence.

    On the gate: identical to ``CrossViewAdapter``'s, including the fields it reads from
    ``EvoConfig`` (``cross_view_gamma_scale`` / ``cross_view_gate_init`` /
    ``cross_view_gate_per_channel`` / ``cross_view_gate_lr``). ``gate_init=0.0`` makes this
    module an exact identity, which is what the unit tests and the smoke test's step-0 loss
    check rely on.
    """

    def __init__(
        self,
        embed_dim: int,
        num_tokens: int = 16,
        num_heads: int = 14,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        gamma_scale: float = 0.1,
        gate_init: float = 0.2,
        gate_per_channel: bool = True,
        view_partition: bool = True,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be positive, got {num_tokens}")
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        if gamma_scale <= 0.0:
            raise ValueError(f"gamma_scale must be positive, got {gamma_scale}")
        if view_partition and num_tokens < 2:
            raise ValueError("view_partition needs at least 2 bottleneck tokens (one per view)")

        self.embed_dim = int(embed_dim)
        self.num_tokens = int(num_tokens)
        # Which z tokens belong to which view under the partition: [0, num_tp_tokens) read
        # TP only, [num_tp_tokens, K) read Wrist only. Index-defined, so the step-2 memo-read
        # shares below are computed the same way whether or not the mask is on.
        self.view_partition = bool(view_partition)
        self.num_tp_tokens = self.num_tokens // 2
        self.gamma_scale = float(gamma_scale)
        self.gate_per_channel = bool(gate_per_channel)

        # The bottleneck itself, plus the two view tags that make step 1's concatenated key
        # sequence distinguishable. 0.02 is the usual small-init scale for learned tokens.
        self.z = nn.Parameter(torch.randn(self.num_tokens, self.embed_dim) * 0.02)
        self.view_embed_tp = nn.Parameter(torch.randn(self.embed_dim) * 0.02)
        self.view_embed_wrist = nn.Parameter(torch.randn(self.embed_dim) * 0.02)

        # Step 1: z <- [TP ; Wrist ; z].
        self.z_q_norm = nn.LayerNorm(embed_dim)
        self.z_kv_norm = nn.LayerNorm(embed_dim)
        self.z_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.z_mlp_norm = nn.LayerNorm(embed_dim)
        self.z_mlp = _make_mlp(embed_dim, mlp_ratio)

        # Step 2: each view reads z. Separate weights per view -- a TP patch and a Wrist
        # patch are asking different questions of the same summary.
        self.tp_q_norm = nn.LayerNorm(embed_dim)
        self.tp_kv_norm = nn.LayerNorm(embed_dim)
        self.tp_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.tp_mlp_norm = nn.LayerNorm(embed_dim)
        self.tp_mlp = _make_mlp(embed_dim, mlp_ratio)

        self.wrist_q_norm = nn.LayerNorm(embed_dim)
        self.wrist_kv_norm = nn.LayerNorm(embed_dim)
        self.wrist_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.wrist_mlp_norm = nn.LayerNorm(embed_dim)
        self.wrist_mlp = _make_mlp(embed_dim, mlp_ratio)

        # Names are load-bearing: train.py's is_cross_view_gate() routes anything matching
        # "bottleneck_fusion.alpha_" into the gate learning-rate group.
        gate_shape = (self.embed_dim,) if self.gate_per_channel else ()
        self.alpha_tp = nn.Parameter(torch.full(gate_shape, float(gate_init)))
        self.alpha_wrist = nn.Parameter(torch.full(gate_shape, float(gate_init)))

        # Set by the training loop on log steps only; see residual_summary().
        self.collect_stats = False
        self._last_stats = {}
        # Diagnostics only, off in training: with capture_attention the raw head-averaged
        # attention of both steps is kept instead of only the scalars residual_summary()
        # reduces them to, so a caller can render them over the camera images. It implies
        # collect_stats (that is what asks nn.MultiheadAttention for weights at all) and
        # holds one forward's worth of tensors, replaced on every call.
        self.capture_attention = False
        self._last_attn = {}

    def gamma_tp(self) -> torch.Tensor:
        return self.gamma_scale * torch.tanh(self.alpha_tp)

    def gamma_wrist(self) -> torch.Tensor:
        return self.gamma_scale * torch.tanh(self.alpha_wrist)

    @torch.no_grad()
    def gate_summary(self) -> dict:
        """Current gamma values, same keys as CrossViewAdapter.gate_summary()."""

        summary = {}
        for name, gamma in (("gamma_tp", self.gamma_tp()), ("gamma_wrist", self.gamma_wrist())):
            gamma = gamma.detach().float()
            if gamma.ndim == 0:
                summary[name] = float(gamma)
            else:
                summary[f"{name}_mean"] = float(gamma.mean())
                summary[f"{name}_absmax"] = float(gamma.abs().max())
        return summary

    @torch.no_grad()
    def residual_summary(self) -> Optional[dict]:
        """Evidence that the module is doing something, populated only on collecting steps.

        ``residual_cos_*`` / ``residual_ratio_*`` are the adapter's, unchanged, so the two
        families are directly comparable. ``z_share_tp`` / ``z_share_wrist`` are specific to
        this topology: how much of step 1's attention mass ``z`` spends on each view. Under
        ``view_partition`` they are exactly 0.5 / 0.5 by construction (half the tokens read
        each view), and the live question moves to step 2: ``tp_reads_wrist`` is the share of
        a TP patch's attention that lands on the Wrist-group memos, ``wrist_reads_tp`` the
        mirror. Either one collapsing toward 0 means the patches learned to ignore the memos
        that carry the opposite view -- the same failure the old z_share caught, one step
        later. Both start near 0.5 (random z, random view tags).
        """

        return dict(self._last_stats) if self._last_stats else None

    @torch.no_grad()
    def attention_summary(self) -> Optional[dict]:
        """Raw attention of the last forward, when capture_attention is on. See __init__."""

        return dict(self._last_attn) if self._last_attn else None

    @staticmethod
    @torch.no_grad()
    def _z_share(weights: torch.Tensor, num_tp: int, num_wrist: int) -> dict:
        """Share of z's step-1 attention that lands on each view.

        ``weights`` is [B, K, num_tp + num_wrist + K], head-averaged by
        nn.MultiheadAttention. The trailing K columns are z attending to itself, which says
        nothing about fusion, so they are dropped and the remainder renormalized -- the two
        numbers then sum to 1 and a drift toward 0/1 is immediately readable. Padded keys
        already carry exactly zero weight (key_padding_mask), so summing whole blocks is
        correct without re-masking. fp32 because a bf16 sum over 512 columns loses about two
        significant digits.
        """

        weights = weights.detach().float()
        tp_mass = weights[:, :, :num_tp].sum(dim=-1)
        wrist_mass = weights[:, :, num_tp:num_tp + num_wrist].sum(dim=-1)
        denominator = (tp_mass + wrist_mass).clamp_min(1e-6)
        return {
            "z_share_tp": float((tp_mass / denominator).mean()),
            "z_share_wrist": float((wrist_mass / denominator).mean()),
        }

    def _partition_mask(self, num_tp: int, num_wrist: int, device) -> torch.Tensor:
        """[K, num_tp + num_wrist + K] bool, True = may NOT attend (nn.MultiheadAttention's
        convention). TP-group rows are blind to Wrist keys, Wrist-group rows to TP keys; the
        trailing K z keys stay visible to every row, which also guarantees no row is ever
        fully masked (the NaN case the key_padding guard in forward() is about)."""

        mask = torch.zeros(
            self.num_tokens, num_tp + num_wrist + self.num_tokens, dtype=torch.bool, device=device
        )
        mask[: self.num_tp_tokens, num_tp : num_tp + num_wrist] = True
        mask[self.num_tp_tokens :, :num_tp] = True
        return mask

    @torch.no_grad()
    def _memo_read_share(
        self,
        tp_weights: torch.Tensor,
        wrist_weights: torch.Tensor,
        tp_padding_mask: torch.Tensor,
        wrist_padding_mask: torch.Tensor,
    ) -> dict:
        """Step-2 diagnostic: how much of each view's patch attention lands on the memos that
        belong to the OTHER view's z group. ``tp_weights`` is [B, num_tp, K] head-averaged,
        ``wrist_weights`` [B, num_wrist, K]. Padded queries are excluded from the mean."""

        k = self.num_tp_tokens
        tp_keep = (~tp_padding_mask).float()
        wrist_keep = (~wrist_padding_mask).float()
        tp_w = tp_weights.detach().float()
        wrist_w = wrist_weights.detach().float()
        tp_reads_wrist = (tp_w[:, :, k:].sum(dim=-1) * tp_keep).sum() / tp_keep.sum().clamp_min(1.0)
        wrist_reads_tp = (wrist_w[:, :, :k].sum(dim=-1) * wrist_keep).sum() / wrist_keep.sum().clamp_min(1.0)
        return {
            "tp_reads_wrist": float(tp_reads_wrist),
            "wrist_reads_tp": float(wrist_reads_tp),
        }

    def forward(
        self,
        x_tp: torch.Tensor,
        x_wrist: torch.Tensor,
        tp_padding_mask: Optional[torch.Tensor] = None,
        wrist_padding_mask: Optional[torch.Tensor] = None,
        pair_valid: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Signature matches CrossViewAdapter.forward minus its attn_* overrides, so
        InternVL3Embedder can dispatch to either module on one line."""

        if x_tp.ndim != 3 or x_wrist.ndim != 3:
            raise ValueError(
                f"x_tp/x_wrist must be [B, N, D], got {tuple(x_tp.shape)} and {tuple(x_wrist.shape)}"
            )
        if (
            x_tp.shape[0] != x_wrist.shape[0]
            or x_tp.shape[-1] != self.embed_dim
            or x_wrist.shape[-1] != self.embed_dim
        ):
            raise ValueError(
                f"Incompatible TP/Wrist shapes for embed_dim={self.embed_dim}: "
                f"{tuple(x_tp.shape)} and {tuple(x_wrist.shape)}"
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

        if self.capture_attention:
            self.collect_stats = True
        if self.collect_stats:
            # A collecting step whose batch has no valid pair must report nothing, not an
            # earlier step's numbers.
            self._last_stats = {}
            self._last_attn = {}

        fused_tp = x_tp.clone()
        fused_wrist = x_wrist.clone()
        valid_indices = pair_valid.nonzero(as_tuple=False).flatten()
        if valid_indices.numel() > 0:
            tp_valid = x_tp.index_select(0, valid_indices)
            wrist_valid = x_wrist.index_select(0, valid_indices)
            tp_pad_valid = tp_padding_mask.index_select(0, valid_indices)
            wrist_pad_valid = wrist_padding_mask.index_select(0, valid_indices)
            num_valid, num_tp, _ = tp_valid.shape
            num_wrist = wrist_valid.shape[1]

            tagged_tp = tp_valid + self.view_embed_tp.to(tp_valid.dtype)
            tagged_wrist = wrist_valid + self.view_embed_wrist.to(wrist_valid.dtype)
            z = self.z.unsqueeze(0).expand(num_valid, -1, -1).to(tp_valid.dtype)

            # --- Step 1: z reads both views (and itself) --------------------------------
            kv = torch.cat([tagged_tp, tagged_wrist, z], dim=1)
            kv_padding_mask = torch.cat(
                [
                    tp_pad_valid,
                    wrist_pad_valid,
                    torch.zeros(num_valid, self.num_tokens, dtype=torch.bool, device=device),
                ],
                dim=1,
            )
            # The z block is never padded, so this can only fire if the mask is malformed.
            # Kept because nn.MultiheadAttention answers an all-padded row with NaN rather
            # than an error, and a NaN here would reach the LLM as a valid-looking feature.
            if kv_padding_mask.all(dim=1).any():
                raise ValueError("Bottleneck fusion received a sample with every key padded")

            normalized_kv = self.z_kv_norm(kv)
            # The view partition is a fixed [K, S] mask merged with the per-sample key
            # padding; it has no parameters and adds nothing to the state_dict.
            attn_mask = (
                self._partition_mask(num_tp, num_wrist, device) if self.view_partition else None
            )
            # need_weights=True disables nn.MultiheadAttention's fused fast path, so it is
            # paid only on the ~1-in-log_interval steps that actually read z_share_*.
            z_attn_out, z_weights = self.z_attn(
                self.z_q_norm(z),
                normalized_kv,
                normalized_kv,
                key_padding_mask=kv_padding_mask,
                attn_mask=attn_mask,
                need_weights=bool(self.collect_stats),
            )
            # .to(dtype) at every residual junction: under torch.amp.autocast -- which
            # Evo1_server.py enables and DeepSpeed training does NOT -- nn.LayerNorm is an
            # fp32-list op and returns fp32, so a residual add can silently promote the
            # whole chain and index_copy() below then raises on a dtype mismatch. Training
            # never sees it; the first attempt to serve the checkpoint does.
            z_state = z + z_attn_out.to(z.dtype)
            z_fused = z_state + self.z_mlp(self.z_mlp_norm(z_state)).to(z_state.dtype)

            # --- Step 2: each view reads the updated z ----------------------------------
            # No key padding needed: all K bottleneck tokens are real.
            tp_kv = self.tp_kv_norm(z_fused)
            delta_tp, tp_weights = self.tp_attn(
                self.tp_q_norm(tagged_tp), tp_kv, tp_kv, need_weights=bool(self.collect_stats)
            )
            delta_tp = self.tp_mlp(self.tp_mlp_norm(delta_tp)).to(tp_valid.dtype)

            wrist_kv = self.wrist_kv_norm(z_fused)
            delta_wrist, wrist_weights = self.wrist_attn(
                self.wrist_q_norm(tagged_wrist), wrist_kv, wrist_kv, need_weights=bool(self.collect_stats)
            )
            delta_wrist = self.wrist_mlp(self.wrist_mlp_norm(delta_wrist)).to(wrist_valid.dtype)

            # The MLP has bias terms, so MLP(0) != 0: a padded position would otherwise
            # receive a non-zero residual and be spliced back into the LLM as content.
            delta_tp = delta_tp.masked_fill(tp_pad_valid.unsqueeze(-1), 0.0)
            delta_wrist = delta_wrist.masked_fill(wrist_pad_valid.unsqueeze(-1), 0.0)

            residual_tp = self.gamma_tp().to(delta_tp.dtype) * delta_tp
            residual_wrist = self.gamma_wrist().to(delta_wrist.dtype) * delta_wrist
            tp_updated = tp_valid + residual_tp.to(tp_valid.dtype)
            wrist_updated = wrist_valid + residual_wrist.to(wrist_valid.dtype)
            fused_tp = fused_tp.index_copy(0, valid_indices, tp_updated)
            fused_wrist = fused_wrist.index_copy(0, valid_indices, wrist_updated)

            if self.collect_stats:
                self._last_stats = _residual_stats(
                    self,
                    (tp_valid, tp_updated, tp_pad_valid),
                    (wrist_valid, wrist_updated, wrist_pad_valid),
                )
                if z_weights is not None:
                    self._last_stats.update(self._z_share(z_weights, num_tp, num_wrist))
                if tp_weights is not None and wrist_weights is not None:
                    self._last_stats.update(
                        self._memo_read_share(tp_weights, wrist_weights, tp_pad_valid, wrist_pad_valid)
                    )
                if self.capture_attention:
                    self._last_attn = {
                        # [B, K, num_tp + num_wrist + K] -- step 1, where each bottleneck
                        # token looks. Under the partition the first K//2 rows have zero mass
                        # on the Wrist block and vice versa.
                        "z_attn": z_weights.detach().float().cpu() if z_weights is not None else None,
                        # [B, num_tp, K] / [B, num_wrist, K] -- step 2, how much each patch
                        # reads from each bottleneck token.
                        "tp_attn": tp_weights.detach().float().cpu() if tp_weights is not None else None,
                        "wrist_attn": wrist_weights.detach().float().cpu() if wrist_weights is not None else None,
                        "num_tp": int(num_tp), "num_wrist": int(num_wrist),
                        "num_tp_tokens": int(self.num_tp_tokens), "num_tokens": int(self.num_tokens),
                    }

        # Keep every parameter in the distributed autograd graph even on a rank whose
        # current batch has no valid camera pair; ZeRO otherwise stalls on the missing
        # gradient. Numerically this is +0.
        graph_zero = _first_parameter_zero(self).to(dtype=fused_tp.dtype)
        return fused_tp + graph_zero, fused_wrist + graph_zero
