import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  
        self.register_buffer('pe', pe)

    def forward(self, seq_len: int):
        if seq_len > self.pe.size(1):
            self._extend_pe(seq_len)
        return self.pe[:, :seq_len, :]

    def _extend_pe(self, new_max_len):
        old_max_len, dim = self.pe.size(1), self.pe.size(2)
        if new_max_len <= old_max_len:
            return
        extra_positions = torch.arange(old_max_len, new_max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
        extra_pe = torch.zeros(new_max_len - old_max_len, dim)
        extra_pe[:, 0::2] = torch.sin(extra_positions * div_term)
        extra_pe[:, 1::2] = torch.cos(extra_positions * div_term)
        extra_pe = extra_pe.unsqueeze(0)
        new_pe = torch.cat([self.pe, extra_pe.to(self.pe.device)], dim=1)
        self.pe = new_pe

class CategorySpecificLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_categories: int = 1):
        super().__init__()
        self.num_categories = num_categories
        if num_categories <= 1:
            self.linear = nn.Linear(in_dim, out_dim)
        else:
            self.weight = nn.Parameter(torch.empty(num_categories, in_dim, out_dim))
            self.bias = nn.Parameter(torch.zeros(num_categories, out_dim))
            nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, category_id: torch.LongTensor):

        if self.num_categories <= 1:
            if x.dtype != self.linear.weight.dtype:
                x = x.to(dtype=self.linear.weight.dtype)
            return self.linear(x)

        if x.dtype != self.weight.dtype:
            x = x.to(dtype=self.weight.dtype)

        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1]) 
        if category_id.dim() == 0:
       
            cid = category_id.item()
            out = x_flat @ self.weight[cid] + self.bias[cid]
        else:
           
            category_id = category_id.reshape(-1)
            if category_id.numel() != x_flat.size(0):
                raise ValueError(
                    f"category_id length {category_id.numel()} does not match flattened batch {x_flat.size(0)}"
                )
            weight_selected = self.weight[category_id]        
            bias_selected = self.bias[category_id]        
            out = torch.bmm(x_flat.unsqueeze(1), weight_selected).squeeze(1) + bias_selected
        out_shape = orig_shape[:-1] + (out.shape[-1],)
        return out.view(out_shape)

class CategorySpecificMLP(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_categories: int = 1):
        super().__init__()
        self.fc1 = CategorySpecificLinear(input_dim, hidden_dim, num_categories)
        self.fc2 = CategorySpecificLinear(hidden_dim, output_dim, num_categories)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, category_id: torch.LongTensor):
        out = self.activation(self.fc1(x, category_id))
        out = self.fc2(out, category_id)
        return out

class MultiEmbodimentActionEncoder(nn.Module):

    def __init__(self, action_dim: int, embed_dim: int, hidden_dim: int, horizon: int, num_categories: int = 1,
                 use_pos_embed: bool = False):
        super().__init__()
        self.horizon = horizon
        self.embed_dim = embed_dim
        self.num_categories = num_categories
        
        self.W1 = CategorySpecificLinear(action_dim, hidden_dim, num_categories)
        self.W2 = CategorySpecificLinear(hidden_dim, hidden_dim, num_categories)
        self.W3 = CategorySpecificLinear(hidden_dim, embed_dim, num_categories)
   
        # FLOWER adds an additive position to the action tokens only when RoPE is off
        # (flower.py:1087 `if not self.use_rope and not self.use_nope`). FlowerSelfAttention
        # applies RoPE, so adding this as well puts two position signals on the same tokens.
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim, max_len=horizon) if use_pos_embed else None
        self.activation = nn.ReLU(inplace=True)

    def forward(self, action_seq: torch.Tensor, category_id: torch.LongTensor):

        B, H, D = action_seq.shape
        assert H == self.horizon, "Action sequence length must match horizon"
       
        x = action_seq.reshape(B * H, D) 
      
        if category_id.dim() == 0:
           
            cat_ids = category_id.expand(H * B)
        else:
            cat_ids = category_id.unsqueeze(1).expand(B, H).reshape(B * H)
        out = self.activation(self.W1(x, cat_ids))            
    
        if self.pos_encoding is not None:
            pos_enc = self.pos_encoding(H).to(device=out.device, dtype=out.dtype)
            out = (out.view(B, H, -1) + pos_enc).view(B * H, -1)
        out = self.activation(self.W2(out, cat_ids))         
        out = self.W3(out, cat_ids)                        
        out = out.view(B, H, self.embed_dim)
        return out

class BasicTransformerBlock(nn.Module):
    """Cross-attention to the VLM context, plus optional self-attention over the action chunk.

    The self-attention branch is an addition (2026-08-26). Without it the block's only
    attention is `action -> context`, so the `horizon` action steps never see each other:
    every step of the chunk is denoised from the VLM context alone, with nothing in the
    architecture coordinating step k with step k+1. FLOWER's FlowBlock
    (flower/models/networks/transformers.py:545-554) carries self_attn, cross_attn and mlp
    as three separate branches; this is the first of those two differences.

    That gap lines up with what the CALVIN sweep measured on step_12500: tasks that need a
    coherent trajectory fail (rotate_pink_block_right 1/8, push_blue_block_right 2/6) while
    tasks that mostly need a direction succeed (turn_on_led 24/24, move_slider_left 16/16).
    Both attempts to impose coherence from outside the model made it worse -- replanning
    every 3 steps scored 1.96 and Gaussian-smoothing the chunk ~2.07, against 2.41 for the
    untouched baseline.

    `out_proj` of the self-attention is zero-initialised, so at step 0 the branch is an exact
    no-op and the block reproduces the previous architecture bit-for-bit; it only departs from
    it once trained. Self-attention is bidirectional, not causal: flow matching denoises the
    whole chunk in one pass, so masking step k from the steps after it withholds part of the
    trajectory being jointly denoised (FLOWER's own comment at transformers.py makes the same
    point about its hardcoded causal flag).

    `context_key_padding_mask` is True at positions to IGNORE, matching
    nn.MultiheadAttention's convention.
    """

    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, dropout: float = 0.0,
                 use_self_attn: bool = True):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

        self.use_self_attn = use_self_attn
        if use_self_attn:
            self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout,
                                                   batch_first=True)
            self.norm0 = nn.LayerNorm(embed_dim)
            nn.init.zeros_(self.self_attn.out_proj.weight)
            nn.init.zeros_(self.self_attn.out_proj.bias)

    def forward(self, action_tokens: torch.Tensor, context_tokens: torch.Tensor,
                time_emb: torch.Tensor, context_key_padding_mask: torch.Tensor = None,
                global_adaln=None):
        # global_adaln is accepted and ignored: this block has no adaLN to add it to. The
        # shared call site passes it unconditionally, which made `action_block: basic` --
        # the path every pre-FLOWER checkpoint takes -- raise TypeError at the first block.

        if self.use_self_attn:
            s = self.norm0(action_tokens)
            self_out, _ = self.self_attn(s, s, s, need_weights=False)
            action_tokens = action_tokens + self_out

        x = self.norm1(action_tokens)
        attn_out, _ = self.attn(x, context_tokens, context_tokens,
                                key_padding_mask=context_key_padding_mask,
                                need_weights=False)

        x = action_tokens + attn_out

        x2 = self.norm2(x)

        if time_emb is not None:
            x2 = x2 + time_emb.unsqueeze(1)
        ff_out = self.ff(x2)
        x = x + ff_out
        return x

# =====================================================================================
# FLOWER-style DiT block, ported from flower/models/networks/transformers.py
# (FlowBlock at :494, FlowerAttention at :157, FlowerCrossAttention at :241, SwiGlu at :39,
# modulate at :27, precompute_freqs_1d at :78, apply_rotary_pos_emb at :127).
#
# Deliberate differences from that source, and why:
#   * width stays at embed_dim (896, = InternVL3's LLM hidden size) rather than FLOWER's
#     dit_dim 1024. FLOWER needs a width change because its LLM is 960-wide; here the
#     dimensions already line up, and forcing 1024 would add projections on both the action
#     tokens and the context for no design reason.
#   * the adaLN pair is initialised to shift 0 / scale 0 / gate 1, i.e. identity modulation,
#     so an untrained block computes the unmodulated `residual + branch(norm(x))` that the
#     previous BasicTransformerBlock computed. FLOWER makes the same choice for its
#     cross-attention triple and says why: gating the only route from the VLM to the actions
#     near zero at init is not a small effect, and it would confound the ablation.
#   * self-attention is not causal. Flow matching denoises the whole chunk in one pass, so
#     masking step k from the steps after it withholds part of what is being denoised;
#     FLOWER's own comment flags its hardcoded causal flag as questionable for that reason.
#   * the cross-attention "sink" token is NOT ported: it is this fork's own experiment, off
#     by default there (use_sink_token: False), not part of the published design.
# =====================================================================================

def _find_multiple(n: int, k: int) -> int:
    return n if n % k == 0 else n + k - (n % k)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """x * (1 + scale) + shift, broadcast over the sequence axis."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def precompute_freqs_1d(dim: int, max_seq_len: int, theta: float = 32.0):
    freqs = torch.arange(0, dim, 2).float()
    freqs = theta ** (-freqs / dim)
    positions = torch.arange(max_seq_len).float()
    angles = positions.unsqueeze(1) * freqs.unsqueeze(0)
    return angles.cos(), angles.sin()


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    seq_len = q.size(-2)
    pos = torch.arange(seq_len, device=q.device)
    cos = cos.to(q.dtype)[pos].unsqueeze(0).unsqueeze(0)
    sin = sin.to(q.dtype)[pos].unsqueeze(0).unsqueeze(0)
    q1, q2 = q.chunk(2, dim=-1)
    k1, k2 = k.chunk(2, dim=-1)
    q_rot = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
    return q_rot, k_rot


def stateless_norm(x: torch.Tensor) -> torch.Tensor:
    """FLOWER's transformers.py:19 -- per-sample standardisation with no learned scale."""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + 1e-6)


class TimestepEmbedder(nn.Module):
    """FLOWER's TimestepEmbedder (transformers.py:638).

    Sinusoidal features of the *continuous* t, then a learned two-layer MLP. What this
    replaces was a lookup into a fixed sinusoidal table at int(t*999): no learnable
    parameters, and quantised to 1000 bins. adaLN is the only path the timestep takes into
    the model, so that left the time conditioning with no capacity at all.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0):
        # The leading 1000x is FLOWER's, and is what maps t in [0,1] onto the frequency
        # range a diffusion timestep index would have covered. Kept verbatim.
        half = dim // 2
        freqs = 1000 * torch.exp(
            -math.log(max_period)
            * torch.arange(0, half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().reshape(-1)[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor):
        dtype = next(self.parameters()).dtype
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size).to(dtype))


class DDPMSchedule:
    """Diffusion Policy's noise schedule and its two samplers, written out rather than imported.

    `diffusers` is installed in the Evo1 env but cannot be imported there (it requires
    peft>=0.17.0 and the LoRA path pins 0.10.0), so the ~40 lines Diffusion Policy actually
    uses from DDPMScheduler / DDIMScheduler are reproduced here, and
    tests/test_action_objective.py checks them against diffusers' formulas.

    Diffusion Policy's scheduler config (diffusion_policy/config/task/*.yaml):
        num_train_timesteps 100, beta_schedule squaredcos_cap_v2, clip_sample True,
        prediction_type epsilon; DDIM with set_alpha_to_one True, steps_offset 0.
    Timestep spacing is diffusers' default "leading": stride = K // S, timesteps
    [(S-1)*stride, ..., stride, 0], prev = t - stride, and abar_prev = 1 once prev < 0.

    Holds plain tensors, not buffers: the schedule is a function of the config, so it must
    not appear in the state_dict (a flow and a ddpm checkpoint have identical keys).
    """

    def __init__(self, num_train_steps: int, beta_schedule: str = "squaredcos_cap_v2"):
        K = int(num_train_steps)
        if K <= 0:
            raise ValueError(f"ddpm_train_steps must be positive, got {K}")
        if beta_schedule == "squaredcos_cap_v2":
            # diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar, max_beta=0.999
            def alpha_bar(t):
                return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            betas = [min(1.0 - alpha_bar((i + 1) / K) / alpha_bar(i / K), 0.999) for i in range(K)]
            betas = torch.tensor(betas, dtype=torch.float64)
        elif beta_schedule == "linear":
            betas = torch.linspace(1e-4, 2e-2, K, dtype=torch.float64)
        else:
            raise ValueError(f"unknown ddpm_beta_schedule {beta_schedule!r}")
        self.num_train_steps = K
        self.beta_schedule = beta_schedule
        self.betas = betas
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Forward process x_k = sqrt(abar_k) x0 + sqrt(1 - abar_k) eps, k a (B,) long tensor."""
        abar = self.alphas_cumprod.to(x0.device)[k].to(x0.dtype)
        while abar.dim() < x0.dim():
            abar = abar.unsqueeze(-1)
        return abar.sqrt() * x0 + (1.0 - abar).sqrt() * noise

    def inference_timesteps(self, num_inference_steps: int):
        S = int(num_inference_steps)
        if not 1 <= S <= self.num_train_steps:
            raise ValueError(f"num_inference_timesteps must be in [1, {self.num_train_steps}], got {S}")
        stride = self.num_train_steps // S
        return [i * stride for i in range(S)][::-1], stride

    def _abar(self, t: int, stride: int):
        abar_t = self.alphas_cumprod[t].item()
        prev = t - stride
        abar_prev = self.alphas_cumprod[prev].item() if prev >= 0 else 1.0
        return abar_t, abar_prev

    @staticmethod
    def _x0_eps(x_t, out, abar_t, prediction):
        if prediction == "epsilon":
            eps = out
            x0 = (x_t - math.sqrt(1.0 - abar_t) * eps) / math.sqrt(abar_t)
        elif prediction == "sample":
            x0 = out
            eps = (x_t - math.sqrt(abar_t) * x0) / math.sqrt(1.0 - abar_t)
        else:
            raise ValueError(f"unknown ddpm_prediction {prediction!r}")
        return x0, eps

    def ddim_step(self, x_t, out, t: int, stride: int, eta: float = 0.0, clip: bool = True,
                  prediction: str = "epsilon"):
        """diffusers DDIMScheduler.step with use_clipped_model_output=False (Diffusion
        Policy's call): x0 is clipped, eps is not recomputed from the clipped x0."""
        abar_t, abar_prev = self._abar(t, stride)
        x0, eps = self._x0_eps(x_t, out, abar_t, prediction)
        if clip:
            x0 = x0.clamp(-1.0, 1.0)
        sigma = eta * math.sqrt((1.0 - abar_prev) / (1.0 - abar_t)) * math.sqrt(1.0 - abar_t / abar_prev)
        x_prev = math.sqrt(abar_prev) * x0 + math.sqrt(max(1.0 - abar_prev - sigma ** 2, 0.0)) * eps
        if eta > 0.0:
            x_prev = x_prev + sigma * torch.randn_like(x_t)
        return x_prev

    def ddpm_step(self, x_t, out, t: int, stride: int, clip: bool = True, prediction: str = "epsilon"):
        """diffusers DDPMScheduler.step, variance_type fixed_small (its default)."""
        abar_t, abar_prev = self._abar(t, stride)
        alpha_t = abar_t / abar_prev
        beta_t = 1.0 - alpha_t
        x0, _ = self._x0_eps(x_t, out, abar_t, prediction)
        if clip:
            x0 = x0.clamp(-1.0, 1.0)
        coeff_x0 = math.sqrt(abar_prev) * beta_t / (1.0 - abar_t)
        coeff_xt = math.sqrt(alpha_t) * (1.0 - abar_prev) / (1.0 - abar_t)
        x_prev = coeff_x0 * x0 + coeff_xt * x_t
        if t > 0:
            variance = max((1.0 - abar_prev) / (1.0 - abar_t) * beta_t, 1e-20)
            x_prev = x_prev + math.sqrt(variance) * torch.randn_like(x_t)
        return x_prev


def training_target(config, actions_gt: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """What the network is regressed onto, given the noise the head drew. Flow matching: the
    velocity actions_gt - noise. DDPM: the noise itself (epsilon) or the clean action
    (sample). The loss code in train.py is objective-agnostic because of this."""
    if getattr(config, "action_objective", "flow") == "ddpm":
        if getattr(config, "ddpm_prediction", "epsilon") == "epsilon":
            return noise
        return actions_gt
    return actions_gt - noise


class SwiGlu(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        n_hidden = _find_multiple(int(2 * hidden_dim / 3), 256)
        self.fc1 = nn.Linear(dim, n_hidden, bias=False)
        self.fc2 = nn.Linear(dim, n_hidden, bias=False)
        self.proj = nn.Linear(n_hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.proj(self.dropout(F.silu(self.fc1(x)) * self.fc2(x)))


class FlowerSelfAttention(nn.Module):
    """Self-attention over the action chunk, with per-head q/k RMSNorm and RoPE."""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0,
                 use_rope: bool = True, max_seq_len: int = 64, rope_theta: float = 1000.0,
                 is_causal: bool = False):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads, self.head_dim = n_heads, dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.attn_p = dropout
        self.resid_dropout = nn.Dropout(dropout)
        # FLOWER runs is_causal=True (conf/model/flower.yaml use_causal_attention). Off here
        # on purpose -- flow matching denoises the whole chunk in one pass, so masking token k
        # from the tokens after it withholds part of the trajectory being jointly denoised.
        self.is_causal = is_causal
        self.use_rope = use_rope
        if use_rope:
            cos, sin = precompute_freqs_1d(self.head_dim, max_seq_len, theta=rope_theta)
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.use_rope:
            q, k = apply_rotary_pos_emb(q, k, self.cos, self.sin)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_p if self.training else 0.0,
            is_causal=self.is_causal)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.resid_dropout(self.proj(out))


class FlowerCrossAttention(nn.Module):
    """Action tokens attend over the VLM context.

    FLOWER RMSNorms both the queries and the keys (transformers.py:280, applied at :431).
    q/k norm is a pair: normalising only the queries leaves the logit scale riding on the raw
    magnitude of the VLM features, which is the very thing cond_norm/cond_linear exists to
    control. `key_padding_mask` is True at positions to ignore.
    """

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0, k_norm: bool = True):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads, self.head_dim = n_heads, dim // n_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if k_norm else None
        self.attn_p = dropout
        self.resid_dropout = nn.Dropout(dropout)
        # Diagnostics only. F.scaled_dot_product_attention does not return weights, so with
        # capture_attention on they are recomputed alongside it -- the OUTPUT still comes from
        # SDPA, so a capturing run is numerically identical to a scoring run. (The bottleneck
        # module's capture is not: there it flips need_weights, which changes the kernel.)
        # Accumulated as a running sum over every call between resets, i.e. over the denoising
        # steps of one action chunk, since one map per chunk is what is readable.
        self.capture_attention = False
        self._attn_sum = None
        self._attn_calls = 0

    def reset_attention(self):
        self._attn_sum = None
        self._attn_calls = 0

    def attention_mean(self):
        """[B, T, S] head-averaged attention, meaned over the calls since the last reset."""
        if self._attn_sum is None or self._attn_calls == 0:
            return None
        return self._attn_sum / self._attn_calls

    def forward(self, x, context, key_padding_mask=None):
        B, T, C = x.shape
        S = context.shape[1]
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = ~key_padding_mask[:, None, None, :].expand(B, self.n_heads, T, S)
        if self.capture_attention:
            with torch.no_grad():
                logits = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(self.head_dim)
                if key_padding_mask is not None:
                    # A row that is entirely masked would softmax to NaN; SDPA has the same
                    # hazard and the caller guarantees at least the state token is real.
                    logits = logits.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
                w = logits.softmax(dim=-1).mean(dim=1).cpu()      # [B, T, S], mean over heads
                self._attn_sum = w if self._attn_sum is None else self._attn_sum + w
                self._attn_calls += 1
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.attn_p if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.resid_dropout(self.proj(out))


class FlowerGatedCrossAttention(nn.Module):
    """Action-query-adapter ablation: two independent K/V sources feeding one joint softmax.

    Query is `x`. `query_context` (InternVL3's own action-query hidden states -- a block of
    zero-init learnable tokens run through its 14 layers of causal self-attention alongside
    the vision/text tokens -- plus the state token) is always attended to: it is the branch
    VLA-Adapter calls the "adapter"/already-fused source. `raw_context` (the VLM's raw
    image-patch hidden states, restricted to real, unmasked positions via
    raw_key_padding_mask) is attended to through a per-block learnable gate: its attention
    *logits* are scaled by tanh(gate) before joining query_context's logits in ONE softmax --
    not summed as two independent attentions computed afterwards. At gate=0 the raw branch's
    logits are all equal (0), so softmax gives it only a small, content-independent share of
    the mass and the block's output is dominated by query_context; training can grow the gate
    if the raw signal turns out to help.

    query_context always has at least one valid position (the query tokens and the state
    token are never padded), so the concatenated softmax always has at least one finite logit
    even if every raw_context position is masked (e.g. all camera views occluded -- possible
    only at eval, never in training data). No NaN, no special-cased fallback needed.
    """

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0, k_norm: bool = True,
                 gate_init: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads, self.head_dim = n_heads, dim // n_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_query = nn.Linear(dim, dim, bias=False)
        self.v_query = nn.Linear(dim, dim, bias=False)
        self.k_raw = nn.Linear(dim, dim, bias=False)
        self.v_raw = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm_query = nn.RMSNorm(self.head_dim, eps=1e-6) if k_norm else None
        self.k_norm_raw = nn.RMSNorm(self.head_dim, eps=1e-6) if k_norm else None
        self.attn_p = dropout
        self.resid_dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def _split_heads(self, t, B, L):
        return t.reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, raw_context, query_context, raw_key_padding_mask=None):
        B, T, C = x.shape
        S_raw, S_query = raw_context.shape[1], query_context.shape[1]

        q = self.q_norm(self._split_heads(self.q_proj(x), B, T))

        k_raw = self._split_heads(self.k_raw(raw_context), B, S_raw)
        v_raw = self._split_heads(self.v_raw(raw_context), B, S_raw)
        if self.k_norm_raw is not None:
            k_raw = self.k_norm_raw(k_raw)

        k_query = self._split_heads(self.k_query(query_context), B, S_query)
        v_query = self._split_heads(self.v_query(query_context), B, S_query)
        if self.k_norm_query is not None:
            k_query = self.k_norm_query(k_query)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores_raw = torch.matmul(q, k_raw.transpose(-2, -1)) * scale * torch.tanh(self.gate)
        if raw_key_padding_mask is not None:
            mask = raw_key_padding_mask[:, None, None, :].expand(B, self.n_heads, T, S_raw)
            scores_raw = scores_raw.masked_fill(mask, float("-inf"))
        scores_query = torch.matmul(q, k_query.transpose(-2, -1)) * scale

        weights = torch.softmax(torch.cat([scores_query, scores_raw], dim=-1), dim=-1)
        weights = F.dropout(weights, p=self.attn_p, training=self.training)
        out = torch.matmul(weights, torch.cat([v_query, v_raw], dim=2))
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.resid_dropout(self.proj(out))


class FlowerFlowBlock(nn.Module):
    """FLOWER's FlowBlock: adaLN-modulated self-attn -> cross-attn -> SwiGlu, all gated."""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0, lora_dim: int = 256,
                 max_seq_len: int = 64, rope_theta: float = 1000.0,
                 is_causal: bool = False, cross_k_norm: bool = True,
                 full_rank_adaln: bool = False, use_action_query_adapter: bool = False,
                 action_query_gate_init: float = 0.0):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.norm3 = nn.RMSNorm(dim, eps=1e-6)
        self.self_attn = FlowerSelfAttention(dim, n_heads, dropout,
                                             max_seq_len=max_seq_len, rope_theta=rope_theta,
                                             is_causal=is_causal)
        # Off (default): the unchanged single-source FlowerCrossAttention, byte-for-bit the
        # same module this block used before this ablation existed. On: the dual-source
        # gated version above.
        self.use_action_query_adapter = bool(use_action_query_adapter)
        if self.use_action_query_adapter:
            self.cross_attn = FlowerGatedCrossAttention(dim, n_heads, dropout, k_norm=cross_k_norm,
                                                         gate_init=action_query_gate_init)
        else:
            self.cross_attn = FlowerCrossAttention(dim, n_heads, dropout, k_norm=cross_k_norm)
        self.mlp = SwiGlu(dim, dropout=dropout)
        # full_rank_adaln=True is the "no parameter-saving" ablation: this block's own
        # adaLN becomes an ordinary independent dim->6*dim projection instead of the
        # lora_dim bottleneck FLOWER pairs with the shared SharedAdaLNController
        # (FlowmatchingActionHead.global_adaln) to keep the per-layer cost down.
        if full_rank_adaln:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim, 6 * dim),
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim, lora_dim),
                nn.Linear(lora_dim, 6 * dim),
            )
        up = self.adaLN_modulation[-1]
        with torch.no_grad():
            up.weight.zero_()
            up.bias.zero_()
            up.bias[2 * dim:3 * dim].fill_(1.0)   # gate_msa = 1
            up.bias[5 * dim:6 * dim].fill_(1.0)   # gate_mlp = 1

    def forward(self, x, context, cond, context_key_padding_mask=None, global_adaln=None,
                query_context=None):
        """`cond` is FLOWER's global_cond: the mask-weighted mean of the VLM context plus the
        timestep embedding (flower.py:1140-1145), not the timestep alone. `global_adaln` is the
        shared controller's 6 signals, added to this block's own (transformers.py:600).
        `query_context` is only used when use_action_query_adapter=True: `context` is then the
        gated raw branch (InternVL3's image-patch hidden states) and `query_context` the
        always-on branch (its action-query hidden states + state)."""
        signals = self.adaLN_modulation(cond).chunk(6, dim=1)
        if global_adaln is not None:
            signals = [signals[i] + global_adaln[i] for i in range(6)]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = signals

        x = x + gate_msa.unsqueeze(1) * self.self_attn(modulate(self.norm1(x), shift_msa, scale_msa))
        if self.use_action_query_adapter:
            x = x + self.cross_attn(self.norm2(x), context, query_context,
                                     raw_key_padding_mask=context_key_padding_mask)
        else:
            x = x + self.cross_attn(self.norm2(x), context, key_padding_mask=context_key_padding_mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x


class FlowmatchingActionHead(nn.Module):

    def __init__(self, config=None,
                 embed_dim: int = 896, 
                 hidden_dim: int = 1024,
                 action_dim: int = 16*7,
                 horizon: int = 16,
                 per_action_dim: int = 7,
                 num_heads: int = 8,
                 num_layers: int = 8,
                 dropout: float = 0.0,
                 num_inference_timesteps: int = 20,
                 num_categories: int = 1):
        super().__init__()

        if config is not None:
      
            embed_dim = getattr(config, "embed_dim", embed_dim)
            hidden_dim = getattr(config, "hidden_dim", hidden_dim)
            action_dim = getattr(config, "action_dim", action_dim)
            horizon = getattr(config, "horizon", horizon)
            num_heads = getattr(config, "num_heads", num_heads)
            num_layers = getattr(config, "num_layers", num_layers)
            dropout = getattr(config, "dropout", dropout)
            num_inference_timesteps = getattr(config, "num_inference_timesteps", num_inference_timesteps)
            num_categories = getattr(config, "num_categories", num_categories)
            self.config = config
        else:
            from config import EvoConfig
            self.config = EvoConfig(
                embed_dim=embed_dim, hidden_dim=hidden_dim,
                action_horizon=horizon, per_action_dim=per_action_dim,
                num_heads=num_heads, num_layers=num_layers,
                dropout=dropout, num_inference_timesteps=num_inference_timesteps,
                num_categories=num_categories
            )
        logger.info("FlowmatchingActionHead num_inference_timesteps=%s", num_inference_timesteps)
        # See EvoConfig.action_objective. The schedule is derived from the config and holds no
        # parameters, so the state_dict is identical under both objectives.
        self.action_objective = getattr(self.config, "action_objective", "flow")
        if self.action_objective == "ddpm":
            self.ddpm = DDPMSchedule(getattr(self.config, "ddpm_train_steps", 100),
                                     getattr(self.config, "ddpm_beta_schedule", "squaredcos_cap_v2"))
            logger.info("FlowmatchingActionHead action_objective=ddpm K=%d schedule=%s prediction=%s "
                        "sampler=%s eta=%s", self.ddpm.num_train_steps, self.ddpm.beta_schedule,
                        getattr(self.config, "ddpm_prediction", "epsilon"),
                        getattr(self.config, "ddpm_sampler", "ddim"), getattr(self.config, "ddim_eta", 0.0))
        elif self.action_objective == "flow":
            self.ddpm = None
        else:
            raise ValueError(f"unknown action_objective {self.action_objective!r}")
        self.embed_dim = embed_dim
        self.horizon = horizon
        self.per_action_dim = getattr(self.config, "per_action_dim", per_action_dim)
        self.action_dim = getattr(self.config, "action_dim", action_dim)


        # FLOWER's TimestepEmbedder, or the original fixed lookup table. See EvoConfig.
        if getattr(self.config, "action_time_embed", "mlp") == "table":
            self.t_embedder = None
            self.time_pos_enc = SinusoidalPositionalEncoding(embed_dim, max_len=1000)
        else:
            self.t_embedder = TimestepEmbedder(embed_dim)
            self.time_pos_enc = None

        block_kind = getattr(self.config, "action_block", "flower")
        # Action-query adapter (see EvoConfig.use_action_query_adapter). Only meaningful for
        # "flower" blocks; False leaves every FlowerFlowBlock byte-for-bit unchanged.
        self.use_action_query_adapter = getattr(self.config, "use_action_query_adapter", False)
        action_query_gate_init = getattr(self.config, "action_query_raw_gate_init", 0.0)
        if block_kind == "flower":
            # action_global_adaln=False ablates FLOWER's whole parameter-saving scheme for
            # adaLN (the shared controller below, plus every block's lora_dim bottleneck):
            # each block gets an ordinary independent full-rank adaLN instead.
            full_rank_adaln = not getattr(self.config, "action_global_adaln", True)
            self.transformer_blocks = nn.ModuleList([
                FlowerFlowBlock(dim=embed_dim, n_heads=num_heads, dropout=dropout,
                                lora_dim=getattr(self.config, "action_adaln_dim", 256),
                                max_seq_len=max(self.horizon, 8),
                                rope_theta=getattr(self.config, "action_rope_theta", 1000.0),
                                is_causal=getattr(self.config, "action_causal_attn", False),
                                cross_k_norm=getattr(self.config, "action_cross_k_norm", True),
                                full_rank_adaln=full_rank_adaln,
                                use_action_query_adapter=self.use_action_query_adapter,
                                action_query_gate_init=action_query_gate_init)
                for _ in range(num_layers)
            ])
            # FLOWER's SharedAdaLNController (transformers.py:673): one extra set of
            # modulation signals produced once from global_cond and added to every block's
            # own. Zero-initialised there and here, so it contributes nothing until trained
            # and the identity-modulation start is preserved.
            # FLOWER instantiates one per action type (flower.py:857, action_type_adaln);
            # CALVIN is a single embodiment, so one controller is the whole of it.
            if getattr(self.config, "action_global_adaln", True):
                self.global_adaln = nn.Sequential(
                    nn.SiLU(), nn.Linear(embed_dim, 6 * embed_dim, bias=False))
                nn.init.zeros_(self.global_adaln[-1].weight)
            else:
                self.global_adaln = None
        else:
            self.global_adaln = None
            use_self_attn = getattr(self.config, "action_self_attn", True)
            self.transformer_blocks = nn.ModuleList([
                BasicTransformerBlock(embed_dim=embed_dim, num_heads=num_heads,
                                       hidden_dim=embed_dim*4, dropout=dropout,
                                       use_self_attn=use_self_attn)
                for _ in range(num_layers)
            ])
       
        # FLOWER normalises and re-projects the VLM features before its DiT reads them
        # (flower/models/flower.py:813-815 cond_norm/cond_linear, applied at :1096). Evo-1
        # passed the LLM's hidden states through untouched. The dimensions already match
        # here (embed_dim 896 == InternVL3's LLM hidden size), so cond_linear is square and
        # is not needed for shape -- what the pair buys is a scale the action head controls,
        # instead of inheriting whatever the LLM's own final RMSNorm was calibrated for,
        # which was chosen for language modelling and not for this cross-attention's
        # query-key products.
        self.use_adaln_cond = getattr(self.config, "action_adaln_cond", False)
        self.use_cond_proj = getattr(self.config, "action_cond_proj", True)
        if self.use_cond_proj:
            self.cond_norm = nn.RMSNorm(embed_dim)
            self.cond_linear = nn.Linear(embed_dim, embed_dim, bias=False)
            # Identity init: at step 0 the pair reduces to cond_norm alone, so the only
            # change from the previous behaviour is the rescale, not a random re-basis.
            with torch.no_grad():
                self.cond_linear.weight.copy_(torch.eye(embed_dim))

        self.norm_out = nn.LayerNorm(embed_dim)
        # FLOWER decodes each action token on its own (flower.py:1688: action_decoders is an
        # nn.Linear(dit_dim, action_dim) applied over [B, T, D]). "pooled" is the original
        # path: flatten the chunk, squeeze it through one embed_dim vector, expand that back
        # into the whole chunk -- a single bottleneck placed right after self-attention and
        # RoPE have built per-step representations.
        if getattr(self.config, "action_decode", "per_token") == "pooled":
            self.seq_pool_proj = nn.Linear(self.horizon * self.embed_dim, self.embed_dim)
            self.mlp_head = CategorySpecificMLP(input_dim=embed_dim, hidden_dim=hidden_dim,
                                                output_dim=action_dim, num_categories=num_categories)
        else:
            self.seq_pool_proj = None
            self.mlp_head = CategorySpecificMLP(input_dim=embed_dim, hidden_dim=hidden_dim,
                                                output_dim=self.per_action_dim,
                                                num_categories=num_categories)

        self.state_encoder = None
        if hasattr(self.config, "state_dim") and self.config.state_dim is not None:
       
            state_hidden = getattr(self.config, "state_hidden_dim", embed_dim)
        
            self.state_encoder = CategorySpecificMLP(input_dim=self.config.state_dim,
                                                    hidden_dim=state_hidden,
                                                    output_dim=embed_dim,
                                                    num_categories=num_categories)

        self.action_encoder = None
        if horizon > 1:
          
            per_action_dim = getattr(self.config, "per_action_dim", None)
            if per_action_dim is None:
            
                per_action_dim = action_dim // horizon if action_dim % horizon == 0 else action_dim
            self.action_encoder = MultiEmbodimentActionEncoder(action_dim=per_action_dim,
                                                               embed_dim=embed_dim,
                                                               hidden_dim=embed_dim,  
                                                               horizon=horizon,
                                                               num_categories=num_categories,
                                                               use_pos_embed=getattr(
                                                                   self.config, "action_token_pos_embed", False))
            self.single_action_proj = None
        else:
            self.action_encoder = None
            self.single_action_proj = nn.Linear(self.per_action_dim, self.embed_dim)


    def _project_actions(self, action_seq: torch.Tensor, embodiment_id: torch.LongTensor) -> torch.Tensor:
        if self.horizon > 1 and self.action_encoder is not None:
            return self.action_encoder(action_seq, embodiment_id)
        if self.single_action_proj is None:
            raise RuntimeError("single_action_proj is not initialized for horizon <= 1.")
        return self.single_action_proj(action_seq)

    def _expand_action_mask(self, action_mask: torch.Tensor, batch_size: int, per_action_dim: int, device, dtype):
        if action_mask is None:
            raise ValueError("action_mask must be provided for flow matching inference.")

        if action_mask.dim() == 2:
            expected_last_dim = self.horizon * per_action_dim
            if action_mask.shape == (batch_size, expected_last_dim):
                expanded_mask = action_mask.reshape(batch_size, self.horizon, per_action_dim)
            elif action_mask.shape == (batch_size, per_action_dim):
                expanded_mask = action_mask.unsqueeze(1).expand(batch_size, self.horizon, per_action_dim)
            else:
                raise ValueError(
                    f"Expected action_mask shape {(batch_size, expected_last_dim)} or {(batch_size, per_action_dim)}, got {tuple(action_mask.shape)}"
                )
        elif action_mask.dim() == 3:
            expected_shape = (batch_size, self.horizon, per_action_dim)
            if tuple(action_mask.shape) != expected_shape:
                raise ValueError(f"Expected action_mask shape {expected_shape}, got {tuple(action_mask.shape)}")
            expanded_mask = action_mask
        else:
            raise ValueError(f"Unsupported action_mask rank: {action_mask.dim()}")

        return expanded_mask.to(device=device, dtype=dtype)

    def _global_cond(self, context_tokens, time_emb, ctx_pad):
        """FLOWER's global_cond and shared adaLN signals (flower.py:1133-1154).

        FLOWER ships use_adaln_cond=false, so global_cond is the timestep embedding alone and
        the VLM reaches the action tokens only through cross-attention. action_adaln_cond=True
        takes flower.py:1133's other branch and adds the mask-weighted mean of the context, so
        that <pad> and masked-out views do not dilute the summary -- the same reason the
        cross-attention masks them. Returns (cond, global_adaln_signals or None).
        """
        cond = time_emb
        if self.use_adaln_cond:
            if ctx_pad is not None:
                w = (~ctx_pad).to(context_tokens.dtype).unsqueeze(-1)
                vlm_token = (context_tokens * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
            else:
                vlm_token = context_tokens.mean(dim=1)
            cond = vlm_token + time_emb.to(vlm_token.dtype)
        if self.global_adaln is None:
            return cond, None
        return cond, self.global_adaln(cond).chunk(6, dim=-1)

    def _time_embedding(self, t: torch.Tensor, dtype):
        """FLOWER's t_emb (flower.py:1092). Its other two terms, the frequency and proprio
        embeddings, are zero here -- this model has no FreqEmbedder and every FLOWER run it is
        measured against sets use_proprio=False -- so stateless_norm of the timestep MLP is the
        whole of it. The `table` branch is the original fixed lookup at int(t*999)."""
        if self.t_embedder is not None:
            return stateless_norm(self.t_embedder(t)).to(dtype)
        time_index = (t * 999).long().clamp_(0, 999)
        return self.time_pos_enc(1000)[:, time_index, :].squeeze(0).to(dtype)

    def _decode(self, x, embodiment_id, B):
        """Velocity out of the block stack. `per_token` applies one decoder to each action
        token, as FLOWER does; `pooled` is the original single-bottleneck path."""
        if self.seq_pool_proj is None:
            H = x.shape[1]
            cat_ids = embodiment_id.reshape(B, 1).expand(B, H).reshape(B * H)
            out = self.mlp_head(x.reshape(B * H, self.embed_dim), cat_ids)
            return out.reshape(B, -1)
        if self.horizon > 1:
            x_pooled = self.seq_pool_proj(x.reshape(B, -1))
        else:
            x_pooled = x.squeeze(1)
        return self.mlp_head(x_pooled, embodiment_id)

    @staticmethod
    def _extend_context_mask(context_mask, context_tokens, appended: int):
        """context_mask covers fused_tokens; context_tokens may have `appended` extra
        real tokens (the state embedding). Grow the mask so the two line up, and flip to
        nn.MultiheadAttention's convention (True = ignore)."""
        if context_mask is None:
            return None
        keep = context_mask.to(torch.bool)
        if appended:
            keep = torch.cat(
                [keep, keep.new_ones(keep.shape[0], appended)], dim=1)
        assert keep.shape[1] == context_tokens.shape[1], (
            f"context mask width {keep.shape[1]} != context length {context_tokens.shape[1]}")
        return ~keep

    def forward(self, fused_tokens: torch.Tensor, state: torch.Tensor = None,
                actions_gt: torch.Tensor = None, embodiment_id: torch.LongTensor = None,
                state_mask: torch.Tensor = None, action_mask: torch.Tensor = None,
                context_mask: torch.Tensor = None, action_query_tokens: torch.Tensor = None,
                action_query_img_mask: torch.Tensor = None):

        if actions_gt is None:
            return self.get_action(fused_tokens, state=state, embodiment_id=embodiment_id,
                                   action_mask=action_mask, context_mask=context_mask,
                                   action_query_tokens=action_query_tokens,
                                   action_query_img_mask=action_query_img_mask)
        B = fused_tokens.size(0)
        device = fused_tokens.device

        if embodiment_id is None:
            embodiment_id = torch.zeros(B, dtype=torch.long, device=device)

        # state_emb is appended AFTER the projection, matching FLOWER's ordering for its
        # register tokens (flower.py:1104-1111): cond_norm is a per-token RMSNorm, so
        # normalising a differently-scaled non-VLM token alongside the VLM ones would erase
        # exactly the scale that token carries.
        # fused_tokens can arrive as float32 while the module's parameters are bf16 -- the
        # original code handled that with a `.to(dtype=target_dtype)` further down, which is
        # after this projection. Cast here so cond_norm/cond_linear see a matching dtype.
        context_tokens = fused_tokens.to(dtype=self.dtype)
        query_tokens = None
        if self.use_action_query_adapter:
            assert action_query_tokens is not None, (
                "use_action_query_adapter=True requires action_query_tokens (from "
                "InternVL3Embedder.get_fused_image_text_embedding_from_tensor_images)")
            query_tokens = action_query_tokens.to(dtype=self.dtype)

        if self.use_cond_proj:
            if self.use_action_query_adapter:
                # Same cond_norm/cond_linear for both branches: they come from the same
                # InternVL3 final-layer output space, just different sequence positions, so
                # there is no reason to rescale them with two different learned projections.
                S = context_tokens.shape[1]
                combined = torch.cat([context_tokens, query_tokens], dim=1)
                combined = self.cond_linear(self.cond_norm(combined))
                context_tokens, query_tokens = combined[:, :S, :], combined[:, S:, :]
            else:
                context_tokens = self.cond_linear(self.cond_norm(context_tokens))
        if state is not None and self.state_encoder is not None:


            state_emb = self.state_encoder(state, embodiment_id)
            state_emb = state_emb.unsqueeze(1)

            # State always joins the always-on branch (query_tokens), never the gated raw
            # branch: it must never be dampened by a gate that might settle on a small value.
            if self.use_action_query_adapter:
                query_tokens = torch.cat([query_tokens, state_emb], dim=1)
            else:
                context_tokens = torch.cat([context_tokens, state_emb], dim=1)

        # FLOWER's sampling_type 'uniform' (flower.py:1005-1007) is a stratified sampler: one
        # shared random offset plus evenly spaced strata, so a batch covers [0,1) evenly
        # instead of clumping. Evo-1's Beta(2,2) concentrated t near 0.5 and clamped the ends
        # away, which trains the middle of the path well and its two ends least -- and the
        # ends are exactly where few-step Euler sampling takes its largest strides.
        ddpm_k = None
        if self.action_objective == "ddpm":
            # Diffusion Policy's forward process: an integer step k ~ U{0..K-1}. It is embedded
            # as k/K so the same TimestepEmbedder serves both objectives; its internal 1000x
            # turns that back into an integer-like frequency argument (1000/K per step).
            ddpm_k = torch.randint(0, self.ddpm.num_train_steps, (B,), device=device)
            t = (ddpm_k.to(torch.float32) / self.ddpm.num_train_steps).to(dtype=self.dtype)
        elif getattr(self.config, "flow_time_sampling", "stratified_uniform") == "beta":
            t = torch.distributions.Beta(2, 2).sample((B,)).clamp(0.02, 0.98).to(device).to(dtype=self.dtype)
        else:
            eps = 1e-5
            t = ((torch.rand(1, device=device) + torch.arange(B, device=device) / B) % (1 - eps)
                 ).to(dtype=self.dtype)

        
                    
        time_emb = self._time_embedding(t, context_tokens.dtype)
    
        action_shape = actions_gt.shape[1]  
    

        actions_gt_seq = actions_gt  


        # Gaussian, matching FLOWER (flower.py:1013 torch.randn_like). Evo-1 drew uniform
        # [-1,1]: same support as a bounds-normalised action, but a distribution with hard
        # corners, and the transport path from it to the data is not the one rectified flow
        # is derived for. Inference must draw from the same distribution -- see get_action.
        if self.action_objective != "ddpm" and getattr(self.config, "flow_noise", "gaussian") == "uniform":
            noise = torch.rand_like(actions_gt) * 2 - 1
        else:
            noise = torch.randn_like(actions_gt)

        if action_mask is not None:
            action_mask = action_mask.to(dtype=noise.dtype, device=noise.device)
            assert action_mask.shape == noise.shape, f"action_mask shape {action_mask.shape} != noise shape {noise.shape}"
            noise = noise * action_mask


        if self.horizon > 1:
            noise_seq = noise.view(B, self.horizon, self.per_action_dim)
            
        else:
            noise_seq = noise.unsqueeze(1)

        if self.horizon > 1:
            t_broadcast = t.view(B, 1, 1)
        else:
            t_broadcast = t.view(B, 1)
        if self.action_objective == "ddpm":
            action_intermediate_seq = self.ddpm.add_noise(actions_gt_seq, noise_seq, ddpm_k)
        else:
            action_intermediate_seq = (1 - t_broadcast) * noise_seq + t_broadcast * actions_gt_seq  

        action_tokens = self._project_actions(action_intermediate_seq, embodiment_id)
        
        target_dtype = self.dtype
        action_tokens = action_tokens.to(dtype=target_dtype)
        context_tokens = context_tokens.to(dtype=target_dtype)
        time_emb = time_emb.to(dtype=target_dtype)

        ctx_pad = self._extend_context_mask(context_mask, context_tokens,
                                            context_tokens.shape[1] - fused_tokens.shape[1])
        if self.use_action_query_adapter and action_query_img_mask is not None:
            # Restrict the gated raw branch to actual image-patch positions (VLA-Adapter's
            # task_hidden_states), not the full image+text sequence: query_tokens already
            # carries the language instruction (InternVL3 fused it in via causal
            # self-attention before the action head ever sees it), so keeping text here too
            # would just duplicate that signal without adding anything new.
            not_img = ~action_query_img_mask.to(torch.bool)
            ctx_pad = not_img if ctx_pad is None else (ctx_pad | not_img)
        cond, gadaln = self._global_cond(context_tokens, time_emb, ctx_pad)
        x = action_tokens
        for block in self.transformer_blocks:
            if self.use_action_query_adapter:
                x = block(x, context_tokens, cond, context_key_padding_mask=ctx_pad,
                          global_adaln=gadaln, query_context=query_tokens)
            else:
                x = block(x, context_tokens, cond, context_key_padding_mask=ctx_pad,
                          global_adaln=gadaln)

        x = self.norm_out(x)
        pred_velocity = self._decode(x, embodiment_id, B)

        return pred_velocity, noise

    def get_action(self, fused_tokens: torch.Tensor, state: torch.Tensor = None, embodiment_id: torch.LongTensor = None, action_mask: torch.Tensor = None, context_mask: torch.Tensor = None,
                   action_query_tokens: torch.Tensor = None, action_query_img_mask: torch.Tensor = None):

        B = fused_tokens.size(0)
        device = fused_tokens.device
        if embodiment_id is None:
            embodiment_id = torch.zeros(B, dtype=torch.long, device=device)

        # fused_tokens can arrive as float32 while the module's parameters are bf16 -- the
        # original code handled that with a `.to(dtype=target_dtype)` further down, which is
        # after this projection. Cast here so cond_norm/cond_linear see a matching dtype.
        context_tokens = fused_tokens.to(dtype=self.dtype)
        query_tokens = None
        if self.use_action_query_adapter:
            assert action_query_tokens is not None, (
                "use_action_query_adapter=True requires action_query_tokens (from "
                "InternVL3Embedder.get_fused_image_text_embedding_from_tensor_images)")
            query_tokens = action_query_tokens.to(dtype=self.dtype)

        if self.use_cond_proj:
            if self.use_action_query_adapter:
                S = context_tokens.shape[1]
                combined = torch.cat([context_tokens, query_tokens], dim=1)
                combined = self.cond_linear(self.cond_norm(combined))
                context_tokens, query_tokens = combined[:, :S, :], combined[:, S:, :]
            else:
                context_tokens = self.cond_linear(self.cond_norm(context_tokens))
        if state is not None and self.state_encoder is not None:

            state_emb = self.state_encoder(state, embodiment_id).unsqueeze(1)
            if self.use_action_query_adapter:
                query_tokens = torch.cat([query_tokens, state_emb], dim=1)
            else:
                context_tokens = torch.cat([context_tokens, state_emb], dim=1)

        action_dim_total = getattr(self.config, "action_dim", None)
        if action_dim_total is None:
          
            action_dim_total = self.action_dim
       
        if self.horizon > 1:
            per_action_dim = getattr(self.config, "per_action_dim", action_dim_total // self.horizon)
        else:
            per_action_dim = action_dim_total

        # Same distribution the velocity field was trained to transport from. Drawing uniform
        # here while training on Gaussian would start the integration off-distribution.
        if self.action_objective != "ddpm" and getattr(self.config, "flow_noise", "gaussian") == "uniform":
            action = (torch.rand(B, action_dim_total, device=device, dtype=context_tokens.dtype) * 2 - 1)
        else:
            action = torch.randn(B, action_dim_total, device=device, dtype=context_tokens.dtype)

        if self.horizon > 1:
            action_seq = action.view(B, self.horizon, per_action_dim)
        else:
            action_seq = action.view(B, 1, per_action_dim)

        action_mask = self._expand_action_mask(
            action_mask, batch_size=B, per_action_dim=per_action_dim, device=action_seq.device, dtype=action_seq.dtype
        )
        action_seq = action_seq * action_mask
        
        target_dtype = self.dtype
        context_tokens = context_tokens.to(dtype=target_dtype)
        
        ctx_pad = self._extend_context_mask(context_mask, context_tokens,
                                            context_tokens.shape[1] - fused_tokens.shape[1])
        if self.use_action_query_adapter and action_query_img_mask is not None:
            not_img = ~action_query_img_mask.to(torch.bool)
            ctx_pad = not_img if ctx_pad is None else (ctx_pad | not_img)
        N = int(getattr(self.config, "num_inference_timesteps", 32))
        if N <= 0:
            raise ValueError(f"num_inference_timesteps must be positive, got {N}")

        def run_net(seq, t_value):
            # One pass of the network at scalar time t_value: the velocity under flow
            # matching, epsilon (or x0) under DDPM. The block stack is the same either way.
            time_emb = self._time_embedding(
                torch.full((B,), t_value, device=device, dtype=torch.float32), context_tokens.dtype)
            action_tokens = self._project_actions(seq, embodiment_id)
            action_tokens = action_tokens.to(dtype=target_dtype)
            time_emb = time_emb.to(dtype=target_dtype)
            cond, gadaln = self._global_cond(context_tokens, time_emb, ctx_pad)
            x = action_tokens
            for block in self.transformer_blocks:
                if self.use_action_query_adapter:
                    x = block(x, context_tokens, cond, context_key_padding_mask=ctx_pad,
                              global_adaln=gadaln, query_context=query_tokens)
                else:
                    x = block(x, context_tokens, cond, context_key_padding_mask=ctx_pad,
                              global_adaln=gadaln)
            x = self.norm_out(x)
            return self._decode(x, embodiment_id, B)

        if self.action_objective == "ddpm":
            # Diffusion Policy's reverse process (DDPMSchedule). The sampler state is float32:
            # the 100-step ancestral chain would otherwise accumulate 100 bf16 roundings, a
            # source of jitter that belongs to the precision, not to the method. The flow
            # loop below stays in the model dtype so every existing flow eval reproduces.
            sampler = getattr(self.config, "ddpm_sampler", "ddim")
            eta = float(getattr(self.config, "ddim_eta", 0.0))
            clip = bool(getattr(self.config, "action_clamp_output", True))
            prediction = getattr(self.config, "ddpm_prediction", "epsilon")
            timesteps, stride = self.ddpm.inference_timesteps(N)
            net_dtype = action_seq.dtype
            x = action_seq.to(torch.float32)
            mask32 = action_mask.to(torch.float32)
            for t_int in timesteps:
                x = x * mask32
                out = run_net(x.to(net_dtype), t_int / self.ddpm.num_train_steps)
                out = out.to(torch.float32).view_as(x)
                if sampler == "ddim":
                    x = self.ddpm.ddim_step(x, out, t_int, stride, eta=eta, clip=clip, prediction=prediction)
                elif sampler == "ddpm":
                    x = self.ddpm.ddpm_step(x, out, t_int, stride, clip=clip, prediction=prediction)
                else:
                    raise ValueError(f"unknown ddpm_sampler {sampler!r}")
            action_seq = x.to(net_dtype)
        else:
            dt = 1.0 / N
            for i in range(N):
                t = i / N
                action_seq = action_seq * action_mask
                pred = run_net(action_seq, t)
                action = action + dt * pred
                if self.horizon > 1:
                    action_seq = action.view(B, self.horizon, per_action_dim)
                else:
                    action_seq = action.view(B, 1, per_action_dim)

        action_seq = action_seq * action_mask
        # FLOWER's sample_actions ends on z.clamp(-1, 1) (flower.py:1063). Under bounds
        # normalisation the actions live in [-1, 1] by construction, so an overshoot here
        # denormalises to a command outside the range the data ever contained.
        if getattr(self.config, "action_clamp_output", True):
            action_seq = action_seq.clamp(-1, 1)
        return action_seq.reshape(B, -1)

    def set_capture_attention(self, on: bool):
        """Turn cross-attention capture on for every block. Diagnostics only; see
        FlowerCrossAttention.capture_attention for why it does not change the output."""

        for block in self.transformer_blocks:
            ca = getattr(block, "cross_attn", None)
            if hasattr(ca, "capture_attention"):
                ca.capture_attention = bool(on)

    def reset_attention(self):
        for block in self.transformer_blocks:
            ca = getattr(block, "cross_attn", None)
            if hasattr(ca, "reset_attention"):
                ca.reset_attention()

    @torch.no_grad()
    def attention_summary(self):
        """Per-block [T, S] maps and their mean, for one action chunk.

        T is the action horizon -- one row per predicted step, kept separate because the early
        and late steps of a chunk need not look at the same thing -- and S indexes the context
        the head cross-attends over: the VLM's own sequence positions followed by the state
        token.
        """

        per_block = []
        for block in self.transformer_blocks:
            ca = getattr(block, "cross_attn", None)
            m = ca.attention_mean() if hasattr(ca, "attention_mean") else None
            if m is not None:
                per_block.append(m[0].float().numpy())
        if not per_block:
            return None
        import numpy as _np
        stacked = _np.stack(per_block)                      # [n_blocks, T, S]
        return {"per_block": stacked, "mean": stacked.mean(axis=0)}

    @property
    def device(self):
      
        return next(self.parameters()).device

    @property
    def dtype(self):
        
        return next(self.parameters()).dtype

