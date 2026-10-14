"""Guided Real-Time Chunking (RTC) for lerobot's Diffusion Policy, applied at inference only.

RTC (Black et al. 2025, https://www.physicalintelligence.company/research/real_time_chunking)
generates the next action chunk while the robot is still executing the current one, and treats
the overlap as inpainting: the actions that will already have been executed by the time the new
chunk is ready are frozen to the previous chunk's values, and the following ones are softly pulled
towards it with decaying weights. This removes the jump between chunks without retraining.

lerobot 0.6.2 implements RTC only for flow-matching policies (policies/rtc/modeling_rtc.py). This
module adapts the same guidance to the Diffusion Policy's DDIM sampler (epsilon prediction):

  DDIM state  x_t = a * x0 + s * eps,   a = sqrt(alpha_bar_t), s = sqrt(1 - alpha_bar_t)

is exactly a flow-matching state z = (1 - time) * x0 + time * eps after rescaling,
z = x_t / (a + s) with time = s / (a + s). So each denoising step converts to flow time, applies
lerobot's RTC correction and guidance weight unchanged, and converts back:

  x0_hat   = (x_t - s * eps_hat) / a                       (clean-chunk estimate, with grad)
  corr     = (a + s) * J^T [ W * (prefix - x0_hat) ]       (vector-Jacobian product, d/dz)
  w        = min(beta, lerobot's c(tau) * inv_r2(tau)),  tau = 1 - time
  eps_tilde = eps_hat - (1 - time) * w * corr

which moves the DDIM x0 estimate by + time * w * corr, the same shift RTC applies in flow space.
The DDIM step itself (and clip_sample) is the checkpoint's own scheduler, unchanged.
"""
from __future__ import annotations

import torch
from torch import Tensor

from lerobot.configs import RTCAttentionSchedule
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.modeling_rtc import RTCProcessor


def prefix_weights(frozen: int, overlap: int, horizon: int, schedule: str = "LINEAR") -> Tensor:
    """(horizon,) guidance weights: 1 for the first `frozen` positions, decaying to 0 at `overlap`
    (the last position the previous chunk still covers), 0 after. lerobot's own schedule code."""
    processor = RTCProcessor(RTCConfig(prefix_attention_schedule=RTCAttentionSchedule[schedule]))
    return processor.get_prefix_weights(frozen, overlap, horizon)


def guidance_weight(tau: float, max_guidance_weight: float) -> float:
    """lerobot's RTC guidance weight (modeling_rtc.py denoise_step), clamped to max_guidance_weight."""
    beta = torch.as_tensor(max_guidance_weight, dtype=torch.float64)
    tau_t = torch.as_tensor(tau, dtype=torch.float64)
    sq = (1 - tau_t) ** 2
    inv_r2 = (sq + tau_t ** 2) / sq
    c = torch.nan_to_num((1 - tau_t) / tau_t, posinf=float(beta))
    return float(torch.minimum(torch.nan_to_num(c * inv_r2, posinf=float(beta)), beta))


def guided_conditional_sample(
    diffusion,                          # lerobot DiffusionModel (policy.diffusion)
    global_cond: Tensor,                # (1, global_cond_dim), from _prepare_global_conditioning
    prefix: Tensor | None = None,       # (1, horizon, action_dim) normalized target, or None
    weights: Tensor | None = None,      # (horizon,) prefix weights; ignored without prefix
    max_guidance_weight: float = 10.0,
    noise: Tensor | None = None,
) -> Tensor:
    """DiffusionModel.conditional_sample with RTC prefix guidance. Without a prefix it performs the
    same steps as lerobot's unguided sampler. Returns the full (1, horizon, action_dim) chunk."""
    cfg = diffusion.config
    scheduler = diffusion.noise_scheduler
    unet = diffusion.unet
    device, dtype = global_cond.device, global_cond.dtype
    sample = noise if noise is not None else torch.randn(
        (1, cfg.horizon, cfg.action_feature.shape[0]), dtype=dtype, device=device)
    scheduler.set_timesteps(diffusion.num_inference_steps)
    global_cond = global_cond.detach()
    if prefix is not None:
        w_pos = weights.to(device=device, dtype=dtype).view(1, -1, 1)
        alphas_cumprod = scheduler.alphas_cumprod.to(device=device, dtype=torch.float64)

    for t in scheduler.timesteps:
        t_batch = torch.full((1,), t, dtype=torch.long, device=device)
        if prefix is None:
            with torch.no_grad():
                eps = unet(sample, t_batch, global_cond=global_cond)
        else:
            a = float(alphas_cumprod[t].sqrt())
            s = float((1 - alphas_cumprod[t]).sqrt())
            time = s / (a + s)
            with torch.enable_grad():
                x = sample.detach().requires_grad_(True)
                eps_hat = unet(x, t_batch, global_cond=global_cond)
                x0_hat = (x - s * eps_hat) / a
                err = ((prefix - x0_hat) * w_pos).detach()
                corr = torch.autograd.grad(x0_hat, x, grad_outputs=err)[0] * (a + s)
            w = guidance_weight(1.0 - time, max_guidance_weight)
            eps = (eps_hat - (1.0 - time) * w * corr).detach()
        sample = scheduler.step(eps, t, sample).prev_sample
    return sample
