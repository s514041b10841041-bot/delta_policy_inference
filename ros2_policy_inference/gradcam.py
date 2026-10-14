"""
Grad-CAM for the lerobot policies (act, depth_act, diffusion*, depth_inference*): which parts of each
camera image drive the predicted actions.

Where: the last ResNet stage (layer4) of the vision backbone, once per camera. ACT shares one
backbone across cameras; the diffusion checkpoints have one encoder per camera and see the image
after a 240x240 center crop, so their maps cover only the crop (the 8 px border stays empty).

What is explained: the predicted action chunk's distance from "do nothing", in the model's
normalized action space,

    y = 1/2 * sum_{t in steps, d in dims} (a[t, d] - r[d])^2

with r the normalized no-op action (zero motion, gripper 0.5). Its gradient is
sum (a - r) * da, so the map shows the evidence for the action the policy is actually predicting,
in whatever direction it points. dims follows `target`: motion (dx..drz, the default), position,
rotation, gripper or all; steps are the actions executed from this observation (ACT: the first
`horizon`, default the checkpoint's n_action_steps; diffusion: positions n_obs_steps-1 onwards).
Standard Grad-CAM then: channel weights = spatial mean of dy/dA, map = ReLU(sum_k w_k A_k),
upsampled to the image.

All cameras share one scale (the largest value over the cameras), so a weak camera looks weak; the
caption gives each camera's share of the total map.

Runs on its own thread (GradCAMWorker) and CUDA stream at a low rate and never touches the policy's
action queue, ensembler or RTC state: ACT calls the network directly, diffusion samples with its own
copy of the noise scheduler from a fixed seed (so successive maps differ only by the observation).
The forward hooks only act on that thread, and torch.autograd.grad computes no parameter gradients.
"""
from __future__ import annotations

import copy
import threading
import time
from collections import deque

import cv2
import numpy as np

TARGETS = {
    "motion": [0, 1, 2, 3, 4, 5],
    "position": [0, 1, 2],
    "rotation": [3, 4, 5],
    "gripper": [6],
    "all": [0, 1, 2, 3, 4, 5, 6],
}
_NOOP_RAW = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]   # zero motion; gripper halfway between open and closed
OBS_IMAGES = "observation.images"
OBS_STATE = "observation.state"


class GradCAM:
    """Grad-CAM maps for an ACTPolicyModel / DiffusionPolicyModel (or their depth variants)."""

    def __init__(self, policy, target: str = "motion", horizon: int = 0, seed: int = 0):
        import torch

        if target not in TARGETS:
            raise ValueError(f"gradcam target '{target}' unknown (options: {', '.join(TARGETS)})")
        cfg = getattr(policy, "cfg", None)
        kind = getattr(cfg, "type", None)
        if kind not in ("act", "diffusion"):
            raise ValueError("Grad-CAM needs a lerobot act or diffusion policy (act, depth_act, diffusion*, "
                             "depth_inference*)")
        if not cfg.image_features:
            raise ValueError("the checkpoint has no image inputs")
        self._torch = torch
        self.policy, self.cfg, self.kind = policy, cfg, kind
        self.cameras = list(cfg.image_features)
        self.target = target
        self._dims = TARGETS[target]
        self.device = policy.device
        self._seed = seed
        self._tls = threading.local()
        self._captured = []

        if kind == "act":
            self.net = policy.policy.model
            self._layers = [self.net.backbone.layer4]
            start, steps = 0, horizon or cfg.n_action_steps
            # Temporal ensembling forces n_action_steps to 1; explain the chunk the checkpoint
            # was trained to execute instead.
            if not horizon and cfg.temporal_ensemble_coeff is not None:
                steps = 10
            self._steps = slice(start, min(start + steps, cfg.chunk_size))
            self.n_frames = 1
        else:
            self.net = policy.policy.diffusion
            enc = self.net.rgb_encoder
            encoders = list(enc) if cfg.use_separate_rgb_encoder_per_camera else [enc]
            self._layers = [e.backbone[-1] for e in encoders]
            self._encoder = encoders[0]
            self._scheduler = copy.deepcopy(self.net.noise_scheduler)   # set_timesteps mutates it
            self._scheduler.set_timesteps(self.net.num_inference_steps)
            start = cfg.n_obs_steps - 1
            self._steps = slice(start, min(start + (horizon or cfg.n_action_steps), cfg.horizon))
            self.n_frames = cfg.n_obs_steps
        for layer in self._layers:
            layer.register_forward_hook(self._hook)

        # Normalized no-op action: the post-processor's unnormalization is affine per dimension.
        with torch.no_grad():
            a_dim = cfg.output_features["action"].shape[0]
            zero = self._unnormalize(torch.zeros(1, a_dim, device=self.device))
            one = self._unnormalize(torch.ones(1, a_dim, device=self.device))
            noop = torch.tensor(_NOOP_RAW[:a_dim], dtype=torch.float32, device=self.device)
            span = (one - zero).reshape(-1)
            span = torch.where(span.abs() > 1e-12, span, torch.ones_like(span))
            self._noop = (noop - zero.reshape(-1)) / span

    # ---------- public ----------

    def compute(self, frames: list[dict]) -> dict:
        """frames: the last n_frames node observations, oldest first (fewer are padded with the
        oldest, as lerobot's queues do). Returns
          cams     {camera key: (H, W) float32, 0..1 on the shared scale}
          images   {camera key: (H, W, 3) uint8, what the camera fed the policy}
          share    {camera key: fraction of the summed map}
          actions  (steps, A) unnormalized actions that were explained
          ms       compute time"""
        torch = self._torch
        t0 = time.perf_counter()
        frames = list(frames)[-self.n_frames:]
        frames = [frames[0]] * (self.n_frames - len(frames)) + frames
        raw = [self.policy._build_batch(f) for f in frames]
        latest_imgs = {k: _to_uint8(raw[-1][k]) for k in self.cameras}
        self._captured = []
        self._tls.on = True
        try:
            with torch.enable_grad():
                actions = self._forward([self.policy.pre(dict(r)) for r in raw])
                sel = actions[:, self._steps][..., self._dims]
                y = 0.5 * ((sel - self._noop[self._dims]) ** 2).sum()
                grads = torch.autograd.grad(y, self._captured)
        finally:
            self._tls.on = False
        feats = self._per_camera(self._captured)
        grads = self._per_camera(grads)

        cams = {}
        for k in self.cameras:
            a, g = feats[k].detach(), grads[k]
            cam = torch.relu((g.mean(dim=(-2, -1), keepdim=True) * a).sum(dim=-3))[0]      # (h, w)
            cams[k] = self._to_image_frame(cam, latest_imgs[k].shape[:2])
        total = sum(float(c.sum()) for c in cams.values())
        peak = max(float(c.max()) for c in cams.values())
        with torch.no_grad():
            explained = self._unnormalize(actions[0, self._steps].detach()).float().cpu().numpy()
        return {
            "cams": {k: (c / peak if peak > 0 else c).astype(np.float32) for k, c in cams.items()},
            "images": latest_imgs,
            "share": {k: (float(c.sum()) / total if total > 0 else 0.0) for k, c in cams.items()},
            "actions": explained,
            "ms": 1e3 * (time.perf_counter() - t0),
        }

    # ---------- internals ----------

    def _hook(self, _module, _inp, out):
        """On the Grad-CAM thread only: swap layer4's output for a leaf that collects dy/dA."""
        if not getattr(self._tls, "on", False):
            return None
        leaf = out.detach().requires_grad_(True)
        self._captured.append(leaf)
        return leaf

    def _forward(self, batches: list[dict]):
        """Preprocessed per-frame batches -> normalized action chunk (1, T, A), with grad."""
        torch = self._torch
        if self.kind == "act":
            b = dict(batches[-1])
            b[OBS_IMAGES] = [b[k] for k in self.cameras]
            return self.net(b)[0]
        batch = {
            OBS_STATE: torch.stack([b[OBS_STATE] for b in batches], dim=1),
            OBS_IMAGES: torch.stack([torch.stack([b[k] for k in self.cameras], dim=-4) for b in batches], dim=1),
        }
        global_cond = self.net._prepare_global_conditioning(batch)
        gen = torch.Generator(device=self.device).manual_seed(self._seed)
        a_dim = self.cfg.output_features["action"].shape[0]
        sample = torch.randn((1, self.cfg.horizon, a_dim), device=self.device, generator=gen)
        for t in self._scheduler.timesteps:          # lerobot's conditional_sample, own scheduler
            out = self.net.unet(sample, torch.full((1,), t, dtype=torch.long, device=self.device),
                                global_cond=global_cond)
            sample = self._scheduler.step(out, t, sample, generator=gen).prev_sample
        return sample

    def _per_camera(self, tensors) -> dict:
        """Hook outputs in call order -> {camera: (1, C, h, w)} for the latest frame."""
        if self.kind == "act" or len(tensors) == len(self.cameras):
            # ACT: one call per camera; diffusion with one encoder per camera: (n_frames, C, h, w) each.
            return {k: t[-1:] for k, t in zip(self.cameras, tensors)}
        # Diffusion with a shared encoder: one call over (frame, camera) pairs.
        t = tensors[0].reshape(self.n_frames, len(self.cameras), *tensors[0].shape[1:])
        return {k: t[-1, i:i + 1] for i, k in enumerate(self.cameras)}

    def _to_image_frame(self, cam, hw) -> np.ndarray:
        """(h, w) map on the network input -> (H, W) on the camera image (undoing the center crop)."""
        torch = self._torch
        H, W = hw
        if self.kind == "diffusion" and getattr(self._encoder, "resize", None) is not None:
            H0, W0 = self._encoder.resize.size if hasattr(self._encoder.resize, "size") else (H, W)
        else:
            H0, W0 = H, W
        ch, cw = (self._encoder.center_crop.size if self.kind == "diffusion" and self._encoder.do_crop
                  else (H0, W0))
        up = torch.nn.functional.interpolate(cam[None, None].float(), size=(ch, cw), mode="bilinear",
                                             align_corners=False)[0, 0].cpu().numpy()
        full = np.zeros((H0, W0), np.float32)
        top, left = (H0 - ch) // 2, (W0 - cw) // 2
        full[top:top + ch, left:left + cw] = up
        return full if (H0, W0) == (H, W) else cv2.resize(full, (W, H), interpolation=cv2.INTER_LINEAR)

    def _unnormalize(self, a):
        out = self.policy.post(a)
        return (out if isinstance(out, self._torch.Tensor) else self._torch.as_tensor(out)).to(a.device).float()


def _to_uint8(img) -> np.ndarray:
    """1CHW float [0, 1] tensor -> HWC uint8."""
    return (img[0].permute(1, 2, 0).clamp(0, 1) * 255).round().byte().cpu().numpy()


def render(result: dict, target: str, scale: int = 2, alpha: float = 0.45) -> np.ndarray:
    """Grad-CAM result -> RGB uint8 image: per camera, the input with the map overlaid (jet: blue =
    no influence, red = most), side by side, captioned with the camera's share."""
    panels = []
    for key, cam in result["cams"].items():
        img = result["images"][key]
        heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(np.clip(cam, 0, 1) * 255), cv2.COLORMAP_JET),
                            cv2.COLOR_BGR2RGB)
        over = np.uint8(np.clip((1 - alpha) * img + alpha * heat, 0, 255))
        h, w = over.shape[:2]
        over = cv2.resize(over, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)
        name = key.rsplit(".", 1)[-1]
        _caption(over, f"{name}  {100 * result['share'][key]:.0f}%", 0)
        panels.append(over)
    out = np.hstack(panels)
    a = result["actions"]
    move = a[:, :3].sum(axis=0) * 1000
    _caption(out, f"target {target} | next {len(a)} steps: d=({move[0]:+.1f},{move[1]:+.1f},{move[2]:+.1f}) mm"
                  f" gripper {a[0, -1]:.2f} | {result['ms']:.0f} ms", 1)
    return out


def _caption(img, text, row):
    """row 0 = top line; 1 = bottom line, 2 = the line above it, ..."""
    y = 22 if row == 0 else img.shape[0] - 10 - 24 * (row - 1)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


class GradCAMWorker:
    """Runs GradCAM on its own thread at up to rate_hz on the newest observations and hands each
    rendered image to publish(rgb_uint8). submit() is called every control tick and only stores the
    observation, so the control loop never waits for Grad-CAM."""

    def __init__(self, gradcam: GradCAM, publish, rate_hz: float, logger=None):
        self.gc, self._publish, self._period, self._log = gradcam, publish, 1.0 / rate_hz, logger
        self._frames = deque(maxlen=gradcam.n_frames)
        self._lock = threading.Lock()
        self._new = threading.Event()
        self._stop = threading.Event()
        torch = gradcam._torch
        self._stream = torch.cuda.Stream() if str(gradcam.device).startswith("cuda") else None
        self._thread = threading.Thread(target=self._run, name="gradcam", daemon=True)
        self._thread.start()

    def submit(self, obs: dict):
        with self._lock:
            self._frames.append(obs)
        self._new.set()

    def reset(self):
        """Forget older frames (episode boundary), like the policy's own reset."""
        with self._lock:
            self._frames.clear()

    def stop(self):
        self._stop.set()
        self._new.set()
        self._thread.join(timeout=2.0)

    def _run(self):
        torch = self.gc._torch
        errors = 0
        while not self._stop.is_set():
            self._new.wait()
            self._new.clear()
            if self._stop.is_set():
                break
            t0 = time.monotonic()
            with self._lock:
                frames = list(self._frames)
            if not frames:
                continue
            try:
                if self._stream is not None:
                    with torch.cuda.stream(self._stream):
                        res = self.gc.compute(frames)
                else:
                    res = self.gc.compute(frames)
                self._publish(render(res, self.gc.target))
                errors = 0
            except Exception as e:  # diagnostics only, never take the control loop down
                errors += 1
                if self._log is not None and errors <= 3:
                    self._log.error(f"Grad-CAM failed: {e!r}" + (" (further errors not logged)" if errors == 3 else ""))
            self._stop.wait(max(0.0, self._period - (time.monotonic() - t0)))
