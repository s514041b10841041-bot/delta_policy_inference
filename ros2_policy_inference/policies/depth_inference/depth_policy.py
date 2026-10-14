"""Diffusion Policy with the wrist depth image as a second camera (policy_name depth_inference).

Everything except the depth input is policies/diffusion/diffusion_policy.py: same checkpoint
format, 30 Hz rate, warm-up and chunking. The checkpoint expects observation.images.wrist_depth,
built here from obs["depth"] exactly as Robot_Friday's preprocessing_depth.py built the training
data:

    uint16 millimetres (aligned_depth_to_color, resized nearest-neighbour to image_size, the same
    as ros2_data_collector records it)  ->  0 = no reading, 255 = depth_min_m ... 1 = depth_max_m
    (farther clipped to 1)  ->  repeated in 3 channels  ->  float [0, 1], like the RGB image.
"""
from __future__ import annotations

import os

import numpy as np
import yaml

from ..diffusion.diffusion_policy import DiffusionPolicyModel

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_CONFIG_PATH) as _f:
    _CFG = yaml.safe_load(_f)

DEPTH_KEY = "observation.images.wrist_depth"


def encode_depth(depth, d_min: float, d_max: float) -> np.ndarray:
    """(H, W) depth -> (H, W, 3) uint8. uint16 is millimetres (training format); float is metres
    (a 32FC1 topic). 0, NaN and inf are "no reading"."""
    depth = np.asarray(depth)
    if np.issubdtype(depth.dtype, np.floating):
        d = depth.astype(np.float32)
        invalid = ~np.isfinite(d) | (d <= 0)
        d = np.where(invalid, 0.0, d)
    else:
        invalid = depth == 0
        d = depth.astype(np.float32) / 1000.0
    x = (np.clip(d, d_min, d_max) - d_min) / (d_max - d_min)          # 0 near .. 1 far
    enc = np.rint(255.0 - 254.0 * x).astype(np.uint8)                 # 255 near .. 1 far
    enc[invalid] = 0
    return np.repeat(enc[..., None], 3, axis=-1)


class DepthInputMixin:
    """Adds observation.images.wrist_depth to a policy's batch via the _extra_images() hook of
    DiffusionPolicyModel / ACTPolicyModel. Used by DiffusionDepthPolicyModel and
    policies/act_depth's ACTDepthPolicyModel."""

    def _setup_depth(self, checkpoint_path: str, d_min: float, d_max: float):
        """Call before the base __init__ (its warm-up already calls _extra_images)."""
        self._d_min, self._d_max = float(d_min), float(d_max)
        if not self._d_max > self._d_min >= 0:
            raise ValueError(f"need 0 <= depth_min_m < depth_max_m, got {self._d_min}, {self._d_max}")
        # The checkpoint's own record of the encoding, if it was copied along (it is not part of
        # pretrained_model/ by default), is checked so a mismatch fails loudly at startup.
        meta = os.path.join(checkpoint_path, "depth_encoding.json")
        if os.path.exists(meta):
            import json
            with open(meta) as f:
                m = json.load(f)
            if (m["depth_min_m"], m["depth_max_m"]) != (self._d_min, self._d_max):
                raise ValueError(f"config.yaml depth range {self._d_min}-{self._d_max} m differs from "
                                 f"{meta}: {m['depth_min_m']}-{m['depth_max_m']} m")

    def _check_depth_checkpoint(self, checkpoint_path: str, rgb_policy_name: str):
        """Call after the base __init__ (needs self.cfg)."""
        if DEPTH_KEY not in self.cfg.image_features:
            raise ValueError(f"{checkpoint_path} has no {DEPTH_KEY} input; use policy_name {rgb_policy_name}")
        self._info(f"depth input {DEPTH_KEY}: {self._d_min:g}-{self._d_max:g} m -> 255-1, no reading -> 0")

    def _extra_images(self, obs: dict) -> dict:
        if obs.get("depth") is None:
            raise ValueError("obs['depth'] missing: this policy needs the aligned depth topic")
        enc = encode_depth(obs["depth"], self._d_min, self._d_max)
        if DEPTH_KEY in self.cfg.image_features:
            want = tuple(self.cfg.image_features[DEPTH_KEY].shape[1:])
            if enc.shape[:2] != want:
                raise ValueError(f"depth is {enc.shape[:2]}, checkpoint expects {want} (image_size)")
        torch = self._torch
        img = torch.from_numpy(enc).to(self.device)
        return {DEPTH_KEY: img.permute(2, 0, 1).float().div_(255.0).unsqueeze(0)}


class DiffusionDepthPolicyModel(DepthInputMixin, DiffusionPolicyModel):
    def __init__(self, checkpoint_path: str = _CFG["checkpoint_dir"], use_rtc: bool = False, **kwargs):
        self._setup_depth(checkpoint_path, _CFG["depth_min_m"], _CFG["depth_max_m"])
        super().__init__(checkpoint_path=checkpoint_path, use_rtc=use_rtc, **kwargs)
        self._check_depth_checkpoint(checkpoint_path, "diffusion")
