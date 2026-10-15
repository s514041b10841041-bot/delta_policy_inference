"""Diffusion Policy with the wrist depth image as its ONLY camera (policy_name diffusion_depth_only).

Everything except the inputs is policies/diffusion/diffusion_policy.py: same checkpoint format,
30 Hz rate, warm-up and chunking. The checkpoint's inputs are observation.images.wrist_depth and
observation.state, with no observation.images.wrist: the depth image is encoded exactly as for
policies/depth_inference (encode_depth), and obs["rgb"] is not fed to the model.
"""
from __future__ import annotations

import os

import yaml

from ..depth_inference.depth_policy import DEPTH_KEY, DepthInputMixin
from ..diffusion.diffusion_policy import DiffusionPolicyModel

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_CONFIG_PATH) as _f:
    _CFG = yaml.safe_load(_f)


class DiffusionDepthOnlyPolicyModel(DepthInputMixin, DiffusionPolicyModel):
    def __init__(self, checkpoint_path: str = _CFG["checkpoint_dir"], use_rtc: bool = False, **kwargs):
        self._setup_depth(checkpoint_path, _CFG["depth_min_m"], _CFG["depth_max_m"])
        super().__init__(checkpoint_path=checkpoint_path, use_rtc=use_rtc, **kwargs)
        self._check_depth_checkpoint(checkpoint_path, "diffusion")
        others = [k for k in self.cfg.image_features if k != DEPTH_KEY]
        if others:
            raise ValueError(f"{checkpoint_path} also takes {', '.join(others)}, not depth only; "
                             f"use policy_name depth_inference")
