"""ACT with the wrist depth image as a second camera (policy_name depth_act).

Everything except the depth input is policies/act/act_policy.py: same checkpoint format, 30 Hz
rate, warm-up, chunking and temporal ensembling (policies/act/config.yaml). The checkpoint expects
observation.images.wrist_depth, encoded from obs["depth"] the same way as the depth diffusion
policy (policies/depth_inference/depth_policy.py, encode_depth).
"""
from __future__ import annotations

import os

import yaml

from ..act.act_policy import ACTPolicyModel
from ..depth_inference.depth_policy import DepthInputMixin

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_CONFIG_PATH) as _f:
    _CFG = yaml.safe_load(_f)


class ACTDepthPolicyModel(DepthInputMixin, ACTPolicyModel):
    def __init__(self, checkpoint_path: str = _CFG["checkpoint_dir"], **kwargs):
        self._setup_depth(checkpoint_path, _CFG["depth_min_m"], _CFG["depth_max_m"])
        super().__init__(checkpoint_path=checkpoint_path, **kwargs)
        self._check_depth_checkpoint(checkpoint_path, "act")
