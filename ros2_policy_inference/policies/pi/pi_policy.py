"""π₀.₅ (pi05) policy: runs a LeRobot pi05 checkpoint, configured by config.yaml."""
import dataclasses
import json
import os
import tempfile
import time
from contextlib import nullcontext

import numpy as np
import yaml
from rclpy.logging import get_logger

from ..base_policy import ACTION_DIM, PolicyModel

try:
    import transformers  # noqa: F401  needed by lerobot's pi05, only installed with lerobot[pi]
except ImportError as exc:
    raise ImportError("pi policy needs transformers — rebuild the image (docker/Dockerfile installs lerobot[pi])") from exc

import torch
from lerobot.configs import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors, prepare_observation_for_inference
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.utils.constants import ACTION, OBS_STATE

# colcon only installs .py files, so fall back to the mounted source tree in the container
_CONFIG_PATH = next(
    (p for p in (
        os.path.join(os.path.dirname(__file__), "config.yaml"),
        "/workspace/src/ros2_policy_inference/ros2_policy_inference/policies/pi/config.yaml",
    ) if os.path.isfile(p)),
    None,
)

_STATE_SEGMENTS = {"joint_pos": 6, "joint_vel": 6, "ee_pos": 3, "ee_quat": 4, "gripper_state": 1, "ft_wrench": 6}

# 20-dim observation.state, in the order ros2_data_collector recorded it
_STATE_KEYS = ("joint_pos", "joint_vel", "ee_pos", "ee_quat", "gripper_state")

# Fields the forked lerobot writes into config.json; stock PI05Config rejects them.
# Safe to drop only while these features are off.
_FORK_FEATURE_FLAGS = ("use_visual_memory", "use_proprioceptive_memory")


class PIPolicyModel(PolicyModel):
    def __init__(self):
        with open(_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}
        self._log = get_logger("pi_policy")

        ckpt_path = cfg["ckpt_path"]
        self._task = cfg["task"]
        self._robot_type = cfg.get("robot_type", "")

        policy_cfg = self._load_config(ckpt_path, cfg)
        self._device = torch.device(policy_cfg.device)
        self._use_amp = policy_cfg.use_amp and self._device.type == "cuda"

        # empty_camera_* are placeholders pi05 fills with blank frames itself
        cams = [k for k in policy_cfg.image_features if "empty_camera" not in k]
        self._rgb_key = cfg.get("rgb_key") or next(k for k in cams if "depth" not in k)
        self._depth_key = cfg.get("depth_key") or next((k for k in cams if "depth" in k), None)
        if self._depth_key:
            with open(cfg["depth_info_path"]) as f:
                info = json.load(f)
            self._depth_min, self._depth_max = float(info["depth_min"]), float(info["depth_max"])  # mm

        state_dim = policy_cfg.input_features[OBS_STATE].shape[0]
        self._state_keys = cfg.get("state_keys") or _STATE_KEYS
        assert sum(_STATE_SEGMENTS[k] for k in self._state_keys) == state_dim, "state_keys don't match checkpoint"

        self._policy = get_policy_class(policy_cfg.type).from_pretrained(ckpt_path, config=policy_cfg)
        self._policy.to(self._device).eval()
        overrides = {"device_processor": {"device": policy_cfg.device}}
        # the fork saves its tokenizer next to the checkpoint
        if os.path.isdir(os.path.join(ckpt_path, "tokenizer")):
            overrides["tokenizer_processor"] = {"tokenizer_name": os.path.join(ckpt_path, "tokenizer")}
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg, pretrained_path=ckpt_path, preprocessor_overrides=overrides
        )

        self._log.info(
            f"pi05 loaded from {ckpt_path}: task='{self._task}', state={list(self._state_keys)}, "
            f"rgb={self._rgb_key}, depth={self._depth_key}, n_action_steps={policy_cfg.n_action_steps}"
        )
        if cfg.get("warmup", True):
            self._warmup(policy_cfg.input_features[self._rgb_key].shape[1:])

    def reset(self):
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()

    def predict(self, obs: dict) -> np.ndarray:
        frame = {self._rgb_key: np.ascontiguousarray(obs["rgb"], dtype=np.uint8), OBS_STATE: self._build_state(obs)}
        if self._depth_key:
            frame[self._depth_key] = self._depth_frame(obs["depth"])
        return self._infer(frame)

    def _load_config(self, ckpt_path, cfg):
        with open(os.path.join(ckpt_path, "config.json")) as f:
            raw = json.load(f)
        known = {f.name for f in dataclasses.fields(PI05Config)} | {"type"}
        unknown = set(raw) - known
        if any(raw.get(k) for k in _FORK_FEATURE_FLAGS):
            raise ValueError(f"{ckpt_path} uses fork-only memory features stock lerobot lacks")
        for k in unknown:
            del raw[k]
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "config.json"), "w") as f:
                json.dump(raw, f)
            policy_cfg = PreTrainedConfig.from_pretrained(tmp)

        policy_cfg.pretrained_path = ckpt_path
        policy_cfg.rtc_config = None  # select_action() refuses RTC
        if cfg.get("device"):
            policy_cfg.device = cfg["device"]
        if cfg.get("n_action_steps"):
            policy_cfg.n_action_steps = cfg["n_action_steps"]
        if cfg.get("num_inference_steps"):
            policy_cfg.num_inference_steps = cfg["num_inference_steps"]
        assert policy_cfg.output_features[ACTION].shape[0] == ACTION_DIM, "checkpoint action dim mismatch"
        return policy_cfg

    def _build_state(self, obs):
        return np.concatenate([np.asarray(obs[k], dtype=np.float32).ravel() for k in self._state_keys])

    def _depth_frame(self, depth):
        """Same encoding as the dataset conversion: clip to [min, max], near=bright,
        valid -> 1..255, invalid (0/NaN) -> 0, grey replicated to 3 channels."""
        raw = np.asarray(depth, dtype=np.float32)
        if np.issubdtype(np.asarray(depth).dtype, np.floating):
            raw = raw * 1000.0  # 32FC1 is metres, the range is mm
        valid = np.isfinite(raw) & (raw > 0)
        x = np.clip((raw - self._depth_min) / (self._depth_max - self._depth_min), 0.0, 1.0)
        gray = np.where(valid, 1.0 + np.round((1.0 - x) * 254.0), 0.0).astype(np.uint8)
        return np.repeat(gray[:, :, None], 3, axis=2)

    def _warmup(self, hw):
        frame = {self._rgb_key: np.zeros((*hw, 3), np.uint8),
                 OBS_STATE: np.zeros(sum(_STATE_SEGMENTS[k] for k in self._state_keys), np.float32)}
        if self._depth_key:
            frame[self._depth_key] = np.zeros((*hw, 3), np.uint8)
        self._infer(frame)
        self.reset()

    def _infer(self, frame):
        started = time.perf_counter()
        autocast = torch.autocast(device_type=self._device.type) if self._use_amp else nullcontext()
        with torch.inference_mode(), autocast:
            batch = prepare_observation_for_inference(frame, self._device, self._task, self._robot_type)
            action = self._postprocessor(self._policy.select_action(self._preprocessor(batch)))
        elapsed_ms = (time.perf_counter() - started) * 1e3
        if elapsed_ms > 20.0:  # a new chunk was denoised (queue pops are ~free)
            self._log.info(f"denoised new chunk in {elapsed_ms:.0f} ms")
        return action.squeeze(0).float().cpu().numpy()
