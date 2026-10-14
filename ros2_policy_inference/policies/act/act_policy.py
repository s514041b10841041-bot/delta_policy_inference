"""ACT backend: wraps a lerobot ACT checkpoint trained on the fan-grasp dataset.

Same data, action convention and observation format as the diffusion backend
(policies/diffusion/diffusion_policy.py): the checkpoint predicts 7-D per-step deltas
[dx, dy, dz, drx, dry, drz, gripper], position in metres in the WORLD frame, rotation as an
axis-angle in radians applied by LEFT multiplication (R_next = R(drot) @ R_now), gripper 0/1 with
1 = open. One action corresponds to one camera period, so this class declares inference_rate_hz
(30 Hz, from policies/act/config.yaml), which the node uses in place of config/params.yaml's rate.

Chunking: lerobot's ACTPolicy keeps its own action queue. select_action() runs the transformer only
when the queue is empty (predicting chunk_size = 50 actions and keeping n_action_steps = 10 of
them) and otherwise pops the next queued action, so this wrapper calls select_action() exactly once
per predict() and does no chunking of its own. With temporal ensembling enabled in config.yaml the
model runs on every tick instead and the overlapping chunks are averaged.

The checkpoint was trained with lerobot 0.6.2 (conda env lerobot312), the version docker/Dockerfile
installs. Its training run used all 97 episodes, 375k steps at batch 32, and the same
normalization as the diffusion checkpoint (QUANTILES action, MIN_MAX state, MEAN_STD image).

Image handling: identical to the diffusion backend - obs["rgb"] is HWC uint8 RGB at
image_size x image_size, produced by the same ros_bridge/preprocessing functions the recorder used.
ACT sees the full frame (no crop at train or test time).
"""
from __future__ import annotations

import os

import numpy as np
import yaml

from ..base_policy import ACTION_DIM, PolicyModel

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_CONFIG_PATH) as _f:
    _CFG = yaml.safe_load(_f)


class ACTPolicyModel(PolicyModel):
    """lerobot ACTPolicy checkpoint, driven one action per predict() call."""

    def __init__(self, checkpoint_path: str = _CFG["checkpoint_dir"], device: str = _CFG["device"],
                 task: str = _CFG["task"], warmup_iters: int = _CFG["warmup_iters"],
                 n_action_steps: int | None = _CFG.get("n_action_steps"),
                 temporal_ensemble_coeff: float | None = _CFG.get("temporal_ensemble_coeff"),
                 logger=None):
        # Imported lazily: torch/lerobot are only needed when this policy is selected.
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        self._torch = torch
        self.inference_rate_hz = float(_CFG["inference_rate_hz"])
        self.task = task
        self.device = device
        self._log = logger

        cfg = PreTrainedConfig.from_pretrained(checkpoint_path)
        cfg.device = device
        # The checkpoint holds every backbone tensor, so the ImageNet init ACTPolicy would otherwise
        # download (torchvision, ~45 MB, on every fresh container) is overwritten anyway - and the
        # download fails offline. Skipping it loads bit-identical weights.
        cfg.pretrained_backbone_weights = None
        # Inference-only overrides. They must be set on the config BEFORE the policy is built:
        # ACTPolicy creates its temporal ensembler and action queue from the config in __init__.
        if temporal_ensemble_coeff is not None:
            cfg.temporal_ensemble_coeff = float(temporal_ensemble_coeff)
            cfg.n_action_steps = 1                 # lerobot requires 1 with ensembling
        elif n_action_steps is not None:
            cfg.n_action_steps = int(n_action_steps)
        if not 1 <= cfg.n_action_steps <= cfg.chunk_size:
            raise ValueError(f"n_action_steps={cfg.n_action_steps} must be in [1, chunk_size={cfg.chunk_size}]")
        self.cfg = cfg
        self.checkpoint_dir = checkpoint_path   # read by the node for diagnostics (state_range.py)
        self.policy = ACTPolicy.from_pretrained(checkpoint_path, config=cfg).to(device).eval()
        # Normalization statistics saved with the checkpoint; loading them from `pretrained_path`
        # rather than recomputing keeps deployment identical to training, and the post-processor
        # maps the network output back into metres/radians.
        self.pre, self.post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=checkpoint_path)

        # The RGB camera: observation.images.wrist when present (depth checkpoints also have
        # observation.images.wrist_depth, filled by _extra_images()), else the first image feature.
        self._image_key = ("observation.images.wrist" if "observation.images.wrist" in cfg.image_features
                           else next(iter(cfg.image_features), "observation.images.wrist"))
        self._state_dim = cfg.input_features["observation.state"].shape[0]
        mode = (f"temporal ensembling (coeff {cfg.temporal_ensemble_coeff})"
                if cfg.temporal_ensemble_coeff is not None else f"n_action_steps={cfg.n_action_steps}")
        self._info(
            f"loaded {cfg.type} from {checkpoint_path} on {device}: "
            f"chunk_size={cfg.chunk_size}, {mode}, image key {self._image_key}"
        )
        self._warmup(warmup_iters)

    # ---------- PolicyModel interface ----------

    def reset(self):
        """Clear the policy's action queue (or temporal ensemble) between episodes.

        Without this the first actions of a new episode are leftovers from the previous episode's
        chunk.
        """
        self.policy.reset()

    def predict(self, obs: dict) -> np.ndarray:
        batch = self._build_batch(obs)
        with self._torch.no_grad():
            action = self.post(self.policy.select_action(self.pre(batch)))
        action = np.asarray(action.squeeze(0).float().cpu().numpy(), dtype=np.float64)
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"policy returned {action.shape}, expected ({ACTION_DIM},)")
        return action

    # ---------- internals ----------

    def _build_batch(self, obs: dict) -> dict:
        """obs (ROS node format) -> the batch the checkpoint expects.

        Only two of the node's observation keys are used: the policy was trained on the wrist image
        and a 7-D state of [joint_0..joint_5, gripper]. joint_vel, ee_pos, ee_quat and ft_wrench are
        deliberately ignored, and depth is used only by depth checkpoints (_extra_images).
        """
        torch = self._torch
        rgb = obs["rgb"]                                   # HWC uint8 RGB, already image_size square
        if rgb.dtype != np.uint8:
            raise ValueError(f"obs['rgb'] must be uint8, got {rgb.dtype}")
        img = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device)
        img = img.permute(2, 0, 1).float().div_(255.0).unsqueeze(0)   # HWC uint8 -> 1CHW float [0,1]

        joint_pos = np.asarray(obs["joint_pos"], dtype=np.float32).reshape(-1)
        state = np.concatenate([joint_pos, [np.float32(obs["gripper_state"])]]).astype(np.float32)
        if state.shape[0] != self._state_dim:
            raise ValueError(
                f"state has {state.shape[0]} dims (joint_pos {joint_pos.shape[0]} + gripper), "
                f"checkpoint expects {self._state_dim}"
            )
        return {
            self._image_key: img,
            **self._extra_images(obs),
            "observation.state": torch.from_numpy(state).to(self.device).unsqueeze(0),
            # Carried through for parity with training batches; ACT does not condition on language.
            "task": [self.task],
        }

    def _extra_images(self, obs: dict) -> dict:
        """More image inputs for the batch; none for RGB-only checkpoints. policies/act_depth
        overrides it to add the encoded depth image."""
        return {}

    def _warmup(self, iters: int):
        """Spend the first-call cost (CUDA context, cuDNN autotune, lazy kernels) before the robot
        moves. reset() before each call makes every dummy call run the full model."""
        if iters <= 0:
            return
        torch = self._torch
        h, w = (self.cfg.image_features[self._image_key].shape[1:3] if self.cfg.image_features else (256, 256))
        dummy = {
            "rgb": np.zeros((h, w, 3), dtype=np.uint8),
            "joint_pos": np.zeros(self._state_dim - 1, dtype=np.float32),
            "gripper_state": np.float32(1.0),
            "depth": np.zeros((h, w), dtype=np.uint16),   # only read by depth checkpoints
        }
        with torch.no_grad():
            for _ in range(iters):
                self.policy.reset()
                self.post(self.policy.select_action(self.pre(self._build_batch(dummy))))
        self.policy.reset()          # drop the dummy chunk from the queue
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._info(f"warmed up with {iters} dummy inferences")

    def _info(self, msg: str):
        if self._log is not None:
            self._log.info(msg)
        else:
            print(f"[act_policy] {msg}")
