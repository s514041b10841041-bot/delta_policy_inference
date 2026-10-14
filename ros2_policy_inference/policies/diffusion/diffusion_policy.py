"""Diffusion Policy backend: wraps a lerobot checkpoint trained on the fan-grasp dataset.

The checkpoint predicts 7-D per-step deltas [dx, dy, dz, drx, dry, drz, gripper], where the
position delta is in metres in the WORLD frame, the rotation delta is an axis-angle (rotation
vector) in radians applied by LEFT multiplication (R_next = R(drot) @ R_now), and the gripper is
0/1 with 1 = open. One action corresponds to one camera period, so the node must tick at the rate
the training data was recorded at: this class declares inference_rate_hz (30 Hz, from
policies/diffusion/config.yaml), which the node uses in place of config/params.yaml's rate.

Chunking has two modes, chosen by the policy name (policies/__init__.py):

* policy_name diffusion_rtc - Real-Time Chunking (rtc.py), tuned by config.yaml's rtc block. All model work runs on one background thread; predict()
  only reads the current chunk and never waits on the GPU after the first chunk. Chunk geometry
  follows lerobot: position p of a chunk generated from the observation at tick k is the action for
  tick k + p - 1 (position 0 lies in the past because n_obs_steps = 2). Once n_action_steps - d
  actions of the current chunk have run (d = inference delay in ticks), the next chunk is started
  from the latest observations and guided towards the current chunk's remaining actions: weight 1
  on positions 0..d (executed before it can arrive), decaying to 0 where the current chunk ends.
  When it arrives, execution switches to it at the position matching the current tick, so no
  action is skipped or repeated. Guided sampling costs ~93 ms (vs ~60 ms unguided) on an
  RTX 4070 Ti SUPER, i.e. d = 3 ticks at 30 Hz, absorbed by running ahead of time.
* policy_name diffusion - synchronous: lerobot's own select_action() queues, one call per predict().
  The sampler runs when its queue is empty, so one tick in every n_action_steps takes ~62 ms
  and overruns the 33 ms period, and consecutive chunks are sampled independently.

Image handling note: obs["rgb"] arrives as HWC uint8 RGB at image_size x image_size, produced by
ros_bridge._compressed_msg_to_array (cv2.imdecode + BGR->RGB) and preprocessing.resize_image
(cv2.INTER_AREA). The recorder that produced the training data uses byte-for-byte the same two
functions, duplicated in ros2_data_collector/ros_bridge.py and ros2_data_collector/recorder.py.
Editing either copy silently breaks the match between training and deployment images — change both
or neither.
"""
from __future__ import annotations

import math
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import yaml

from ..base_policy import ACTION_DIM, PolicyModel

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_CONFIG_PATH) as _f:
    _CFG = yaml.safe_load(_f)

_STATS_EVERY = 100          # log an RTC timing summary every this many chunks


class DiffusionPolicyModel(PolicyModel):
    """lerobot DiffusionPolicy checkpoint, driven one action per predict() call."""

    def __init__(self, checkpoint_path: str = _CFG["checkpoint_dir"], device: str = _CFG["device"],
                 task: str = _CFG["task"], warmup_iters: int = _CFG["warmup_iters"],
                 use_rtc: bool = False, rtc: dict | None = None, logger=None):
        # Imported lazily: torch/lerobot are only needed when this policy is selected.
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        from lerobot.policies.factory import make_pre_post_processors

        self._torch = torch
        self.inference_rate_hz = float(_CFG["inference_rate_hz"])
        self.task = task
        self.device = device
        self._log = logger

        cfg = PreTrainedConfig.from_pretrained(checkpoint_path)
        cfg.device = device
        self.cfg = cfg
        self.checkpoint_dir = checkpoint_path   # read by the node for diagnostics (state_range.py)
        self.policy = DiffusionPolicy.from_pretrained(checkpoint_path, config=cfg).to(device).eval()
        # Normalization statistics saved with the checkpoint (QUANTILES for the action, MIN_MAX for
        # the state, MEAN_STD for the image). Loading them from `pretrained_path` rather than
        # recomputing is what keeps deployment identical to training; the post-processor maps the
        # network's [-1, 1] output back into metres/radians.
        self.pre, self.post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=checkpoint_path)

        # The RGB camera: "observation.images.wrist" when present (a depth checkpoint also has
        # observation.images.wrist_depth, filled by _extra_images()), else the first image feature.
        self._image_key = ("observation.images.wrist" if "observation.images.wrist" in cfg.image_features
                           else next(iter(cfg.image_features), "observation.images.wrist"))
        self._state_dim = cfg.input_features["observation.state"].shape[0]

        # rtc overrides config.yaml's rtc settings (tests); use_rtc decides whether RTC runs at all.
        rtc_cfg = dict(_CFG.get("rtc") or {}) if rtc is None else dict(rtc)
        self._rtc = bool(use_rtc)
        if self._rtc:
            from .rtc import guided_conditional_sample, prefix_weights
            self._sample = guided_conditional_sample
            self._prefix_weights = prefix_weights
            self._beta = float(rtc_cfg.get("max_guidance_weight", 10.0))
            self._schedule = str(rtc_cfg.get("prefix_attention_schedule", "LINEAR")).upper()
            fixed = rtc_cfg.get("inference_delay")
            self._fixed_delay = None if fixed is None else int(fixed)
            if self._fixed_delay is not None and not 0 <= self._fixed_delay <= cfg.horizon - 2:
                raise ValueError(f"rtc.inference_delay={fixed} must be in [0, horizon - 2 = {cfg.horizon - 2}]")
            # One worker = all GPU work on one thread, in order; predict() stays CPU-only.
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="diffusion_rtc")
            self._latencies = deque(maxlen=20)      # guided inference wall times (s)
            self._future = None
            self._reset_rtc_state()

        mode = (f"RTC (guided, max_guidance_weight={self._beta}, {self._schedule} prefix weights)"
                if self._rtc else "synchronous chunking")
        self._info(
            f"loaded {cfg.type} from {checkpoint_path} on {device}: "
            f"horizon={cfg.horizon}, n_action_steps={cfg.n_action_steps}, n_obs_steps={cfg.n_obs_steps}, "
            f"{cfg.noise_scheduler_type}/{cfg.num_inference_steps} denoising steps, image key {self._image_key}, "
            f"{mode}"
        )
        self._warmup(warmup_iters)

    # ---------- PolicyModel interface ----------

    def reset(self):
        """Clear observation history and queued actions between episodes.

        Without this the first actions of a new episode are leftovers from the previous episode's
        chunk, and the n_obs_steps history starts with stale frames.
        """
        if self._rtc:
            self._reset_rtc_state()
        else:
            self.policy.reset()

    def predict(self, obs: dict) -> np.ndarray:
        if self._rtc:
            action = self._predict_rtc(obs)
        else:
            batch = self._build_batch(obs)
            with self._torch.no_grad():
                action = self.post(self.policy.select_action(self.pre(batch)))
            action = np.asarray(action.squeeze(0).float().cpu().numpy(), dtype=np.float64)
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"policy returned {action.shape}, expected ({ACTION_DIM},)")
        return action

    # ---------- Real-Time Chunking ----------

    def _reset_rtc_state(self):
        if self._future is not None:
            self._future.result()           # let an in-flight inference finish, then drop it
            self._future = None
        self._hist = deque(maxlen=self.cfg.n_obs_steps)   # raw CPU observations, newest last
        self._tick = -1
        self._chunk_norm = None             # (1, horizon, A) normalized, on the GPU (worker-owned)
        self._chunk_actions = None          # (horizon, A) unnormalized numpy
        self._chunk_tick = None             # tick of the observation the chunk was generated from
        self._stats = {"chunks": 0, "stalls": 0, "late": 0, "max_delay": 0}

    def _predict_rtc(self, obs: dict) -> np.ndarray:
        horizon = self.cfg.horizon
        self._tick += 1
        tick = self._tick
        frame = {
            "rgb": np.array(obs["rgb"], copy=True),
            "joint_pos": np.array(obs["joint_pos"], dtype=np.float32, copy=True),
            "gripper_state": np.float32(obs["gripper_state"]),
        }
        if "depth" in obs:                               # used by depth checkpoints (_extra_images)
            frame["depth"] = np.array(obs["depth"], copy=True)
        if frame["rgb"].dtype != np.uint8:
            raise ValueError(f"obs['rgb'] must be uint8, got {frame['rgb'].dtype}")
        while len(self._hist) < self._hist.maxlen - 1:   # first tick: repeat it, as lerobot does
            self._hist.append(frame)
        self._hist.append(frame)

        if self._future is not None and self._future.done():
            self._adopt()
        if self._chunk_actions is None:                  # first chunk: the only blocking wait
            self._launch(tick)
            self._adopt()
        pos = 1 + tick - self._chunk_tick
        if pos > horizon - 1:
            # The current chunk ran out before the next one arrived (inference slower than the
            # margin): wait for it, and if even that one is too old, start over unguided.
            self._stats["stalls"] += 1
            if self._future is not None:
                self._adopt()
                pos = 1 + tick - self._chunk_tick
            if pos > horizon - 1:
                self._launch(tick)
                self._adopt()
                pos = 1
        if self._future is None:
            delay = self._delay_ticks()
            offset = tick - self._chunk_tick
            if offset >= max(1, self.cfg.n_action_steps - delay):
                self._launch(tick, offset=offset, delay=delay)
        return np.asarray(self._chunk_actions[pos], dtype=np.float64)

    def _delay_ticks(self) -> int:
        if self._fixed_delay is not None:
            return self._fixed_delay
        if not self._latencies:
            return 0
        return min(self.cfg.horizon - 2, math.ceil(max(self._latencies) * self.inference_rate_hz))

    def _launch(self, tick: int, offset: int | None = None, delay: int = 0):
        """Start an inference on the worker from the current observation history; guided towards
        the current chunk when offset is given, unguided otherwise."""
        prev = self._chunk_norm if offset is not None else None
        self._future = self._executor.submit(self._infer, list(self._hist), prev, offset, 1 + delay)
        self._future_tick = tick
        self._future_delay = delay

    def _adopt(self):
        """Switch to the finished (or, blocking, the in-flight) inference's chunk."""
        chunk_norm, chunk_actions, latency, guided = self._future.result()
        self._future = None
        actual = self._tick - self._future_tick       # ticks that passed while it was computed
        self._chunk_norm, self._chunk_actions, self._chunk_tick = chunk_norm, chunk_actions, self._future_tick
        if guided:
            self._latencies.append(latency)
            s = self._stats
            s["chunks"] += 1
            s["max_delay"] = max(s["max_delay"], actual)
            s["late"] += actual > self._future_delay   # switched past the frozen positions
            if s["chunks"] % _STATS_EVERY == 0:
                self._info(
                    f"RTC: {s['chunks']} chunks, guided inference mean "
                    f"{1e3 * np.mean(self._latencies):.0f} ms / max {1e3 * max(self._latencies):.0f} ms "
                    f"(last {len(self._latencies)}), delay {self._delay_ticks()} ticks, "
                    f"max observed {s['max_delay']}, late {s['late']}, stalls {s['stalls']}"
                )

    def _infer(self, frames: list, prev, offset: int | None, frozen: int):
        """Worker thread: observations -> (normalized chunk, unnormalized chunk, seconds, guided)."""
        torch = self._torch
        t0 = time.perf_counter()
        horizon = self.cfg.horizon
        with torch.no_grad():
            per_frame = [self.pre(self._build_batch(f)) for f in frames]
            batch = {"observation.state": torch.stack([b["observation.state"] for b in per_frame], dim=1)}
            if self.cfg.image_features:
                batch["observation.images"] = torch.stack(
                    [torch.stack([b[k] for k in self.cfg.image_features], dim=-4) for b in per_frame], dim=1)
            global_cond = self.policy.diffusion._prepare_global_conditioning(batch)
        prefix = weights = None
        if prev is not None:
            prefix = torch.zeros_like(prev)
            prefix[:, : horizon - offset] = prev[:, offset:]      # same ticks, new positions
            weights = self._prefix_weights(frozen, horizon - offset, horizon, self._schedule)
        chunk = self._sample(self.policy.diffusion, global_cond, prefix=prefix, weights=weights,
                             max_guidance_weight=self._beta)
        with torch.no_grad():
            actions = self.post(chunk[0]).float().cpu().numpy()   # (horizon, A); waits for the GPU
        return chunk.detach(), actions, time.perf_counter() - t0, prev is not None

    # ---------- internals ----------

    def _build_batch(self, obs: dict) -> dict:
        """obs (ROS node format) -> the batch the checkpoint expects.

        Only two of the node's observation keys are used: the policy was trained on the wrist image
        and a 7-D state of [joint_0..joint_5, gripper]. depth, joint_vel, ee_pos, ee_quat and
        ft_wrench are deliberately ignored.
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
            # Carried through for parity with training batches; the diffusion policy itself does not
            # condition on language, but the processor pipeline passes the field along.
            "task": [self.task],
        }

    def _extra_images(self, obs: dict) -> dict:
        """Image inputs other than the RGB camera, as {feature key: 1CHW float tensor}. None here;
        policies/depth_inference overrides it to add the encoded depth image."""
        return {}

    def _warmup(self, iters: int):
        """Spend the first-call cost (CUDA context, cuDNN autotune, lazy kernels) before the robot
        moves: measured ~226 ms on the first sampler call versus ~62 ms steady state. With RTC this
        also times guided inference, which seeds the inference-delay estimate."""
        if iters <= 0:
            return
        torch = self._torch
        dummy = {
            "rgb": np.zeros((self.cfg.image_features[self._image_key].shape[1] if self.cfg.image_features else 256,
                             self.cfg.image_features[self._image_key].shape[2] if self.cfg.image_features else 256,
                             3), dtype=np.uint8),
            "joint_pos": np.zeros(self._state_dim - 1, dtype=np.float32),
            "gripper_state": np.float32(1.0),
            "depth": np.zeros((256, 256), dtype=np.uint16),   # only read by depth checkpoints
        }
        if self._rtc:
            frames = [dummy] * self.cfg.n_obs_steps
            offset = max(1, self.cfg.n_action_steps - 3)
            for i in range(iters):
                chunk, _, _, _ = self._executor.submit(self._infer, frames, None, None, 1).result()
                _, _, latency, _ = self._executor.submit(self._infer, frames, chunk, offset, 4).result()
                if i > 0 or iters == 1:                 # the first guided call includes autotuning
                    self._latencies.append(latency)
            self._info(f"warmed up with {iters} unguided + {iters} guided inferences: guided "
                       f"{1e3 * max(self._latencies):.0f} ms -> inference delay {self._delay_ticks()} ticks")
            return
        with torch.no_grad():
            for _ in range(iters):
                self.policy.reset()
                self.post(self.policy.select_action(self.pre(self._build_batch(dummy))))
        self.policy.reset()          # drop the dummy observations from the queues
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._info(f"warmed up with {iters} dummy inferences")

    def _info(self, msg: str):
        if self._log is not None:
            self._log.info(msg)
        else:
            print(f"[diffusion_policy] {msg}")
