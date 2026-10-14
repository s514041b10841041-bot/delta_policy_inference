"""LiteFlow (InternVL3-1B + FLOWER flow-matching action head) PolicyModel.

Wraps the checkpoint-loading and inference logic already verified in the liteflow_delta_deploy
bundle (scripts/liteflow_server.py, scripts/liteflow_model.py, model/) -- copied into this
package rather than reimplemented -- and adapts it to this node's interface:

  * base_policy.PolicyModel.predict() is called once per ROS timer tick (10 Hz) and must
    return exactly one (7,) action. LiteFlow's model instead returns a whole ~1s chunk (10
    steps) per query. This class buffers a chunk and pops one row per predict() call,
    re-querying after `n_exec` popped rows rather than only when the buffer empties -- so the
    arm never runs open-loop on one observation for longer than n_exec * (1/inference_rate_hz)
    seconds. See policies/liteflow/config.yaml for that knob.
  * No websocket / JSON wire: liteflow_server.py's wire format exists only to cross a network
    (base64+zlib image, a client process on the other end). In-process, this class calls the
    same tensor-level functions the wire handlers call, skipping the encode/decode layer.
    Notably, the wire's RGB->BGR->RGB round trip (encode_image flips to BGR; the server's
    decode_image_from_list flips back) cancels out, so obs["rgb"] (already RGB from RosBridge)
    needs no channel flip here at all.
"""
import os

import yaml

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_CONFIG_PATH) as _f:
    _CFG = yaml.safe_load(_f)

# Must be set before transformers/huggingface_hub are ever imported: both read HF_HOME /
# HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE into their own module-level constants at import time,
# so setting os.environ after `from .scripts.liteflow_server import ...` below (which pulls
# transformers in transitively) has no effect -- this bit the first end-to-end test, where the
# offline env vars were set in __init__() and the load silently reached the network anyway.
if _CFG.get("hf_home"):
    os.environ.setdefault("HF_HOME", _CFG["hf_home"])
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# load_model_and_normalizer reads this env var itself (see liteflow_server.py) rather than
# taking it as an argument -- likewise must be set before that import runs.
os.environ["EVO1_INFER_STEPS"] = str(_CFG["infer_steps"])

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from . import action_convert as ac
from .scripts.liteflow_server import load_model_and_normalizer
from ..base_policy import PolicyModel


class LiteFlowPolicyModel(PolicyModel):
    def __init__(self):
        cfg = _CFG
        self._prompt = cfg["prompt"]
        self._arm_key = cfg["arm_key"]
        self._dataset_key = cfg["dataset_key"]
        self._n_exec = int(cfg["n_exec"])

        print(f"[liteflow] loading checkpoint from {cfg['checkpoint_dir']} ...", flush=True)
        self._model, self._normalizer = load_model_and_normalizer(cfg["checkpoint_dir"])
        print("[liteflow] checkpoint loaded.", flush=True)

        self._buffer = []
        self._steps_since_query = 0

    def reset(self):
        self._buffer = []
        self._steps_since_query = 0

    def predict(self, obs: dict) -> np.ndarray:
        if not self._buffer or self._steps_since_query >= self._n_exec:
            self._buffer = list(self._query(obs))
            self._steps_since_query = 0
        row = self._buffer.pop(0)
        self._steps_since_query += 1
        return row.astype(np.float32)

    def _query(self, obs: dict) -> np.ndarray:
        """obs -> (10, ACTION_DIM) float64: one row per ~0.1s policy step (rows_per_step=1)."""
        image = self._prep_image(obs["rgb"])
        state = ac.build_state(obs["ee_pos"], obs["ee_quat"], obs["gripper_state"])
        state_t = torch.tensor(state, dtype=torch.float32, device="cuda").unsqueeze(0)
        norm_state = self._normalizer.normalize_state(
            state_t, self._arm_key, self._dataset_key
        ).to(dtype=torch.float32)

        image_mask = torch.tensor([1], dtype=torch.int32, device="cuda")
        action_mask = torch.tensor(
            [[1] * ac.ACTION_DIM + [0] * (ac.MODEL_DIM - ac.ACTION_DIM)],
            dtype=torch.int32, device="cuda",
        )

        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            action = self._model.run_inference(
                images=[image],
                image_mask=image_mask,
                prompt=self._prompt,
                state_input=norm_state,
                action_mask=action_mask,
            )
            action = action.reshape(1, -1, ac.MODEL_DIM)
            action = self._normalizer.denormalize_action(action[0], self._arm_key, self._dataset_key)

        chunk = action.to(dtype=torch.float32).cpu().numpy()[:, :ac.ACTION_DIM]
        chunk[:, 6] = np.where(chunk[:, 6] > 0.5, 1.0, 0.0)   # the capture's gripper_cmd is 0/1
        return chunk

    @staticmethod
    def _prep_image(rgb: np.ndarray) -> torch.Tensor:
        img = cv2.resize(rgb, (448, 448))
        pil = Image.fromarray(img)
        return transforms.ToTensor()(pil).to("cuda")
