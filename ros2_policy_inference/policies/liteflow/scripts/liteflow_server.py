# evo1_server_json.py

import sys
import os
import asyncio
import logging
import websockets
import numpy as np
import cv2
import json
import base64
import zlib
import torch
from PIL import Image
from torchvision import transforms
try:  # optional: only a FLOP probe uses it, and Jetson images do not carry it
    from fvcore.nn import FlopCountAnalysis
except ImportError:  # pragma: no cover
    FlopCountAnalysis = None



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.liteflow_model import EVO1
from dataset.normalization import NormalizationType  # the loader itself needs av/decord/torchcodec



class Normalizer:
    def __init__(self, stats_or_path, normalization_type: NormalizationType = NormalizationType.BOUNDS):
        if isinstance(stats_or_path, str):
            with open(stats_or_path, "r") as f:
                self.stats_map = json.load(f)
        else:
            self.stats_map = stats_or_path

        if isinstance(normalization_type, str):
            normalization_type = NormalizationType(normalization_type)
        self.normalization_type = normalization_type
        print(f"Using normalization type: {self.normalization_type}")
        self.target_dim = 24
        self._cache_stats = {}

    def _pad_vector(self, values, name):
        tensor = torch.tensor(values, dtype=torch.float32)
        length = tensor.shape[0]
        if length < self.target_dim:
            pad = torch.zeros(self.target_dim - length, dtype=torch.float32)
            tensor = torch.cat([tensor, pad], dim=0)
        elif length > self.target_dim:
            raise ValueError(f"{name} length {length} exceeds expected {self.target_dim}")
        return tensor

    def _prepare_stats(self, stats_dict, stats_name):
        prepared = {}
        for key, values in stats_dict.items():
            prepared[key] = self._pad_vector(values, f"{stats_name}.{key}")
        return prepared

    def _stat_to_device(self, stats_dict, key, device, dtype):
        tensor = stats_dict.get(key)
        if tensor is None:
            return None
        return tensor.to(device=device, dtype=dtype)
    
    def _get_stats_for(self, arm_key, dataset_key, stats_type):
        """get the relevant stats dict for the given arm/dataset and stat type (state/action)"""
        cache_key = (arm_key, dataset_key, stats_type)
        if cache_key in self._cache_stats:
            return self._cache_stats[cache_key]
        
        if arm_key not in self.stats_map:
             raise ValueError(f"Arm key '{arm_key}' not found in normalization stats.")
             
        if "observation.state" in self.stats_map[arm_key] or "action" in self.stats_map[arm_key]:
            raw_stats = self.stats_map[arm_key]
        else:
            if dataset_key not in self.stats_map[arm_key]:
                 raise ValueError(f"Dataset key '{dataset_key}' not found in normalization stats for arm '{arm_key}'.")
            raw_stats = self.stats_map[arm_key][dataset_key]
        
        dict_key = "observation.state" if stats_type == "state" else "action"
        
        if dict_key not in raw_stats:
            raise ValueError(f"Key '{dict_key}' not found in stats for {arm_key}/{dataset_key}")
            
        prepared = self._prepare_stats(raw_stats[dict_key], dict_key)
        self._cache_stats[cache_key] = prepared
        return prepared

    def _normalize_tensor(self, tensor: torch.Tensor, stats_dict, clamp: bool) -> torch.Tensor:
        eps = 1e-8
        device, dtype = tensor.device, tensor.dtype
        norm_type = self.normalization_type
        
        current_dim = tensor.shape[-1]

        if norm_type == NormalizationType.NORMAL:
            mean = self._stat_to_device(stats_dict, "mean", device, dtype)
            std = self._stat_to_device(stats_dict, "std", device, dtype)
            if mean is not None: mean = mean[..., :current_dim]
            if std is not None: std = std[..., :current_dim]
            
            if mean is None or std is None:
                raise ValueError("Normal normalization selected but mean/std are missing in norm_stats.json")
            return (tensor - mean) / (std + eps)

        low_key, high_key = ("min", "max")
        if norm_type == NormalizationType.BOUNDS_Q99:
            low_key, high_key = ("q01", "q99")

        low = self._stat_to_device(stats_dict, low_key, device, dtype)
        high = self._stat_to_device(stats_dict, high_key, device, dtype)

        if (low is None or high is None) and norm_type == NormalizationType.BOUNDS_Q99:
            logging.warning("Missing q01/q99 stats; falling back to min/max bounds normalization.")
            low = self._stat_to_device(stats_dict, "min", device, dtype)
            high = self._stat_to_device(stats_dict, "max", device, dtype)

        if low is None or high is None:
            raise ValueError("Bounds normalization selected but min/max stats are missing in norm_stats.json")

        low = low[..., :current_dim]
        high = high[..., :current_dim]

        normalized = 2 * (tensor - low) / (high - low + eps) - 1
        if clamp:
            normalized = torch.clamp(normalized, -1.0, 1.0)
        return normalized

    def _denormalize_tensor(self, tensor: torch.Tensor, stats_dict) -> torch.Tensor:
        eps = 1e-8
        device, dtype = tensor.device, tensor.dtype
        norm_type = self.normalization_type

        if norm_type == NormalizationType.NORMAL:
            mean = self._stat_to_device(stats_dict, "mean", device, dtype)
            std = self._stat_to_device(stats_dict, "std", device, dtype)
            if mean is None or std is None:
                raise ValueError("Normal denormalization requested but mean/std stats are missing")
            return tensor * (std + eps) + mean

        low_key, high_key = ("min", "max")
        if norm_type == NormalizationType.BOUNDS_Q99:
            low_key, high_key = ("q01", "q99")

        low = self._stat_to_device(stats_dict, low_key, device, dtype)
        high = self._stat_to_device(stats_dict, high_key, device, dtype)

        if (low is None or high is None) and norm_type == NormalizationType.BOUNDS_Q99:
            logging.warning("Missing q01/q99 stats; falling back to min/max bounds denormalization.")
            low = self._stat_to_device(stats_dict, "min", device, dtype)
            high = self._stat_to_device(stats_dict, "max", device, dtype)

        if low is None or high is None:
            raise ValueError("Bounds denormalization requested but min/max stats are missing")
        
        current_dim = tensor.shape[-1]
        if low.shape[-1] > current_dim:
            low = low[..., :current_dim]
            high = high[..., :current_dim]

        return (tensor + 1.0) / 2.0 * (high - low + eps) + low

    def normalize_state(self, state: torch.Tensor, arm_key: str, dataset_key: str) -> torch.Tensor:
        stats = self._get_stats_for(arm_key, dataset_key, "state")
        norm_state = self._normalize_tensor(state, stats, clamp=True)

        if norm_state.shape[-1] < self.target_dim:
            padding_size = self.target_dim - norm_state.shape[-1]
            pad_tensor = torch.zeros(
                (*norm_state.shape[:-1], padding_size), 
                dtype=norm_state.dtype, 
                device=norm_state.device
            )
            norm_state = torch.cat([norm_state, pad_tensor], dim=-1)
            
        return norm_state

    def denormalize_action(self, action: torch.Tensor, arm_key: str, dataset_key: str) -> torch.Tensor:
        if action.ndim == 1:
            action = action.view(1, -1)
        stats = self._get_stats_for(arm_key, dataset_key, "action")
        denorm_action = self._denormalize_tensor(action, stats)

        # Padding if action dim is less than target_dim
        if denorm_action.shape[-1] < self.target_dim:
            padding_size = self.target_dim - denorm_action.shape[-1]
            pad_tensor = torch.zeros(
                (*denorm_action.shape[:-1], padding_size), 
                dtype=denorm_action.dtype, 
                device=denorm_action.device
            )
            denorm_action = torch.cat([denorm_action, pad_tensor], dim=-1)

        return denorm_action


def load_model_and_normalizer(ckpt_dir):
    config_dict = json.load(open(os.path.join(ckpt_dir, "config.json")))
    stats = json.load(open(os.path.join(ckpt_dir, "norm_stats.json")))

    config_dict["finetune_vlm"] = False
    config_dict["finetune_action_head"] = False
    # Euler steps for the flow-matching integration (flow_matching.py get_action). Purely an
    # inference knob -- training never runs this loop -- so it can be swept on a finished
    # checkpoint. 50 is what every released Evo-1 config uses; FLOWER's num_sampling_steps is
    # 4 with the same Euler scheme, and cost is linear in it: 50 steps means 50 action-head
    # passes per query, which is most of an evaluation's wall clock.
    config_dict["num_inference_timesteps"] = int(os.environ.get("EVO1_INFER_STEPS", "50"))
    # Same idea for a DDPM checkpoint's reverse process (flow_matching.py DDPMSchedule): which
    # sampler integrates the trained network and DDIM's eta. Inference-only, and ignored by
    # flow-matching checkpoints, so a flow eval with these set is unaffected.
    if "EVO1_DDPM_SAMPLER" in os.environ:
        config_dict["ddpm_sampler"] = os.environ["EVO1_DDPM_SAMPLER"]
    if "EVO1_DDIM_ETA" in os.environ:
        config_dict["ddim_eta"] = float(os.environ["EVO1_DDIM_ETA"])

    from config import EvoConfig
    config = EvoConfig.from_dict(config_dict)
    model = EVO1(config).eval()
    objective = getattr(config, "action_objective", "flow")
    sampler = (f"{config.ddpm_sampler} eta={config.ddim_eta}" if objective == "ddpm" else "euler")
    print(f"[info] action_objective={objective} sampler={sampler} "
          f"num_inference_timesteps={config.num_inference_timesteps}", flush=True)
    ckpt_path = os.path.join(ckpt_dir, "mp_rank_00_model_states.pt")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["module"], strict=True)
    # CVP is a training-only supervision head. Load it strictly so checkpoint
    # corruption cannot hide, then remove it before serving; the shared
    # Cross-View Adapter remains part of the inference path.
    model.remove_training_only_modules()

    # Retrain-free ablation: gamma = gamma_scale*tanh(0) = 0 makes whichever fusion module
    # this checkpoint carries an exact identity, so the eval measures the trained weights
    # without the module's inference-time contribution. The score it loses is what the
    # module is actually worth at inference. Done here, after strict loading, so the
    # checkpoint is still verified complete. The first such ablation was run with a
    # temporarily patched server (checkpoints/libero_logs/crossview_gamma0_ablation.log);
    # this is that, made a real switch.
    if os.environ.get("EVO1_ZERO_FUSION_GATE") == "1":
        zeroed = []
        for attribute in ("cross_view_adapter", "bottleneck_fusion"):
            module = getattr(model.embedder, attribute, None)
            if module is None:
                continue
            with torch.no_grad():
                module.alpha_tp.zero_()
                module.alpha_wrist.zero_()
            zeroed.append(attribute)
        # flush=True is load-bearing: the benchmark wrappers redirect this server's stdout to
        # a file, where print() is block-buffered, and the process then blocks forever in the
        # websocket loop. Without the flush this line does not reach the log until the server
        # is killed -- i.e. never, for the operator who is told to look for it.
        if zeroed:
            print(f"[info] fusion gate forced to 0 ({', '.join(zeroed)})", flush=True)
        else:
            print(
                "[info] fusion gate forced to 0 requested, but this checkpoint has no fusion module",
                flush=True,
            )
    if ATTN_DUMP_DIR:
        os.makedirs(ATTN_DUMP_DIR, exist_ok=True)
        module = getattr(model.embedder, "bottleneck_fusion", None)
        if module is None:
            print("[info] no bottleneck_fusion on this checkpoint; dumping the action head only", flush=True)
        else:
            module.capture_attention = True
        if hasattr(model.action_head, "set_capture_attention"):
            model.action_head.set_capture_attention(True)
            print("[info] action-head cross-attention capture on", flush=True)
        print(f"[info] attention dump on -> {ATTN_DUMP_DIR}", flush=True)
    model = model.to("cuda")

    normalization_type = config_dict.get("normalization_type")
    if normalization_type is None:
        # Checkpoints written before EvoConfig carried the field. Training took the value
        # from its dataset config, so "bounds" is only a guess here -- say so rather than
        # silently denormalizing with the wrong bounds.
        normalization_type = NormalizationType.BOUNDS.value
        print(
            f"[warning] {ckpt_dir}/config.json has no 'normalization_type'; assuming "
            f"'{normalization_type}'. If this checkpoint was trained with a dataset config "
            f"that set anything else, the actions below will be wrong. Check the "
            f"'normalization_type' of the dataset config it was trained on."
        )
    normalizer = Normalizer(stats, normalization_type=normalization_type)
    return model, normalizer




# Opt-in, additive diagnostic: off unless EVO1_ACTION_LOG is set, so every existing client
# (RoboTwin, LIBERO, MetaWorld, ...) that doesn't send "meta" is byte-for-byte unaffected.
# Appends one JSON line per query: the returned chunk's first 7 dims (what every CALVIN client
# actually executes) plus whatever the client tagged the request with in "meta", so a chunk's
# magnitude/direction can be correlated back to a specific episode/hop/step after the run.
ACTION_LOG_PATH = os.environ.get("EVO1_ACTION_LOG")
if ACTION_LOG_PATH:
    print(f"[action-log] logging every returned chunk's first 7 dims to {ACTION_LOG_PATH}")


def log_action_chunk(meta, chunk_7d):
    if not ACTION_LOG_PATH:
        return
    import time
    record = {"t": time.time(), "meta": meta, "chunk": chunk_7d}
    with open(ACTION_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def decode_image_payload(entry):
    """One client image entry -> HxWx3 uint8 array.

    Two wire formats are accepted and yield bit-identical arrays, so a checkpoint scores the
    same either way:

    * a nested Python list -- the original format. Every existing client (CALVIN, MetaWorld,
      RoboTwin, LIBERO, LIBERO-Occ, libero-plus) sends this and is unaffected.
    * {"codec": "zlib"|"raw", "shape": [h, w, c], "data": "<base64>"} -- lossless, and the
      reason this exists: `np.array(<448x448x3 nested list>)` costs ~66 ms per view and
      `json.dumps/loads` of it another ~120 ms, on a 6 MB payload. Measured on this box that
      is ~340 ms of the ~850 ms round trip, i.e. the eval was spending more time marshalling
      pixels than running the model (64 ms). zlib level 1 brings the payload to ~64 KB and
      the encode+decode to ~1 ms.
    """
    if isinstance(entry, dict):
        codec = entry.get("codec")
        if codec == "zlib":
            raw = zlib.decompress(base64.b64decode(entry["data"]))
        elif codec == "raw":
            raw = base64.b64decode(entry["data"])
        else:
            raise ValueError(f"unknown image codec: {codec!r}")
        # frombuffer is read-only; cv2 wants a writable, contiguous array.
        return np.frombuffer(raw, dtype=np.uint8).reshape(tuple(entry["shape"])).copy()
    return np.array(entry, dtype=np.uint8)


def decode_image_from_list(img_list):
    img_array = decode_image_payload(img_list)
    img = cv2.resize(img_array, (448, 448))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    return transforms.ToTensor()(pil).to("cuda")



# EVO1_ATTN_DUMP_DIR turns on a per-request dump of the fusion module's raw attention plus
# the two camera frames it acted on, one .npz per inference call. Diagnostics only: it forces
# need_weights=True inside the module, which disables nn.MultiheadAttention's fused path, so
# leave it unset for scoring runs. Off by default; a client that does not set it is unaffected.
ATTN_DUMP_DIR = os.environ.get("EVO1_ATTN_DUMP_DIR", "")
_attn_call = [0]


def dump_attention(model, images, data):
    """Write one .npz: step-1 and step-2 attention, both frames, and the client's meta.

    Frames are stored as uint8 HWC in the model's own colour order (i.e. after the server's
    BGR2RGB), so an overlay lines up with exactly what the network saw.
    """

    module = getattr(model.embedder, "bottleneck_fusion", None)
    attn = module.attention_summary() if module is not None else None
    head = getattr(model, "action_head", None)
    head_attn = head.attention_summary() if hasattr(head, "attention_summary") else None
    if not attn and not head_attn:
        return
    _attn_call[0] += 1
    meta = data.get("meta") or {}
    name = "call_{:05d}".format(_attn_call[0])
    if isinstance(meta, dict) and "task" in meta and "ep" in meta and "step" in meta:
        name = "t{}_ep{}_s{:03d}".format(meta["task"], meta["ep"], meta["step"])
    attn = attn or {}
    payload = {
        "z_attn": attn["z_attn"].numpy() if attn.get("z_attn") is not None else np.zeros(0),
        "tp_attn": attn["tp_attn"].numpy() if attn.get("tp_attn") is not None else np.zeros(0),
        "wrist_attn": attn["wrist_attn"].numpy() if attn.get("wrist_attn") is not None else np.zeros(0),
        "num_tp": attn.get("num_tp", 0), "num_wrist": attn.get("num_wrist", 0),
        "num_tp_tokens": attn.get("num_tp_tokens", 0), "num_tokens": attn.get("num_tokens", 0),
        # Action head cross-attention: [n_blocks, horizon, S] and its block-mean [horizon, S].
        # img_token_mask marks which of those S positions are <IMG_CONTEXT> slots, in sequence
        # order -- the first 256 are view 0's patches, the next 256 view 1's.
        "head_attn": head_attn["per_block"] if head_attn else np.zeros(0),
        "head_attn_mean": head_attn["mean"] if head_attn else np.zeros(0),
        "img_token_mask": (getattr(model.embedder, "_last_img_token_mask", None)[0].cpu().numpy()
                           if getattr(model.embedder, "_last_img_token_mask", None) is not None
                           else np.zeros(0)),
        "img_tp": (images[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8),
        "img_wrist": (images[1].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8),
        "prompt": np.array(data.get("prompt", ""), dtype=object),
        "meta": np.array(json.dumps(meta), dtype=object),
    }
    np.savez_compressed(os.path.join(ATTN_DUMP_DIR, name + ".npz"), **payload)


def infer_from_json_dict(data: dict, model, normalizer, arm_key, dataset_key):
    device = "cuda"
    model_dtype = next(model.parameters()).dtype
    # Prefer per-request arm_key/dataset_key from the client payload; fall back to the
    # server startup defaults. Needed for multi-task benchmarks (RoboTwin sends
    # arm_key=aloha_joint and a per-task dataset_key=robotwin_<task>, and the norm
    # stats are keyed per task).
    arm_key = data.get("arm_key", arm_key)
    dataset_key = data.get("dataset_key", dataset_key)

  
    images = [decode_image_from_list(img) for img in data["image"]]
    # The count must equal the max_views the checkpoint was trained with -- the model splices
    # 256 token slots per view into the LLM sequence, so a different count shifts every text
    # token's position. It is NOT fixed at 3: that is only the default in the repo's dataset
    # configs, and a suite with two cameras (CALVIN) should train and serve at 2.
    assert len(images) == len(data["image_mask"]), (
        f"image count {len(images)} != image_mask length {len(data['image_mask'])}")
    for img in images:
        assert img.shape == (3, 448, 448), "image_size must be (3,448,448)"

 
    state = torch.tensor(data["state"], dtype=torch.float32, device=device)
    if state.ndim == 1:
        state = state.unsqueeze(0)
    # if state.shape[1] < 24:
        # state = torch.cat([state, torch.zeros((1, 24 - state.shape[1]), device=device)], dim=1)
    norm_state = normalizer.normalize_state(state, arm_key, dataset_key).to(dtype=torch.float32)

    
    prompt = data["prompt"]
    image_mask = torch.tensor(data["image_mask"], dtype=torch.int32, device=device)
    action_mask = torch.tensor([data["action_mask"]],dtype=torch.int32, device=device)

    print(f"image_mask,{image_mask}")
    print(f"action_mask,{action_mask}")
    
    if ATTN_DUMP_DIR and hasattr(model.action_head, "reset_attention"):
        # One map per action chunk: clear the running sum before the denoising loop so the
        # dump is this call's attention, not everything since the server started.
        model.action_head.reset_attention()
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        action = model.run_inference(
            images=images,
            image_mask=image_mask,
            prompt=prompt,
            state_input=norm_state,
            action_mask=action_mask
        )
        action = action.reshape(1, -1, 24)
        action = normalizer.denormalize_action(action[0], arm_key, dataset_key)
        action_list = action.cpu().numpy().tolist()
        if ACTION_LOG_PATH:
            log_action_chunk(data.get("meta"), [row[:7] for row in action_list])
        if ATTN_DUMP_DIR:
            dump_attention(model, images, data)
        return action_list


async def handle_request(websocket, model, normalizer, arm_key, dataset_key):
    print("Client connected")
    try:
        async for message in websocket:
           
            json_data = json.loads(message)
            print(f"Received JSON observation")
            actions = infer_from_json_dict(json_data, model, normalizer, arm_key, dataset_key)
            await websocket.send(json.dumps(actions))
            print("Sent action chunk")


    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")
 


if __name__ == "__main__":
    # EVO1_CKPT_DIR / EVO1_PORT let a benchmark script point the server at a checkpoint
    # without editing this file, which matters when several are evaluated in a row.
    # The defaults are what they always were.
    ckpt_dir = os.environ.get("EVO1_CKPT_DIR", "/home/miislab-server11/tzu/Evo-1/checkpoints/Evo1_LIBERO")
    #Example: ckpt_dir = "/your/path/checkpoints/Evo1/Evo1_MetaWorld/"

    port = int(os.environ.get("EVO1_PORT", "9000"))

    # Specialized for different downstream tasks
    # Old-version LIBERO checkpoint (MINT-SJTU/Evo1_LIBERO): norm_stats.json has a single
    # "libero_robot" key holding observation.state/action directly, so dataset_key is unused.
    arm_key = "libero_robot" # Keep the same name as in your norm_stats.json, which is usually the same as the arm name in config.yaml
    dataset_key = "libero_4_datasets"

    print("Loading LiteFlow model...")
    model, normalizer = load_model_and_normalizer(ckpt_dir)

    async def main():
        print(f"LiteFlow server running at ws://0.0.0.0:{port}")
        async with websockets.serve(
            lambda ws: handle_request(ws, model, normalizer, arm_key, dataset_key),
            "0.0.0.0", port, max_size=100_000_000, ping_interval=None, ping_timeout=None
        ):
            await asyncio.Future()

    asyncio.run(main())
