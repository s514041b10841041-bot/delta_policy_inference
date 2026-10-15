"""Registry mapping policy_name (ROS param) -> PolicyModel implementation.

Add a branch here when you add a new policies/<x>.py module. Imports are
local to each branch so unrelated policies' dependencies (e.g. a heavy ML
framework) don't need to be installed unless that policy is selected.

The checkpoint parameter (run_docker.sh --checkpoint, launch arg checkpoint:=)
overrides the policy's config.yaml checkpoint_dir; see resolve_checkpoint().
"""
import os

from .base_policy import PolicyModel

_OPTIONS = ("dummy, act, depth_act, pi, liteflow, diffusion, diffusion_rtc, depth_inference, "
            "depth_inference_rtc, diffusion_depth_only, diffusion_depth_only_rtc")


def resolve_checkpoint(name: str, default: str) -> str:
    """Turn a --checkpoint value into a checkpoint directory.

    default is the policy's config.yaml checkpoint_dir, e.g.
    /checkpoints/diffusion/checkpoint_diffusion_375: its parent (/checkpoints/diffusion) is the
    policy's folder, and that folder's parent (/checkpoints) the mount root. Tried in order:

      ""                          -> default
      an absolute path            -> as given
      <policy folder>/<name>      e.g. checkpoint_diffusion_50
      <mount root>/<name>         e.g. diffusion/checkpoint_diffusion_50
      <policy folder>/checkpoint_<name>, also with a trailing "k" dropped,
                                  e.g. diffusion_50k -> checkpoint_diffusion_50
    """
    if not name:
        return default
    folder = os.path.dirname(default.rstrip("/"))
    if os.path.isabs(name):
        candidates = [name]
    else:
        root = os.path.dirname(folder)
        short = [name] + ([name[:-1]] if name.lower().endswith("k") else [])
        candidates = ([os.path.join(folder, name), os.path.join(root, name)]
                      + [os.path.join(folder, f"checkpoint_{s}") for s in short])
    for c in candidates:
        if os.path.isfile(os.path.join(c, "config.json")):
            return c
    available = sorted(d for d in os.listdir(folder)
                       if os.path.isfile(os.path.join(folder, d, "config.json"))) if os.path.isdir(folder) else []
    raise ValueError(f"checkpoint '{name}' not found (tried {', '.join(candidates)}); "
                     f"checkpoints in {folder}: {', '.join(available) or 'none'}")


def create_policy(name: str, checkpoint: str = "") -> PolicyModel:
    if name == "act":
        from .act import act_policy
        return act_policy.ACTPolicyModel(
            checkpoint_path=resolve_checkpoint(checkpoint, act_policy._CFG["checkpoint_dir"]))
    if name in ("depth_act", "act_depth"):
        # ACT checkpoint trained with the wrist depth image as a second camera (policies/act_depth);
        # same backend and chunking as act.
        from .act_depth import act_depth_policy
        return act_depth_policy.ACTDepthPolicyModel(
            checkpoint_path=resolve_checkpoint(checkpoint, act_depth_policy._CFG["checkpoint_dir"]))
    if name in ("diffusion", "diffusion_rtc"):
        # Same checkpoint and backend; diffusion_rtc adds guided Real-Time Chunking at inference
        # (policies/diffusion/rtc.py), diffusion keeps lerobot's synchronous chunking.
        from .diffusion import diffusion_policy
        return diffusion_policy.DiffusionPolicyModel(
            checkpoint_path=resolve_checkpoint(checkpoint, diffusion_policy._CFG["checkpoint_dir"]),
            use_rtc=(name == "diffusion_rtc"))
    if name in ("depth_inference", "depth_inference_rtc"):
        # Diffusion checkpoint trained with the wrist depth image as a second camera
        # (policies/depth_inference); same backend and chunking choice as diffusion / diffusion_rtc.
        from .depth_inference import depth_policy
        return depth_policy.DiffusionDepthPolicyModel(
            checkpoint_path=resolve_checkpoint(checkpoint, depth_policy._CFG["checkpoint_dir"]),
            use_rtc=(name == "depth_inference_rtc"))
    if name in ("diffusion_depth_only", "diffusion_depth_only_rtc"):
        # Diffusion checkpoint trained on the wrist depth image as its only camera, no RGB
        # (policies/diffusion_depth_only); same backend and chunking choice as diffusion / diffusion_rtc.
        from .diffusion_depth_only import depth_only_policy
        return depth_only_policy.DiffusionDepthOnlyPolicyModel(
            checkpoint_path=resolve_checkpoint(checkpoint, depth_only_policy._CFG["checkpoint_dir"]),
            use_rtc=(name == "diffusion_depth_only_rtc"))

    if name not in ("dummy", "pi", "liteflow"):
        raise ValueError(f"unknown policy_name '{name}' (options: {_OPTIONS})")
    if checkpoint:
        raise ValueError(f"policy_name '{name}' does not take a checkpoint (only act, depth_act, "
                         f"diffusion*, depth_inference*, diffusion_depth_only* do); drop --checkpoint")
    if name == "dummy":
        from .dummy_policy import DummyPolicyModel
        return DummyPolicyModel()
    if name == "pi":
        from .pi_policy import PIPolicyModel
        return PIPolicyModel()
    if name == "liteflow":
        from .liteflow.liteflow_policy import LiteFlowPolicyModel
        return LiteFlowPolicyModel()
    raise ValueError(f"unknown policy_name '{name}' (options: {_OPTIONS})")
