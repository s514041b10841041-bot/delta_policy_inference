"""
Training range of observation.state, read from a lerobot checkpoint's saved
normalizer statistics, for diagnostics (e.g. the IL hand-off log in
PolicyInferenceNode). Independent of how the policy itself normalizes the
state: values are reported range-normalized, so the training data spans
exactly [-1, 1] per dimension and anything outside it was never seen.
"""
import glob
import os

import numpy as np


def load_state_range(checkpoint_dir):
    """(min, max) arrays of observation.state from the checkpoint's
    preprocessor normalizer file, or None if unavailable."""
    if not checkpoint_dir:
        return None
    files = glob.glob(os.path.join(checkpoint_dir, "policy_preprocessor_step_*_normalizer_processor.safetensors"))
    if not files:
        return None
    from safetensors.numpy import load_file
    stats = load_file(files[0])
    lo, hi = stats.get("observation.state.min"), stats.get("observation.state.max")
    if lo is None or hi is None:
        return None
    return lo.astype(np.float64).reshape(-1), hi.astype(np.float64).reshape(-1)


def range_normalize(x, state_range):
    """Map x into the training range's [-1, 1] (MIN_MAX), per dimension."""
    lo, hi = state_range
    return 2.0 * (np.asarray(x, dtype=np.float64) - lo) / np.maximum(hi - lo, 1e-8) - 1.0
