"""Rotation/state conventions LiteFlow's checkpoint was trained and verified against.

Vendored verbatim (only the functions this in-process policy actually needs) from the
liteflow_delta_deploy bundle's delta_arm/delta_client.py, which verified these against all 107
grasp-fan demos (translation error 0.0, rotation 2e-8, gripper 0 mismatches). Not reimplemented
here: getting real-robot action math subtly wrong is worse than carrying a little dead code.
"""
import numpy as np

ACTION_DIM = 7          # d_pos (3) | d_rpy (3) | gripper_cmd (1) -- matches base_policy.ACTION_DIM
MODEL_DIM = 24          # the checkpoint's padded action/state width; action_mask must be this long


def canonical_quat_xyzw(q) -> np.ndarray:
    """(..., 4) -> the same rotation with w >= 0. THE sign rule the checkpoint's state uses."""
    q = np.asarray(q, dtype=np.float64)
    return q * np.where(q[..., 3:4] < 0, -1.0, 1.0)


def rpy_to_matrix(rpy) -> np.ndarray:
    """(..., 3) [roll, pitch, yaw] -> (..., 3, 3) with R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    rpy = np.asarray(rpy, dtype=np.float64)
    r, p, y = rpy[..., 0], rpy[..., 1], rpy[..., 2]
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    R = np.empty(rpy.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = cy * cp
    R[..., 0, 1] = cy * sp * sr - sy * cr
    R[..., 0, 2] = cy * sp * cr + sy * sr
    R[..., 1, 0] = sy * cp
    R[..., 1, 1] = sy * sp * sr + cy * cr
    R[..., 1, 2] = sy * sp * cr - cy * sr
    R[..., 2, 0] = -sp
    R[..., 2, 1] = cp * sr
    R[..., 2, 2] = cp * cr
    return R


def matrix_to_rpy(R) -> np.ndarray:
    """Inverse of rpy_to_matrix. (..., 3, 3) -> (..., 3)."""
    R = np.asarray(R, dtype=np.float64)
    roll = np.arctan2(R[..., 2, 1], R[..., 2, 2])
    pitch = -np.arcsin(np.clip(R[..., 2, 0], -1.0, 1.0))
    yaw = np.arctan2(R[..., 1, 0], R[..., 0, 0])
    return np.stack([roll, pitch, yaw], axis=-1)


def matrix_to_rotvec(R) -> np.ndarray:
    """(..., 3, 3) -> (..., 3) axis * angle. For the small deltas an action carries (< ~3 rad)."""
    R = np.asarray(R, dtype=np.float64)
    skew = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    s = np.linalg.norm(skew, axis=-1, keepdims=True)                 # 2 sin(angle)
    angle = np.arctan2(s, (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1.0)[..., None])
    # angle / (2 sin angle) -> 1/2 as angle -> 0; the series keeps that limit exact.
    scale = np.where(s > 1e-8, angle / np.maximum(s, 1e-300), 0.5 + angle * angle / 12.0)
    return skew * scale


def rotvec_to_matrix(r) -> np.ndarray:
    """(3,) axis * angle -> (3, 3), Rodrigues."""
    r = np.asarray(r, dtype=np.float64)
    angle = np.linalg.norm(r)
    K = np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])
    if angle < 1e-8:
        return np.eye(3) + K + 0.5 * K @ K
    return np.eye(3) + np.sin(angle) / angle * K + (1 - np.cos(angle)) / angle ** 2 * K @ K


def build_state(real_ee_pos, real_ee_quat, gripper_cmd) -> np.ndarray:
    """The capture's own fields, concatenated in its own order -> the 8-value state.

    A pass-through: the values are used as given. The quaternion's sign is the only thing
    touched, and that changes nothing about the rotation it stands for.
    """
    pos = np.asarray(real_ee_pos, dtype=np.float64).ravel()
    quat = np.asarray(real_ee_quat, dtype=np.float64).ravel()
    grip = np.asarray(gripper_cmd, dtype=np.float64).ravel()
    return np.concatenate([pos, canonical_quat_xyzw(quat), grip])


def split_rows(chunk: np.ndarray, n: int) -> np.ndarray:
    """(H, 7) policy steps -> (H * n, 7) rows at the recorder's rate, n sub-deltas per step.

    Applying a step's n rows in order reproduces the step exactly: the translation is divided
    by n, the rotation is replaced by its n-th root (dividing the three Euler angles is only a
    first-order approximation), and the gripper command is held.
    """
    chunk = np.asarray(chunk, dtype=np.float64)
    if n == 1:
        return chunk.copy()
    out = np.empty((len(chunk) * n, ACTION_DIM), dtype=np.float64)
    for i, row in enumerate(chunk):
        sub = row.copy()
        sub[:3] = row[:3] / n
        sub[3:6] = matrix_to_rpy(rotvec_to_matrix(matrix_to_rotvec(rpy_to_matrix(row[3:6])) / n))
        out[i * n:(i + 1) * n] = sub
    return out
