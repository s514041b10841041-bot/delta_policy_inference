"""
Observation preprocessing shared by PolicyInferenceNode: image resize and
the fixed EE-frame reprojection (tcp_pose_topic's reported frame -> the
true EE frame), matching the transform ros2_data_collector applies before
recording so the policy sees observations in the same frame/units it was
trained on.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def resize_image(img, size, is_depth):
    """Resize an (H,W) / (H,W,C) image to size x size. Depth uses
    nearest-neighbour interpolation so real depth values are never blended."""
    if img is None or getattr(img, "ndim", 0) < 2:
        return img
    if img.shape[0] == size and img.shape[1] == size:
        return img
    import cv2
    interp = cv2.INTER_NEAREST if is_depth else cv2.INTER_AREA
    return np.ascontiguousarray(cv2.resize(img, (size, size), interpolation=interp))


def apply_ee_offset(pos, quat_xyzw, pos_offset: np.ndarray, rot_offset: Rotation):
    """pos (3,), quat_xyzw (4,) in the topic's reported frame -> (pos, quat)
    in the true EE frame, given a fixed offset expressed in the topic
    frame's local axes (same math as ros2_data_collector.recorder)."""
    r_ref = Rotation.from_quat(quat_xyzw.astype(np.float64))
    new_pos = pos.astype(np.float64) + r_ref.apply(pos_offset)
    new_rot = r_ref * rot_offset
    return new_pos, new_rot.as_quat()
