"""
Pluggable IL/VLA policy interface for PolicyInferenceNode.

To deploy a trained model: add a new PolicyModel subclass alongside this
file (e.g. policies/il_policy.py) whose predict() loads/wraps your
checkpoint, then import it in inference_node.main() in place of
DummyPolicyModel. Nothing else in this package needs to change.
"""
from abc import ABC, abstractmethod

import numpy as np

# [dx, dy, dz, drx, dry, drz, gripper] — matches the action definition
# ros2_data_collector records training data in (position delta in metres,
# rotation delta as XYZ-Euler radians, gripper in [0, 1]).
ACTION_DIM = 7


class PolicyModel(ABC):
    """Interface every IL/VLA policy backend must implement."""

    # Set by a backend whose actions are tied to a fixed control period (e.g.
    # per-camera-frame deltas): the node then ticks at this rate instead of
    # the inference_rate_hz parameter. None = use the parameter.
    inference_rate_hz = None

    def reset(self):
        """Called when the node starts and, with reset_on_handoff, at every IL
        hand-off from the sim. Override to reset any internal state (e.g. an
        action-chunking buffer or recurrent hidden state) between episodes.
        No-op by default."""

    @abstractmethod
    def predict(self, obs: dict) -> np.ndarray:
        """obs -> action, a float array of shape (ACTION_DIM,).

        obs keys:
          rgb          (image_size, image_size, 3) uint8
          depth        (image_size, image_size) uint16/float32
          joint_pos    (6,) float32
          joint_vel    (6,) float32
          ee_pos       (3,) float32
          ee_quat      (4,) float32, xyzw
          gripper_state  scalar float32, 1.0=open / 0.0=closed — the
                         policy's own previous gripper output (or the
                         initial "open" state before the first prediction)
          ft_wrench    (6,) float32 [fx,fy,fz,tx,ty,tz]
        """
