import numpy as np

from .base_policy import ACTION_DIM, PolicyModel


class DummyPolicyModel(PolicyModel):
    """Placeholder used until a trained IL/VLA checkpoint is wired in.

    Outputs a no-op action — zero pose delta, gripper held at its current
    state — rather than a random one, since this node may run against a
    real arm before a real model is plugged in.
    """

    def predict(self, obs: dict) -> np.ndarray:
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        action[6] = obs["gripper_state"]
        return action
