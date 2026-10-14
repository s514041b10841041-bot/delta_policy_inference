import numpy as np

from .base_policy import ACTION_DIM, PolicyModel


class PIPolicyModel(PolicyModel):   
    def predict(self, obs: dict) -> np.ndarray:
        pass
