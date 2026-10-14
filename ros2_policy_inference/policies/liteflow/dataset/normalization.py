"""NormalizationType on its own module so the inference path (Evo1_server.py, the efficiency
bench) can import it without pulling in the LeRobot loader's training-only dependencies (av,
decord, torchcodec, pandas, accelerate) -- none of which have wheels on a Jetson.
lerobot_dataset_pretrain_mp re-exports it, so train.py's import is unchanged."""
from enum import Enum


class NormalizationType(str, Enum):
    NORMAL = "normal"
    BOUNDS = "bounds"
    BOUNDS_Q99 = "bounds_q99"
