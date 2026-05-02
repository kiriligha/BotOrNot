import random
import numpy as np


def set_all_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs(*paths) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
