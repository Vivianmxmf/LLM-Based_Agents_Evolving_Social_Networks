# src/utils/seed.py
from __future__ import annotations
import os, random
import numpy as np

def set_all(seed: int | None) -> int:
    """
    Set Python and NumPy RNG seeds. If None, derive from env or random.
    Returns the resolved seed for logging.
    """
    if seed is None:
        env_seed = os.getenv("SEED")
        seed = int(env_seed) if env_seed is not None else random.randint(1, 2_147_483_647)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    return seed