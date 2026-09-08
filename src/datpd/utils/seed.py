"""Determinism helpers.

Winning solutions must be reproducible (competition rule). Seed everything, log the seed, and
prefer `deterministic=True` while debugging even though it costs throughput.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Seed python, numpy and torch (if installed).

    Args:
        seed: The seed to use everywhere.
        deterministic: Force deterministic cuDNN kernels. Slower, but makes two runs of the
            same config comparable — worth it when measuring a small log-loss change.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
