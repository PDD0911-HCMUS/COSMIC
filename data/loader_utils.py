"""Path validation and worker seeding shared by dataset loaders."""

from pathlib import Path
import random

import numpy as np
import torch


def resolve_under(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"Expected a relative path inside {root}: {relative}")
    return path


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
