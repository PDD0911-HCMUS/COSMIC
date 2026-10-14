"""Seed a run and prepare its output paths/config snapshot."""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import random

import numpy as np
import torch

from config import save_config
from data.osg_storage import write_json_atomic


@dataclass(frozen=True)
class RunContext:
    directory: Path
    visualization_dir: Path
    checkpoint_dir: Path
    device: str


def resolve_device(requested):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable; set device to cpu or auto")
        index = 0 if requested == "cuda" else int(requested.split(":")[1])
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index {index} is unavailable")
    return requested


def prepare_run(config, action=None):
    device = resolve_device(config.device)
    random.seed(config.seed)
    np.random.seed(config.seed % 2**32)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    directory = Path(config.output.root).expanduser().resolve()
    if config.output.timestamp:
        directory /= datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    context = RunContext(directory, directory / config.output.visualizations,
                         directory / config.save.directory, device)
    directory.mkdir(parents=True, exist_ok=True)
    if config.output.save_config:
        save_config(config, directory / "resolved_config.yaml")
        write_json_atomic(directory / "run.json", {
            "name": config.name, "mode": config.mode, "seed": config.seed,
            "device": device, "output_dir": str(directory),
            "checkpoint_dir": str(context.checkpoint_dir),
            "started_at_utc": datetime.now(timezone.utc).isoformat(), "action": action,
        })
    return context
