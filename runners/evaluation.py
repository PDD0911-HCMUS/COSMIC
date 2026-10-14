"""Validate a completed COSMIC checkpoint against explicitly configured retrieval data."""

from pathlib import Path

import torch

from config import config_to_dict


def load_evaluation_checkpoint(config):
    if config.save.resume is None:
        raise ValueError("Evaluation requires a trained checkpoint: --resume outputs/gsv/checkpoints/best.pt")
    path = Path(config.save.resume)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(payload, dict) or payload.get("format_version") != 1
            or type(payload.get("epoch")) is not int or payload["epoch"] < 1
            or not isinstance(payload.get("model"), dict) or not payload["model"]
            or not all(isinstance(value, torch.Tensor) for value in payload["model"].values())
            or not isinstance(payload.get("config"), dict)):
        raise ValueError("Evaluation requires a completed TrainEngine checkpoint")
    current, saved = config_to_dict(config), payload["config"]
    if current["model"] != saved.get("model"):
        raise ValueError("Evaluation model settings differ from the checkpoint")
    preprocessing = ("image_size", "mean", "std", "graph")
    evaluation_data = current["validation"].get("data") or current["data"]
    if any(evaluation_data[key] != saved.get("data", {}).get(key) for key in preprocessing):
        raise ValueError("Evaluation image/graph preprocessing differs from the checkpoint")
    print(f"Evaluate checkpoint: {path} | completed epoch={payload['epoch']}")
    return payload["model"]
