"""Lightweight scene-scoring head, independent of encoders and dataset labels."""

from collections.abc import Mapping

import torch
from torch import nn

from model.config import SceneSimilarityConfig


class GlobalSceneSimilarity(nn.Module):
    """Map a verified pair descriptor [B,D] to raw logits and sigmoid scores.

    Input is a verification mapping or its verified_descriptor tensor. The
    head never reads labels, performs thresholding or discards unmatched pairs.
    A zero descriptor (no verified regions) is still scored using the learned
    MLP biases, providing a prior rather than a forced negative prediction.
    These scores require training and are not calibrated probabilities.
    """

    def __init__(self, d_model=256, config: SceneSimilarityConfig | None = None):
        super().__init__()
        self.config = config or SceneSimilarityConfig()
        if type(d_model) is not int or d_model <= 0:
            raise ValueError("Scene similarity d_model must be a positive integer")
        self.d_model = d_model
        self.mlp = nn.Sequential(nn.Linear(d_model, self.config.hidden_dim), nn.GELU(),
                                 nn.Dropout(self.config.dropout), nn.Linear(self.config.hidden_dim, 1))

    def forward(self, verification):
        available = None
        if isinstance(verification, Mapping):
            descriptor = verification.get("verified_descriptor")
            available = verification.get("has_verified_matches")
        else:
            descriptor = verification
        if (not isinstance(descriptor, torch.Tensor) or descriptor.ndim != 2 or descriptor.shape[0] <= 0
                or descriptor.shape[1] != self.d_model or not descriptor.is_floating_point()
                or not torch.isfinite(descriptor).all()):
            raise ValueError("Scene similarity requires finite floats verified_descriptor [B,d_model]")
        if available is not None and (not isinstance(available, torch.Tensor) or available.dtype != torch.bool
                                      or available.shape != descriptor.shape[:1]):
            raise ValueError("has_verified_matches must be bool [B] when supplied")
        reference = self.mlp[0].weight
        descriptor = descriptor.to(device=reference.device, dtype=reference.dtype)
        logits = self.mlp(descriptor).squeeze(-1)
        return {
            "scene_logits": logits,
            "scene_score": torch.sigmoid(logits.float()),
            "has_verified_matches": available.to(reference.device) if available is not None else None,
        }
