"""Shared visual encoder: DINOv2 patch tokens or ResNet + Transformer.

Inputs are RGB tensors already resized and normalized by the joint image/OSG
preprocessor. No crop, resize or second normalization occurs in this module.
Tokens remain in row-major spatial order; CLS is excluded from the DINO grid.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F

from model.config import VisualEncoderConfig
from model.visual_encoder.dino import _build_dino, _extract_dino_tokens
from model.visual_encoder.resnet import (
    _build_resnet, _build_transformer, _extract_resnet_tokens, _encode_resnet_tokens,
)


class VisualEncoder(nn.Module):
    """Encode M images independently, without requiring a paired observation.

    Returns visual_tokens [M,N,D], spatial_features [M,D,Hf,Wf], unit-length
    global_descriptor [M,G], grid_size (Hf,Wf) and image_size (H,W).
    N=Hf*Wf and spatial_features is the row-major reshape of visual_tokens.
    The descriptor uses mean pooling, a learned projection and L2 normalization.
    """

    def __init__(self, config: VisualEncoderConfig | None = None):
        super().__init__()
        self.config = config or VisualEncoderConfig()
        self.is_dino = self.config.backbone.startswith("dinov2")
        if self.is_dino:
            self.backbone = _build_dino(self.config)
            channels = self.backbone.config.hidden_size
            self.patch_size = self.backbone.config.patch_size
            self.transformer = nn.Identity()  # DINO already contains its ViT encoder.
        else:
            self.backbone, channels = _build_resnet(self.config)
            self.patch_size = None
            self.transformer = _build_transformer(self.config)
        self.projection = nn.Sequential(nn.Linear(channels, self.config.d_model),
                                        nn.LayerNorm(self.config.d_model))
        self.global_head = nn.Linear(self.config.d_model, self.config.global_dim)
        if self.config.freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.config.freeze_backbone:
            self.backbone.eval()
        return self

    @staticmethod
    def _validate_images(images):
        if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected normalized RGB images [M,3,H,W]")
        if not images.is_floating_point() or min(images.shape) <= 0:
            raise ValueError("Images must be nonempty floating-point tensors")

    def forward(self, images):
        self._validate_images(images)
        image_size = tuple(images.shape[-2:])
        context = torch.no_grad() if self.config.freeze_backbone else nullcontext()
        with context:
            if self.is_dino:
                tokens, height, width = _extract_dino_tokens(self.backbone, images, self.patch_size)
            else:
                tokens, height, width = _extract_resnet_tokens(self.backbone, images)
        tokens = self.projection(tokens)
        if not self.is_dino:
            tokens = _encode_resnet_tokens(tokens, height, width, self.transformer)
        spatial = tokens.transpose(1, 2).reshape(images.shape[0], self.config.d_model,
                                                height, width).contiguous()
        descriptor = F.normalize(self.global_head(tokens.mean(dim=1)), p=2, dim=-1, eps=1e-6)
        return {"visual_tokens": tokens, "spatial_features": spatial,
                "global_descriptor": descriptor, "grid_size": (height, width),
                "image_size": image_size}


def build_visual_encoder(config: VisualEncoderConfig | None = None) -> VisualEncoder:
    return VisualEncoder(config)
