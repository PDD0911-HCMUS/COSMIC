"""ResNet feature extraction and its additional spatial Transformer."""

from pathlib import Path
import math

import torch
from torch import nn


def _build_resnet(config):
    from torchvision import models

    network = getattr(models, config.backbone)(weights=None)
    if config.checkpoint is not None:
        if not config.checkpoint.is_file():
            raise FileNotFoundError(f"Missing ResNet checkpoint: {config.checkpoint}")
        state = torch.load(config.checkpoint, map_location="cpu", weights_only=True)
        network.load_state_dict(state, strict=True)
    elif config.pretrained:
        weights = models.get_model_weights(config.backbone).DEFAULT
        cache = config.cache_dir / "resnet"
        cached = cache / Path(weights.url).name
        if config.local_files_only:
            if not cached.is_file():
                raise FileNotFoundError(f"Missing cached ResNet weights: {cached}; "
                                        "set local_files_only=false to download once")
            state = torch.load(cached, map_location="cpu", weights_only=True)
        else:
            state = torch.hub.load_state_dict_from_url(
                weights.url, model_dir=str(cache), map_location="cpu", check_hash=True,
                weights_only=True,
            )
        network.load_state_dict(state, strict=True)
    stop = 7 if config.resnet_stage == "layer3" else 8
    backbone = nn.Sequential(*list(network.children())[:stop])
    channels = {"resnet18": (256, 512), "resnet50": (1024, 2048)}[config.backbone]
    return backbone, channels[config.resnet_stage == "layer4"]


def _position_encoding_2d(height, width, dimension, device, dtype):
    """Fixed x/y sine-cosine encoding, supporting rectangular feature grids."""
    y, x = torch.meshgrid(
        (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height,
        (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width,
        indexing="ij",
    )
    frequencies = 10000.0 ** (-torch.arange(dimension // 4, device=device, dtype=torch.float32)
                              / (dimension // 4))
    x = x.flatten()[:, None] * (2 * math.pi) * frequencies
    y = y.flatten()[:, None] * (2 * math.pi) * frequencies
    return torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=-1).to(dtype).unsqueeze(0)


def _build_transformer(config):
    layer = nn.TransformerEncoderLayer(
        d_model=config.d_model, nhead=config.transformer_heads,
        dim_feedforward=config.transformer_ffn_dim,
        dropout=config.transformer_dropout, activation="gelu",
        batch_first=True, norm_first=True,
    )
    transformer = nn.TransformerEncoder(
        layer, num_layers=config.transformer_layers,
        norm=nn.LayerNorm(config.d_model), enable_nested_tensor=False,
    )
    # PyTorch clones the initial encoder layer. Initialize each layer
    # independently so stacked layers do not start with identical weights.
    for block in transformer.layers:
        nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
        nn.init.xavier_uniform_(block.self_attn.out_proj.weight)
        nn.init.xavier_uniform_(block.linear1.weight)
        nn.init.xavier_uniform_(block.linear2.weight)
    return transformer


def _extract_resnet_tokens(backbone, images):
    feature_map = backbone(images)
    height, width = feature_map.shape[-2:]
    return feature_map.flatten(2).transpose(1, 2), height, width


def _encode_resnet_tokens(tokens, height, width, transformer):
    tokens = tokens + _position_encoding_2d(
        height, width, tokens.shape[-1], tokens.device, tokens.dtype)
    return transformer(tokens)
