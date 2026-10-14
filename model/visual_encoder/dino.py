"""DINOv2 loading and spatial patch-token extraction (without CLS)."""

_DINO_MODELS = {
    "dinov2_vits14": ("facebook/dinov2-small", 384, 6),
    "dinov2_vitb14": ("facebook/dinov2-base", 768, 12),
}


def _build_dino(config):
    # Lazy import: dataset export/preview never constructs/downloads DINO.
    from transformers import Dinov2Config, Dinov2Model

    model_id, hidden_size, heads = _DINO_MODELS[config.backbone]
    if config.checkpoint is not None:
        if not config.checkpoint.is_dir():
            raise FileNotFoundError("DINO checkpoint must be a local Hugging Face model directory "
                                    f"containing config.json and weights: {config.checkpoint}")
        architecture = Dinov2Config.from_pretrained(str(config.checkpoint), local_files_only=True)
        if (architecture.hidden_size, architecture.num_attention_heads,
                architecture.num_hidden_layers, architecture.patch_size, architecture.num_channels) != (
                hidden_size, heads, 12, 14, 3):
            raise ValueError(f"Local DINO checkpoint architecture does not match {config.backbone}")
        return Dinov2Model.from_pretrained(str(config.checkpoint), config=architecture, local_files_only=True)
    if config.pretrained:
        return Dinov2Model.from_pretrained(
            model_id, cache_dir=str(config.cache_dir / "dinov2"),
            local_files_only=config.local_files_only,
        )
    return Dinov2Model(Dinov2Config(
        hidden_size=hidden_size, num_hidden_layers=12, num_attention_heads=heads,
        image_size=518, patch_size=14, layerscale_value=1.0,
    ))


def _extract_dino_tokens(backbone, images, patch_size):
    image_size = images.shape[-2:]
    if any(size % patch_size for size in image_size):
        raise ValueError(f"DINO image height/width must be divisible by patch_size={patch_size} "
                         "to preserve the full image/bbox grid")
    output = backbone(pixel_values=images, return_dict=True)
    tokens = output.last_hidden_state[:, 1:, :]
    height, width = (size // patch_size for size in image_size)
    if tokens.shape[1] != height * width:
        raise RuntimeError("DINO patch tokens do not match the image's spatial grid")
    return tokens, height, width
