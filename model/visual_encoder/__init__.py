"""Visual encoder package with a shared interface for DINOv2 and ResNet."""

from model.visual_encoder.encoder import VisualEncoder, build_visual_encoder

__all__ = ["VisualEncoder", "build_visual_encoder"]
