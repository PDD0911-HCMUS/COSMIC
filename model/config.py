"""Model settings, independent of model construction or downloads."""

from dataclasses import dataclass, field
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VISUAL_BACKBONES = ("resnet18", "resnet50", "dinov2_vits14", "dinov2_vitb14")


@dataclass(frozen=True)
class VisualEncoderConfig:
    backbone: str = "dinov2_vits14"
    pretrained: bool = True
    freeze_backbone: bool = True
    checkpoint: Path | None = None
    cache_dir: Path = PROJECT_ROOT / ".cache" / "visual_backbones"
    local_files_only: bool = False
    d_model: int = 256
    global_dim: int = 256
    resnet_stage: str = "layer3"
    transformer_layers: int = 2
    transformer_heads: int = 8
    transformer_ffn_dim: int = 1024
    transformer_dropout: float = 0.1

    def __post_init__(self):
        if self.backbone not in VISUAL_BACKBONES:
            raise ValueError(f"Unknown visual backbone {self.backbone!r}. Available: {', '.join(VISUAL_BACKBONES)}")
        for key in ("pretrained", "freeze_backbone", "local_files_only"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"visual_encoder.{key} must be boolean")
        for key in ("d_model", "global_dim", "transformer_layers", "transformer_heads", "transformer_ffn_dim"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"visual_encoder.{key} must be a positive integer")
        if self.resnet_stage not in ("layer3", "layer4"):
            raise ValueError("visual_encoder.resnet_stage must be layer3 or layer4")
        dropout = self.transformer_dropout
        if type(dropout) not in (int, float) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("visual_encoder.transformer_dropout must be in [0, 1)")
        if self.backbone.startswith("resnet"):
            if self.d_model % 4 or self.d_model % self.transformer_heads:
                raise ValueError("ResNet d_model must be divisible by 4 and transformer_heads")
        for key in ("checkpoint", "cache_dir"):
            value = getattr(self, key)
            if value is None:
                if key == "cache_dir":
                    raise ValueError("visual_encoder.cache_dir must identify a directory")
                continue
            if not isinstance(value, (str, Path)) or not str(value).strip():
                raise ValueError(f"visual_encoder.{key} must be a path")
            path = Path(value).expanduser()
            object.__setattr__(self, key, path if path.is_absolute() else PROJECT_ROOT / path)


@dataclass(frozen=True)
class OSGEncoderConfig:
    backbone: str = "bert-base-uncased"
    checkpoint: Path | None = PROJECT_ROOT / "data/OvSGTR/weights/bert-base-uncased"
    cache_dir: Path = PROJECT_ROOT / ".cache/osg_encoder"
    local_files_only: bool = True
    freeze_backbone: bool = True
    d_model: int = 256
    global_dim: int = 256
    max_length: int = 64
    text_batch_size: int = 64
    cache_size: int = 4096
    pooling: str = "confidence"
    use_bbox_embedding: bool = False

    def __post_init__(self):
        if self.backbone != "bert-base-uncased":
            raise ValueError("osg_encoder.backbone must be bert-base-uncased")
        for key in ("local_files_only", "freeze_backbone", "use_bbox_embedding"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"osg_encoder.{key} must be boolean")
        for key in ("d_model", "global_dim", "max_length", "text_batch_size"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"osg_encoder.{key} must be a positive integer")
        if self.max_length < 5:
            raise ValueError("osg_encoder.max_length must be at least 5")
        if type(self.cache_size) is not int or self.cache_size < 0:
            raise ValueError("osg_encoder.cache_size must be a nonnegative integer")
        if self.pooling not in ("confidence", "mean"):
            raise ValueError("osg_encoder.pooling must be confidence or mean")
        for key in ("checkpoint", "cache_dir"):
            value = getattr(self, key)
            if value is None:
                if key == "cache_dir":
                    raise ValueError("osg_encoder.cache_dir must identify a directory")
                continue
            if not isinstance(value, (str, Path)) or not str(value).strip():
                raise ValueError(f"osg_encoder.{key} must be a path")
            path = Path(value).expanduser()
            object.__setattr__(self, key, path if path.is_absolute() else PROJECT_ROOT / path)


@dataclass(frozen=True)
class GroundingConfig:
    num_heads: int = 4
    ffn_dim: int = 512
    dropout: float = 0.1
    use_self_attention: bool = True
    use_bbox_bias: bool = True
    bbox_bias_strength: float = 2.0
    pooling: str = "confidence"

    def __post_init__(self):
        for key in ("num_heads", "ffn_dim"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"grounding.{key} must be a positive integer")
        for key in ("use_self_attention", "use_bbox_bias"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"grounding.{key} must be boolean")
        if (type(self.dropout) not in (int, float) or not math.isfinite(self.dropout)
                or not 0 <= self.dropout < 1):
            raise ValueError("grounding.dropout must be in [0, 1)")
        if (type(self.bbox_bias_strength) not in (int, float)
                or not math.isfinite(self.bbox_bias_strength) or self.bbox_bias_strength < 0):
            raise ValueError("grounding.bbox_bias_strength must be finite and nonnegative")
        if self.pooling not in ("confidence", "mean"):
            raise ValueError("grounding.pooling must be confidence or mean")


@dataclass(frozen=True)
class AlignmentConfig:
    num_heads: int = 4
    temperature: float = 0.1
    unmatched_score: float = 0.5
    learn_unmatched_score: bool = True
    use_hungarian: bool = True

    def __post_init__(self):
        if type(self.num_heads) is not int or self.num_heads <= 0:
            raise ValueError("alignment.num_heads must be a positive integer")
        if (type(self.temperature) not in (int, float) or not math.isfinite(self.temperature)
                or self.temperature <= 0):
            raise ValueError("alignment.temperature must be finite and positive")
        if (type(self.unmatched_score) not in (int, float) or not math.isfinite(self.unmatched_score)
                or not -1 < self.unmatched_score < 1):
            raise ValueError("alignment.unmatched_score must be in (-1,1), on the cosine similarity scale")
        for key in ("learn_unmatched_score", "use_hungarian"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"alignment.{key} must be boolean")


@dataclass(frozen=True)
class VerificationConfig:
    num_heads: int = 4
    ffn_dim: int = 512
    dropout: float = 0.1
    max_patches: int = 32
    pair_chunk_size: int = 32
    use_graph_guidance: bool = True
    pooling: str = "confidence"

    def __post_init__(self):
        for key in ("num_heads", "ffn_dim", "max_patches", "pair_chunk_size"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"verification.{key} must be a positive integer")
        if self.max_patches < 2:
            raise ValueError("verification.max_patches must be at least 2 to retain both triplet endpoints")
        if (type(self.dropout) not in (int, float) or not math.isfinite(self.dropout)
                or not 0 <= self.dropout < 1):
            raise ValueError("verification.dropout must be in [0,1)")
        if type(self.use_graph_guidance) is not bool:
            raise ValueError("verification.use_graph_guidance must be boolean")
        if self.pooling not in ("confidence", "mean"):
            raise ValueError("verification.pooling must be confidence or mean")


@dataclass(frozen=True)
class SceneSimilarityConfig:
    hidden_dim: int = 256
    dropout: float = 0.1

    def __post_init__(self):
        if type(self.hidden_dim) is not int or self.hidden_dim <= 0:
            raise ValueError("scene_similarity.hidden_dim must be a positive integer")
        if (type(self.dropout) not in (int, float) or not math.isfinite(self.dropout)
                or not 0 <= self.dropout < 1):
            raise ValueError("scene_similarity.dropout must be in [0,1)")


@dataclass(frozen=True)
class CriteriaConfig:
    retrieval_weight: float = 1.0
    verification_weight: float = 1.0
    retrieval_temperature: float = 0.1

    def __post_init__(self):
        for key in ("retrieval_weight", "verification_weight"):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"criteria.{key} must be finite and nonnegative")
        if not (self.retrieval_weight > 0 or self.verification_weight > 0):
            raise ValueError("At least one criteria weight must be positive")
        value = self.retrieval_temperature
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("criteria.retrieval_temperature must be finite and positive")


@dataclass(frozen=True)
class ModelConfig:
    visual_encoder: VisualEncoderConfig = field(default_factory=VisualEncoderConfig)
    osg_encoder: OSGEncoderConfig = field(default_factory=OSGEncoderConfig)
    grounding: GroundingConfig = field(default_factory=GroundingConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    scene_similarity: SceneSimilarityConfig = field(default_factory=SceneSimilarityConfig)
    criteria: CriteriaConfig = field(default_factory=CriteriaConfig)

    def __post_init__(self):
        if self.visual_encoder.d_model != self.osg_encoder.d_model:
            raise ValueError("visual_encoder.d_model and osg_encoder.d_model must match for grounding")
        if self.visual_encoder.d_model % self.grounding.num_heads:
            raise ValueError("d_model must be divisible by grounding.num_heads")
        if self.visual_encoder.d_model % self.alignment.num_heads:
            raise ValueError("d_model must be divisible by alignment.num_heads")
        if self.visual_encoder.d_model % self.verification.num_heads:
            raise ValueError("d_model must be divisible by verification.num_heads")
