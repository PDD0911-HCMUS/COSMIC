"""Validated experiment settings; paths are independent of the shell cwd."""

from dataclasses import asdict, dataclass, field, replace
import math
import os
from pathlib import Path
import tempfile

import yaml

from data.config import DataConfig, PROJECT_ROOT, parse_data_config
from model.config import AlignmentConfig, CriteriaConfig, GroundingConfig, ModelConfig, OSGEncoderConfig, SceneSimilarityConfig, VerificationConfig, VisualEncoderConfig


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative_number(value, name, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be a finite {'positive' if positive else 'nonnegative'} number")


def _boolean_fields(instance, *names):
    for name in names:
        if type(getattr(instance, name)) is not bool:
            raise ValueError(f"{name} must be boolean")


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 0.0001
    weight_decay: float = 0.0001

    def __post_init__(self):
        if self.name not in ("adamw", "adam", "sgd"):
            raise ValueError("optimizer.name must be adamw, adam or sgd")
        _nonnegative_number(self.lr, "optimizer.lr", positive=True)
        _nonnegative_number(self.weight_decay, "optimizer.weight_decay")


@dataclass(frozen=True)
class SchedulerConfig:
    name: str = "cosine"
    warmup_epochs: int = 0
    min_lr: float = 0.000001

    def __post_init__(self):
        if self.name not in ("cosine", "none"):
            raise ValueError("scheduler.name must be cosine or none")
        if type(self.warmup_epochs) is not int or self.warmup_epochs < 0:
            raise ValueError("scheduler.warmup_epochs must be a nonnegative integer")
        _nonnegative_number(self.min_lr, "scheduler.min_lr")


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 50
    precision: str = "fp32"
    accumulation_steps: int = 1
    grad_clip_norm: float | None = 1.0
    validate_every: int = 1
    log_every: int = 50
    max_batches_per_epoch: int | None = None
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    def __post_init__(self):
        for name in ("epochs", "accumulation_steps", "validate_every", "log_every"):
            _positive_integer(getattr(self, name), f"training.{name}")
        if self.max_batches_per_epoch is not None:
            _positive_integer(self.max_batches_per_epoch, "training.max_batches_per_epoch")
        if self.precision not in ("fp32", "amp_fp16", "amp_bf16"):
            raise ValueError("training.precision must be fp32, amp_fp16 or amp_bf16")
        if self.grad_clip_norm is not None:
            _nonnegative_number(self.grad_clip_norm, "training.grad_clip_norm", positive=True)
        if self.scheduler.warmup_epochs >= self.epochs:
            raise ValueError("scheduler.warmup_epochs must be smaller than training.epochs")
        if self.scheduler.min_lr > self.optimizer.lr:
            raise ValueError("scheduler.min_lr must not exceed optimizer.lr")


@dataclass(frozen=True)
class SaveConfig:
    enabled: bool = True
    directory: str = "checkpoints"
    save_every: int = 5
    keep_last: int = 3
    save_best: bool = True
    monitor: str = "recall@1"
    mode: str = "max"
    resume: Path | None = None

    def __post_init__(self):
        _boolean_fields(self, "enabled", "save_best")
        _positive_integer(self.save_every, "save.save_every")
        _positive_integer(self.keep_last, "save.keep_last")
        _relative_directory(self.directory, "save.directory")
        if self.mode not in ("min", "max"):
            raise ValueError("save.mode must be min or max")
        if not isinstance(self.monitor, str) or not self.monitor.strip():
            raise ValueError("save.monitor must be nonempty")


def _relative_directory(value, name):
    if (not isinstance(value, str) or not value.strip() or Path(value).anchor
            or ".." in Path(value).parts):
        raise ValueError(f"{name} must be a relative directory inside the output directory")


@dataclass(frozen=True)
class OutputConfig:
    root: Path = PROJECT_ROOT / "outputs"
    timestamp: bool = False
    visualizations: str = "visualizations"
    show: bool = True
    save_config: bool = True

    def __post_init__(self):
        if self.root is None:
            raise ValueError("output.root must identify a directory")
        _boolean_fields(self, "timestamp", "show", "save_config")
        _relative_directory(self.visualizations, "output.visualizations")


@dataclass(frozen=True)
class ValidationConfig:
    recall_k: tuple[int, ...] = (1, 5, 10, 20)
    query_chunk_size: int = 128
    database_chunk_size: int = 4096
    pair_threshold: float = 0.5
    log_every: int = 50
    save_reports: bool = True
    directory: str = "validation"
    data: DataConfig | None = None

    def __post_init__(self):
        if (not isinstance(self.recall_k, (tuple, list)) or not self.recall_k
                or any(type(k) is not int or k <= 0 for k in self.recall_k)
                or len(set(self.recall_k)) != len(self.recall_k)):
            raise ValueError("validation.recall_k must contain distinct positive integers")
        object.__setattr__(self, "recall_k", tuple(sorted(self.recall_k)))
        for key in ("query_chunk_size", "database_chunk_size", "log_every"):
            _positive_integer(getattr(self, key), f"validation.{key}")
        _nonnegative_number(self.pair_threshold, "validation.pair_threshold")
        if self.pair_threshold > 1:
            raise ValueError("validation.pair_threshold must be in [0,1]")
        _boolean_fields(self, "save_reports")
        _relative_directory(self.directory, "validation.directory")
        if self.data is not None:
            if not isinstance(self.data, DataConfig):
                raise ValueError("validation.data must contain parsed data settings")
            if (not self.data.load_osg or self.data.shuffle or self.data.drop_last
                    or self.data.horizontal_flip_prob != 0):
                raise ValueError("Validation requires OSGs, no shuffle/drop_last, and no random horizontal flip")


@dataclass(frozen=True)
class ExportConfig:
    max_items: int | None = None
    overwrite: bool = False
    object_threshold: float = 0.0
    relation_threshold: float = 0.0
    graph_format: str = "json_gzip"

    def __post_init__(self):
        if self.graph_format not in ("json", "json_gzip"):
            raise ValueError("export.graph_format must be json or json_gzip")
        if self.max_items is not None:
            _positive_integer(self.max_items, "export.max_items")
        _boolean_fields(self, "overwrite")
        for key in ("object_threshold", "relation_threshold"):
            value = getattr(self, key)
            _nonnegative_number(value, f"export.{key}")
            if value > 1:
                raise ValueError(f"export.{key} must be in [0, 1]")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "cosmic"
    mode: str = "train"
    seed: int = 42
    device: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    save: SaveConfig = field(default_factory=SaveConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be nonempty")
        if self.mode not in ("prepare", "train", "evaluate"):
            raise ValueError("mode must be prepare, train or evaluate")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if (not isinstance(self.device, str) or self.device not in ("auto", "cpu", "cuda")
                and not (self.device.startswith("cuda:") and self.device[5:].isdigit())):
            raise ValueError("device must be auto, cpu, cuda or cuda:<index>")
        # Keep sampler/worker seeds aligned after CLI overrides too.
        object.__setattr__(self, "data", replace(self.data, seed=self.seed))
        if self.validation.data is not None:
            object.__setattr__(self, "validation", replace(self.validation,
                               data=replace(self.validation.data, seed=self.seed)))


def _section(cls, payload, name):
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be an object")
    unknown = set(payload) - cls.__dataclass_fields__.keys()
    if unknown:
        raise ValueError(f"Unknown {name} keys: {sorted(unknown)}")
    return dict(payload)


def _project_path(value):
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


class _ConfigLoader(yaml.SafeLoader):
    """Reject duplicate/non-string settings instead of silently overriding."""

    def construct_mapping(self, node, deep=False):
        keys = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError("Config mapping keys must be strings")
            if key in keys:
                raise ValueError(f"Duplicate config key {key!r} at line {key_node.start_mark.line + 1}")
            keys.add(key)
        return super().construct_mapping(node, deep=deep)


def load_config(path=DEFAULT_CONFIG_PATH) -> ExperimentConfig:
    """Read and validate the experiment YAML."""
    path = Path(path)
    source = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError("Config file must have a .yaml or .yml extension")
    payload = yaml.load(source, Loader=_ConfigLoader)
    if not isinstance(payload, dict):
        raise ValueError("Config must be an object")
    payload = _section(ExperimentConfig, payload, "experiment config")
    data_payload = payload.pop("data", {})
    if not isinstance(data_payload, dict):
        raise ValueError("data must be an object")
    if "seed" in data_payload:
        if "seed" in payload and payload["seed"] != data_payload["seed"]:
            raise ValueError("Use one seed for the experiment and data sampler")
        payload.setdefault("seed", data_payload["seed"])
    payload["data"] = parse_data_config(data_payload)
    model = _section(ModelConfig, payload.pop("model", {}), "model")
    model["visual_encoder"] = VisualEncoderConfig(**_section(
        VisualEncoderConfig, model.get("visual_encoder", {}), "model.visual_encoder"))
    model["osg_encoder"] = OSGEncoderConfig(**_section(
        OSGEncoderConfig, model.get("osg_encoder", {}), "model.osg_encoder"))
    model["grounding"] = GroundingConfig(**_section(
        GroundingConfig, model.get("grounding", {}), "model.grounding"))
    model["alignment"] = AlignmentConfig(**_section(
        AlignmentConfig, model.get("alignment", {}), "model.alignment"))
    model["verification"] = VerificationConfig(**_section(
        VerificationConfig, model.get("verification", {}), "model.verification"))
    model["scene_similarity"] = SceneSimilarityConfig(**_section(
        SceneSimilarityConfig, model.get("scene_similarity", {}), "model.scene_similarity"))
    model["criteria"] = CriteriaConfig(**_section(
        CriteriaConfig, model.get("criteria", {}), "model.criteria"))
    payload["model"] = ModelConfig(**model)
    training = _section(TrainingConfig, payload.pop("training", {}), "training")
    training["optimizer"] = OptimizerConfig(**_section(
        OptimizerConfig, training.get("optimizer", {}), "training.optimizer"))
    training["scheduler"] = SchedulerConfig(**_section(
        SchedulerConfig, training.get("scheduler", {}), "training.scheduler"))
    payload["training"] = TrainingConfig(**training)
    validation = _section(ValidationConfig, payload.pop("validation", {}), "validation")
    if validation.get("data") is not None:
        inherited = {key: getattr(payload["data"], key) for key in ("image_size", "mean", "std")}
        inherited["graph"] = asdict(payload["data"].graph)
        inherited.update(_section(DataConfig, validation["data"], "validation.data"))
        validation["data"] = parse_data_config(inherited)
    payload["validation"] = ValidationConfig(**validation)
    save = _section(SaveConfig, payload.pop("save", {}), "save")
    if "resume" in save:
        save["resume"] = _project_path(save["resume"])
    payload["save"] = SaveConfig(**save)
    output = _section(OutputConfig, payload.pop("output", {}), "output")
    if "root" in output:
        output["root"] = _project_path(output["root"])
    payload["output"] = OutputConfig(**output)
    payload["export"] = ExportConfig(**_section(ExportConfig, payload.pop("export", {}), "export"))
    return ExperimentConfig(**payload)


def config_to_dict(config):
    def portable(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: portable(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [portable(item) for item in value]
        return value
    return portable(asdict(config))


def save_config(config, path):
    """Atomically save the effective configuration as readable YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            yaml.safe_dump(config_to_dict(config), stream, sort_keys=False, allow_unicode=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
