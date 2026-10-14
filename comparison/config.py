"""Comparison settings; relative paths resolve from the project root."""

from dataclasses import dataclass, field
from pathlib import Path
import re

import yaml

from config import _ConfigLoader
from data.config import PROJECT_ROOT


def project_path(value):
    if value is None:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def section(payload, allowed, name):
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a mapping")
    unknown = set(payload) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown {name} keys: {sorted(unknown)}")
    return dict(payload)


@dataclass(frozen=True)
class MethodConfig:
    adapter: str
    checkpoint: Path | None = None
    repo: Path | None = None
    config: Path | None = None
    pca: Path | None = None
    batch_size: int | None = None
    stages: tuple[str, ...] = ("retrieval",)
    rerank_top_k: int = 100
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.adapter, str) or not self.adapter:
            raise ValueError("method.adapter must be a nonempty name")
        if self.batch_size is not None:
            positive(self.batch_size, "method.batch_size")
        positive(self.rerank_top_k, "method.rerank_top_k")
        if (not isinstance(self.stages, (list, tuple)) or not self.stages
                or any(not isinstance(stage, str) or stage not in ("retrieval", "rerank")
                       for stage in self.stages)
                or len(set(self.stages)) != len(self.stages)):
            raise ValueError("method.stages must contain retrieval and/or rerank without duplicates")
        object.__setattr__(self, "stages", tuple(self.stages))
        if "rerank" in self.stages and self.adapter not in ("cosmic", "fol"):
            raise ValueError("Re-ranking adapters currently available: cosmic, fol")
        if not isinstance(self.params, dict):
            raise ValueError("method.params must be a mapping")


@dataclass(frozen=True)
class ComparisonConfig:
    dataset: str = "amstertime"
    data_root: Path = PROJECT_ROOT / "Dataset/AmsterTime"
    osg_root: Path | None = None
    expected_pairs: int = 1231
    device: str = "cpu"
    seed: int = 42
    batch_size: int = 8
    num_workers: int = 0
    cpu_threads: int = 2
    recall_k: tuple[int, ...] = (1, 5, 10)
    metric: str = "l2"
    query_chunk_size: int = 128
    database_chunk_size: int = 4096
    log_every: int = 20
    max_queries: int | None = None
    output_root: Path = PROJECT_ROOT / "outputs/comparison"
    hub_dir: Path = PROJECT_ROOT / ".cache/comparison_hub"
    methods: dict[str, MethodConfig] = field(default_factory=dict)

    def __post_init__(self):
        if self.dataset != "amstertime":
            raise ValueError("Comparison datasets currently available: amstertime")
        for name in ("expected_pairs", "batch_size", "cpu_threads", "query_chunk_size", "database_chunk_size", "log_every"):
            positive(getattr(self, name), name)
        for name in ("seed", "num_workers"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.max_queries is not None:
            positive(self.max_queries, "max_queries")
        if (not isinstance(self.device, str) or not re.fullmatch(r"auto|cpu|cuda(?::\d+)?", self.device)):
            raise ValueError("device must be auto, cpu, cuda or cuda:<index>")
        if (not isinstance(self.recall_k, (list, tuple)) or not self.recall_k
                or any(type(k) is not int or k <= 0 for k in self.recall_k)
                or len(set(self.recall_k)) != len(self.recall_k)):
            raise ValueError("recall_k must contain distinct positive integers")
        object.__setattr__(self, "recall_k", tuple(sorted(self.recall_k)))
        if self.metric not in ("l2", "cosine"):
            raise ValueError("metric must be l2 or cosine")
        if not self.methods:
            raise ValueError("Configure at least one comparison method")
        for name, method in self.methods.items():
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
                raise ValueError("Method names may contain letters, digits, underscores and hyphens")
            if "rerank" in method.stages and method.rerank_top_k < max(self.recall_k):
                raise ValueError(f"{name}: rerank_top_k must cover all recall_k values")


def load_comparison_config(path):
    path = project_path(path)
    payload = yaml.load(path.read_text(encoding="utf-8-sig"), Loader=_ConfigLoader)
    payload = section(payload, ComparisonConfig.__dataclass_fields__, "comparison")
    methods = payload.pop("methods", {})
    if not isinstance(methods, dict):
        raise ValueError("methods must map experiment names to adapter settings")
    parsed = {}
    for name, settings in methods.items():
        settings = section(settings, MethodConfig.__dataclass_fields__, f"methods.{name}")
        for key in ("checkpoint", "repo", "config", "pca"):
            if key in settings:
                settings[key] = project_path(settings[key])
        parsed[name] = MethodConfig(**settings)
    for key in ("data_root", "osg_root", "output_root", "hub_dir"):
        if key in payload:
            payload[key] = project_path(payload[key])
    for key in ("data_root", "output_root", "hub_dir"):
        if payload.get(key, True) is None:
            raise ValueError(f"{key} must identify a directory")
    return ComparisonConfig(methods=parsed, **payload)
