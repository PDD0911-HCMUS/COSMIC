"""Dataset settings parsed by the central experiment YAML reader."""

from dataclasses import dataclass, field
from pathlib import Path

from data.preprocessing import GraphFilterConfig, IMAGENET_MEAN, IMAGENET_STD, RGBToTensor


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "gsv"
    root: Path = PROJECT_ROOT / "Dataset" / "GSV_CITITES"
    osg_root: Path | None = None
    batch_size: int = 8
    num_workers: int = 0
    shuffle: bool = False
    pin_memory: bool = False
    drop_last: bool = False
    seed: int = 42
    load_osg: bool = True
    image_size: tuple[int, int] = (224, 224)
    mean: tuple[float, float, float] | None = IMAGENET_MEAN
    std: tuple[float, float, float] | None = IMAGENET_STD
    horizontal_flip_prob: float = 0.0
    images_per_place: int = 4
    graph: GraphFilterConfig = field(default_factory=GraphFilterConfig)

    def __post_init__(self):
        from registry import get_dataset

        adapter = get_dataset(self.dataset)
        if self.root is None:
            raise ValueError("root must identify a dataset directory")
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if type(self.images_per_place) is not int or self.images_per_place < 2:
            raise ValueError("images_per_place must be an integer >= 2")
        if type(self.num_workers) is not int or self.num_workers < 0:
            raise ValueError("num_workers must be a nonnegative integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not 0 <= self.horizontal_flip_prob <= 1:
            raise ValueError("horizontal_flip_prob must be in [0, 1]")
        for name in ("shuffle", "pin_memory", "drop_last", "load_osg"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        RGBToTensor(self.image_size, self.mean, self.std)
        if adapter.validate_config is not None:
            adapter.validate_config(self)


def parse_data_config(payload: dict) -> DataConfig:
    """Parse a data section; also used by the experiment configuration."""
    from registry import get_dataset

    payload = dict(payload)
    allowed = set(DataConfig.__dataclass_fields__)
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"Unknown data config keys: {sorted(unknown)}")
    if "root" not in payload:
        payload["root"] = get_dataset(payload.get("dataset", "gsv")).default_root
    for name in ("root", "osg_root"):
        if name in payload and payload[name] is not None:
            value = Path(payload[name]).expanduser()
            payload[name] = value if value.is_absolute() else PROJECT_ROOT / value
    for name in ("image_size", "mean", "std"):
        if name in payload and payload[name] is not None:
            payload[name] = tuple(payload[name])
    payload["graph"] = GraphFilterConfig(**payload.get("graph", {}))
    return DataConfig(**payload)
