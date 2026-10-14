"""Named dataset adapters for training data, retrieval validation and OSG export."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from data.amstertime_loader import create_amstertime_data_loader
from data.gsv_cities_loader import create_gsv_data_loader, validate_pair_batch_settings
from data.pitts30k_loader import build_pitts30k_validation


@dataclass(frozen=True)
class DatasetAdapter:
    name: str
    loader_factory: Callable | None
    exporter: Callable
    default_root: Path
    export_limit: str
    extra_loader_fields: tuple[str, ...] = ()
    validate_config: Callable | None = None
    validation_factory: Callable | None = None
    evaluation_only: bool = False

    def build_loader(self, config):
        if self.loader_factory is None:
            raise ValueError(f"Dataset {self.name!r} uses separate query/database loaders; use build_validation_inputs")
        kwargs = {key: getattr(config, key) for key in (
            "root", "osg_root", "batch_size", "num_workers", "shuffle", "pin_memory",
            "drop_last", "seed", "image_size", "mean", "std", "load_osg", "horizontal_flip_prob")}
        kwargs["graph_config"] = config.graph
        kwargs.update({key: getattr(config, key) for key in self.extra_loader_fields})
        return self.loader_factory(**kwargs)


DATASET_REGISTRY: dict[str, DatasetAdapter] = {}
# Canonical names accepted by YAML; register_dataset updates this list.
DATASET_NAMES: list[str] = []


def register_dataset(adapter: DatasetAdapter, aliases=()):
    """Register one adapter; aliases resolve to the same implementation."""
    names = (adapter.name, *aliases)
    if any(not isinstance(name, str) or not name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError("Dataset names/aliases must be nonempty and unique")
    duplicates = set(names) & DATASET_REGISTRY.keys()
    if duplicates:
        raise ValueError(f"Datasets already registered: {sorted(duplicates)}")
    DATASET_REGISTRY.update({name: adapter for name in names})
    DATASET_NAMES.append(adapter.name)
    return adapter


def get_dataset(name: str) -> DatasetAdapter:
    adapter = DATASET_REGISTRY.get(name) if isinstance(name, str) else None
    # Existing aliases (e.g. gsv_cities) resolve to a canonical listed name.
    if adapter is None or adapter.name not in DATASET_NAMES:
        raise ValueError(f"Unknown dataset {name!r}. Available: {', '.join(sorted(DATASET_NAMES))}")
    return adapter


def build_data_loader(config, *, require_complete_osg=False):
    """Build paired data; training can require a complete source-graph export."""
    loader = get_dataset(config.dataset).build_loader(config)
    if require_complete_osg:
        ensure_complete = getattr(loader.dataset, "ensure_complete_graphs", None)
        if not callable(ensure_complete):
            raise ValueError(f"Dataset {config.dataset!r} has no complete-OSG readiness check")
        ensure_complete()
    return loader


def _export_amstertime(**kwargs):
    from data.prepare_data import prepare_amstertime_osg
    return prepare_amstertime_osg(**kwargs)


def _export_gsv(**kwargs):
    from data.prepare_data import prepare_gsv_osg
    return prepare_gsv_osg(**kwargs)


def _export_pitts30k(**kwargs):
    from data.pitts30k_export import prepare_pitts30k_osg
    return prepare_pitts30k_osg(**kwargs)


def _validate_gsv(config):
    validate_pair_batch_settings(config.batch_size, config.images_per_place)


_DATA_ROOT = Path(__file__).resolve().parent / "Dataset"
register_dataset(DatasetAdapter(
    name="gsv", loader_factory=create_gsv_data_loader,
    exporter=_export_gsv, default_root=_DATA_ROOT / "GSV_CITITES",
    export_limit="max_images",
    extra_loader_fields=("images_per_place",), validate_config=_validate_gsv,
), aliases=("gsv_cities",))
register_dataset(DatasetAdapter(
    name="amstertime", loader_factory=create_amstertime_data_loader,
    exporter=_export_amstertime, default_root=_DATA_ROOT / "AmsterTime",
    export_limit="max_pairs",
    evaluation_only=True,
))
register_dataset(DatasetAdapter(
    name="pitts30k", loader_factory=None,
    exporter=_export_pitts30k, default_root=_DATA_ROOT.parent / "Pitts30k",
    export_limit="max_images",
    validation_factory=build_pitts30k_validation, evaluation_only=True,
))
