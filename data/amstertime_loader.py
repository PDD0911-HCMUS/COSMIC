"""Load AmsterTime as paired <image, OSG> inputs for COSMIC."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import operator
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data.loader_utils import resolve_under, seed_worker
from data.pair_collation import collate_image_osg_pairs
from data.osg_storage import file_sha256, read_json, validate_graph
from data.preprocessing import (
    GraphFilterConfig, ImageGraphPreprocessor, IMAGENET_MEAN, IMAGENET_STD,
)


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "Dataset" / "AmsterTime"


@dataclass(frozen=True)
class ImagePair:
    image_id_a: str
    image_id_b: str
    path_a: Path
    path_b: Path


class AmsterTimeDataset(Dataset):
    """Read CSV pairs and source OSGs, then apply joint preprocessing.

    A sample contains a/b branches with image, osg, image_id, image_path,
    osg_path and original_size, plus label and pair_index. Graph tensors
    stay on CPU; collation preserves variable-size graphs as lists.
    load_osg=False supports data export and image-only previews.
    """

    def __init__(self, root: str | Path = DEFAULT_DATA_ROOT,
                 transform: Callable[[Image.Image], torch.Tensor] | None = None,
                 image_size=(224, 224), mean=IMAGENET_MEAN, std=IMAGENET_STD,
                 load_osg: bool = True, osg_root: str | Path | None = None,
                 graph_config: GraphFilterConfig | None = None, horizontal_flip_prob=0.0):
        self.root = Path(root).expanduser().resolve()
        self.osg_root = self.root / "osg" if osg_root is None else Path(osg_root).expanduser().resolve()
        self.load_osg = load_osg
        self.graph_config = graph_config or GraphFilterConfig()
        if load_osg and transform is not None:
            raise ValueError("Independent image transforms cannot preserve OSG geometry; use joint preprocessing")
        if transform is not None and horizontal_flip_prob != 0:
            raise ValueError("Configure augmentation in the custom transform for image-only loading")
        self.custom_transform = transform
        self.preprocessor = ImageGraphPreprocessor(image_size, mean, std, self.graph_config, horizontal_flip_prob)
        self.pairs = self._read_pairs()
        self.osg_entries = self._read_osg_index() if load_osg else None

    def _read_osg_index(self):
        index_path = self.osg_root / "index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"Missing OSG index: {index_path}. Run python main.py --prepare-data first.")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if (index["schema_version"] != 1 or index["image_path_base"] != "dataset_root"
                or index["osg_path_base"] != "index_directory"):
            raise ValueError(f"Unsupported OSG index: {index_path}")
        if index["source_csv_sha256"] != file_sha256(self.root / "image_pairs.csv"):
            raise ValueError("OSG index does not match image_pairs.csv; export the dataset again")
        if index["total_dataset_pairs"] != len(self.pairs):
            raise ValueError("OSG index pair count differs from the source CSV")
        for key, threshold in (("object_threshold", self.graph_config.object_threshold),
                               ("relation_threshold", self.graph_config.relation_threshold)):
            if threshold < index["generation"][key]:
                raise ValueError(f"Loader {key} is below the export floor; lower the floor and re-export")
        self.generation_id = index["generation_id"]
        records = {}
        for record in index["pairs"]:
            pair_index = record["pair_index"]
            if type(pair_index) is not int or not 0 <= pair_index < len(self.pairs) or pair_index in records:
                raise ValueError("OSG index contains invalid or duplicate pair indices")
            if record["label"] != 1:
                raise ValueError("AmsterTime CSV pairs must be positive matches")
            pair = self.pairs[pair_index]
            for side, path, image_id in (("a", pair.path_a, pair.image_id_a), ("b", pair.path_b, pair.image_id_b)):
                entry = record[side]
                if entry["image_id"] != image_id or resolve_under(self.root, entry["image_path"]) != path:
                    raise ValueError(f"OSG index image mismatch at pair {pair_index}, side {side}")
                graph_path = resolve_under(self.osg_root, entry["osg_path"])
                if not graph_path.is_file():
                    raise FileNotFoundError(f"Missing source OSG: {graph_path}")
            records[pair_index] = record
        if len(records) != len(self.pairs):
            raise ValueError("OSG index is incomplete; run python main.py --prepare-data without --max-pairs")
        return records

    def _read_pairs(self) -> list[ImagePair]:
        csv_path = self.root / "image_pairs.csv"
        required = {"old_id", "old_num", "new_id", "new_num"}
        pairs = []
        with csv_path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{csv_path}: missing columns {sorted(missing)}")
            for row_number, row in enumerate(reader, start=2):
                try:
                    old_num, new_num = int(row["old_num"]), int(row["new_num"])
                    if old_num < 0 or new_num < 0:
                        raise ValueError("negative image number")
                    old_id, new_id = row["old_id"].strip(), row["new_id"].strip()
                    if not old_id or not new_id:
                        raise ValueError("empty image ID")
                except (TypeError, ValueError, AttributeError) as exc:
                    raise ValueError(f"{csv_path}: invalid pair at row {row_number}") from exc
                pair = ImagePair(
                    image_id_a=old_id,
                    image_id_b=new_id,
                    path_a=self.root / "old" / f"{old_num:04d}.jpg",
                    path_b=self.root / "new" / f"{new_num:04d}.png",
                )
                for path in (pair.path_a, pair.path_b):
                    if not path.is_file():
                        raise FileNotFoundError(f"Missing image at CSV row {row_number}: {path}")
                pairs.append(pair)
        if not pairs:
            raise ValueError(f"{csv_path}: no image pairs found")
        return pairs

    def __len__(self):
        return len(self.pairs)

    def _load_branch(self, index, side, path, image_id):
        graph_path = None
        graph = None
        if self.load_osg:
            entry = self.osg_entries[index][side]
            graph_path = resolve_under(self.osg_root, entry["osg_path"])
            try:
                graph = read_json(graph_path)
                validate_graph(graph)
                if graph["image_id"] != image_id or graph["image_path"] != entry["image_path"]:
                    raise ValueError("Image identity mismatch")
                if graph["provenance"]["generation_id"] != self.generation_id:
                    raise ValueError("Generation fingerprint mismatch")
                if graph["provenance"]["image_sha256"] != file_sha256(path):
                    raise ValueError("Image changed after OSG export")
            except (ValueError, KeyError, TypeError, OSError) as exc:
                raise ValueError(f"Invalid source OSG {graph_path}: {exc}") from exc
        with Image.open(path) as source:
            image = source.convert("RGB")
            width, height = image.size
            if self.custom_transform is not None:
                tensor, osg = self.custom_transform(image), None
            else:
                tensor, osg = self.preprocessor(image, graph)
        return {
            "image": tensor, "osg": osg, "image_id": image_id,
            "image_path": str(path), "osg_path": str(graph_path) if graph_path else None,
            "original_size": torch.tensor([height, width]),
        }

    def __getitem__(self, index):
        index = operator.index(index)
        pair = self.pairs[index]
        index = index % len(self)
        return {
            "a": self._load_branch(index, "a", pair.path_a, pair.image_id_a),
            "b": self._load_branch(index, "b", pair.path_b, pair.image_id_b),
            "label": torch.tensor(1.0, dtype=torch.float32),
            "pair_index": index,
        }


def create_amstertime_data_loader(root: str | Path = DEFAULT_DATA_ROOT, batch_size=8, shuffle=False,
                       num_workers=0, image_size=(224, 224), transform=None, pin_memory=False,
                       drop_last=False, generator=None, mean=IMAGENET_MEAN, std=IMAGENET_STD,
                       load_osg=True, osg_root=None, graph_config=None, horizontal_flip_prob=0.0,
                       seed=42) -> DataLoader:
    """Create CPU batches; frozen OvSGTR is not loaded or called by workers."""
    dataset = AmsterTimeDataset(
        root=root, transform=transform, image_size=image_size, mean=mean, std=std,
        load_osg=load_osg, osg_root=osg_root, graph_config=graph_config,
        horizontal_flip_prob=horizontal_flip_prob,
    )
    if generator is None:
        generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=pin_memory, drop_last=drop_last, generator=generator,
        collate_fn=collate_image_osg_pairs, worker_init_fn=seed_worker,
    )
