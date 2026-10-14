"""CSV-backed GSV images and balanced positive/negative A/B pair batches."""

from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
import json
import operator
import os
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from data.loader_utils import resolve_under, seed_worker
from data.osg_storage import file_sha256, graph_file_candidates, read_json, validate_graph
from data.pair_collation import collate_image_osg_pairs
from data.samplers import PairBatchSampler, validate_pair_batch_settings
from data.preprocessing import (
    GraphFilterConfig, ImageGraphPreprocessor, IMAGENET_MEAN, IMAGENET_STD,
)


DEFAULT_GSV_ROOT = Path(__file__).resolve().parents[1] / "Dataset" / "GSV_CITITES"


@dataclass(frozen=True)
class GSVImage:
    city: str
    place_id: int
    label: int
    filename: str

    @property
    def relative_path(self):
        return f"Images/{self.city}/{self.filename}"

    @property
    def graph_path(self):
        return f"{self.city}/{Path(self.filename).stem}.json"


class GSVCitiesDataset(Dataset):
    """One <image, OSG> per item; sampler supplies positive/negative groups.

    Metadata is authoritative, including the full panoid (which can contain
    underscores). CSV coordinate strings are preserved when resolving names.
    Labels are unique across cities. No model inference occurs in workers.
    """

    def __init__(self, root=DEFAULT_GSV_ROOT, image_size=(224, 224),
                 mean=IMAGENET_MEAN, std=IMAGENET_STD, load_osg=True,
                 osg_root=None, graph_config=None, horizontal_flip_prob=0.0):
        self.root = Path(root).expanduser().resolve()
        self.osg_root = self.root / "osg" if osg_root is None else Path(osg_root).expanduser().resolve()
        self.load_osg = load_osg
        self.graph_config = graph_config or GraphFilterConfig()
        self.preprocessor = ImageGraphPreprocessor(
            image_size, mean, std, self.graph_config, horizontal_flip_prob)
        self.csv_paths = sorted((self.root / "Dataframes").glob("*.csv"))
        if not self.csv_paths:
            raise FileNotFoundError(f"No GSV-Cities CSVs in {self.root / 'Dataframes'}")
        self.images = []
        self.place_to_indices = defaultdict(list)
        self.place_keys = []
        labels = {}
        self.city_counts = {}
        required = {"place_id", "year", "month", "northdeg", "city_id", "lat", "lon", "panoid"}
        for csv_path in self.csv_paths:
            directory = self.root / "Images" / csv_path.stem
            if not directory.is_dir():
                raise FileNotFoundError(f"Missing extracted image directory: {directory}")
            # One directory scan instead of hundreds of thousands of stat calls.
            filenames = {entry.name.casefold(): entry.name for entry in os.scandir(directory)
                         if entry.is_file() and Path(entry.name).suffix.lower() in {".jpg", ".jpeg", ".png"}}
            seen = set()
            with csv_path.open(newline="", encoding="utf-8-sig") as stream:
                reader = csv.DictReader(stream)
                if required - set(reader.fieldnames or []):
                    raise ValueError(f"Missing metadata columns in {csv_path}")
                for row_number, row in enumerate(reader, start=2):
                    city, place = row["city_id"], int(row["place_id"])
                    if city != csv_path.stem or place < 0:
                        raise ValueError(f"Invalid city/place in {csv_path}:{row_number}")
                    expected = (f"{city}_{place:07d}_{int(row['year']):04d}_{int(row['month']):02d}_"
                                f"{int(row['northdeg']):03d}_{row['lat']}_{row['lon']}_{row['panoid']}.jpg")
                    filename = filenames.get(expected.casefold())
                    if filename is None:
                        raise FileNotFoundError(f"Missing image for {csv_path.name}:{row_number}: {directory / expected}")
                    if filename in seen:
                        raise ValueError(f"Duplicate image in {csv_path}:{row_number}: {filename}")
                    seen.add(filename)
                    key = (city, place)
                    if key not in labels:
                        labels[key] = len(labels)
                        self.place_keys.append(key)
                    label = labels[key]
                    self.place_to_indices[label].append(len(self.images))
                    self.images.append(GSVImage(city, place, label, filename))
            self.city_counts[csv_path.stem] = len(seen)
        if not self.images:
            raise ValueError("GSV-Cities metadata contains no images")
        self.place_to_indices = dict(self.place_to_indices)
        self.generation_id = None
        self.manifest = None
        if load_osg:
            self._read_manifest()

    def metadata_hashes(self):
        return {path.relative_to(self.root).as_posix(): file_sha256(path) for path in self.csv_paths}

    def _read_manifest(self):
        path = self.osg_root / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing GSV-Cities OSG manifest: {path}. "
                                    "Export source graphs with --prepare-data first.")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (manifest.get("schema_version") != 1 or manifest.get("dataset") != "GSV-Cities"
                or manifest.get("total_images") != len(self)
                or manifest.get("image_path_base") != "dataset_root"):
            raise ValueError("Unsupported GSV-Cities OSG manifest")
        if manifest["source_csv_sha256"] != self.metadata_hashes():
            raise ValueError("GSV-Cities CSVs changed after OSG export")
        for key in ("object_threshold", "relation_threshold"):
            if getattr(self.graph_config, key) < manifest["generation"][key]:
                raise ValueError(f"Loader {key} is below the export floor")
        self.generation_id = manifest["generation_id"]
        self.graph_format = manifest.get("graph_format", "json")
        graph_file_candidates(Path("graph.json"), self.graph_format)
        self.manifest = manifest

    def ensure_complete_graphs(self):
        """Reject unfinished exports and missing files before any training update."""
        if not self.load_osg:
            raise ValueError("COSMIC training requires exported GSV OSGs")
        count = len(self)
        if (self.manifest.get("complete_dataset") is not True
                or self.manifest.get("requested_images") != count
                or self.manifest.get("completed_images") != count):
            raise ValueError("GSV OSG export is incomplete; finish/resume --prepare-data "
                             "without --max-images before training")
        required = defaultdict(list)
        for entry in self.images:
            required[entry.city].append(entry)
        # One directory scan per city avoids a stat call for every image on G:.
        for city, entries in required.items():
            directory = self.osg_root / city
            if not directory.is_dir():
                raise FileNotFoundError(f"Missing GSV OSG directory: {directory}. Resume --prepare-data.")
            with os.scandir(directory) as scan:
                filenames = {entry.name for entry in scan if entry.is_file()}
            for entry in entries:
                candidates = graph_file_candidates(Path(entry.graph_path), self.graph_format)
                if not any(path.name in filenames for path in candidates):
                    raise FileNotFoundError(f"Missing training OSG: {self.osg_root / candidates[0]}. "
                                            "Resume --prepare-data.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        index = operator.index(index)
        entry = self.images[index]
        path = self.root / entry.relative_path
        graph, graph_path = None, None
        if self.load_osg:
            candidates = graph_file_candidates(resolve_under(self.osg_root, entry.graph_path), self.graph_format)
            graph_path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
            if not graph_path.is_file():
                raise FileNotFoundError(f"Missing GSV-Cities source OSG: {graph_path}. Resume --prepare-data.")
            try:
                graph = read_json(graph_path)
                validate_graph(graph)
                if graph["image_id"] != entry.relative_path or graph["image_path"] != entry.relative_path:
                    raise ValueError("Image identity mismatch")
                if graph["provenance"]["generation_id"] != self.generation_id:
                    raise ValueError("Generation fingerprint mismatch")
                if graph["provenance"]["image_sha256"] != file_sha256(path):
                    raise ValueError("Image changed after OSG export")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid source OSG {graph_path}: {exc}") from exc
        with Image.open(path) as source:
            image = source.convert("RGB")
            width, height = image.size
            tensor, osg = self.preprocessor(image, graph)
        return {"image": tensor, "osg": osg, "label": entry.label,
                "city": entry.city, "place_id": entry.place_id,
                "image_id": entry.relative_path, "image_path": str(path),
                "osg_path": str(graph_path) if graph_path else None,
                "original_size": torch.tensor([height, width]), "image_index": index % len(self)}


class GSVCitiesPairDataset(Dataset):
    """Load (image_index_a, image_index_b) keys supplied by the pair sampler.

    Length is the nominal number of sampled pairs per epoch before dropping
    the tail. The per-image dataset remains available for export/inference.
    """

    def __init__(self, image_dataset, images_per_place=4):
        self.image_dataset = image_dataset
        self.num_pairs = sum(len(indices) >= images_per_place
                             for indices in image_dataset.place_to_indices.values()) * images_per_place // 2

    def __len__(self):
        return self.num_pairs

    def ensure_complete_graphs(self):
        self.image_dataset.ensure_complete_graphs()

    def __getitem__(self, indices):
        if not isinstance(indices, tuple) or len(indices) != 2:
            raise TypeError("GSV pair items require two image indices from PairBatchSampler")
        a_index, b_index = map(operator.index, indices)
        if not (0 <= a_index < len(self.image_dataset) and 0 <= b_index < len(self.image_dataset)):
            raise IndexError("GSV pair image index out of range")
        if a_index == b_index:
            raise ValueError("A pair must contain two distinct images")
        a, b = self.image_dataset[a_index], self.image_dataset[b_index]
        a["place_label"], b["place_label"] = a.pop("label"), b.pop("label")
        return {"a": a, "b": b,
                "label": torch.tensor(float(a["place_label"] == b["place_label"]), dtype=torch.float32)}


def create_gsv_data_loader(root=DEFAULT_GSV_ROOT, batch_size=8, images_per_place=4,
                           num_workers=0, shuffle=True, drop_last=True,
                           pin_memory=False, seed=42, **dataset_kwargs):
    image_dataset = GSVCitiesDataset(root=root, **dataset_kwargs)
    sampler = PairBatchSampler(image_dataset, batch_size, images_per_place, shuffle, drop_last, seed)
    dataset = GSVCitiesPairDataset(image_dataset, images_per_place)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers,
                      pin_memory=pin_memory, generator=torch.Generator().manual_seed(seed),
                      collate_fn=collate_image_osg_pairs, worker_init_fn=seed_worker)
