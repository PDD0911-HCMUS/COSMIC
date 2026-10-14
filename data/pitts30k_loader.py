"""Pitts30k-val observations, sequential loaders and 25-meter retrieval ground truth."""

import operator
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import torch
from torch.utils.data import BatchSampler, Dataset, DataLoader, SequentialSampler, Subset

from data.loader_utils import resolve_under, seed_worker
from data.osg_storage import file_sha256, graph_file_candidates, read_json, validate_graph, write_json_atomic
from data.pitts30k_metadata import DEFAULT_ROOT, read_pitts30k_metadata
from data.preprocessing import GraphFilterConfig, ImageGraphPreprocessor, IMAGENET_MEAN, IMAGENET_STD


def retrieval_positives(metadata):
    """All database images within the official UTM radius; no visual-label inference."""
    database = np.asarray([entry.utm for entry in metadata.database], dtype=np.float64)
    queries = np.asarray([entry.utm for entry in metadata.queries], dtype=np.float64)
    neighbors = cKDTree(database).query_ball_point(queries, metadata.radius)
    return {query.image_id: [metadata.database[i].image_id for i in sorted(indices)]
            for query, indices in zip(metadata.queries, neighbors)}


def prepare_validation_metadata(root=DEFAULT_ROOT):
    metadata = read_pitts30k_metadata(root)
    positives = retrieval_positives(metadata)
    ground_truth_path = metadata.root / "metadata/ground_truth_25m.json.gz"
    write_json_atomic(ground_truth_path, {"schema_version":1, "source_mat_sha256":metadata.source_sha256,
                                        "positive_radius_m":metadata.radius, "positives":positives})
    manifest = {"schema_version":1, "dataset":"Pitts30k", "split":"val",
                "source_mat_sha256":metadata.source_sha256, "positive_radius_m":metadata.radius,
                "query_ids":[entry.image_id for entry in metadata.queries],
                "database_ids":[entry.image_id for entry in metadata.database],
                "queries":len(metadata.queries), "database":len(metadata.database),
                "empty_ground_truth_queries":sum(not values for values in positives.values()),
                "ground_truth_path":ground_truth_path.relative_to(metadata.root).as_posix(),
                "ground_truth_sha256":file_sha256(ground_truth_path)}
    write_json_atomic(metadata.root / "metadata/validation_manifest.json", manifest)
    return manifest


class Pitts30kDataset(Dataset):
    """Single <image, OSG> observations, database first then queries, without augmentation."""

    def __init__(self, root=DEFAULT_ROOT, *, load_osg=True, osg_root=None, image_size=(224, 224),
                 mean=IMAGENET_MEAN, std=IMAGENET_STD, graph_config=None, horizontal_flip_prob=0.0):
        if horizontal_flip_prob != 0:
            raise ValueError("Pitts30k-val observations must not use random horizontal flips")
        self.metadata = read_pitts30k_metadata(root)
        self.root = self.metadata.root
        self.images = self.metadata.images
        self.osg_root = self.root / "osg" if osg_root is None else Path(osg_root).expanduser().resolve()
        self.load_osg = load_osg
        self.graph_config = graph_config or GraphFilterConfig()
        self.preprocessor = ImageGraphPreprocessor(image_size=image_size, mean=mean, std=std,
                                                   graph_config=self.graph_config, horizontal_flip_prob=0.0)
        self.manifest = None
        if load_osg:
            self.manifest = read_json(self.osg_root / "manifest.json")
            if (self.manifest.get("schema_version") != 1 or self.manifest.get("dataset") != "Pitts30k"
                    or self.manifest.get("split") != "val"
                    or self.manifest.get("source_mat_sha256") != self.metadata.source_sha256
                    or self.manifest.get("total_images") != len(self.images)
                    or self.manifest.get("image_path_base") != "dataset_root"):
                raise ValueError("Pitts30k OSG manifest differs from the source validation split")
            for key in ("object_threshold", "relation_threshold"):
                if getattr(self.graph_config, key) < self.manifest["generation"][key]:
                    raise ValueError(f"Pitts30k loader {key} is below the export floor")
            graph_file_candidates(Path("graph.json"), self.manifest["graph_format"])

    def graph_path(self, entry):
        candidates = graph_file_candidates(resolve_under(self.osg_root, entry.graph_path.as_posix()),
                                           self.manifest["graph_format"])
        return next((path for path in candidates if path.is_file()), candidates[0])

    def ensure_complete_graphs(self):
        if not self.load_osg:
            raise ValueError("COSMIC validation requires exported OSGs")
        if not self.manifest.get("complete_dataset") or self.manifest.get("completed_images") != len(self.images):
            raise ValueError("Pitts30k OSG export is incomplete; resume --prepare-data without --max-images")
        for entry in self.images:
            if not self.graph_path(entry).is_file():
                raise FileNotFoundError(f"Missing validation OSG: {self.graph_path(entry)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        entry = self.images[operator.index(index)]
        path = resolve_under(self.root, entry.relative_path)
        graph = graph_path = None
        if self.load_osg:
            graph_path = self.graph_path(entry)
            graph = read_json(graph_path)
            validate_graph(graph)
            if (graph["image_id"] != entry.image_id or graph["image_path"] != entry.relative_path
                    or graph["provenance"]["generation_id"] != self.manifest["generation_id"]
                    or graph["provenance"]["image_sha256"] != file_sha256(path)):
                raise ValueError(f"OSG image identity/provenance mismatch: {graph_path}")
        with Image.open(path) as source:
            image = source.convert("RGB")
            width, height = image.size
            tensor, osg = self.preprocessor(image, graph)
        return {"image": tensor, "osg": osg, "image_id": entry.image_id, "image_path": str(path),
                "osg_path": str(graph_path) if graph_path else None,
                "original_size": torch.tensor([height, width]), "role": entry.role}


def collate_observations(samples):
    if not samples:
        raise ValueError("Cannot collate an empty observation batch")
    presence = [sample["osg"] is not None for sample in samples]
    if any(presence) and not all(presence):
        raise ValueError("Cannot mix graph and image-only validation observations")
    return {"image": torch.stack([sample["image"] for sample in samples]),
            "osg": [sample["osg"] for sample in samples] if all(presence) else None,
            "image_id": [sample["image_id"] for sample in samples],
            "image_path": [sample["image_path"] for sample in samples],
            "osg_path": [sample["osg_path"] for sample in samples],
            "original_size": torch.stack([sample["original_size"] for sample in samples])}


def build_pitts30k_validation(config):
    if not config.load_osg:
        raise ValueError("COSMIC validation requires exported Pitts30k OSGs")
    if config.shuffle or config.drop_last or config.horizontal_flip_prob != 0:
        raise ValueError("Pitts30k validation requires no shuffle/drop_last or random horizontal flip")
    source = Pitts30kDataset(root=config.root, osg_root=config.osg_root, load_osg=config.load_osg,
                            image_size=config.image_size, mean=config.mean, std=config.std,
                            graph_config=config.graph, horizontal_flip_prob=config.horizontal_flip_prob)
    source.ensure_complete_graphs()
    count = len(source.metadata.database)
    def loader(indices):
        dataset = Subset(source, indices)
        batch_sampler = BatchSampler(SequentialSampler(dataset), batch_size=config.batch_size, drop_last=False)
        return DataLoader(dataset, batch_sampler=batch_sampler,
                          num_workers=config.num_workers, pin_memory=config.pin_memory,
                          collate_fn=collate_observations, worker_init_fn=seed_worker,
                          generator=torch.Generator().manual_seed(config.seed))
    positives = retrieval_positives(source.metadata)
    print(f"Pitts30k-val: {len(source.metadata.queries)} queries / {count} database; "
          f"positive radius={source.metadata.radius:g}m; empty GT={sum(not values for values in positives.values())}")
    return {"query_loader": loader(range(count, len(source))), "database_loader": loader(range(count)),
            "positives": positives}


