"""AmsterTime observations and CSV ground truth, independent of method weights."""

from dataclasses import dataclass
import hashlib
import json

from torch.utils.data import BatchSampler, Dataset, DataLoader, SequentialSampler
import torch

from data.amstertime_loader import AmsterTimeDataset
from data.loader_utils import seed_worker
from data.osg_storage import file_sha256


@dataclass(frozen=True)
class Benchmark:
    pairs: tuple
    database_indices: tuple[int, ...]
    query_indices: tuple[int, ...]
    database_ids: tuple[str, ...]
    query_ids: tuple[str, ...]
    positives: dict[str, list[str]]
    protocol: dict


def read_benchmark(config):
    source = AmsterTimeDataset(root=config.data_root, load_osg=False)
    pairs = tuple(source.pairs)
    if len(pairs) != config.expected_pairs:
        raise ValueError(f"Expected {config.expected_pairs} AmsterTime pairs, found {len(pairs)}")
    for side in ("a", "b"):
        paths = [getattr(pair, f"path_{side}") for pair in pairs]
        if len(set(paths)) != len(paths):
            raise ValueError(f"Duplicate AmsterTime image paths on side {side}; unique counterpart protocol required")
    database_indices = tuple(sorted(range(len(pairs)), key=lambda i: pairs[i].path_b.name))
    query_indices = tuple(sorted(range(len(pairs)), key=lambda i: pairs[i].path_a.name))
    if config.max_queries is not None:
        query_indices = query_indices[:config.max_queries]
    def image_id(index, side):
        return getattr(pairs[index], f"path_{side}").relative_to(source.root).as_posix()
    database_ids = tuple(image_id(i, "b") for i in database_indices)
    query_ids = tuple(image_id(i, "a") for i in query_indices)
    positives = {image_id(i, "a"): [image_id(i, "b")] for i in query_indices}
    identity = {"database_ids": database_ids, "query_ids": query_ids, "positives": positives}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    protocol = {
        "dataset": "amstertime", "task": "retrieval", "source_csv_sha256": file_sha256(source.root / "image_pairs.csv"),
        "manifest_sha256": fingerprint, "queries": len(query_ids), "database": len(database_ids),
        "total_pairs": len(pairs), "full_benchmark": len(query_ids) == len(pairs),
        "query_selection": "all" if len(query_ids) == len(pairs) else "smoke: first sorted historical images",
        "database_selection": "all contemporary images, including distractors",
        "ground_truth": "unique CSV counterpart", "inference_order": "database filenames, then query filenames",
        "metric": config.metric, "ranking_ties": "lexical database image_id",
        "recall_k": list(config.recall_k),
    }
    return Benchmark(pairs, database_indices, query_indices, database_ids, query_ids, positives, protocol)


class ObservationDataset(Dataset):
    """Read one branch only; combined order preserves cross-image batch boundaries."""

    def __init__(self, config, benchmark, adapter):
        self.source = AmsterTimeDataset(root=config.data_root, osg_root=config.osg_root,
                                        **adapter.data_settings)
        if tuple(self.source.pairs) != benchmark.pairs:
            raise ValueError("AmsterTime CSV changed between benchmark preparation and loading")
        self.records = [(i, "b", image_id) for i, image_id in zip(benchmark.database_indices, benchmark.database_ids)]
        self.records += [(i, "a", image_id) for i, image_id in zip(benchmark.query_indices, benchmark.query_ids)]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        pair_index, side, image_id = self.records[index]
        pair = self.source.pairs[pair_index]
        branch = self.source._load_branch(pair_index, side, getattr(pair, f"path_{side}"),
                                          getattr(pair, f"image_id_{side}"))
        branch["source_image_id"] = branch["image_id"]
        branch["image_id"] = image_id
        return branch


def collate_observations(samples):
    return {"image": torch.stack([sample["image"] for sample in samples]),
            "osg": [sample["osg"] for sample in samples],
            "image_id": [sample["image_id"] for sample in samples]}


def build_observation_loader(config, benchmark, adapter, batch_size):
    dataset = ObservationDataset(config, benchmark, adapter)
    batch_sampler = BatchSampler(SequentialSampler(dataset), batch_size=batch_size, drop_last=False)
    return DataLoader(dataset, batch_sampler=batch_sampler,
                      num_workers=config.num_workers,
                      collate_fn=collate_observations, worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(config.seed),
                      pin_memory=adapter.device.type == "cuda")
