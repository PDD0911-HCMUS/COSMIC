"""Describe loader/batch structure without dataset-specific branches."""

from collections.abc import Mapping

import torch


def describe_loader(loader, dataset_name=None):
    """Report loader and sampler settings without reading images or graphs."""
    name = dataset_name or type(loader.dataset).__name__
    batch_sampler = loader.batch_sampler
    sampler = getattr(batch_sampler, "sampler", None)
    print(f"Loaded {name}: {len(loader.dataset)} samples, {len(loader)} batches")
    print(f"Batch sampler: {type(batch_sampler).__name__} | "
          f"batch_size={getattr(batch_sampler, 'batch_size', loader.batch_size)} | "
          f"drop_last={getattr(batch_sampler, 'drop_last', loader.drop_last)} | workers={loader.num_workers}")
    if sampler is not None:
        print(f"Sampler: {type(sampler).__name__}")


def describe_data_loader(loader, batch, dataset_name=None):
    """Print counts, tensor shapes and graph sizes; never modify the batch.

    Nested mappings cover both per-image and paired batches, regardless of
    branch names. Other metadata sequences are summarized by item count.
    """
    describe_loader(loader, dataset_name)

    def describe(value, path):
        if isinstance(value, torch.Tensor):
            print(f"{path}: shape={tuple(value.shape)}, dtype={value.dtype}")
        elif isinstance(value, Mapping):
            for key, item in value.items():
                describe(item, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            if value and all(isinstance(item, Mapping) and "node_ids" in item and "edge_index" in item
                             for item in value):
                nodes = [len(graph["node_ids"]) for graph in value]
                edges = [graph["edge_index"].shape[1] for graph in value]
                suffix = " ..." if len(value) > 8 else ""
                print(f"{path}: {len(value)} graphs, nodes={nodes[:8]}{suffix}, edges={edges[:8]}{suffix}")
            else:
                print(f"{path}: {len(value)} items")
        elif value is None or isinstance(value, (str, int, float, bool)):
            print(f"{path}: {value}")
        else:
            print(f"{path}: {type(value).__name__}")

    describe(batch, "batch")
