"""Validate/pad processed loader graphs without changing their node ordering."""

import torch


def _tensor(graph, key, shape, integer=False):
    value = graph[key]
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise ValueError(f"OSG {key} must be a tensor of shape {shape}")
    if integer and value.dtype != torch.long:
        raise ValueError(f"OSG {key} must have dtype int64")
    if not torch.isfinite(value).all():
        raise ValueError(f"OSG {key} contains nonfinite values")
    return value


def _label(text):
    return " ".join(text.replace("_", " ").lower().split())


def pad_graphs(graphs, device):
    """Return dense metadata/masks plus flat text in matching row-major order.

    True masks denote real nodes/edges; padded IDs/endpoints use -1. Metadata
    is copied into new tensors, so neither source graphs nor boxes are mutated.
    Text triplets are rebuilt from directed endpoints and current edge labels,
    preserving predicate changes from joint horizontal-flip preprocessing.
    """
    if not isinstance(graphs, (list, tuple)) or not graphs:
        raise ValueError("OSG encoder expects a nonempty list of processed loader graphs")
    counts = []
    for graph in graphs:
        if not isinstance(graph, dict):
            raise ValueError("Missing processed OSG; enable data.load_osg and generate graphs first")
        try:
            nodes, edges = graph["node_labels"], graph["edge_labels"]
            if not isinstance(nodes, (list, tuple)) or not isinstance(edges, (list, tuple)):
                raise ValueError("OSG node_labels/edge_labels must be sequences")
            if any(not isinstance(label, str) or not _label(label) for label in (*nodes, *edges)):
                raise ValueError("OSG labels must be nonempty strings")
            n, e = len(nodes), len(edges)
            ids = _tensor(graph, "node_ids", (n,), integer=True)
            if ids.tolist() != list(range(n)):
                raise ValueError("Processed OSG node IDs must be contiguous in token order")
            source_ids = _tensor(graph, "source_node_ids", (n,), integer=True)
            if (source_ids < 0).any() or len(set(source_ids.tolist())) != n:
                raise ValueError("OSG source node IDs must be nonnegative and unique")
            endpoints = _tensor(graph, "edge_index", (2, e), integer=True)
            if ((endpoints < 0).any() or (endpoints >= n).any()
                    or (endpoints[0] == endpoints[1]).any()):
                raise ValueError("OSG edge_index contains invalid directed endpoints")
            for key, length in (("node_scores", n), ("edge_scores", e), ("triplet_scores", e)):
                scores = _tensor(graph, key, (length,))
                if ((scores < 0) | (scores > 1)).any():
                    raise ValueError(f"OSG {key} must be in [0,1]")
            for key in ("boxes", "boxes_original", "boxes_normalized"):
                boxes = _tensor(graph, key, (n, 4))
                if (boxes[:, 2:] <= boxes[:, :2]).any():
                    raise ValueError(f"OSG {key} must contain positive xyxy boxes")
                if key == "boxes_normalized" and ((boxes < -1e-6) | (boxes > 1 + 1e-6)).any():
                    raise ValueError("OSG boxes_normalized must lie in [0,1]")
            for key in ("image_size", "original_size"):
                size = _tensor(graph, key, (2,), integer=True)
                if (size <= 0).any():
                    raise ValueError(f"OSG {key} must be positive (height,width)")
            counts.append((n, e))
        except KeyError as exc:
            raise ValueError(f"Missing processed OSG field: {exc.args[0]}; use the data loader/preprocessor") from exc

    count = len(graphs)
    max_nodes, max_edges = max(n for n, _ in counts), max(e for _, e in counts)
    output = {
        "node_mask": torch.zeros(count, max_nodes, dtype=torch.bool, device=device),
        "edge_mask": torch.zeros(count, max_edges, dtype=torch.bool, device=device),
        "edge_index": torch.full((count, 2, max_edges), -1, dtype=torch.long, device=device),
        "node_ids": torch.full((count, max_nodes), -1, dtype=torch.long, device=device),
        "source_node_ids": torch.full((count, max_nodes), -1, dtype=torch.long, device=device),
    }
    for key in ("boxes", "boxes_original", "boxes_normalized"):
        output[key] = torch.zeros(count, max_nodes, 4, device=device)
    for key, length in (("node_scores", max_nodes), ("edge_scores", max_edges), ("triplet_scores", max_edges)):
        output[key] = torch.zeros(count, length, device=device)
    for key in ("image_size", "original_size"):
        output[key] = torch.stack([graph[key].to(device) for graph in graphs])
    output["node_labels"] = [list(graph["node_labels"]) for graph in graphs]
    output["edge_labels"] = [list(graph["edge_labels"]) for graph in graphs]
    output["triplet_texts"] = []
    node_texts, predicate_texts, triplet_texts = [], [], []
    for row, (graph, (n, e)) in enumerate(zip(graphs, counts)):
        output["node_mask"][row, :n] = True
        output["edge_mask"][row, :e] = True
        for key in ("node_ids", "source_node_ids", "node_scores", "boxes", "boxes_original", "boxes_normalized"):
            output[key][row, :n] = graph[key].to(device)
        for key in ("edge_scores", "triplet_scores"):
            output[key][row, :e] = graph[key].to(device)
        output["edge_index"][row, :, :e] = graph["edge_index"].to(device)
        nodes = [_label(label) for label in graph["node_labels"]]
        predicates = [_label(label) for label in graph["edge_labels"]]
        texts = [f"{nodes[s]} {predicate} {nodes[o]}."
                 for (s, o), predicate in zip(graph["edge_index"].t().tolist(), predicates)]
        output["triplet_texts"].append(texts)
        node_texts.extend(nodes)
        predicate_texts.extend(predicates)
        triplet_texts.extend(texts)
    return output, node_texts, predicate_texts, triplet_texts
