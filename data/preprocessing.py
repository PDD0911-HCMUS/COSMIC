"""Joint image/OSG preprocessing; source graphs are never modified."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageOps


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalization_tensors(mean, std):
    if mean is None and std is None:
        return None
    if mean is None or std is None:
        raise ValueError("mean and std must both be provided, or both be None")
    mean, std = torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)
    if mean.shape != (3,) or std.shape != (3,):
        raise ValueError("mean and std must each contain three RGB values")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Normalization requires finite values and positive std")
    return mean.view(3, 1, 1), std.view(3, 1, 1)


@dataclass(frozen=True)
class GraphFilterConfig:
    object_threshold: float = 0.25
    relation_threshold: float = 0.20
    max_nodes: int | None = 32
    max_edges: int | None = 100
    edge_rank_by: str = "triplet_score"

    def __post_init__(self):
        for value in (self.object_threshold, self.relation_threshold):
            if not 0 <= value <= 1:
                raise ValueError("Graph thresholds must be in [0, 1]")
        for value in (self.max_nodes, self.max_edges):
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError("Graph budgets must be positive integers or None")
        if self.edge_rank_by not in ("triplet_score", "confidence"):
            raise ValueError("edge_rank_by must be triplet_score or confidence")


class RGBToTensor:
    """Resize to (height, width), convert RGB to CHW and normalize."""

    def __init__(self, image_size=(224, 224), mean=IMAGENET_MEAN, std=IMAGENET_STD):
        if len(image_size) != 2 or any(type(size) is not int or size <= 0 for size in image_size):
            raise ValueError("image_size must be positive integer (height, width)")
        self.image_size = tuple(image_size)
        self.mean, self.std = mean, std
        self.normalization = normalization_tensors(mean, std)

    def __call__(self, image):
        height, width = self.image_size
        image = image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
        pixels = np.array(image, dtype=np.float32, copy=True) / 255.0
        tensor = torch.from_numpy(pixels).permute(2, 0, 1).contiguous()
        if self.normalization is not None:
            mean, std = self.normalization
            tensor = (tensor - mean) / std
        return tensor


def preprocess_graph(graph: dict, image_size: tuple[int, int], config: GraphFilterConfig,
                     horizontal_flipped: bool = False) -> dict:
    """Select nodes/edges, remap endpoints, and derive boxes and text triplets."""
    objects = sorted((obj for obj in graph["objects"] if obj["confidence"] >= config.object_threshold),
                     key=lambda obj: (-obj["confidence"], obj["id"]))[:config.max_nodes]
    mapping = {obj["id"]: index for index, obj in enumerate(objects)}
    relations = sorted((edge for edge in graph["relations"]
                        if edge["subject"] in mapping and edge["object"] in mapping
                        and edge["confidence"] >= config.relation_threshold),
                       key=lambda edge: (-edge[config.edge_rank_by], edge["subject"],
                                         edge["object"], edge["predicate"]))[:config.max_edges]
    original_boxes = torch.tensor([obj["bbox"] for obj in objects], dtype=torch.float32).reshape(-1, 4)
    width, height = graph["image_size"]["width"], graph["image_size"]["height"]
    target_height, target_width = image_size
    boxes = original_boxes * torch.tensor([target_width / width, target_height / height] * 2)
    if horizontal_flipped:
        left, right = boxes[:, 0].clone(), boxes[:, 2].clone()
        boxes[:, 0], boxes[:, 2] = target_width - right, target_width - left
    normalized_boxes = boxes / torch.tensor([target_width, target_height] * 2)
    edge_index = torch.tensor([[mapping[edge["subject"]], mapping[edge["object"]]]
                               for edge in relations], dtype=torch.long).reshape(-1, 2).t().contiguous()
    labels = [obj["label"] for obj in objects]
    predicates = [edge["predicate"] for edge in relations]
    if horizontal_flipped:
        flip_predicates = {"left of": "right of", "right of": "left of",
                           "to the left of": "to the right of", "to the right of": "to the left of"}
        predicates = [flip_predicates.get(predicate, predicate) for predicate in predicates]
    triplets = [(labels[mapping[edge["subject"]]], predicate, labels[mapping[edge["object"]]])
                for edge, predicate in zip(relations, predicates)]
    return {
        "node_ids": torch.arange(len(objects)),
        "source_node_ids": torch.tensor([obj["id"] for obj in objects], dtype=torch.long),
        "node_labels": labels,
        "node_scores": torch.tensor([obj["confidence"] for obj in objects], dtype=torch.float32),
        "boxes_original": original_boxes,
        "boxes": boxes,
        "boxes_normalized": normalized_boxes,
        "edge_index": edge_index,
        "edge_labels": predicates,
        "edge_scores": torch.tensor([edge["confidence"] for edge in relations], dtype=torch.float32),
        "triplet_scores": torch.tensor([edge["triplet_score"] for edge in relations], dtype=torch.float32),
        "triplets": triplets,
        "triplet_texts": [f"{subject} {predicate} {target}." for subject, predicate, target in triplets],
        "original_size": torch.tensor([height, width]),
        "image_size": torch.tensor(image_size),
        "horizontal_flipped": horizontal_flipped,
    }


class ImageGraphPreprocessor:
    def __init__(self, image_size=(224, 224), mean=IMAGENET_MEAN, std=IMAGENET_STD,
                 graph_config: GraphFilterConfig | None = None, horizontal_flip_prob=0.0):
        self.image_transform = RGBToTensor(image_size, mean, std)
        self.graph_config = graph_config or GraphFilterConfig()
        if not 0 <= horizontal_flip_prob <= 1:
            raise ValueError("horizontal_flip_prob must be in [0, 1]")
        self.horizontal_flip_prob = horizontal_flip_prob

    def __call__(self, image: Image.Image, graph: dict | None):
        if graph is not None and image.size != (graph["image_size"]["width"], graph["image_size"]["height"]):
            raise ValueError("OSG dimensions do not match the original image")
        flipped = self.horizontal_flip_prob > 0 and torch.rand(()).item() < self.horizontal_flip_prob
        if flipped:
            image = ImageOps.mirror(image)
        tensor = self.image_transform(image)
        osg = None if graph is None else preprocess_graph(
            graph, self.image_transform.image_size, self.graph_config, horizontal_flipped=flipped)
        return tensor, osg
