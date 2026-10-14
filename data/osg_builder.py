"""Build per-image observed scene graphs with the author's ECCV OvSGTR model.

Read original images and use OvSGTR's aspect-preserving 800/1333 resize.
Candidate vocabularies are shared across images, never derived from match
labels. Defaults use all VG150 objects/predicates; custom vocabularies can
be supplied without changing the downstream COSMIC graph schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "data" / "OvSGTR"
DEFAULT_CONFIG = REPO / "config" / "GroundingDINO_SwinT_OGC_ovdr.py"
DEFAULT_CHECKPOINT = REPO / "weights" / "vg-ovdr-swint.pth"


def default_vocabulary() -> tuple[list[str], list[str]]:
    """Load the original shared VG150 vocabulary retained for inference."""
    vocabulary = json.loads((REPO / "config" / "vocabulary.json").read_text(encoding="utf-8"))
    return vocabulary["objects"], vocabulary["relations"]


def _validate_vocabulary(names: list[str]) -> list[str]:
    if not names or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Vocabulary must contain nonempty strings")
    names = [name.lower().strip() for name in names]
    if len(set(names)) != len(names) or any("." in name or name == "[unk]" for name in names):
        raise ValueError("Vocabulary entries must be unique and exclude dots/background tokens")
    return names


class OvSGTRBuilder:
    """Frozen inference model; object boxes are xyxy pixels of the source image."""

    def __init__(
        self,
        config: str | Path = DEFAULT_CONFIG,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        device: str = "cuda",
        object_names: list[str] | None = None,
        relation_names: list[str] | None = None,
        object_threshold: float = 0.25,
        relation_threshold: float = 0.2,
        max_relations: int | None = 100,
    ):
        if not 0 <= object_threshold <= 1 or not 0 <= relation_threshold <= 1:
            raise ValueError("Confidence thresholds must be in [0, 1]")
        if max_relations is not None and max_relations <= 0:
            raise ValueError("max_relations must be positive or None (keep all)")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use device='cpu' for CPU inference")
        self.config = Path(config).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        bert_path = REPO / "weights" / "bert-base-uncased"
        for path in (self.config, self.checkpoint, bert_path / "pytorch_model.bin"):
            if not path.is_file():
                raise FileNotFoundError(f"Missing OvSGTR asset: {path}; run scripts/download_ovsgtr_assets.py")

        # Keep model/cache access local after setup, including old Transformers.
        os.environ["HF_HOME"] = str(ROOT / ".cache" / "huggingface")
        os.environ.pop("TRANSFORMERS_CACHE", None)
        os.environ["HF_HUB_CACHE"] = str(ROOT / ".cache" / "huggingface" / "hub")
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        for directory in (REPO, REPO / "GroundingDINO"):
            if str(directory) not in sys.path:
                sys.path.insert(0, str(directory))

        from groundingdino.models.GroundingDINO import build_groundingdino
        from groundingdino.datasets import transforms as T

        settings = {key: value for key, value in runpy.run_path(str(self.config)).items() if not key.startswith("_")}
        if not settings.get("sg_ovd_mode") or not settings.get("sg_ovr_mode"):
            raise ValueError("COSMIC requires an OvD+R-SGG config")
        settings.update(device=str(self.device), text_encoder_type=str(bert_path),
                        use_checkpoint=False, use_transformer_ckpt=False)
        model, _, processors = build_groundingdino(argparse.Namespace(**settings))
        with torch.serialization.safe_globals([argparse.Namespace]):
            weights = torch.load(self.checkpoint, map_location="cpu", weights_only=True)["model"]
        weights = {key.removeprefix("module."): value for key, value in weights.items()}
        # Transformers >=4.31 no longer serializes this deterministic buffer.
        # Verify it agrees with the runtime buffer before dropping the legacy key.
        position_ids = weights.pop("bert.embeddings.position_ids", None)
        if position_ids is not None and not torch.equal(position_ids, model.bert.embeddings.position_ids):
            raise ValueError("Checkpoint BERT position_ids differs from the runtime buffer")
        model.load_state_dict(weights, strict=True)
        del weights
        self.model = model.to(self.device).eval().requires_grad_(False)
        self.processor = processors["bbox"].to(self.device).eval()
        for name in ("rln_proj", "rln_classifier", "rln_freq_bias"):
            setattr(self.processor, name, getattr(self.model, name, None))

        defaults = default_vocabulary()
        self.object_names = _validate_vocabulary(defaults[0] if object_names is None else object_names)
        self.relation_names = _validate_vocabulary(defaults[1] if relation_names is None else relation_names)
        self.processor.name2classes = {name: index for index, name in enumerate(self.object_names, start=1)}
        self.processor.name2predicates = {"[UNK]": 0, **{name: index for index, name in enumerate(self.relation_names, start=1)}}
        # Keep upstream postprocessing behavior; filter after graph inference so
        # its zero/single-object special cases cannot invent COSMIC nodes/edges.
        self.transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.object_threshold = object_threshold
        self.relation_threshold = relation_threshold
        self.max_relations = max_relations
        self.source_manifest = json.loads((REPO / "source_manifest.json").read_text(encoding="utf-8"))
        self.repo_commit = self.source_manifest["commit"]

    @torch.inference_mode()
    def build(self, image_path: str | Path, image_id: str | None = None) -> dict:
        """Infer one OSG without receiving any pair/ground-truth information."""
        image_path = Path(image_path).resolve()
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            width, height = image.size
            tensor, _ = self.transform(image, None)
        outputs = self.model(
            tensor.unsqueeze(0).to(self.device),
            captions=[". ".join(self.object_names) + "."],
            rel_captions=[". ".join(self.relation_names) + "."],
        )
        sizes = torch.tensor([[height, width]], device=self.device)
        prediction = self.processor(outputs, sizes)[0]["graph"]
        # The upstream single-node fallback duplicates that node for its
        # evaluator. Keep only actual detections in the COSMIC graph.
        node_count = prediction["node_id"].numel()
        boxes = prediction["pred_boxes"].reshape(-1, 4)[:node_count]
        scores = prediction["pred_boxes_score"].reshape(-1)[:node_count]
        labels = prediction["pred_boxes_class"].reshape(-1)[:node_count]
        objects, index_map = [], {}
        for index, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            if float(score) < self.object_threshold or not 1 <= int(label) <= len(self.object_names):
                continue
            x1, y1, x2, y2 = box.tolist()
            bbox = [max(0.0, min(width, x1)), max(0.0, min(height, y1)),
                    max(0.0, min(width, x2)), max(0.0, min(height, y2))]
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            index_map[index] = len(objects)
            objects.append({"id": len(objects), "label": self.object_names[int(label) - 1],
                            "bbox": bbox, "confidence": float(score)})

        relations = []
        for pair, probabilities in zip(prediction["all_node_pairs"], prediction["all_relation"]):
            subject, obj = pair.tolist()
            if subject == obj or subject not in index_map or obj not in index_map:
                continue
            confidence, predicate = probabilities[1:].max(dim=0)
            if float(confidence) < self.relation_threshold:
                continue
            relations.append({"subject": index_map[subject], "predicate": self.relation_names[int(predicate)],
                              "object": index_map[obj], "confidence": float(confidence),
                              "triplet_score": float(confidence * scores[subject] * scores[obj])})
        relations.sort(key=lambda relation: relation["triplet_score"], reverse=True)
        vocabulary = {"objects": self.object_names, "relations": self.relation_names}
        manifest_path = REPO / "weights" / "assets_manifest.json"
        return {
            "schema_version": 1,
            "image_id": image_id or f"{image_path.parent.name}/{image_path.name}",
            "image_path": str(image_path),
            "image_size": {"width": width, "height": height},
            "bbox_format": "xyxy_pixels",
            "objects": objects,
            "relations": relations[:self.max_relations],
            "provenance": {
                "model": "OvSGTR", "setting": "OvD+R-SGG", "repo_commit": self.repo_commit,
                "config": str(self.config), "checkpoint": str(self.checkpoint),
                "assets": json.loads(manifest_path.read_text()) if manifest_path.exists() else None,
                "vocabulary": vocabulary,
                "vocabulary_sha256": hashlib.sha256(json.dumps(vocabulary, sort_keys=True).encode()).hexdigest(),
                "object_threshold": self.object_threshold, "relation_threshold": self.relation_threshold,
                "max_relations": self.max_relations, "resize": {"short_side": 800, "max_side": 1333},
                "postprocessing": {
                    "num_select": self.processor.num_select,
                    "detections_per_img": self.processor.detections_per_img,
                    "nms_iou_threshold": self.processor.nms_iou_threshold,
                    "score_threshold": self.processor.score_threshold,
                    "predicate_selection": "highest_scoring_non_background_per_directed_pair",
                },
            },
        }

    def build_and_save(self, image_path: str | Path, output_path: str | Path, image_id: str | None = None) -> dict:
        graph = self.build(image_path, image_id=image_id)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
        return graph
