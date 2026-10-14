"""COSMIC retrieval descriptors and cached single-pair scene verification."""

import torch

from config import load_config, config_to_dict
from model.model import build_model
from comparison.adapters.base import state_dict_from_payload


class CosmicAdapter:
    def __init__(self, spec, device):
        if spec.params or spec.pca is not None:
            raise ValueError("COSMIC uses its experiment config and native descriptor dimension")
        self.spec, self.device = spec, torch.device(device)
        self.experiment = load_config(spec.config)
        if self.experiment.data.horizontal_flip_prob != 0:
            raise ValueError("Comparison requires deterministic preprocessing: horizontal_flip_prob=0")
        payload = torch.load(spec.checkpoint, map_location="cpu", weights_only=True)
        if (not isinstance(payload, dict) or "model" not in payload or not isinstance(payload.get("config"), dict)
                or type(payload.get("epoch")) is not int or payload["epoch"] < 1):
            raise ValueError("COSMIC comparison requires a completed TrainEngine checkpoint with model/config/epoch")
        if "rerank" in spec.stages and self.experiment.model.criteria.verification_weight <= 0:
            raise ValueError("COSMIC re-ranking requires a checkpoint trained with verification_weight > 0")
        resolved = config_to_dict(self.experiment)
        if payload["config"]["model"] != resolved["model"]:
            raise ValueError("COSMIC model config differs from the checkpoint's training configuration")
        for key in ("image_size", "mean", "std", "graph"):
            if payload["config"]["data"][key] != resolved["data"][key]:
                raise ValueError(f"COSMIC preprocessing data.{key} differs from the training checkpoint")
        self.model = build_model(self.experiment.model, device)
        self.model.load_state_dict(state_dict_from_payload(payload), strict=True)
        self.model.eval()
        data = self.experiment.data
        self.data_settings = {"load_osg": True, "image_size": data.image_size,
                              "mean": data.mean, "std": data.std, "graph_config": data.graph}
        self.metadata = {"backbone": self.experiment.model.visual_encoder.backbone,
                         "image_size": list(data.image_size), "normalization": {"mean": data.mean, "std": data.std},
                         "uses_osg": True, "pca_applied": False, "checkpoint_epoch": payload["epoch"],
                         "model_config": resolved["model"], "graph_filter": resolved["data"]["graph"],
                         "source_url": None, "config": str(spec.config)}

    def encode(self, observation, *, retain_features=False):
        output = self.model(observation)
        cache = None
        if retain_features:
            cache = []
            keys = {"visual": ("visual_tokens", "grid_size", "image_size"),
                    "osg": ("boxes_normalized", "node_mask", "edge_mask", "edge_index", "image_size"),
                    "grounding": ("fused_tokens", "token_mask", "token_types")}
            for row in range(len(observation["image_id"])):
                streams = {}
                for stream, names in keys.items():
                    streams[stream] = {name: output[stream][name][row:row+1].detach().cpu().clone()
                                       if isinstance(output[stream][name], torch.Tensor) else output[stream][name]
                                       for name in names}
                cache.append(streams)
        return output["grounding"]["fused_descriptor"], output["grounding"]["graph_mask"], cache

    def pair_score(self, a, b):
        alignment = self.model.align(a, b, compute_hard=True)
        verification = self.model.verify(a, b, alignment)
        if not bool(verification["has_verified_matches"][0]):
            return None
        value = self.model.score(verification)["scene_logits"]
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite COSMIC re-ranking scene logits")
        return float(value[0])
