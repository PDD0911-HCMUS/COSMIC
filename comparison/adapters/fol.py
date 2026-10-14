"""FoL global retrieval and local matching from the official evaluation forward."""

import math
import torch

from comparison.adapters.base import ImageAdapter, load_module, untrained_dinov2_source
from comparison.config import project_path, section


class FoLAdapter(ImageAdapter):
    def __init__(self, spec, device):
        params = section(spec.params, ("backbone", "dinov2_repo", "local_threshold"), "fol params")
        threshold = params.get("local_threshold", .7)
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("FoL local_threshold must be finite and in [0,1]")
        name = params.get("backbone", "vitl")
        architectures = {"vitl": ("dinov2_vitl14", 1024), "vitb": ("dinov2_vitb14", 768)}
        if name not in architectures:
            raise ValueError("FoL backbone must be vitl or vitb")
        model_name, channels = architectures[name]
        with untrained_dinov2_source(project_path(params.get("dinov2_repo"))):
            model = load_module(spec.repo, "network_FoL.py", "fol_network").FoLNet(
                num_channels=channels, model_name=model_name, num_trainable_blocks=4)
        super().__init__(spec, device, model, size=(504, 504), backbone=model_name,
                         source_url="https://github.com/chenshunpeng/FoL",
                         preprocessing_order="tensor_normalize_resize")
        self.local_threshold = float(threshold)
        self.metadata["local_matching"] = "mutual nearest neighbors; count cosine similarity > threshold"
        self.metadata["local_threshold"] = self.local_threshold

    def forward_image(self, image):
        return self.model(image, test=True)

    def encode(self, observation, *, retain_features=False):
        output = self.forward_image(observation["image"].to(self.device))
        descriptor = self.project_descriptor(output[0])
        cached = None
        if retain_features:
            local = output[1]
            if (not isinstance(local, torch.Tensor) or local.ndim != 3 or min(local.shape) <= 0
                    or local.shape[0] != len(observation["image_id"]) or not local.is_floating_point()
                    or not torch.isfinite(local).all()):
                raise ValueError("FoL evaluation must return finite local features [B,L,D]")
            cached = [{"local_tokens": row.detach().cpu().clone()} for row in local]
        return descriptor, None, cached

    def pair_score(self, a, b):
        return float(mutual_match_count(a["local_tokens"].to(self.device),
                                        b["local_tokens"].to(self.device), self.local_threshold))


def mutual_match_count(query, database, threshold=.7):
    """FoL's local_sim criterion, without importing its CUDA/Linux-only test script.

    Official local tokens are normalized; padded zeros fail the cosine threshold.
    Match only mutual nearest neighbors and count their cosine values > threshold.
    """
    if (query.ndim != 2 or database.ndim != 2 or query.shape[1] != database.shape[1]
            or not torch.isfinite(query).all() or not torch.isfinite(database).all()):
        raise ValueError("FoL local matching requires finite [L,D] tokens of the same dimension")
    if not len(query) or not len(database):
        return 0
    similarity = database @ query.t()
    best_database = similarity.argmax(dim=0)
    best_query = similarity.argmax(dim=1)
    queries = torch.arange(len(query), device=query.device)
    mutual = best_query[best_database] == queries
    return int((mutual & (similarity[best_database, queries] > threshold)).sum())
