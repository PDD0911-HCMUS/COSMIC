"""Strict checkpoint loading and shared image-descriptor adapter contracts."""

from contextlib import contextmanager
import importlib.util
from pathlib import Path
import subprocess
import sys

import torch
from torch import nn
import torchvision.transforms as T

from data.osg_storage import file_sha256


def load_module(repo, relative_path, name):
    path = Path(repo) / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Official source file not found: {path}")
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location(f"_comparison_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_state_dict(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return state_dict_from_payload(payload)


def state_dict_from_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must contain a state_dict mapping")
    state = payload.get("state_dict", payload.get("model_state_dict", payload.get("model", payload)))
    if not isinstance(state, dict) or not state or any(not isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Checkpoint does not contain tensor model weights")
    if any(not isinstance(key, str) for key in state):
        raise ValueError("State dict keys must be strings")
    return {key.removeprefix("module."): value for key, value in state.items()}


@contextmanager
def untrained_dinov2_source(local_repo=None):
    """Official constructors need DINO source, but full VPR weights replace initialization.

    Each method runs in its own worker. Avoid a redundant DINO weight download;
    an optional local DINO repo also avoids downloading its source.
    """
    original = torch.hub.load
    def load(repo, model, *args, **kwargs):
        if str(repo).split(":")[0] == "facebookresearch/dinov2":
            kwargs["pretrained"] = False
            if local_repo is not None:
                repo, kwargs["source"] = str(local_repo), "local"
            else:
                kwargs["trust_repo"] = True
        return original(repo, model, *args, **kwargs)
    torch.hub.load = load
    try:
        yield
    finally:
        torch.hub.load = original


class DescriptorModel(nn.Module):
    """Same backbone/aggregator state names as official MixVPR and BoQ models."""

    def __init__(self, backbone, aggregator):
        super().__init__()
        self.backbone, self.aggregator = backbone, aggregator

    def forward(self, image):
        return self.aggregator(self.backbone(image))


class ImageAdapter:
    def __init__(self, spec, device, model, *, size, interpolation="bilinear", backbone, source_url,
                 preprocessing_order="tensor_resize_normalize"):
        self.spec, self.device = spec, torch.device(device)
        model.load_state_dict(read_state_dict(spec.checkpoint), strict=True)
        self.model = model.to(self.device).eval()
        modes = {"bilinear": T.InterpolationMode.BILINEAR, "bicubic": T.InterpolationMode.BICUBIC}
        resize = T.Resize(tuple(size), interpolation=modes[interpolation], antialias=True)
        tensor = T.ToTensor()
        normalize = T.Normalize([.485, .456, .406], [.229, .224, .225])
        orders = {"pil_resize_tensor_normalize": [resize, tensor, normalize],
                  "tensor_resize_normalize": [tensor, resize, normalize],
                  "tensor_normalize_resize": [tensor, normalize, resize]}
        transform = T.Compose(orders[preprocessing_order])
        self.data_settings = {"load_osg": False, "transform": transform}
        self.metadata = {"backbone": backbone, "image_size": list(size), "interpolation": interpolation,
                         "normalization": "ImageNet", "source_url": source_url, "uses_osg": False,
                         "preprocessing_order": preprocessing_order}
        self.pca = None
        if spec.pca is not None:
            pca = torch.load(spec.pca, map_location="cpu", weights_only=True)
            if (not isinstance(pca, dict) or set(pca) != {"mean", "components"}
                    or not all(isinstance(v, torch.Tensor) and v.is_floating_point() and torch.isfinite(v).all()
                               for v in pca.values())
                    or pca["mean"].ndim != 1 or pca["components"].ndim != 2
                    or min(pca["components"].shape) <= 0 or pca["components"].shape[1] != len(pca["mean"])):
                raise ValueError("PCA file requires finite mean [D] and components [D_out,D] tensors")
            self.pca = {k: v.to(self.device, dtype=torch.float32) for k, v in pca.items()}
        self.metadata["pca_applied"] = self.pca is not None

    def encode(self, observation, *, retain_features=False):
        output = self.forward_image(observation["image"].to(self.device))
        descriptor = output[0] if isinstance(output, (tuple, list)) else output
        return self.project_descriptor(descriptor), None, None

    def project_descriptor(self, descriptor):
        if self.pca is not None:
            if descriptor.shape[-1] != len(self.pca["mean"]):
                raise ValueError("PCA input dimension does not match this checkpoint")
            descriptor = (descriptor.float() - self.pca["mean"]) @ self.pca["components"].t()
        return descriptor

    def forward_image(self, image):
        return self.model(image)


def provenance(adapter):
    spec = adapter.spec
    commit = None
    if spec.repo is not None:
        try:
            result = subprocess.run(["git", "-C", str(spec.repo), "rev-parse", "HEAD"],
                                    capture_output=True, text=True, check=True)
            commit = result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return {**adapter.metadata, "checkpoint": str(spec.checkpoint), "checkpoint_sha256": file_sha256(spec.checkpoint),
            "repo": str(spec.repo) if spec.repo else None, "repo_commit": commit,
            "pca": str(spec.pca) if spec.pca else None,
            "pca_sha256": file_sha256(spec.pca) if spec.pca else None}
