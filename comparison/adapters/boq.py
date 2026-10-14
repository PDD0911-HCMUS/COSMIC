"""Official BoQ ResNet50 and DINOv2 checkpoint architectures."""

from comparison.adapters.base import DescriptorModel, ImageAdapter, load_module, untrained_dinov2_source
from comparison.config import project_path, section


class BoQAdapter(ImageAdapter):
    def __init__(self, spec, device):
        params = section(spec.params, ("backbone", "dinov2_repo"), "boq params")
        name = params.get("backbone", "resnet50")
        backbones = load_module(spec.repo, "src/backbones.py", "boq_backbones")
        if name == "resnet50":
            backbone = backbones.ResNet(backbone_name=name, pretrained=False, crop_last_block=True)
            channels, size = 512, (384, 384)
        elif name == "dinov2":
            with untrained_dinov2_source(project_path(params.get("dinov2_repo"))):
                backbone = backbones.DinoV2(backbone_name="dinov2_vitb14")
            channels, size = 384, (322, 322)
        else:
            raise ValueError("BoQ backbone must be resnet50 or dinov2")
        aggregator = load_module(spec.repo, "src/boq.py", "boq_aggregator").BoQ(
            in_channels=backbone.out_channels, proj_channels=channels, num_queries=64, num_layers=2, row_dim=32)
        super().__init__(spec, device, DescriptorModel(backbone, aggregator), size=size, interpolation="bicubic",
                         backbone="DINOv2 ViT-B/14" if name == "dinov2" else "ResNet50",
                         source_url="https://github.com/amaralibey/Bag-of-Queries")
