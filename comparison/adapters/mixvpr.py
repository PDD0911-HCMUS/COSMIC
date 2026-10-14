"""Official MixVPR inference modules, without its Lightning training dependencies."""

from comparison.adapters.base import DescriptorModel, ImageAdapter, load_module


class MixVPRAdapter(ImageAdapter):
    def __init__(self, spec, device):
        if spec.params:
            raise ValueError("MixVPR adapter uses the official ResNet50 / 4096-dimensional configuration")
        backbone = load_module(spec.repo, "models/backbones/resnet.py", "mixvpr_resnet").ResNet(
            model_name="resnet50", pretrained=False, layers_to_freeze=2, layers_to_crop=[4])
        aggregator = load_module(spec.repo, "models/aggregators/mixvpr.py", "mixvpr_aggregator").MixVPR(
            in_channels=1024, in_h=20, in_w=20, out_channels=1024, mix_depth=4, mlp_ratio=1, out_rows=4)
        super().__init__(spec, device, DescriptorModel(backbone, aggregator), size=(320, 320),
                         backbone="ResNet50", source_url="https://github.com/amaralibey/MixVPR",
                         preprocessing_order="pil_resize_tensor_normalize")
