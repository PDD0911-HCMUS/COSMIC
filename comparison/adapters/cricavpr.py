"""Official CricaVPR; retain combined database/query inference batch boundaries."""

from comparison.adapters.base import ImageAdapter, load_module


class CricaVPRAdapter(ImageAdapter):
    def __init__(self, spec, device):
        if spec.params:
            raise ValueError("CricaVPR adapter uses the official ViT-B/14 architecture")
        model = load_module(spec.repo, "network.py", "cricavpr_network").CricaVPRNet()
        super().__init__(spec, device, model, size=(224, 224), backbone="DINOv2 ViT-B/14",
                         source_url="https://github.com/Lu-Feng/CricaVPR",
                         preprocessing_order="tensor_normalize_resize")
        self.metadata["cross_image_inference"] = True
        self.metadata["raw_descriptor_dimension"] = 10752
