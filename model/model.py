"""COSMIC model: encode one image/OSG observation batch."""

from collections.abc import Mapping

import torch
from torch import nn

from model.config import ModelConfig
from model.visual_encoder import build_visual_encoder
from model.osg_encoder import build_osg_encoder
from model.grounding import ImageOSGGrounding
from model.cross_osg_alignment import CrossOSGAlignment
from model.image_graph_verification import ImageGraphCrossVerification
from model.global_scene_similarity import GlobalSceneSimilarity


class COSMIC(nn.Module):
    """Encode one observation batch, usable for training, queries or a database.

    Input: observation['image'] [B,3,H,W] and ['osg'] containing B
    preprocessed graphs. Images must already be resized and normalized.
    Images are moved to the visual encoder's device; the OSG encoder moves
    graph tensors while padding. The input observation is not modified.

    Output: {'visual': ..., 'osg': ..., 'grounding': ...}. Each stream
    retains its tokens, descriptors and spatial metadata. This stage only
    extracts fused image/OSG features. align() compares two encoded observations
    separately; verify() checks their matched image regions and score() turns
    the verified pair descriptor into a scene logit and sigmoid score.
    """

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        self.visual_encoder = build_visual_encoder(self.config.visual_encoder)
        self.osg_encoder = build_osg_encoder(self.config.osg_encoder)
        self.grounding = ImageOSGGrounding(self.config.visual_encoder.d_model, self.config.grounding)
        self.cross_osg_alignment = CrossOSGAlignment(self.config.visual_encoder.d_model, self.config.alignment)
        self.image_graph_verification = ImageGraphCrossVerification(self.config.visual_encoder.d_model,
                                                                   self.config.verification)
        self.global_scene_similarity = GlobalSceneSimilarity(self.config.visual_encoder.d_model,
                                                            self.config.scene_similarity)

    @staticmethod
    def _validate_observation(observation):
        if (not isinstance(observation, Mapping)
                or "image" not in observation or "osg" not in observation):
            raise ValueError("COSMIC expects one observation containing 'image' and 'osg'; "
                             "for a paired loader batch, call model(batch['a']) and model(batch['b'])")
        images, graphs = observation["image"], observation["osg"]
        if (not isinstance(images, torch.Tensor) or images.ndim != 4
                or images.shape[1] != 3 or not images.is_floating_point()
                or min(images.shape) <= 0):
            raise ValueError("observation['image'] must be nonempty RGB floats [B,3,H,W]")
        if (not isinstance(graphs, (list, tuple)) or len(graphs) != images.shape[0]
                or any(not isinstance(graph, Mapping) for graph in graphs)):
            raise ValueError("observation['osg'] must contain one processed graph per image; "
                             "enable data.load_osg and prepare the source OSGs first")

    def forward(self, observation, *, return_attention=False):
        self._validate_observation(observation)
        device = self.visual_encoder.projection[0].weight.device
        visual = self.visual_encoder(observation["image"].to(device))
        osg = self.osg_encoder(observation["osg"])
        grounding = self.grounding(visual, osg, return_attention=return_attention)
        return {"visual": visual, "osg": osg, "grounding": grounding}

    def align(self, features_a, features_b, *, compute_hard=None, return_attention=False):
        """Compare encoded/cached observations without re-running either backbone."""
        return self.cross_osg_alignment(features_a, features_b, compute_hard=compute_hard,
                                        return_attention=return_attention)

    def verify(self, features_a, features_b, alignment=None, *, return_attention=False):
        """Verify matched regions of cached/full features without running encoders.

        An omitted alignment is computed with hard matching enabled. Pass an
        existing alignment to reuse it. This does not change forward's single-
        observation interface or produce a final same-place score.
        """
        if alignment is None:
            alignment = self.align(features_a, features_b, compute_hard=True)
        return self.image_graph_verification(features_a, features_b, alignment,
                                             return_attention=return_attention)

    def score(self, verification):
        """Score a verified/cached pair; scene_logits is the input to pair BCE."""
        return self.global_scene_similarity(verification)


def build_model(config: ModelConfig | None = None, device="cpu") -> COSMIC:
    """Construct COSMIC on the target device; the caller selects train/eval mode."""
    return COSMIC(config).to(device)
