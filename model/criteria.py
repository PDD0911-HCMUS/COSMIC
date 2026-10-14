"""Joint retrieval and pair-verification objectives for COSMIC.

SupCon follows the mean of positive log probabilities (Eq. 2), using place
identities and excluding self-comparisons. Reference:
https://arxiv.org/abs/2004.11362
Pair BCE receives raw scene logits and image-pair labels, never token matches.
No backbone, scoring head, optimizer or training loop is created here.
"""

from collections.abc import Mapping
import math

import torch
from torch import nn
from torch.nn import functional as F

from model.config import CriteriaConfig


class SupConLoss(nn.Module):
    """Supervise [N,D] descriptors using int64 [N] place labels.

    Every valid image can be a key; only images with another same-place image
    are anchors. Masked/zero-norm descriptors are excluded from both roles.
    No usable anchors yields a differentiable zero, including an empty batch.
    Calculations run in float32 even under autocast. Unlike an image-pair
    binary label, a place label can identify all positives across a batch.
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("SupCon temperature must be finite and positive")
        self.temperature = float(temperature)

    def forward(self, embeddings, place_labels, valid_mask=None, *, return_details=False):
        if type(return_details) is not bool:
            raise ValueError("return_details must be boolean")
        if (not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2 or embeddings.shape[1] <= 0
                or not embeddings.is_floating_point() or not torch.isfinite(embeddings).all()):
            raise ValueError("SupCon embeddings must be finite floats [N,D]")
        count = embeddings.shape[0]
        if (not isinstance(place_labels, torch.Tensor) or place_labels.dtype != torch.long
                or place_labels.shape != (count,)):
            raise ValueError("SupCon place_labels must be int64 [N], not binary pair labels")
        if valid_mask is None:
            valid_mask = torch.ones(count, dtype=torch.bool, device=embeddings.device)
        elif (not isinstance(valid_mask, torch.Tensor) or valid_mask.dtype != torch.bool
              or valid_mask.shape != (count,)):
            raise ValueError("SupCon valid_mask must be bool [N]")
        valid_mask, place_labels = valid_mask.to(embeddings.device), place_labels.to(embeddings.device)
        if (place_labels[valid_mask] < 0).any():
            raise ValueError("Valid place labels must be nonnegative globally unique IDs")
        with torch.autocast(device_type=embeddings.device.type, enabled=False):
            full = embeddings.float()
            keep = valid_mask & (full.norm(dim=-1) > 1e-6)
            features, labels = full[keep], place_labels[keep]
            valid_images, anchors = len(features), 0
            loss = full.sum() * 0
            if valid_images > 1:
                features = F.normalize(features, p=2, dim=-1, eps=1e-6)
                self_mask = torch.eye(valid_images, dtype=torch.bool, device=full.device)
                positive = labels[:, None].eq(labels[None, :]) & ~self_mask
                positive_count = positive.sum(1)
                anchor_mask = positive_count > 0
                anchors = int(anchor_mask.sum())
                if anchors:
                    logits = (features @ features.t()) / self.temperature
                    if not torch.isfinite(logits).all():
                        raise ValueError("SupCon temperature is too small for finite float32 similarities")
                    log_denominator = torch.logsumexp(logits.masked_fill(self_mask, -torch.inf), dim=1)
                    log_probability = logits - log_denominator[:, None]
                    positive_log_probability = log_probability.masked_fill(~positive, 0).sum(1)
                    loss = -(positive_log_probability[anchor_mask] / positive_count[anchor_mask]).mean()
        details = {"loss": loss, "valid_images": valid_images, "anchors": anchors}
        return details if return_details else loss


def _descriptor(features):
    if not isinstance(features, Mapping):
        raise ValueError("Criteria expects encoded COSMIC features or their grounding mappings")
    grounding = features.get("grounding", features)
    if not isinstance(grounding, Mapping) or "fused_descriptor" not in grounding:
        raise ValueError("Criteria needs grounding.fused_descriptor (h^F) for each observation")
    descriptor = grounding["fused_descriptor"]
    if (not isinstance(descriptor, torch.Tensor) or descriptor.ndim != 2 or min(descriptor.shape) <= 0
            or not descriptor.is_floating_point() or not torch.isfinite(descriptor).all()):
        raise ValueError("fused_descriptor must be nonempty finite floats [B,D]")
    mask = grounding.get("graph_mask")
    if mask is None:
        mask = torch.ones(descriptor.shape[0], dtype=torch.bool, device=descriptor.device)
    elif not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != descriptor.shape[:1]:
        raise ValueError("graph_mask must be bool [B]")
    return descriptor, mask.to(descriptor.device)


class COSMICCriteria(nn.Module):
    """Combine SupCon over 2B independent descriptors with BCE over B pairs.

    forward(features_a, features_b, batch, scene_logits=None) accepts actual
    loader metadata. Retrieval uses batch['a/b']['place_label']; verification
    uses batch['label']. scene_logits must come from the scene-scoring head.
    A weight of zero disables that loss and removes its label/logit requirement.
    graph_mask filters retrieval images; every supplied pair remains in BCE,
    including pairs without hard token matches. Returned losses retain gradients.
    """

    def __init__(self, config: CriteriaConfig | None = None):
        super().__init__()
        self.config = config or CriteriaConfig()
        self.retrieval = SupConLoss(self.config.retrieval_temperature)
        self.verification = nn.BCEWithLogitsLoss()

    def forward(self, features_a, features_b, batch, scene_logits=None):
        a, mask_a = _descriptor(features_a)
        b, mask_b = _descriptor(features_b)
        if a.shape != b.shape or a.device != b.device:
            raise ValueError("Criteria requires A/B descriptors with the same [B,D] shape and device")
        if not isinstance(batch, Mapping):
            raise ValueError("Criteria batch must contain loader labels")
        count = a.shape[0]
        zero = (a.float().sum() + b.float().sum()) * 0
        retrieval_loss, verification_loss = zero, zero
        valid_images = anchors = verification_pairs = 0
        place_labels = []
        if self.config.retrieval_weight > 0:
            for side in ("a", "b"):
                branch = batch.get(side)
                labels = branch.get("place_label") if isinstance(branch, Mapping) else None
                if not isinstance(labels, torch.Tensor) or labels.dtype != torch.long or labels.shape != (count,):
                    raise ValueError(f"Retrieval requires batch['{side}']['place_label'] int64 [B]; "
                                     "use globally unique place IDs, not pair labels")
                place_labels.append(labels.to(a.device))
            details = self.retrieval(torch.cat((a.float(), b.float()), dim=0), torch.cat(place_labels),
                                     torch.cat((mask_a, mask_b)), return_details=True)
            retrieval_loss, valid_images, anchors = details["loss"], details["valid_images"], details["anchors"]
        if self.config.verification_weight > 0:
            if (not isinstance(scene_logits, torch.Tensor) or scene_logits.shape not in ((count,), (count, 1))
                    or not scene_logits.is_floating_point() or not torch.isfinite(scene_logits).all()
                    or scene_logits.device != a.device):
                raise ValueError("Verification requires raw finite scene_logits [B] or [B,1] on the descriptor device; "
                                 "supply the output of a scene-scoring head")
            pair_labels = batch.get("label")
            if (not isinstance(pair_labels, torch.Tensor) or pair_labels.shape != (count,)
                    or not torch.isfinite(pair_labels).all() or not ((pair_labels == 0) | (pair_labels == 1)).all()):
                raise ValueError("batch['label'] must be binary [B]: 1=same place, 0=different place")
            pair_labels = pair_labels.to(device=a.device, dtype=torch.float32)
            if place_labels and not torch.equal(place_labels[0].eq(place_labels[1]), pair_labels.bool()):
                raise ValueError("Pair labels must agree with equality of A/B place labels")
            with torch.autocast(device_type=a.device.type, enabled=False):
                verification_loss = self.verification(scene_logits.reshape(count).float(), pair_labels)
            verification_pairs = count
        weighted_retrieval = self.config.retrieval_weight * retrieval_loss
        weighted_verification = self.config.verification_weight * verification_loss
        return {
            "loss": weighted_retrieval + weighted_verification,
            "retrieval_loss": retrieval_loss, "verification_loss": verification_loss,
            "weighted_retrieval_loss": weighted_retrieval, "weighted_verification_loss": weighted_verification,
            "valid_retrieval_images": valid_images, "retrieval_anchors": anchors,
            "verification_pairs": verification_pairs,
        }


def build_criteria(config: CriteriaConfig | None = None, device="cpu") -> COSMICCriteria:
    """Construct the configured retrieval/pair losses on the selected device."""
    return COSMICCriteria(config).to(device)
