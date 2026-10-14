"""Type-aware soft attention and partial hard matching of fused graph tokens."""

from collections.abc import Mapping
import math

import torch
from torch import nn
from torch.nn import functional as F

from model.config import AlignmentConfig
from model.cross_osg_alignment.matching import hungarian_matches


class CrossOSGAlignment(nn.Module):
    """Compare cached/full COSMIC features without running image/text encoders.

    Inputs may be full COSMIC outputs or their 'grounding' mappings. Each
    contains fused_tokens [B,N,D], token_mask [B,N] and token_types [B,N].
    The same learned projection supplies both query/key directions. Per-head
    cosine similarity and temperature yield soft cross-attention, with one
    learned null score per type (node/triplet). Soft weights are conditional
    attention weights, not calibrated match or scene probabilities.

    Hard pair indices address padded fused-token rows, including the node
    padding before triplet rows. Pair handling leaves original features intact.
    No directional discrepancy embeddings or scene score are produced.
    """

    def __init__(self, d_model=256, config: AlignmentConfig | None = None):
        super().__init__()
        self.config = config or AlignmentConfig()
        if type(d_model) is not int or d_model <= 0 or d_model % self.config.num_heads:
            raise ValueError("Alignment d_model must be positive and divisible by num_heads")
        self.d_model = d_model
        self.projection = nn.Linear(d_model, d_model, bias=False)
        nn.init.eye_(self.projection.weight)
        self.unmatched_logits = nn.Parameter(torch.full((2,), math.atanh(self.config.unmatched_score)),
                                             requires_grad=self.config.learn_unmatched_score)

    def _read_features(self, features):
        if not isinstance(features, Mapping):
            raise ValueError("Alignment expects encoded COSMIC features, not raw observations")
        graph = features.get("grounding", features)
        if not isinstance(graph, Mapping) or any(key not in graph for key in ("fused_tokens", "token_mask", "token_types")):
            raise ValueError("Alignment input requires fused_tokens, token_mask and token_types")
        tokens, mask, types = graph["fused_tokens"], graph["token_mask"], graph["token_types"]
        if (not isinstance(tokens, torch.Tensor) or tokens.ndim != 3 or tokens.shape[0] <= 0
                or tokens.shape[-1] != self.d_model or not tokens.is_floating_point()
                or not torch.isfinite(tokens).all()):
            raise ValueError("fused_tokens must be finite floats [B,N,d_model] with a nonempty batch")
        if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != tokens.shape[:2]
                or not isinstance(types, torch.Tensor) or types.dtype != torch.long or types.shape != mask.shape):
            raise ValueError("token_mask must be bool [B,N] and token_types int64 [B,N]")
        device = self.projection.weight.device
        tokens = tokens.to(device=device, dtype=self.projection.weight.dtype)
        mask, types = mask.to(device), types.to(device)
        if ((types[mask] < 0).any() or (types[mask] > 1).any() or (types[~mask] != -1).any()):
            raise ValueError("Valid token_types must be 0=node or 1=triplet; padding must be -1")
        return tokens, mask, types

    def _project(self, tokens):
        batch, count, _ = tokens.shape
        head_dim = self.d_model // self.config.num_heads
        projected = self.projection(tokens).reshape(batch, count, self.config.num_heads, head_dim).transpose(1, 2)
        return F.normalize(projected.float(), p=2, dim=-1, eps=1e-8)

    def _soft(self, similarities, allowed, query_mask, null_score):
        logits = (similarities / self.config.temperature).masked_fill(~allowed[:, None], -torch.inf)
        null = (null_score / self.config.temperature)[:, None, :, None].expand(-1, self.config.num_heads, -1, 1)
        weights = torch.softmax(torch.cat((logits, null), dim=-1), dim=-1)
        weights = weights * query_mask[:, None, :, None]
        return weights[..., :-1], weights[..., -1]

    def forward(self, features_a, features_b, *, compute_hard=None, return_attention=False):
        if compute_hard is None:
            compute_hard = self.config.use_hungarian
        if type(compute_hard) is not bool or type(return_attention) is not bool:
            raise ValueError("compute_hard and return_attention must be boolean")
        a, mask_a, types_a = self._read_features(features_a)
        b, mask_b, types_b = self._read_features(features_b)
        if a.shape[0] != b.shape[0]:
            raise ValueError("Alignment requires equal batch counts; token counts may differ")
        allowed = mask_a[:, :, None] & mask_b[:, None, :] & (types_a[:, :, None] == types_b[:, None, :])
        similarities = self._project(a) @ self._project(b).transpose(-1, -2)
        null_scores = self.unmatched_logits.float().tanh()
        null_a, null_b = null_scores[types_a.clamp_min(0)], null_scores[types_b.clamp_min(0)]
        heads_ab, heads_null_a = self._soft(similarities, allowed, mask_a, null_a)
        heads_ba, heads_null_b = self._soft(similarities.transpose(-1, -2), allowed.transpose(1, 2), mask_b, null_b)
        soft_ab, soft_ba = heads_ab.mean(1), heads_ba.mean(1)
        scores = similarities.mean(1).masked_fill(~allowed, 0)
        output = {
            "similarity_scores": scores, "pair_mask": allowed,
            "soft_alignment_ab": soft_ab, "soft_alignment_ba": soft_ba,
            "soft_unmatched_a": heads_null_a.mean(1), "soft_unmatched_b": heads_null_b.mean(1),
            "aligned_b_for_a": soft_ab.to(b.dtype) @ b,
            "aligned_a_for_b": soft_ba.to(a.dtype) @ a,
            "token_mask_a": mask_a, "token_mask_b": mask_b,
            "token_types_a": types_a, "token_types_b": types_b,
            "attention_weights_ab": heads_ab if return_attention else None,
            "attention_weights_ba": heads_ba if return_attention else None,
            "null_scores": null_scores,
            "hard_alignment": None, "matched_pairs": None, "matched_scores": None,
            "matched_pair_types": None, "unmatched_a": None, "unmatched_b": None,
        }
        if compute_hard:
            pairs = hungarian_matches(scores, allowed, mask_a, mask_b, null_a, null_b)
            hard = torch.zeros_like(allowed)
            remaining_a, remaining_b = mask_a.clone(), mask_b.clone()
            matched_scores, matched_types = [], []
            for index, pair in enumerate(pairs):
                ids_a, ids_b = pair[:, 0], pair[:, 1]
                hard[index, ids_a, ids_b] = True
                remaining_a[index, ids_a], remaining_b[index, ids_b] = False, False
                matched_scores.append(scores[index, ids_a, ids_b])
                matched_types.append(types_a[index, ids_a])
            output.update(hard_alignment=hard, matched_pairs=pairs, matched_scores=matched_scores,
                          matched_pair_types=matched_types,
                          unmatched_a=[row.nonzero(as_tuple=False).flatten() for row in remaining_a],
                          unmatched_b=[row.nonzero(as_tuple=False).flatten() for row in remaining_b])
        return output
