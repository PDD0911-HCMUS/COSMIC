"""Graph-conditioned, bidirectional cross-attention over matched image regions."""

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from model.config import VerificationConfig
from model.image_graph_verification.regions import gather_regions, read_features


class ImageGraphCrossVerification(nn.Module):
    """Verify local correspondences using already encoded COSMIC observations.

    This block takes full A/B features and hard alignment. It retains the
    original fused row IDs, gathers bbox-overlapping patches, and uses shared
    weights for both cross-image directions. It returns matched-region and
    symmetric pair embeddings, not a binary place decision. Soft-aligned
    counterpart tokens provide graph guidance for the hard-matched regions.
    This keeps a differentiable path to alignment and its learned null scores.
    The pooled pair
    descriptor depends on both images and is for later reasoning/scoring;
    independent database indexing still uses COSMIC.forward().
    """

    def __init__(self, d_model=256, config: VerificationConfig | None = None):
        super().__init__()
        self.config = config or VerificationConfig()
        if type(d_model) is not int or d_model <= 0 or d_model % self.config.num_heads:
            raise ValueError("Verification d_model must be positive and divisible by num_heads")
        self.d_model = d_model
        if self.config.use_graph_guidance:
            self.graph_projection = nn.Linear(d_model, d_model, bias=False)
        self.patch_norm = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(d_model, self.config.num_heads,
                                                    dropout=self.config.dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, self.config.ffn_dim), nn.GELU(),
                                 nn.Dropout(self.config.dropout), nn.Linear(self.config.ffn_dim, d_model))
        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(self.config.dropout)
        self.pair_projection = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.LayerNorm(d_model))

    def _matches(self, alignment, a, b):
        batch = a["tokens"].shape[0]
        if not isinstance(alignment, Mapping) or alignment.get("matched_pairs") is None:
            raise ValueError("Verification requires hard matches; call align(..., compute_hard=True)")
        pairs = alignment["matched_pairs"]
        if not isinstance(pairs, (list, tuple)) or len(pairs) != batch:
            raise ValueError("matched_pairs must contain one int64 [K,2] tensor per observation pair")
        scores = alignment.get("similarity_scores")
        if (not isinstance(scores, torch.Tensor) or not scores.is_floating_point()
                or scores.shape != (batch, a["tokens"].shape[1], b["tokens"].shape[1])
                or not torch.isfinite(scores).all()):
            raise ValueError("Alignment similarity_scores must match the current fused A/B token batches")
        pairs = list(pairs)
        for sample, pair in enumerate(pairs):
            if not isinstance(pair, torch.Tensor) or pair.dtype != torch.long or pair.ndim != 2 or pair.shape[1] != 2:
                raise ValueError("Each matched_pairs entry must be int64 [K,2]")
            pair = pair.to(a["tokens"].device)
            pairs[sample] = pair
            for column, features in enumerate((a, b)):
                ids = pair[:, column]
                if (ids < 0).any() or (ids >= features["tokens"].shape[1]).any():
                    raise ValueError("Matched row IDs are outside the padded fused token range")
                if not features["mask"][sample, ids].all() or len(ids.unique()) != len(ids):
                    raise ValueError("Hard matches must be one-to-one and reference only valid tokens")
            if not torch.equal(a["types"][sample, pair[:, 0]], b["types"][sample, pair[:, 1]]):
                raise ValueError("Verification only accepts node-node and triplet-triplet matches")
        count = max(map(len, pairs))
        indices = torch.full((batch, count, 2), -1, dtype=torch.long, device=a["tokens"].device)
        mask = torch.zeros(batch, count, dtype=torch.bool, device=indices.device)
        values = scores.new_zeros(batch, count).to(indices.device)
        types = torch.full_like(mask, -1, dtype=torch.long)
        unmatched_a, unmatched_b = [], []
        scores = scores.to(indices.device)
        for sample, pair in enumerate(pairs):
            length = len(pair)
            indices[sample, :length], mask[sample, :length] = pair, True
            values[sample, :length] = scores[sample, pair[:, 0], pair[:, 1]]
            types[sample, :length] = a["types"][sample, pair[:, 0]]
            for column, features, remaining in ((0, a, unmatched_a), (1, b, unmatched_b)):
                valid = features["mask"][sample].clone()
                valid[pair[:, column]] = False
                remaining.append(valid.nonzero(as_tuple=False).flatten())
        return indices, mask, values, types, unmatched_a, unmatched_b

    def _direction(self, current, normalized, opposite, current_mask, opposite_mask, return_attention):
        update, attention = self.cross_attention(normalized, opposite, opposite,
                                                key_padding_mask=~opposite_mask, need_weights=return_attention,
                                                average_attn_weights=False)
        output = current + self.dropout(update)
        output = self.output_norm(output + self.dropout(self.ffn(self.ffn_norm(output))))
        output = output * current_mask.unsqueeze(-1)
        if attention is not None:
            attention = attention * current_mask[:, None, :, None]
        return output, attention

    @staticmethod
    def _guided_tokens(features, alignment, context_key, rows):
        """Combine own graph tokens with soft context from the opposite graph.

        Direct callers may supply only hard matches/scores; without soft context,
        guidance uses the original tokens. COSMIC.align supplies both contexts.
        """
        tokens = features["tokens"]
        context = alignment.get(context_key)
        if context is not None:
            if (not isinstance(context, torch.Tensor) or context.shape != tokens.shape
                    or not context.is_floating_point() or not torch.isfinite(context).all()):
                raise ValueError(f"{context_key} must match the fused token batch shape")
            tokens = (tokens + context.to(device=tokens.device, dtype=tokens.dtype)) / 2
        batch = torch.arange(tokens.shape[0], device=tokens.device)[:, None]
        return tokens[batch, rows.clamp_min(0)]

    def forward(self, features_a, features_b, alignment, *, return_attention=False):
        if type(return_attention) is not bool:
            raise ValueError("return_attention must be boolean")
        reference = self.patch_norm.weight
        a = read_features(features_a, self.d_model, reference.device, reference.dtype)
        b = read_features(features_b, self.d_model, reference.device, reference.dtype)
        if a["tokens"].shape[0] != b["tokens"].shape[0]:
            raise ValueError("Verification requires equal A/B batch counts; grids and token counts may differ")
        pairs, match_mask, scores, types, unmatched_a, unmatched_b = self._matches(alignment, a, b)
        regions_a, ids_a, mask_a, weights_a = gather_regions(a, pairs[..., 0], match_mask, self.config.max_patches)
        regions_b, ids_b, mask_b, weights_b = gather_regions(b, pairs[..., 1], match_mask, self.config.max_patches)
        valid = match_mask & mask_a.any(-1) & mask_b.any(-1)
        batch, count = match_mask.shape
        flat_valid = valid.flatten().nonzero(as_tuple=False).flatten()
        tokens_a = tokens_b = None
        if self.config.use_graph_guidance:
            tokens_a = self._guided_tokens(a, alignment, "aligned_b_for_a", pairs[..., 0])
            tokens_b = self._guided_tokens(b, alignment, "aligned_a_for_b", pairs[..., 1])
        flat_a, flat_b = regions_a.flatten(0, 1), regions_b.flatten(0, 1)
        flat_ma, flat_mb = mask_a.flatten(0, 1), mask_b.flatten(0, 1)
        flat_wa, flat_wb = weights_a.flatten(0, 1), weights_b.flatten(0, 1)
        pooled_a = reference.new_zeros(batch * count, self.d_model)
        pooled_b = torch.zeros_like(pooled_a)
        pair_tokens = torch.zeros_like(pooled_a)
        attention_ab = attention_ba = None
        if return_attention:
            attention_ab = reference.new_zeros(batch * count, self.config.num_heads,
                                              self.config.max_patches, self.config.max_patches)
            attention_ba = torch.zeros_like(attention_ab)
        for chunk in flat_valid.split(self.config.pair_chunk_size):
            if not chunk.numel():
                continue
            local_a, local_b = flat_a[chunk], flat_b[chunk]
            ma, mb = flat_ma[chunk], flat_mb[chunk]
            if self.config.use_graph_guidance:
                local_a = local_a + self.graph_projection(tokens_a.flatten(0, 1)[chunk])[:, None]
                local_b = local_b + self.graph_projection(tokens_b.flatten(0, 1)[chunk])[:, None]
            local_a, local_b = local_a * ma[..., None], local_b * mb[..., None]
            normalized_a, normalized_b = self.patch_norm(local_a), self.patch_norm(local_b)
            verified_a, ab = self._direction(local_a, normalized_a, normalized_b, ma, mb, return_attention)
            verified_b, ba = self._direction(local_b, normalized_b, normalized_a, mb, ma, return_attention)
            pa = (verified_a * flat_wa[chunk].to(verified_a.dtype)[..., None]).sum(1)
            pb = (verified_b * flat_wb[chunk].to(verified_b.dtype)[..., None]).sum(1)
            # Symmetric combination; reversing A/B preserves pair features in eval mode.
            joint = self.pair_projection(torch.cat(((pa + pb) / 2, pa * pb), dim=-1))
            pooled_a, pooled_b = pooled_a.index_copy(0, chunk, pa), pooled_b.index_copy(0, chunk, pb)
            pair_tokens = pair_tokens.index_copy(0, chunk, joint)
            if return_attention:
                attention_ab = attention_ab.index_copy(0, chunk, ab)
                attention_ba = attention_ba.index_copy(0, chunk, ba)
        pair_tokens = pair_tokens.reshape(batch, count, self.d_model)
        pooling = valid.float() if self.config.pooling == "mean" else scores.float().clamp_min(0) * valid
        pooling = torch.where(pooling.sum(1, keepdim=True) > 0, pooling, valid.float())
        pooling = pooling / pooling.sum(1, keepdim=True).clamp_min(1e-12)
        pooled = (pair_tokens * pooling.to(pair_tokens.dtype)[..., None]).sum(1)
        has_matches = valid.any(1)
        descriptor = F.normalize(pooled, p=2, dim=-1, eps=1e-6) * has_matches[:, None]
        original_a = (regions_a * weights_a.to(regions_a.dtype)[..., None]).sum(2)
        original_b = (regions_b * weights_b.to(regions_b.dtype)[..., None]).sum(2)
        local_similarity = F.cosine_similarity(original_a.float(), original_b.float(), dim=-1, eps=1e-6) * valid
        return {
            "verified_tokens_a": pooled_a.reshape(batch, count, self.d_model),
            "verified_tokens_b": pooled_b.reshape(batch, count, self.d_model),
            "verified_pair_tokens": pair_tokens, "verified_descriptor": descriptor,
            "matched_pairs": pairs, "match_mask": match_mask, "verification_mask": valid,
            "matched_scores": scores, "matched_pair_types": types,
            "unmatched_a": unmatched_a, "unmatched_b": unmatched_b, "has_verified_matches": has_matches,
            "region_patch_indices_a": ids_a, "region_patch_indices_b": ids_b,
            "region_patch_mask_a": mask_a, "region_patch_mask_b": mask_b,
            "region_patch_weights_a": weights_a, "region_patch_weights_b": weights_b,
            "local_visual_similarity": local_similarity,
            "attention_weights_ab": attention_ab.reshape(batch, count, self.config.num_heads,
                                                         self.config.max_patches, self.config.max_patches) if return_attention else None,
            "attention_weights_ba": attention_ba.reshape(batch, count, self.config.num_heads,
                                                         self.config.max_patches, self.config.max_patches) if return_attention else None,
        }
