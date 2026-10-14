"""One graph self-attention/cross-attention block for image-OSG grounding."""

import torch
from torch import nn
from torch.nn import functional as F

from model.config import GroundingConfig
from model.grounding.spatial import graph_patch_coverage


class ImageOSGGrounding(nn.Module):
    """Ground node/triplet queries in visual patch keys/values for one view.

    H^F is fused_tokens [B,Nmax+Emax,D]; h^F is the unit-length pooled
    fused_descriptor [B,D]. True token masks indicate valid queries.
    Cross-attention weights [B,heads,Nmax+Emax,P] are optional, from this
    block before residual/FFN. Padded queries and empty graphs stay zero.
    """

    def __init__(self, d_model=256, config: GroundingConfig | None = None):
        super().__init__()
        self.config = config or GroundingConfig()
        if d_model <= 0 or d_model % self.config.num_heads:
            raise ValueError("Grounding d_model must be positive and divisible by num_heads")
        self.d_model = d_model
        self.type_embedding = nn.Embedding(2, d_model)
        nn.init.normal_(self.type_embedding.weight, std=0.02)
        if self.config.use_self_attention:
            self.self_norm = nn.LayerNorm(d_model)
            self.self_attention = nn.MultiheadAttention(d_model, self.config.num_heads,
                                                       dropout=self.config.dropout, batch_first=True)
        self.query_norm = nn.LayerNorm(d_model)
        self.visual_norm = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(d_model, self.config.num_heads,
                                                    dropout=self.config.dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, self.config.ffn_dim), nn.GELU(),
                                 nn.Dropout(self.config.dropout), nn.Linear(self.config.ffn_dim, d_model))
        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, visual, osg, *, return_attention=False):
        if type(return_attention) is not bool:
            raise ValueError("return_attention must be boolean")
        patches = visual["visual_tokens"]
        nodes, triplets = osg["node_tokens"], osg["triplet_tokens"]
        mask = torch.cat((osg["node_mask"], osg["edge_mask"]), dim=1)
        batch, node_count, dimension = nodes.shape
        if (patches.ndim != 3 or patches.shape[0] != batch or patches.shape[1] <= 0
                or patches.shape[-1] != self.d_model or dimension != self.d_model
                or triplets.shape != (*osg["edge_mask"].shape, self.d_model)):
            raise ValueError("Grounding expects matching visual/graph batches and feature dimensions")
        grid = tuple(visual["grid_size"])
        if len(grid) != 2 or min(grid) <= 0 or grid[0] * grid[1] != patches.shape[1]:
            raise ValueError("Visual grid_size must match the patch-token count")
        expected_size = osg["image_size"].new_tensor(visual["image_size"])
        if not torch.all(osg["image_size"] == expected_size):
            raise ValueError("OSG image_size must match the preprocessed visual image/bbox geometry")

        token_scores = torch.cat((osg["node_scores"], osg["triplet_scores"]), dim=1)
        types = torch.cat((torch.zeros_like(osg["node_mask"], dtype=torch.long),
                           torch.ones_like(osg["edge_mask"], dtype=torch.long)), dim=1)
        tokens = torch.cat((nodes, triplets), dim=1) + self.type_embedding(types)
        tokens = tokens * mask.unsqueeze(-1)
        graph_mask = mask.any(1)
        active = graph_mask.nonzero(as_tuple=False).flatten()
        fused = torch.zeros_like(tokens)
        weights = tokens.new_zeros(batch, self.config.num_heads, mask.shape[1], patches.shape[1]) if return_attention else None

        # Empty graphs must never reach an attention softmax with all keys masked.
        if active.numel():
            current = tokens.index_select(0, active)
            valid = mask.index_select(0, active)
            if self.config.use_self_attention:
                normalized = self.self_norm(current)
                update, _ = self.self_attention(normalized, normalized, normalized,
                                                key_padding_mask=~valid, need_weights=False)
                current = (current + self.dropout(update)) * valid.unsqueeze(-1)
            attention_bias = None
            if self.config.use_bbox_bias:
                coverage = graph_patch_coverage(osg, grid).index_select(0, active)
                bias = -self.config.bbox_bias_strength * (1 - coverage)
                attention_bias = bias[:, None].expand(-1, self.config.num_heads, -1, -1).reshape(
                    -1, mask.shape[1], patches.shape[1]).to(current.dtype)
            image_tokens = self.visual_norm(patches.index_select(0, active))
            update, attention = self.cross_attention(
                self.query_norm(current), image_tokens, image_tokens,
                attn_mask=attention_bias, need_weights=return_attention, average_attn_weights=False)
            current = current + self.dropout(update)
            current = current + self.dropout(self.ffn(self.ffn_norm(current)))
            current = self.output_norm(current) * valid.unsqueeze(-1)
            fused = fused.index_copy(0, active, current)
            if return_attention:
                attention = attention * valid[:, None, :, None]
                weights = weights.index_copy(0, active, attention)

        pooling_weights = mask.float() if self.config.pooling == "mean" else token_scores.float() * mask
        pooling_weights = torch.where(pooling_weights.sum(1, keepdim=True) > 0, pooling_weights, mask.float())
        pooling_weights = pooling_weights / pooling_weights.sum(1, keepdim=True).clamp_min(1e-12)
        pooled = (fused * pooling_weights.to(fused.dtype).unsqueeze(-1)).sum(1)
        descriptor = F.normalize(pooled, p=2, dim=-1, eps=1e-6) * graph_mask.unsqueeze(-1)
        return {
            "fused_tokens": fused, "fused_descriptor": descriptor,
            "fused_node_tokens": fused[:, :node_count],
            "fused_triplet_tokens": fused[:, node_count:],
            "token_mask": mask, "token_scores": token_scores,
            "token_types": types.masked_fill(~mask, -1), "graph_mask": graph_mask,
            "attention_weights": weights, "grid_size": grid,
        }
