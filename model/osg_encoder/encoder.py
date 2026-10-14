"""Lightweight DCE/GCE graph stream: shared BERT, no graph Transformer."""

import torch
from torch import nn
from torch.nn import functional as F

from model.config import OSGEncoderConfig
from model.osg_encoder.batch import pad_graphs
from model.osg_encoder.text_encoder import BertTextEncoder


def _projection(in_features, out_features):
    return nn.Sequential(nn.Linear(in_features, out_features), nn.LayerNorm(out_features))


def _pool(tokens, scores, mask, mode):
    """Pool real tokens; zero-confidence sets fall back to masked mean."""
    weights = mask.float() if mode == "mean" else scores.float() * mask
    total = weights.sum(1, keepdim=True)
    weights = torch.where(total > 0, weights, mask.float())
    weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-12)
    return (tokens * weights.to(tokens.dtype).unsqueeze(-1)).sum(1)


class OSGEncoder(nn.Module):
    """Encode a list of preprocessed OSGs for one observation batch.

    DCE retains object and predicate features; GCE encodes full directed
    triplet sentences. Every feature stays attached to the source IDs/bboxes.
    No positional embedding is assigned by node or triplet list order.
    """

    def __init__(self, config: OSGEncoderConfig | None = None):
        super().__init__()
        self.config = config or OSGEncoderConfig()
        self.text_encoder = BertTextEncoder(self.config)
        hidden, dimension = self.text_encoder.hidden_size, self.config.d_model
        self.node_projection = _projection(hidden, dimension)
        self.predicate_projection = _projection(hidden, dimension)
        self.triplet_projection = _projection(hidden, dimension)
        self.bbox_projection = _projection(4, dimension) if self.config.use_bbox_embedding else None
        self.dce_projection = _projection(2 * dimension, dimension)
        self.graph_projection = nn.Linear(2 * dimension, self.config.global_dim)

    def clear_text_cache(self):
        self.text_encoder.clear_cache()

    @staticmethod
    def _scatter(features, mask):
        padded = features.new_zeros((*mask.shape, features.shape[-1]))
        padded[mask] = features
        return padded

    def forward(self, graphs):
        device = self.node_projection[0].weight.device
        output, nodes, predicates, triplets = pad_graphs(graphs, device)
        vectors = self.text_encoder(nodes + predicates + triplets)
        vectors = vectors.to(dtype=self.node_projection[0].weight.dtype)
        node_features, predicate_features, triplet_features = vectors.split(
            (len(nodes), len(predicates), len(triplets)), dim=0)
        node_tokens = self._scatter(self.node_projection(node_features), output["node_mask"])
        predicate_tokens = self._scatter(self.predicate_projection(predicate_features), output["edge_mask"])
        triplet_tokens = self._scatter(self.triplet_projection(triplet_features), output["edge_mask"])
        if self.bbox_projection is not None:
            geometry = self.bbox_projection(output["boxes_normalized"].to(node_tokens.dtype))
            node_tokens = (node_tokens + geometry) * output["node_mask"].unsqueeze(-1)

        node_summary = _pool(node_tokens, output["node_scores"], output["node_mask"], self.config.pooling)
        predicate_summary = _pool(predicate_tokens, output["edge_scores"], output["edge_mask"], self.config.pooling)
        graph_mask = output["node_mask"].any(1)
        dce = self.dce_projection(torch.cat((node_summary, predicate_summary), dim=-1))
        dce = dce * graph_mask.unsqueeze(-1)
        gce = _pool(triplet_tokens, output["triplet_scores"], output["edge_mask"], self.config.pooling)
        descriptor = F.normalize(self.graph_projection(torch.cat((dce, gce), dim=-1)),
                                 p=2, dim=-1, eps=1e-6) * graph_mask.unsqueeze(-1)
        output.update(node_tokens=node_tokens, predicate_tokens=predicate_tokens,
                      triplet_tokens=triplet_tokens, dce_descriptor=dce,
                      gce_descriptor=gce, graph_descriptor=descriptor, graph_mask=graph_mask)
        return output


def build_osg_encoder(config: OSGEncoderConfig | None = None) -> OSGEncoder:
    return OSGEncoder(config)
