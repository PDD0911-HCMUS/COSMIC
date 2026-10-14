"""Validate cached geometry and select bounded local patch sets."""

from collections.abc import Mapping

import torch

from model.grounding.spatial import graph_patch_coverage


def read_features(features, d_model, device, dtype):
    """Read full COSMIC features without mutating caches or re-running encoders."""
    if not isinstance(features, Mapping) or any(key not in features for key in ("visual", "osg", "grounding")):
        raise ValueError("Verification requires COSMIC features with visual, osg and grounding metadata")
    visual, osg, fused = (features[key] for key in ("visual", "osg", "grounding"))
    if any(not isinstance(value, Mapping) for value in (visual, osg, fused)):
        raise ValueError("Verification feature streams must be mappings")
    try:
        patches, tokens = visual["visual_tokens"], fused["fused_tokens"]
        mask, types = fused["token_mask"], fused["token_types"]
        boxes, node_mask, edge_mask = osg["boxes_normalized"], osg["node_mask"], osg["edge_mask"]
        edges, image_size = osg["edge_index"], osg["image_size"]
        grid, visual_size = tuple(visual["grid_size"]), tuple(visual["image_size"])
    except (KeyError, TypeError) as exc:
        raise ValueError("Verification needs fused masks/types, visual grid/image size, normalized boxes and edge_index") from exc
    if (not isinstance(patches, torch.Tensor) or patches.ndim != 3 or min(patches.shape) <= 0
            or patches.shape[-1] != d_model or not patches.is_floating_point() or not torch.isfinite(patches).all()):
        raise ValueError("visual_tokens must be finite floats [B,P,d_model]")
    batch = patches.shape[0]
    if (not isinstance(tokens, torch.Tensor) or tokens.ndim != 3 or tokens.shape[0] != batch
            or tokens.shape[-1] != d_model or not tokens.is_floating_point() or not torch.isfinite(tokens).all()):
        raise ValueError("fused_tokens must be finite floats [B,G,d_model]")
    for name, value in (("node_mask", node_mask), ("edge_mask", edge_mask)):
        if not isinstance(value, torch.Tensor) or value.dtype != torch.bool or value.ndim != 2 or value.shape[0] != batch:
            raise ValueError(f"{name} must be bool [B,N/E]")
    n, e = node_mask.shape[1], edge_mask.shape[1]
    if tokens.shape[1] != n + e:
        raise ValueError("Fused rows must contain padded nodes followed by padded triplets")
    if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != (batch, n + e)
            or not isinstance(types, torch.Tensor) or types.dtype != torch.long or types.shape != mask.shape):
        raise ValueError("Fused token_mask/types must be bool/int64 [B,N+E]")
    if (len(grid) != 2 or any(type(value) is not int or value <= 0 for value in grid)
            or grid[0] * grid[1] != patches.shape[1]):
        raise ValueError("grid_size must be positive (height,width) matching the visual patch count")
    if (len(visual_size) != 2 or any(type(value) is not int or value <= 0 for value in visual_size)
            or not isinstance(image_size, torch.Tensor) or image_size.dtype != torch.long
            or image_size.shape != (batch, 2)):
        raise ValueError("Visual and OSG image sizes must be positive (height,width)")
    if (not isinstance(boxes, torch.Tensor) or boxes.shape != (batch, n, 4)
            or not boxes.is_floating_point() or not torch.isfinite(boxes).all()):
        raise ValueError("boxes_normalized must be finite floats [B,N,4]")
    if not isinstance(edges, torch.Tensor) or edges.dtype != torch.long or edges.shape != (batch, 2, e):
        raise ValueError("edge_index must be int64 [B,2,E]")
    node_mask, edge_mask, mask, types = (value.to(device) for value in (node_mask, edge_mask, mask, types))
    boxes, edges, image_size = boxes.to(device=device, dtype=torch.float32), edges.to(device), image_size.to(device)
    expected_mask = torch.cat((node_mask, edge_mask), dim=1)
    expected_types = torch.cat((torch.zeros_like(node_mask, dtype=torch.long),
                                torch.ones_like(edge_mask, dtype=torch.long)), dim=1).masked_fill(~expected_mask, -1)
    if not torch.equal(mask, expected_mask) or not torch.equal(types, expected_types):
        raise ValueError("Fused masks/types must agree with OSG padded node/triplet order")
    if not torch.all(image_size == image_size.new_tensor(visual_size)):
        raise ValueError("OSG image_size must match the preprocessed visual image")
    valid_boxes = boxes[node_mask]
    if ((valid_boxes < -1e-6).any() or (valid_boxes > 1 + 1e-6).any()
            or (valid_boxes[:, 2:] <= valid_boxes[:, :2]).any()):
        raise ValueError("Valid normalized boxes must be positive xyxy regions in [0,1]")
    endpoints = edges.transpose(1, 2)[edge_mask]
    if (endpoints < 0).any() or (endpoints >= n).any() or (endpoints[:, 0] == endpoints[:, 1]).any():
        raise ValueError("Triplet endpoints must be distinct valid node row IDs")
    if endpoints.numel():
        edge_batch = edge_mask.nonzero(as_tuple=False)[:, 0]
        if not node_mask[edge_batch[:, None], endpoints].all():
            raise ValueError("Triplet endpoints cannot reference padded nodes")
    if ((edges.transpose(1, 2)[~edge_mask] != -1).any()):
        raise ValueError("Padded edge_index entries must be -1")
    return {
        "patches": patches.to(device=device, dtype=dtype), "tokens": tokens.to(device=device, dtype=dtype),
        "mask": mask, "types": types, "grid": grid,
        "osg": {"boxes_normalized": boxes, "node_mask": node_mask, "edge_mask": edge_mask, "edge_index": edges},
    }


def _select(coverage, endpoint_coverage, limit):
    """Spread samples through the occupied region, retaining each endpoint."""
    candidates = (coverage > 0).nonzero(as_tuple=False).flatten()
    if len(candidates) <= limit:
        return candidates
    anchors = []
    for endpoint in endpoint_coverage:
        if endpoint.max() > 0:
            anchors.append(endpoint.argmax())
    anchors = torch.stack(anchors).unique() if anchors else candidates[:0]
    remaining = candidates[~torch.isin(candidates, anchors)]
    slots = limit - len(anchors)
    positions = torch.linspace(0, len(remaining) - 1, slots, device=coverage.device).round().long()
    return torch.cat((anchors, remaining[positions])).sort().values


def gather_regions(features, rows, match_mask, max_patches):
    """Return local patches, normalized coverage weights and original grid IDs.

    No patches outside the node bbox/triplet endpoint union are selected.
    Attention never receives empty regions; callers skip them using masks.
    """
    batch, count = rows.shape
    patches = features["patches"]
    ids = torch.full((batch, count, max_patches), -1, dtype=torch.long, device=patches.device)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    weights = patches.new_zeros(batch, count, max_patches, dtype=torch.float32)
    if count:
        coverage = graph_patch_coverage(features["osg"], features["grid"])
        n = features["osg"]["node_mask"].shape[1]
        for sample, pair in match_mask.nonzero(as_tuple=False).tolist():
            row = int(rows[sample, pair])
            endpoints = []
            if row >= n:
                endpoints = [coverage[sample, int(node)] for node in features["osg"]["edge_index"][sample, :, row - n]]
            selected = _select(coverage[sample, row], endpoints, max_patches)
            length = len(selected)
            ids[sample, pair, :length] = selected
            mask[sample, pair, :length] = True
            weights[sample, pair, :length] = coverage[sample, row, selected]
    total = weights.sum(-1, keepdim=True)
    weights = weights / torch.where(total > 0, total, torch.ones_like(total))
    gathered = patches[torch.arange(batch, device=patches.device)[:, None, None], ids.clamp_min(0)]
    gathered = gathered * mask.unsqueeze(-1)
    return gathered, ids, mask, weights
