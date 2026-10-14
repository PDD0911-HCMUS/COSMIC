"""Soft patch-overlap priors for nodes and directed triplet endpoints."""

import torch


def _coverage(boxes, height, width):
    """Fraction of each normalized grid cell covered by each xyxy box."""
    y, x = torch.meshgrid(torch.arange(height, device=boxes.device, dtype=torch.float32),
                          torch.arange(width, device=boxes.device, dtype=torch.float32), indexing="ij")
    patch_left = torch.stack((x.flatten() / width, y.flatten() / height), dim=-1)
    patch_right = patch_left + boxes.new_tensor([1 / width, 1 / height])
    overlap = (torch.minimum(boxes[..., None, 2:], patch_right)
               - torch.maximum(boxes[..., None, :2], patch_left)).clamp_min(0)
    return (overlap.prod(-1) * height * width).clamp(0, 1)


def graph_patch_coverage(osg, grid_size):
    """Node bbox overlap; triplet overlap uses the union of endpoint boxes.

    Union is computed by inclusion-exclusion, retaining the gap between two
    objects rather than treating the enclosing rectangle as occupied.
    Rows are concatenated as padded nodes followed by padded triplets.
    """
    height, width = grid_size
    boxes = osg["boxes_normalized"].float()
    node = _coverage(boxes, height, width) * osg["node_mask"].unsqueeze(-1)
    count, edges = osg["edge_mask"].shape
    if not edges:
        triplet = boxes.new_zeros(count, 0, height * width)
    else:
        indices = osg["edge_index"].clamp_min(0)
        subject = boxes.gather(1, indices[:, 0, :, None].expand(-1, -1, 4))
        target = boxes.gather(1, indices[:, 1, :, None].expand(-1, -1, 4))
        intersection = torch.cat((torch.maximum(subject[..., :2], target[..., :2]),
                                  torch.minimum(subject[..., 2:], target[..., 2:])), dim=-1)
        triplet = (_coverage(subject, height, width) + _coverage(target, height, width)
                   - _coverage(intersection, height, width)).clamp(0, 1)
        triplet = triplet * osg["edge_mask"].unsqueeze(-1)
    return torch.cat((node, triplet), dim=1)
