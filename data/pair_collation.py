"""Shared A/B image-graph collation for sampled and fixed pairs."""

import torch


def collate_image_osg_pairs(samples):
    """Stack pair labels and branch tensors; keep variable graphs as lists."""
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    batch = {"label": torch.stack([sample["label"] for sample in samples])}
    if all("pair_index" in sample for sample in samples):
        batch["pair_index"] = torch.tensor([sample["pair_index"] for sample in samples], dtype=torch.long)
    for side in ("a", "b"):
        branches = [sample[side] for sample in samples]
        presence = [branch["osg"] is not None for branch in branches]
        if any(presence) and not all(presence):
            raise ValueError("A batch cannot mix graph and image-only samples")
        result = {
            "image": torch.stack([branch["image"] for branch in branches]),
            "osg": [branch["osg"] for branch in branches] if all(presence) else None,
            "image_id": [branch["image_id"] for branch in branches],
            "image_path": [branch["image_path"] for branch in branches],
            "osg_path": [branch["osg_path"] for branch in branches] if all(presence) else None,
            "original_size": torch.stack([branch["original_size"] for branch in branches]),
        }
        for key in ("place_label", "place_id", "image_index"):
            if all(key in branch for branch in branches):
                result[key] = torch.tensor([branch[key] for branch in branches], dtype=torch.long)
        if all("city" in branch for branch in branches):
            result["city"] = [branch["city"] for branch in branches]
        batch[side] = result
    return batch
