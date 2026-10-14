"""Display graph-query cross-attention over the corresponding image grid."""

from pathlib import Path

import torch
from torch.nn import functional as F

from data.preprocessing import IMAGENET_MEAN, IMAGENET_STD, normalization_tensors


def visualize_grounding_heatmap(image, features, output_path, *, sample_index=0,
                               token_type="node", token_index=0, head=None,
                               mean=IMAGENET_MEAN, std=IMAGENET_STD, alpha=0.55, show=True):
    """Save source+bbox, raw patch attention and overlay panels.

    image is one normalized [3,H,W] tensor corresponding to sample_index in
    features returned by COSMIC(..., return_attention=True). Node/triplet
    indices refer to preprocessed OSG ordering. head=None averages all heads.
    Colors show the same 0..max attention scale on grid and overlay. This
    visualizes cross-attention weights, not attribution or matching confidence.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    visual, osg, grounding = features["visual"], features["osg"], features["grounding"]
    attention = grounding["attention_weights"]
    if attention is None:
        raise ValueError("Heatmap requires model(observation, return_attention=True)")
    if type(sample_index) is not int or not 0 <= sample_index < attention.shape[0]:
        raise ValueError("sample_index must select an observation in the encoded batch")
    if token_type not in ("node", "triplet"):
        raise ValueError("token_type must be node or triplet")
    token_mask = osg["node_mask"] if token_type == "node" else osg["edge_mask"]
    if (type(token_index) is not int or not 0 <= token_index < token_mask.shape[1]
            or not token_mask[sample_index, token_index]):
        raise ValueError("token_index must select a real node/triplet, not a padded token")
    if head is not None and (type(head) is not int or not 0 <= head < attention.shape[1]):
        raise ValueError("head must select an attention head, or be None for the mean")
    if (not isinstance(image, torch.Tensor) or image.ndim != 3 or image.shape[0] != 3
            or tuple(image.shape[-2:]) != tuple(visual["image_size"]) or not image.is_floating_point()
            or not torch.isfinite(image).all()):
        raise ValueError("image must be one finite preprocessed RGB tensor [3,H,W] matching the visual geometry")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0,1]")

    row = token_index if token_type == "node" else osg["node_mask"].shape[1] + token_index
    values = attention[sample_index, :, row].detach().float().cpu()
    values = values.mean(0) if head is None else values[head]
    grid_size = tuple(visual["grid_size"])
    if values.numel() != grid_size[0] * grid_size[1] or not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("Attention values must be finite nonnegative weights aligned with the visual grid")
    grid = values.reshape(grid_size)
    height, width = image.shape[-2:]
    overlay = F.interpolate(grid[None, None], size=(height, width), mode="bilinear", align_corners=False)[0, 0]
    rgb = image.detach().float().cpu().clone()
    normalization = normalization_tensors(mean, std)
    if normalization is not None:
        channel_mean, channel_std = normalization
        rgb = rgb * channel_std + channel_mean
    rgb = rgb.clamp(0, 1).permute(1, 2, 0).numpy()

    if token_type == "node":
        indices = [token_index]
        label = f"Node #{token_index}: {osg['node_labels'][sample_index][token_index]}"
    else:
        indices = osg["edge_index"][sample_index, :, token_index].tolist()
        label = f"Triplet #{token_index}: {osg['triplet_texts'][sample_index][token_index]}"
    head_label = "mean of heads" if head is None else f"head {head}"
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    figure.suptitle(f"{label} | {head_label}", fontsize=13)
    vmax = max(float(grid.max()), 1e-12)
    axes[0].imshow(rgb)
    axes[0].set_title("Preprocessed image + query bbox")
    heat = axes[1].imshow(grid.numpy(), cmap="inferno", vmin=0, vmax=vmax, interpolation="nearest")
    axes[1].set_title(f"Cross-attention patch grid {grid_size[0]} x {grid_size[1]}")
    figure.colorbar(heat, ax=axes[1], label="Attention weight", shrink=0.75)
    axes[2].imshow(rgb)
    axes[2].imshow(overlay.numpy(), cmap="inferno", vmin=0, vmax=vmax, alpha=alpha)
    axes[2].set_title("Attention overlay + query bbox")
    for axis in (axes[0], axes[2]):
        for position, index in enumerate(indices):
            x1, y1, x2, y2 = osg["boxes"][sample_index, index].detach().cpu().tolist()
            color = ("cyan", "lime")[position % 2]
            axis.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2))
            text = f"#{index} {osg['node_labels'][sample_index][index]}"
            if token_type == "triplet":
                text = ("subject: " if position == 0 else "object: ") + text
            axis.text(x1, y1, text, color=color, fontsize=8, va="top",
                      bbox={"facecolor": "black", "alpha": 0.6, "pad": 2}, clip_on=True)
    for axis in axes:
        axis.set_axis_off()
    output_path = Path(output_path).expanduser().resolve()
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150)
        if show:
            plt.show()
    finally:
        plt.close(figure)
    return output_path
