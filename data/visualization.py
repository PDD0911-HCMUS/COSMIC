"""Visualize normalized RGB/OSG A/B pairs and their match labels."""

from __future__ import annotations

from math import ceil
from pathlib import Path

from data.preprocessing import IMAGENET_MEAN, IMAGENET_STD, normalization_tensors


def visualize_image_pairs(
    batch: dict,
    output_path: str | Path,
    show: bool = True,
    mean: tuple[float, float, float] | None = IMAGENET_MEAN,
    std: tuple[float, float, float] | None = IMAGENET_STD,
    *,
    dataset_name: str = "AmsterTime",
    branch_names: tuple[str, str] = ("A: historical", "B: contemporary"),
) -> Path:
    """Save and optionally display all pairs in a batch, two pairs per row.

    Undo channel normalization on a detached copy before displaying RGB
    tensors. Pass the same mean/std used by the loader, or both None for
    unnormalized images. Model input tensors are not modified. Each pair's
    A/B images are adjacent, with source identity and positive/negative labels.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    images_a, images_b = batch["a"]["image"], batch["b"]["image"]
    normalization = normalization_tensors(mean, std)
    pair_count = len(images_a)
    if pair_count == 0 or len(images_b) != pair_count:
        raise ValueError("The batch must contain equally many nonempty image_a and image_b tensors.")

    pairs_per_row = min(2, pair_count)
    row_count = ceil(pair_count / pairs_per_row)
    figure, axes = plt.subplots(
        row_count,
        2 * pairs_per_row,
        figsize=(6 * pairs_per_row, 3.2 * row_count),
        squeeze=False,
        layout="constrained",
    )
    positive_count = int((batch["label"] == 1).sum())
    figure.suptitle(f"{dataset_name} | {pair_count} pairs: {positive_count} positive, "
                   f"{pair_count - positive_count} negative", fontsize=14)
    for axis in axes.flat:
        axis.set_axis_off()

    for index in range(pair_count):
        row, pair_column = divmod(index, pairs_per_row)
        pair_index = int(batch["pair_index"][index]) if "pair_index" in batch else index + 1
        label = float(batch["label"][index])
        for offset, (images, side, caption) in enumerate(
            (
                (images_a, "a", branch_names[0]),
                (images_b, "b", branch_names[1]),
            )
        ):
            axis = axes[row, 2 * pair_column + offset]
            image = images[index].detach().cpu()
            if normalization is not None:
                channel_mean, channel_std = normalization
                image = image * channel_std + channel_mean
            image = image.clamp(0, 1).permute(1, 2, 0).numpy()
            axis.imshow(image)
            filename = Path(batch[side]["image_path"][index]).name
            identity = filename
            if "city" in batch[side] and "place_id" in batch[side]:
                identity = f"{batch[side]['city'][index]} | place {int(batch[side]['place_id'][index])}"
            graph_caption = ""
            if batch[side]["osg"] is not None:
                graph = batch[side]["osg"][index]
                graph_caption = f"\n{len(graph['node_ids'])} nodes, {graph['edge_index'].shape[1]} edges"
                for node_id, box, node_label in zip(graph["node_ids"], graph["boxes"], graph["node_labels"]):
                    x1, y1, x2, y2 = box.tolist()
                    color = plt.get_cmap("tab20")(int(node_id) % 20)
                    axis.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                             edgecolor=color, linewidth=1))
                    axis.text(x1, y1, f"#{int(node_id)} {node_label}", fontsize=6, va="top", clip_on=True,
                              bbox={"facecolor": color, "alpha": 0.75, "edgecolor": "none", "pad": 0.5})
            axis.set_title(
                f"Pair {pair_index:04d} | {caption} | {'POSITIVE' if label == 1 else 'NEGATIVE'}\n"
                f"{identity}{graph_caption}",
                fontsize=10,
            )

    output_path = Path(output_path).expanduser().resolve()
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150)
        if show:
            plt.show()
    finally:
        plt.close(figure)
    return output_path


def visualize_gsv_batch(batch, output_path, show=True, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Show sampled GSV positive/negative pairs using the common pair layout."""
    return visualize_image_pairs(batch, output_path, show, mean, std,
                                 dataset_name="gsv", branch_names=("A", "B"))
