"""Display an inferred OSG beside its original image and detected boxes."""

from __future__ import annotations

from pathlib import Path

from data.preprocessing import IMAGENET_MEAN, IMAGENET_STD, normalization_tensors


def visualize_record(record: dict, output_path: str | Path, show: bool = True,
                     mean=IMAGENET_MEAN, std=IMAGENET_STD) -> Path:
    """Display the exact preprocessed A/B sample, including both selected OSGs."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    normalization = normalization_tensors(mean, std)
    edge_count = max((branch["osg"]["edge_index"].shape[1]
                      for branch in (record["a"], record["b"]) if branch["osg"] is not None), default=0)
    list_height = max(3, 0.18 * edge_count)
    figure = plt.figure(figsize=(22, 10.5 + list_height), layout="constrained")
    grid = figure.add_gridspec(3, 2, height_ratios=(4, 6, list_height))
    figure.suptitle(f"AmsterTime record {record['pair_index']:04d} | match={float(record['label']):g}", fontsize=18)
    for column, (side, domain) in enumerate((("a", "Historical"), ("b", "Contemporary"))):
        branch = record[side]
        image_axis = figure.add_subplot(grid[0, column])
        graph_axis = figure.add_subplot(grid[1, column])
        list_axis = figure.add_subplot(grid[2, column])
        for axis in (image_axis, graph_axis, list_axis):
            axis.set_axis_off()
        image = branch["image"].detach().cpu()
        if normalization is not None:
            channel_mean, channel_std = normalization
            image = image * channel_std + channel_mean
        image_axis.imshow(image.clamp(0, 1).permute(1, 2, 0).numpy())
        osg = branch["osg"]
        title = f"{side.upper()}: {domain} | {Path(branch['image_path']).name}"
        if osg is None:
            image_axis.set_title(title, fontsize=13)
            graph_axis.text(0.5, 0.5, "Image-only record", ha="center", va="center", transform=graph_axis.transAxes)
            continue
        objects = [{"id": int(node_id), "label": label, "bbox": box.detach().cpu().tolist(),
                    "confidence": float(score)}
                   for node_id, label, box, score in zip(osg["node_ids"], osg["node_labels"], osg["boxes"], osg["node_scores"])]
        relations = [{"subject": int(pair[0]), "predicate": predicate, "object": int(pair[1]),
                      "confidence": float(score)}
                     for pair, predicate, score in zip(osg["edge_index"].t(), osg["edge_labels"], osg["edge_scores"])]
        image_axis.set_title(f"{title}\n{len(objects)} nodes, {len(relations)} edges | preprocessed image", fontsize=13)
        for index, obj in enumerate(objects):
            x1, y1, x2, y2 = obj["bbox"]
            color = plt.get_cmap("tab20")(index % 20)
            image_axis.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                           fill=False, edgecolor=color, linewidth=1.3))
            image_axis.text(x1, y1, f"#{obj['id']} {obj['label']} {obj['confidence']:.2f}",
                            fontsize=7, va="top", clip_on=True,
                            bbox={"facecolor": color, "edgecolor": "none", "alpha": 0.8, "pad": 1})
        _draw_scene_graph(graph_axis, objects, relations)
        list_axis.set_title("Triplets | relation confidence", fontsize=12, loc="left")
        labels = {obj["id"]: obj["label"] for obj in objects}
        rows = [f"#{edge['subject']} {labels[edge['subject']]}  ->  {edge['predicate']}  ->  "
                f"#{edge['object']} {labels[edge['object']]}  ({edge['confidence']:.2f})" for edge in relations]
        list_axis.text(0, 1, "\n".join(rows) or "No selected relations.",
                       transform=list_axis.transAxes, fontsize=9, va="top", linespacing=1.2)
    output_path = Path(output_path).expanduser().resolve()
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150)
        if show:
            plt.show()
    finally:
        plt.close(figure)
    return output_path


def visualize_scene_graph(graph: dict, output_path: str | Path, show: bool = True) -> Path:
    """Save a figure with matching object IDs/colors and directed predicates.

    Render every object and relation in the supplied graph. The relation list
    preserves confidence scores and remains readable when graph edges overlap.
    This function does not filter or modify the model's predictions.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    if graph["bbox_format"] != "xyxy_pixels":
        raise ValueError("Scene graph visualization requires original-image xyxy pixel boxes.")
    objects, relations = graph["objects"], graph["relations"]
    object_by_id = {obj["id"]: obj for obj in objects}
    for relation in relations:
        if relation["subject"] not in object_by_id or relation["object"] not in object_by_id:
            raise ValueError("A relation references an object missing from the graph.")

    figure = plt.figure(figsize=(20, max(11, 5 + 0.22 * len(relations))), layout="constrained")
    grid = figure.add_gridspec(2, 2, width_ratios=(1, 1.8), height_ratios=(1, 1.2))
    image_axis = figure.add_subplot(grid[0, 0])
    list_axis = figure.add_subplot(grid[1, 0])
    graph_axis = figure.add_subplot(grid[:, 1])
    for axis in (image_axis, list_axis, graph_axis):
        axis.set_axis_off()
    image_path = Path(graph["image_path"])
    with Image.open(image_path) as source:
        image_axis.imshow(source.convert("RGB"))
    image_axis.set_title("Detected objects | ID, class, confidence", fontsize=12)
    colors = {obj["id"]: plt.get_cmap("tab20")(index % 20)
              for index, obj in enumerate(objects)}
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        color = colors[obj["id"]]
        image_axis.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       fill=False, edgecolor=color, linewidth=1.6))
        image_axis.text(x1, y1, f"#{obj['id']} {obj['label']} {obj['confidence']:.2f}",
                        fontsize=7, va="bottom", color="black",
                        bbox={"facecolor": color, "alpha": 0.85, "edgecolor": "none", "pad": 1})

    _draw_scene_graph(graph_axis, objects, relations)

    list_axis.set_title("Relations | prediction confidence", fontsize=12, loc="left")
    rows = []
    for relation in relations:
        subject = object_by_id[relation["subject"]]
        target = object_by_id[relation["object"]]
        rows.append(f"#{subject['id']} {subject['label']}  ->  {relation['predicate']}  ->  "
                    f"#{target['id']} {target['label']}   ({relation['confidence']:.2f})")
    list_axis.text(0, 1, "\n".join(rows) or "No relations above the relation threshold.",
                   transform=list_axis.transAxes, fontsize=9, va="top", linespacing=1.5)
    figure.suptitle(f"OvSGTR | {image_path.name} | {len(objects)} objects, {len(relations)} relations",
                   fontsize=16)
    output_path = Path(output_path).expanduser().resolve()
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150)
        if show:
            plt.show()
    finally:
        plt.close(figure)
    return output_path


def _draw_scene_graph(graph_axis, objects, relations):
    """Draw local object IDs and directed predicates on an existing axis."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    import networkx as nx
    import numpy as np

    object_by_id = {obj["id"]: obj for obj in objects}
    colors = {obj["id"]: plt.get_cmap("tab20")(index % 20)
              for index, obj in enumerate(objects)}
    topology = nx.DiGraph()
    topology.add_nodes_from(object_by_id)
    edge_labels = {}
    for relation in relations:
        pair = (relation["subject"], relation["object"])
        topology.add_edge(*pair)
        edge_labels.setdefault(pair, []).append(relation["predicate"])
    graph_axis.set_title("Observed scene graph | subject -> predicate -> object", fontsize=13)
    if objects:
        positions = nx.spring_layout(topology.to_undirected(), seed=42, k=1.5, iterations=200)
        # Separate nearby label boxes while preserving the graph's spring layout.
        node_ids = list(object_by_id)
        widths = {obj['id']: 0.14 + 0.026 * len(f"#{obj['id']} {obj['label']}")
                  for obj in objects}
        for _ in range(240):
            moved = False
            for index, first in enumerate(node_ids):
                for second in node_ids[index + 1:]:
                    delta = positions[second] - positions[first]
                    overlaps = np.array([(widths[first] + widths[second]) / 2, 0.25]) - np.abs(delta)
                    if np.all(overlaps > 0):
                        dimension = int(np.argmin(overlaps))
                        direction = 1 if delta[dimension] >= 0 else -1
                        shift = direction * (overlaps[dimension] + 0.015) / 2
                        positions[first][dimension] -= shift
                        positions[second][dimension] += shift
                        moved = True
            if not moved:
                break
        for (subject, target), predicates in edge_labels.items():
            start, end = positions[subject], positions[target]
            radius = 0.20 if topology.has_edge(target, subject) else 0.06
            graph_axis.add_patch(FancyArrowPatch(
                start, end, arrowstyle="-|>", mutation_scale=16, color="#627084",
                linewidth=1.2, shrinkA=20, shrinkB=20,
                connectionstyle=f"arc3,rad={radius}", zorder=1,
            ))
            delta = end - start
            midpoint = (start + end) / 2 + radius * np.array([delta[1], -delta[0]]) / 2
            graph_axis.text(*midpoint, " / ".join(dict.fromkeys(predicates)), fontsize=8,
                            ha="center", va="center", color="#334155", zorder=2,
                            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1})
        for obj in objects:
            graph_axis.text(*positions[obj["id"]],
                            f"#{obj['id']} {obj['label']}\n{obj['confidence']:.2f}",
                            fontsize=10, ha="center", va="center", zorder=3,
                            bbox={"boxstyle": "round,pad=0.5", "facecolor": colors[obj["id"]],
                                  "edgecolor": "#475569", "linewidth": 1})
        coordinates = np.array(list(positions.values()))
        graph_axis.set_xlim(coordinates[:, 0].min() - 0.4, coordinates[:, 0].max() + 0.4)
        graph_axis.set_ylim(coordinates[:, 1].min() - 0.3, coordinates[:, 1].max() + 0.3)
        graph_axis.set_aspect("auto")
    else:
        graph_axis.text(0.5, 0.5, "No objects above the detection threshold.",
                        transform=graph_axis.transAxes, ha="center", va="center")

