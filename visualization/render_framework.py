"""Render the COSMIC architecture as an editable conference-paper figure.

Run from the project root:
    python -B -m visualization.render_framework

This is an offline drawing utility: it does not load COSMIC or run OvSGTR.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.path import Path as DrawingPath
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
INK = "#263343"
MUTED = "#546275"
PALETTE = {
    "visual": ("#E8EFF8", "#54749C"),
    "graph": ("#F0EAF6", "#84639C"),
    "fusion": ("#E6F2EF", "#4C867B"),
    "alignment": ("#FAF0DB", "#A48A50"),
    "verification": ("#E7EEF3", "#597889"),
    "score": ("#F8E8DF", "#AD775C"),
    "loss": ("#FFFFFF", "#697587"),
}


CAPTION = r"""\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{materials/figures/cosmic_framework.pdf}
    \caption{Overview of COSMIC. A shared single-view encoder extracts visual
    and BERT-based node/triplet features and grounds graph tokens in each image.
    The fused token set $H_X^F$ supports cross-view alignment, while its pooled
    descriptor $h_X^F$ supports independent retrieval. Type-aware soft alignment
    and partial Hungarian matching establish correspondences. Soft counterpart
    context guides bbox-based visual region verification, producing a symmetric
    pair embedding $z_{AB}$ and a same-place score $s_{AB}$. Dashed connections
    denote the supervised contrastive retrieval and binary pair-verification
    objectives; BCE receives raw logits before sigmoid. Input photographs show
    an illustrative AmsterTime pair; graph drawings are schematic.}
    \label{fig:cosmic-framework}
\end{figure*}
"""


def text(ax, x, y, value, *, size=8, bold=False, color=INK, ha="center", background=False):
    return ax.text(
        x, y, value, fontsize=size, fontweight="bold" if bold else "normal",
        color=color, ha=ha, va="center", zorder=10, linespacing=1.16,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5} if background else None,
    )


def block(ax, x, y, width, height, title, detail, kind, *, dashed=False):
    fill, edge = PALETTE[kind]
    patch = Rectangle(
        (x, y), width, height, facecolor=fill, edgecolor=edge, linewidth=0.85,
        linestyle=(0, (3, 2)) if dashed else "solid", zorder=3,
    )
    ax.add_patch(patch)
    title_artist = text(ax, x + width / 2, y + height * 0.67, title, size=8.1, bold=True)
    detail_artist = text(ax, x + width / 2, y + height * 0.25, detail, size=7.25, color=MUTED)
    return patch, title_artist, detail_artist


def arrow(ax, points, *, dashed=False, color=INK, width=0.9, zorder=5, head=True):
    path = DrawingPath(points, [DrawingPath.MOVETO] + [DrawingPath.LINETO] * (len(points) - 1))
    artist = FancyArrowPatch(
        path=path, arrowstyle="-|>" if head else "-", mutation_scale=7.4,
        linewidth=width, edgecolor=color, facecolor=color,
        linestyle=(0, (3.2, 2.4)) if dashed else "solid",
        capstyle="round", joinstyle="round", zorder=zorder,
    )
    ax.add_patch(artist)
    return artist


def photograph(ax, path, x, y, width=27, height=20):
    with Image.open(path) as source:
        rgb = source.convert("RGB")
        pixel_width, pixel_height = rgb.size
        ratio = min(width / pixel_width, height / pixel_height)
        w, h = pixel_width * ratio, pixel_height * ratio
        left, bottom = x + (width - w) / 2, y + (height - h) / 2
        ax.imshow(rgb, extent=(left, left + w, bottom, bottom + h), aspect="auto",
                  interpolation="lanczos", zorder=2)
    ax.add_patch(Rectangle((x, y), width, height, facecolor="none", edgecolor="#A0A9B4",
                           linewidth=0.55, zorder=4))


def graph_icon(ax, x, y, width=27, height=20, *, variant=False):
    """A schematic OSG glyph, explicitly not a model prediction."""
    ax.add_patch(Rectangle((x, y), width, height, facecolor="#FCFCFD",
                           edgecolor="#A0A9B4", linewidth=0.55, zorder=2))
    nodes = [(x + 5, y + 15), (x + 19, y + 15), (x + 9, y + 5), (x + 22, y + 5)]
    if variant:
        nodes = [(x + 5, y + 14), (x + 18, y + 16), (x + 10, y + 5), (x + 23, y + 6)]
    for start, end in ((0, 1), (0, 2), (2, 3), (1, 3)):
        a, b = nodes[start], nodes[end]
        dx, dy = b[0] - a[0], b[1] - a[1]
        norm = (dx * dx + dy * dy) ** 0.5
        a = (a[0] + dx * 2.1 / norm, a[1] + dy * 2.1 / norm)
        b = (b[0] - dx * 2.1 / norm, b[1] - dy * 2.1 / norm)
        arrow(ax, [a, b], color="#8B8296", width=0.6, zorder=3)
    for i, (cx, cy) in enumerate(nodes):
        fill = "#DCD0EC" if i % 2 else "#C4D7E9"
        ax.add_patch(Rectangle((cx - 2.1, cy - 1.7), 4.2, 3.4, facecolor=fill,
                               edgecolor="#6C7590", linewidth=0.55, zorder=4))


def pair_paths(dataset_root, pair_index):
    with (dataset_root / "image_pairs.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not 0 <= pair_index < len(rows):
        raise ValueError(f"pair-index must be in [0,{len(rows) - 1}]")
    row = rows[pair_index]
    return (dataset_root / "old" / f"{int(row['old_num']):04d}.jpg",
            dataset_root / "new" / f"{int(row['new_num']):04d}.png")


def render(image_a, image_b, output, *, dpi=600, width_inches=7.12):
    for path in (image_a, image_b):
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 8,
        "mathtext.fontset": "dejavusans", "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "pdf.compression": 9, "axes.unicode_minus": False,
    })
    figure = plt.figure(figsize=(width_inches, width_inches * 184 / 324), facecolor="white")
    ax = figure.add_axes([0, 0, 1, 1], xlim=(0, 324), ylim=(0, 184))
    ax.set_axis_off()
    boxes = []

    text(ax, 4, 179, "(a) Shared single-view encoding", size=8.8, bold=True, ha="left")
    text(ax, 171, 179, "(b) Cross-view alignment and verification", size=8.8, bold=True, ha="left")
    text(ax, 4, 36, "(c) Objectives", size=8.5, bold=True, ha="left")
    ax.add_patch(Rectangle((34, 43), 112, 122, facecolor="#FAFBFC", edgecolor="#A7AFBA",
                           linewidth=0.7, linestyle=(0, (3, 2.6)), zorder=0))
    text(ax, 90, 165, "Shared weights", size=7.3, color=MUTED, background=True)

    for view, image_path, shift in (("A", image_a, 0), ("B", image_b, -59)):
        photograph(ax, image_path, 4, 142 + shift)
        graph_icon(ax, 4, 112 + shift, variant=view == "B")
        text(ax, 17.5, 167 + shift, rf"Image $I_{view}$", size=8)
        text(ax, 17.5, 136 + shift, rf"OSG $G_{view}$", size=8)

        boxes.append(block(ax, 40, 140 + shift, 42, 22, "Visual Encoder",
                           "DINOv2\n+ projector", "visual"))
        boxes.append(block(ax, 40, 111 + shift, 42, 22, "OSG Encoder",
                           "BERT\n+ projectors", "graph"))
        boxes.append(block(ax, 99, 121 + shift, 39, 34, "Image–OSG\nGrounding",
                           "self-/cross-attn\n+ pooling", "fusion"))

        arrow(ax, [(31, 152 + shift), (40, 152 + shift)])
        arrow(ax, [(31, 122 + shift), (40, 122 + shift)])
        arrow(ax, [(82, 151 + shift), (91, 151 + shift), (91, 146 + shift), (99, 146 + shift)])
        arrow(ax, [(82, 122 + shift), (91, 122 + shift), (91, 130 + shift), (99, 130 + shift)])

    # Fused-token paths. Batch axes are deliberately omitted in the paper figure.
    arrow(ax, [(138, 145), (161, 145), (161, 115), (172, 115)])
    text(ax, 152, 151, r"$H_A^F$", size=8.8, background=True)
    text(ax, 152, 140, r"$L_A\!\times\!D$", size=7, color=MUTED, background=True)
    arrow(ax, [(138, 86), (161, 86), (161, 103), (172, 103)])
    text(ax, 152, 92, r"$H_B^F$", size=8.8, background=True)
    text(ax, 152, 81, r"$L_B\!\times\!D$", size=7, color=MUTED, background=True)

    # Cached visual/geometry outputs originate at the shared-encoder boundary.
    arrow(ax, [(146, 160), (244, 160), (244, 128)], color=MUTED, width=0.75)
    text(ax, 194, 165, r"$V_A,\ b_A$", size=7.4, color=MUTED, background=True)
    arrow(ax, [(146, 48), (244, 48), (244, 92)], color=MUTED, width=0.75)
    text(ax, 182, 54, r"$V_B,\ b_B$", size=7.4, color=MUTED, background=True)

    boxes.append(block(ax, 172, 92, 38, 36, "Cross–OSG\nAlignment",
                       "soft + Hungarian", "alignment"))
    boxes.append(block(ax, 222, 92, 41, 36, "Image–Graph\nVerification",
                       "bbox regions\n+ cross-attention", "verification"))
    boxes.append(block(ax, 283, 92, 36, 36, "Global Scene\nSimilarity",
                       "MLP", "score"))

    arrow(ax, [(210, 109), (222, 109)])
    text(ax, 216, 116, r"$S,M$", size=7.3, background=True)
    text(ax, 216, 85, "matches + soft context", size=7.05, color=MUTED, background=True)
    arrow(ax, [(263, 109), (283, 109)])
    text(ax, 273, 116, r"$z_{AB}$", size=8.1, background=True)
    text(ax, 273, 103, r"$D$", size=7, color=MUTED, background=True)

    arrow(ax, [(303, 92), (303, 83)])
    ax.add_patch(Rectangle((282, 68), 39, 15, facecolor="white", edgecolor="#AD775C",
                           linewidth=0.85, zorder=3))
    text(ax, 301.5, 75.5, r"$s_{AB}=\sigma(\ell_{AB})$", size=7.9)

    # Training-only connections: descriptors for retrieval, raw logits for BCE.
    boxes.append(block(ax, 70, 7, 68, 24, "Retrieval loss",
                       r"$\mathcal{L}_{\mathrm{ret}}=\mathrm{SupCon}(h_A^F,h_B^F)$", "loss", dashed=True))
    boxes.append(block(ax, 159, 7, 68, 24, "Total loss",
                       r"$\lambda_r\mathcal{L}_{\mathrm{ret}}+\lambda_v\mathcal{L}_{\mathrm{ver}}$", "loss", dashed=True))
    boxes.append(block(ax, 250, 7, 70, 24, "Verification loss",
                       r"$\mathcal{L}_{\mathrm{ver}}=\mathrm{BCE}(\ell_{AB},y_{AB})$", "loss", dashed=True))

    arrow(ax, [(107, 121), (107, 114), (88, 114), (88, 35), (97, 35), (97, 31)],
          dashed=True, color=MUTED, width=0.8, zorder=1)
    text(ax, 84, 42, r"$h_A^F$", size=8.1, ha="right", color=MUTED, background=True)
    arrow(ax, [(118, 62), (118, 31)], dashed=True, color=MUTED, width=0.8)
    text(ax, 129, 44, r"$h_B^F$", size=8.1, color=MUTED, background=True)
    arrow(ax, [(283, 92), (275, 92), (275, 31)], dashed=True, color=MUTED, width=0.8)
    text(ax, 267, 61, r"$\ell_{AB}$", size=8.1, color=MUTED, background=True)
    text(ax, 42, 19, "place IDs", size=7.3, color=MUTED)
    arrow(ax, [(59, 19), (70, 19)], dashed=True, color=MUTED, width=0.8)
    text(ax, 305, 43, r"pair label $y_{AB}$", size=7.3, color=MUTED, background=True)
    arrow(ax, [(311, 39), (311, 31)], dashed=True, color=MUTED, width=0.8)
    arrow(ax, [(138, 19), (159, 19)], dashed=True, color=MUTED, width=0.8)
    arrow(ax, [(250, 19), (227, 19)], dashed=True, color=MUTED, width=0.8)
    arrow(ax, [(5, 11), (16, 11)], dashed=True, color=MUTED, width=0.8, head=False)
    text(ax, 21, 11, "training only", size=7.05, color=MUTED, ha="left")

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for rectangle, title, detail in boxes:
        bounds = rectangle.get_window_extent(renderer)
        for artist in (title, detail):
            text_bounds = artist.get_window_extent(renderer)
            if (text_bounds.x0 < bounds.x0 + 1 or text_bounds.x1 > bounds.x1 - 1
                    or text_bounds.y0 < bounds.y0 + 1 or text_bounds.y1 > bounds.y1 - 1):
                raise ValueError(f"Text does not fit its block: {artist.get_text()!r}")

    stem = output / "cosmic_framework"
    figure.savefig(stem.with_suffix(".pdf"), dpi=dpi, facecolor="white",
                   metadata={"Title": "COSMIC architecture", "Author": "COSMIC", "Creator": "Matplotlib"})
    figure.savefig(stem.with_suffix(".svg"), dpi=dpi, facecolor="white",
                   metadata={"Title": "COSMIC architecture", "Description": "Shared image–OSG encoding, cross-view alignment, verification and training objectives."})
    figure.savefig(stem.with_suffix(".png"), dpi=dpi, facecolor="white")
    # A smaller copy is convenient for visual review; the paper PNG stays at dpi.
    figure.savefig(stem.with_name("cosmic_framework_preview.png"), dpi=200, facecolor="white")
    plt.close(figure)
    (output / "cosmic_framework_caption.tex").write_text(CAPTION, encoding="utf-8")
    return stem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "materials/figures")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "Dataset/AmsterTime")
    parser.add_argument("--pair-index", type=int, default=6)
    parser.add_argument("--image-a", type=Path)
    parser.add_argument("--image-b", type=Path)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--width-inches", type=float, default=7.12)
    args = parser.parse_args()
    if args.dpi <= 0 or args.width_inches <= 0:
        parser.error("dpi and width-inches must be positive")
    if (args.image_a is None) != (args.image_b is None):
        parser.error("Supply both --image-a and --image-b")
    image_a, image_b = ((args.image_a, args.image_b) if args.image_a is not None
                        else pair_paths(args.dataset_root, args.pair_index))
    stem = render(image_a, image_b, args.output, dpi=args.dpi, width_inches=args.width_inches)
    print(f"Rendered: {stem}.pdf / .svg / .png")


if __name__ == "__main__":
    main()
