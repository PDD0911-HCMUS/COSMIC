# COSMIC conference figure

The figure follows the current COSMIC code: shared single-observation encoding,
soft/Hungarian cross-OSG alignment, graph-guided visual region verification,
scene scoring, and the two training losses.

## Files

- `cosmic_framework.pdf`: vector diagram with embedded TrueType fonts; use
  this version in the paper. The two example input photographs are raster.
- `cosmic_framework.svg`: editable vector diagram, with embedded photographs
  and text retained as text. Arial should be available when editing it.
- `cosmic_framework.png`: 600 dpi rendering, approximately 7.12 inches wide.
- `cosmic_framework_preview.png`: smaller preview for reviewing the layout.
- `cosmic_framework_caption.tex`: suggested two-column figure and caption.

The diagram uses white backgrounds, muted fills, thin borders, and 7–8.8 pt
labels at the exported size. Panel (a) repeats the same learned modules for A/B;
it does not instantiate independent models. Solid arrows carry features;
dashed connections denote training objectives. The dashed enclosing rectangle
marks the shared encoder.

## Notation

| Symbol | Meaning |
| --- | --- |
| `I_X`, `G_X` | Image and precomputed OSG for view X |
| `V_X`, `b_X` | Cached visual patch tokens and node bboxes |
| `H_X^F` | Fused node/triplet token matrix, `L_X × D` |
| `h_X^F` | Pooled, L2-normalized retrieval descriptor, `D` |
| `S`, `M` | Similarity matrix and partial Hungarian matches |
| Soft context | Opposite-view tokens aggregated by soft alignment |
| `z_AB` | Symmetric verified-pair descriptor, `D` |
| `ell_AB` | Raw scene logit, used by BCE before sigmoid |
| `s_AB` | Same-place score after sigmoid |

Batch axes are omitted to keep the paper figure readable. Current defaults are
`D=256`, `P=256` DINOv2-S patches for 224×224 inputs, and `L_X=N_X+E_X≤132` after
the loader's node/triplet filtering. Actual tensor dtypes and batch shapes are
documented in `../cosmic_pipeline.md`. The visual encoder label shows the
currently configured DINOv2 backbone; ResNet is the supported alternative.

The example photographs come from the CSV-matched AmsterTime pair at row index
6: `old/0006.jpg` and `new/0006.png`. They illustrate model inputs, not training
data selection. GSV is the training dataset. The graph drawings are schematic,
not predictions from these photographs. The caption identifies this distinction.

## Regenerate

From `F:\COSMIC`:

```powershell
python -B -m visualization.render_framework
```

Choose another CSV-matched illustration or supply two photographs:

```powershell
python -B -m visualization.render_framework --pair-index 3
python -B -m visualization.render_framework --image-a path/to/a.jpg --image-b path/to/b.jpg
```

The layout source is `visualization/render_framework.py`. It only uses
Matplotlib and Pillow and does not load a model or use the GPU. Import the PDF
with `graphicx`; adjust the path in the caption template if the figure is copied
into a separate paper directory.
