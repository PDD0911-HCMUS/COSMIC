# OvSGTR in COSMIC

OvSGTR runs directly in this machine's Python 3.10. Run from `F:\COSMIC`.
`main.py --prepare-data` exports source graphs. For a single image, call the
builder and reusable visualization directly:

```python
from pathlib import Path
from data.osg_builder import OvSGTRBuilder
from data.scene_graph_visualization import visualize_scene_graph

graph = OvSGTRBuilder(device="cuda").build_and_save(
    Path("Dataset/AmsterTime/old/0000.jpg"), Path("outputs/osg_example.json"))
visualize_scene_graph(graph, Path("outputs/osg_example.png"), show=True)
```

The figure shows the source image with boxes, directed scene graph and relations.

## Export source graphs

```powershell
python main.py --config configs/amstertime.yaml --prepare-data --max-pairs 1
python main.py --config configs/amstertime.yaml --prepare-data
```

The first command exports two images from the first CSV pair; the second
exports all CSV pairs, reusing valid graphs from the first run. The builder is
loaded once only when inference is needed; completely cached runs skip model
construction. Images repeated in multiple CSV rows are inferred once.

Files are saved to `Dataset/AmsterTime/osg/{old,new}/*.json` with relative image paths,
original image sizes, xyxy pixel boxes, object confidence, and directed
relations with confidence/triplet scores. Node IDs retain object identity;
text triplets can be derived at load time. Images stay in the source dataset.
`Dataset/AmsterTime/osg/index.json` stores pair indices, labels and image/graph paths from
the CSV. A limited export marks `complete_dataset=false`.

Storage defaults use confidence floors of 0 and `max_relations=None`. This
keeps every valid detection after upstream postprocessing and one best
non-background predicate per ordered object pair. It does not preserve the
model's intermediate tensors or alternative predicate scores. Upstream
`num_select=300`, class-aware NMS IoU 0.5 and `detections_per_img=100` still
apply and are recorded in provenance. A 100-node source graph therefore has
9,900 directed relations, including weak predictions to filter later.

Optional export floors are `--save-object-threshold` and
`--save-relation-threshold`. They affect what can later be recovered; model
input node/edge budgets should be applied separately by the loader.
The default directory follows `--data-root` as `<data-root>/osg`.
`--osg-root` selects another output directory; `--overwrite-osg` forces a
fresh export. Existing graphs are reused only when the image hash, source ID,
generation fingerprint and graph validation match. Graph and index writes
use atomic replacement.

## Environment and setup

Tested with PyTorch 2.7.1+cu126, torchvision 0.22.1+cu126, Transformers 4.40.2,
tokenizers 0.19.1, timm 0.9.2, NumPy 1.26.4 and YAPF 0.40.1.
Runtime dependencies are in `data/ovsgtr_requirements.txt` and installed
directly into the selected Python. Standalone training/demo dependencies
such as wandb, supervision and dataset evaluators are not needed here.

The compiled GroundingDINO extension targets Python 3.10 and RTX 3070 (sm_86).
Rebuilding uses CUDA Toolkit 12.6 and Visual Studio 2019 Professional C++ tools;
the compiler path is in `scripts/build_groundingdino.cmd`.

Repeat setup, including dependency installation and inference check:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_ovsgtr.ps1
```

The adapter loads local source paths itself; no global GroundingDINO `.pth`
registration is needed. CUDA source API compatibility changes are reproduced
by `scripts/patch_ovsgtr.py`. The adapter checks the deterministic legacy BERT
`position_ids` buffer before removing that extra checkpoint entry, then loads
all learned parameters with `strict=True`.

## Retained sources and assets

`data/OvSGTR` is an inference snapshot of the author's repository:
https://github.com/gpt4vision/OvSGTR. Its original full commit is retained in
`source_manifest.json`, rather than a nested Git checkout.

```text
OvSGTR/
|-- config/          # Swin-T OvD+R config and shared VG vocabulary
|-- GroundingDINO/   # required Python modules, CUDA binary/sources, setup.py
|-- util/            # helpers imported by model construction
|-- weights/         # OvSGTR checkpoint, BERT and asset revision manifest
|-- source_manifest.json
|-- README.md
`-- LICENSE
```

Training/evaluation entry points, dataset loaders, other configs, demo files,
figures, papers, build intermediates and nested Git metadata were removed.
Some loss/matcher helpers remain because the original model constructor
imports/constructs them, even for inference. Both upstream licenses are kept.

The checkpoint is the original ECCV **Swin-T OvD+R-SGG**
`vg-ovdr-swint.pth`, without MegaSG pretraining. BERT and the final checkpoint
are local; no separate GroundingDINO initialization checkpoint or VG images
are required. `assets_manifest.json` records pinned Hugging Face revisions;
the download script reuses existing files and those revisions.

## Graph format

The builder reads original images, resizes with aspect ratio preserved
(short side 800, max side 1333), then applies ToTensor and ImageNet mean/std.
The loader's normalized 224x224 tensors are used for the COSMIC image stream.
The paired loader reads source OSGs from `<data-root>/osg` and applies the
graph selection in `data.graph` of `configs/amstertime.yaml`. Selected boxes are scaled alongside
the images; batch graphs retain variable sizes as a list for each A/B branch.
Source graphs are read without modifying the saved JSON. See the repository
README for the tensor/graph batch contract and loader commands.

Each JSON graph contains image ID/size/path, objects, directed relations,
confidence scores and provenance. Boxes are **xyxy pixels in the original
image**. Default shared prompts contain 150 VG objects and 50 VG predicates;
custom candidate vocabularies can be passed to `OvSGTRBuilder`.
Model-input graph filters use object/relation confidence thresholds 0.25/0.20,
with at most 100 relations by default. Source export defaults to 0/0 with no
relation cap. No pair labels are used to generate an OSG.

Dataset exports default to lossless `.json.gz` via
`export.graph_format: json_gzip` in the experiment YAML. Manifests/indexes
remain plain JSON, and loaders accept both graph formats. Re-exporting a
valid plain JSON graph in gzip format converts it without inference and keeps
the original file. Use `--graph-format json` for uncompressed source exports.
