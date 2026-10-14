# OvSGTR inference sources for COSMIC

This directory retains the modules needed for frozen scene graph inference.
Original repository: https://github.com/gpt4vision/OvSGTR

The upstream commit is preserved in source_manifest.json. The original
ECCV 2024 Swin-T OvD+R-SGG checkpoint and local BERT files are in weights/;
their source revisions are recorded in weights/assets_manifest.json.
The VG vocabulary was extracted from the author's dataset source and saved
as config/vocabulary.json before removing unused dataset loaders.

See [COSMIC setup and usage](../OVSGTR_SETUP.md). Run from the project root:

```powershell
python -B main.py --config configs/amstertime.yaml --prepare-data --device cuda
```

Training/evaluation/demo assets and nested Git metadata have been removed.
Keep LICENSE and GroundingDINO/LICENSE with these sources.

Paper: Zuyao Chen et al., Expanding Scene Graph Boundaries: Fully
Open-vocabulary Scene Graph Generation via Visual-Concept Alignment and
Retention, ECCV 2024, pp. 108-124.
