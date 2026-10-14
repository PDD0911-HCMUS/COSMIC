# COSMIC

**Cross-modal Observed Scene graph Matching with Image Consistency**

Train on GSV-Cities, select checkpoints on Pitts30k-val, and compare against
MixVPR, BoQ, CricaVPR and FoL on AmsterTime. AmsterTime is reserved for final
testing. Baseline checkpoints and model settings are configured independently
in [configs/comparison.yaml](configs/comparison.yaml).

## Entry points

| Entry point | Purpose |
| --- | --- |
| main.py | Prepare source OSGs, train COSMIC, or validate a trained checkpoint |
| comparison/run.py | Shared AmsterTime retrieval and re-ranking benchmark |
| data/download_pitts30k.py | Download and verify the official Pitts30k-val split |
| scripts/setup_ovsgtr.ps1 | Install/build the local OvSGTR inference dependencies |

There is one experiment configuration reader: config.py::load_config().
The YAML files configure datasets, model modules, optimizer/learning rate,
scheduler, precision, checkpoint saving, outputs and OSG export.

## Training pipeline

main.py creates the components in this order:

    YAML + CLI overrides
      -> GSV train_loader
      -> Pitts30k query_loader + database_loader + ground truth
      -> model + criteria
      -> optimizer + scheduler + AMP scaler
      -> TrainEngine + ValidEngine
      -> training, periodic validation and checkpoints

The loaders check source OSG completeness before model construction.
TrainEngine receives the optimizer/scheduler/scaler created in main.py and
handles backward, gradient clipping, optimizer steps, scheduling and resume.
ValidEngine reports Recall@K in a terminal table and JSON/CSV files.

COSMIC encodes a single observation; A and B share the same model:

    image -> visual encoder ------+
                                 +-> image/OSG grounding -> h^F
    OSG -> graph encoder ---------+

    COSMIC(A), COSMIC(B)
      -> cross-OSG alignment
      -> image/graph verification
      -> global scene similarity logits

Retrieval uses normalized h^F descriptors with supervised contrastive loss.
Pair verification uses scene logits with binary cross-entropy. Loss weights
are configured in model.criteria. DINO supplies its own Transformer; ResNet
uses the additional configured Transformer encoder. The graph encoder uses
BERT text features, node/triplet projections and optional bbox embeddings.
Hungarian matching selects the regions, while soft-aligned counterpart tokens
provide graph guidance in verification. This lets verification loss update
the alignment projection and learned unmatched scores. With graph guidance
disabled, the unmatched thresholds only affect the discrete assignment.
Training retrieval uses h^F; standalone visual/OSG summary descriptors are
auxiliary outputs and their unused projection heads receive no training loss.

## Data and samplers

| Dataset | Loader | Sampling |
| --- | --- | --- |
| GSV-Cities | data/gsv_cities_loader.py | PairBatchSampler in data/samplers.py |
| Pitts30k-val | data/pitts30k_loader.py | BatchSampler over SequentialSampler |
| AmsterTime test | comparison/datasets.py | BatchSampler over SequentialSampler |

GSV batch_size counts pairs. B=8 with K=4 samples four places and sixteen
distinct images, producing four positive and four negative pairs.
B=16 produces eight positive and eight negative pairs. B=32 produces sixteen
positive and sixteen negative pairs, using sixty-four images from sixteen places.
The workstation default is B=32 with four workers, tested on the RTX 3070 with
both frozen backbones and the 32-node/100-triplet graph budget. The sampler resamples
views, partners and branch direction each epoch with seed + epoch.

The local GSV release contains 529,506 images across 23 cities. K=4 keeps
62,514 places with 524,701 eligible source images. Each epoch samples K views
per eligible place; it does not visit every eligible image.

A train batch contains a, b and label. Each branch has image [B,3,H,W],
a list of B processed OSG dictionaries, place_label and image metadata.
A validation/test batch contains single observations and image IDs.
Sequential samplers retain every query/database image and the final short batch.
Pitts30k-val has 7,608 queries and 10,000 database images; positives include
all database images within the official 25-meter UTM radius.

Image normalization, resizing and graph selection are shared in
data/preprocessing.py. Full source graphs retain original xyxy pixel boxes,
confidence scores, relations and provenance in lossless .json.gz files.
The loader applies confidence thresholds and node/edge limits at read time.

registry.py holds DATASET_NAMES and registered dataset factories.
Unknown names raise an error. data/validation.py builds the independent
query/database loaders and ground truth from validation.data.

On this workstation, Dataset is a junction to G:\Datasets. GSV is stored at
Dataset/GSV_CITITES and AmsterTime at Dataset/AmsterTime. Pitts30k is a
project junction to F:\Datasets\Pitts30k.

## Commands

Run from the project root. Training is the default mode in config.yaml:

    python -B main.py --config config.yaml --mode train --device cuda

A short CPU training run after both source OSG exports are complete:

    python -B main.py --config config.yaml --mode train --device cpu --batch-size 8 --num-workers 0 --epochs 1 --max-train-batches 2 --output-dir outputs/train_smoke

Resume using the original optimization settings and epoch horizon:

    python -B main.py --config config.yaml --mode train --resume outputs/gsv/checkpoints/last.pt --device cuda

Validate a completed checkpoint on Pitts30k-val:

    python -B main.py --config configs/pitts30k.yaml --resume outputs/gsv/checkpoints/best.pt --device cuda

Prepare source graphs (the same commands resume an export after its previous
process stops):

    python -B main.py --config config.yaml --prepare-data --device cuda
    python -B main.py --config configs/pitts30k.yaml --prepare-data --device cuda
    python -B main.py --config configs/amstertime.yaml --prepare-data --device cuda

TrainEngine saves training_history.json and full checkpoints containing
model/optimizer/scheduler/scaler states, configuration, RNG state and completed
epochs. Resume starts at the next epoch. Checkpoint selection uses the separate
validation metrics and save.monitor; training loss is not substituted for recall.

Final comparison has its own entry point and configuration:

    python -B -m comparison.run --config configs/comparison.yaml --dry-run
    python -B -m comparison.run --config configs/comparison.yaml

The dry run reports required assets without downloading or evaluating them.
See [comparison/README.md](comparison/README.md) for checkpoint requirements,
shared query/database/ground-truth protocol and retrieval/re-ranking reports.

## Reusable visualizations

Visualizations are separate from the training entry point:

| Module | Function |
| --- | --- |
| data/visualization.py | Image pairs and GSV batches |
| data/scene_graph_visualization.py | One image/OSG record or a scene graph |
| visualization/grounding_heatmap.py | Grounding cross-attention heatmaps |
| data/description.py | Loader/sampler and batch structure summaries |

For a heatmap, encode an observation with
model(observation, return_attention=True), then pass its features and the
corresponding normalized image to visualize_grounding_heatmap().
Train, validation and comparison runners do not open inspection windows.

## Preparation and dependencies

For shared source/data on radish and GPU training on ginger, follow
[the lab deployment guide](.devcontainer/README.md). scripts/sync_to_radish.sh
copies the csMachine working copy to shared storage. On ginger, scripts/lab.sh
builds the project's cosmic:gpu-dev image and starts train/valid containers.
Model cache, logs, checkpoints and results persist on radish.
The shared source is /radish/phamd/duypd-proj/COSMIC, alongside SG-Retrieval,
and the copy script uses cp to update source files. Docker mounts
SG-Retrieval/Datasets directly at /workspace/app/Datasets.

See [data/OVSGTR_SETUP.md](data/OVSGTR_SETUP.md) for local inference dependencies,
CUDA build scripts, assets, provenance and full-graph export. See
[data/PITTS30K.md](data/PITTS30K.md) for official split download/resume and validation.

Runtime requirements are in requirements.txt and data/ovsgtr_requirements.txt.
The retained data/OvSGTR inference sources, licenses and asset revision manifest
are included in the repository. Dataset images/OSGs, downloaded model weights,
compiled extensions, caches and experiment outputs are excluded by .gitignore.
The generated paper figures in materials/figures remain included.

For a direct Python run on Ubuntu, point data.root and validation.data.root in
config.yaml to the transferred GSV_CITITES and Pitts30k directories. With the lab layout:

    data.root: /radish/phamd/duypd-proj/SG-Retrieval/Datasets/GSV_CITITES
    validation.data.root: /radish/phamd/duypd-proj/SG-Retrieval/Datasets/Pitts30k

These are the existing YAML fields to edit, not additional dotted YAML keys.
For the lab Docker setup, retain Dataset/GSV_CITITES and Pitts30k instead;
the runtime bind mounts supply those paths inside the container.
Keep each dataset's images, metadata and osg directories together. Prepared
OSGs use relative image paths and do not require regeneration after transfer.

The default graph encoder also requires the local BERT model/tokenizer directory
at data/OvSGTR/weights/bert-base-uncased, even when all OSGs already exist.
Transfer that directory separately, or run python scripts/download_ovsgtr_assets.py
to download the pinned BERT and OvSGTR assets. DINOv2 downloads into .cache when
not cached and model.visual_encoder.local_files_only is false. GroundingDINO's
compiled extension is needed only for generating new OSGs; it must be built for
the target platform if preparation is required.
