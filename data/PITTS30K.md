# Pitts30k validation

The physical data directory is `F:\Datasets\Pitts30k`. The project junction
`F:\COSMIC\Pitts30k` points there; relative YAML roots use `Pitts30k`.
GSV is used for training, Pitts30k-val for checkpoint selection, and AmsterTime
for final comparisons. Pitts30k is registered as evaluation-only.

## Download and layout

```powershell
python -B -m data.download_pitts30k
```

The downloader uses the [CIIRC NetVLAD server](https://data.ciirc.cvut.cz/public/projects/2015netVLAD/Pittsburgh250k/),
the official `pitts30k_val.mat` split, and the server's published archive MD5s.
It verifies the split SHA256, downloads `queries_real.tar` and database archives
`000.tar` through `003.tar`, and extracts only the required validation images.
HTTP downloads resume from `.part` files. Extraction retains original JPEG
bytes, verifies images, and rejects conflicting existing files.
After verifying the images, it writes a validation manifest and compressed
25-meter ground truth derived from the official UTM coordinates.

```text
Pitts30k/
  metadata/pitts30k_val.mat
  metadata/download_manifest.json
  metadata/validation_manifest.json
  metadata/ground_truth_25m.json.gz
  downloads/                  # Verified archives retained for resuming
  images/val/database/000/...  # 10000 images across shards 000-003
  images/val/queries/000/...   # 7608 images
  osg/val/database/000/...
  osg/val/queries/000/...
  osg/manifest.json
```

Canonical image IDs include role/shard directories; query and database images
with similar names remain distinct. Raw archives require roughly 4 GiB in
addition to the extracted images/OSGs. They are retained until you choose to
remove them after a complete, verified download.

## OSG preparation

```powershell
python -B main.py --config configs/pitts30k.yaml --prepare-data --device cuda
```

Export keeps full objects/relations and source bounding boxes as lossless gzip
JSON. Existing valid graphs are reused only when model/code settings, image
identity, and image SHA256 match. The loader filters graphs at read time using
the same preprocessing as GSV. Partial exports do not qualify as a ready
validation set.

Preparation can run immediately alongside GSV using the `main.py` command
above. For optional waiting when GPU memory is limited, use:

```powershell
python -u -B -m data.pitts30k_prepare --device cuda --min-free-mib 6000
```

An optional `--wait-for-pid <GSV_PID>` also waits for that process to finish.
The queue never stops the GSV process. It writes state to
`outputs/pitts30k/osg_job.json`; `waiting_for_gpu` means inference has not begun.
The background job launched during setup writes stdout to
`outputs/pitts30k/osg_prepare.log` and stderr to
`outputs/pitts30k/osg_prepare.err.log`.
Only one Pitts30k export process should run at a time.

## Validation

`data/pitts30k_loader.py` returns single observations through
`torch.utils.data.DataLoader`. Retrieval positives are every database image
within 25 meters of each query's UTM coordinates, following the
[official MixVPR loader](https://github.com/amaralibey/MixVPR/blob/main/dataloaders/PittsburgDataset.py).
No images are dropped or shuffled during validation.
Each query/database batch has `image` with shape `[B,3,H,W]`, a list of
`B` processed `osg` dictionaries, image IDs/paths and original sizes.
`build_validation_inputs(config)` returns `query_loader`, `database_loader`
and a query-ID-to-positive-database-ID mapping for `ValidEngine`.

GSV training batches retain the paired `a`, `b`, `label` structure. Both
branches use the same image normalization and graph selection as validation;
source boxes and graph identities remain attached to their own images.

The `validation.data` section in `config.yaml` selects Pitts30k-val and
inherits training image normalization, size, and graph filters. In training,
`main.py` obtains query/database loaders and positives through the dataset
registry and binds `ValidEngine` to `TrainEngine`. Model weights are updated
only on GSV batches. Best-checkpoint selection uses `save.monitor: recall@1`.

Standalone validation of a completed GSV checkpoint:

```powershell
python -B main.py --config configs/pitts30k.yaml --mode evaluate --resume outputs/gsv/checkpoints/best.pt --device cuda
```

The evaluator checks checkpoint/model/preprocessing compatibility, runs the
existing global retrieval `ValidEngine`, and exports Recall tables and JSON/CSV.
It does not currently re-rank validation candidates; final AmsterTime
retrieval/re-ranking comparisons remain in `comparison/run.py`.

Validation uses every query and the full database, with all geographic positives.
The loaders explicitly use BatchSampler over SequentialSampler with drop_last=False.
