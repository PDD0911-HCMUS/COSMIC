# COSMIC comparisons

Run independently of the training entry point:

```powershell
python -B -m comparison.run --config configs/comparison.yaml --dry-run
python -B -m comparison.run --config configs/comparison.yaml --list-methods
```

The dry run checks the AmsterTime CSV/images, shared query/database manifest,
configuration, and whether checkpoint/repo paths exist. It does not instantiate
models, download weights, or write benchmark results. It can succeed while
reporting missing assets; ordinary runs record those methods as failed and exit
with a nonzero status.

## Files

| File | Role |
| --- | --- |
| `run.py` | CLI, isolated workers, effective config, aggregate reports |
| `config.py` | Strict comparison YAML parsing and project-relative paths |
| `registry.py` | Available adapter names and lazy construction |
| `datasets.py` | Single observations, shared CSV ground truth and manifest |
| `evaluator.py` | Exact retrieval, Recall@K and candidate-only re-ranking |
| `report.py` | Terminal table and atomic JSON/CSV reports |
| `worker.py` | Evaluate one method in a fresh Python process |
| `adapters/` | Official baseline architectures/preprocessing and COSMIC |

This runner is separate from `main.py`. `configs/comparison.yaml` chooses
methods independently of `config.yaml` and the training dataset registry.
Experiment names are arbitrary; `adapter` must be one of `cosmic`, `mixvpr`,
`boq`, `cricavpr`, or `fol`. Unknown names/settings raise errors.

## Protocol

Train COSMIC on GSV and choose checkpoints/hyperparameters on independent
validation data, such as Pitts30k-val. Keep AmsterTime for final evaluation.
The comparison runner does not train models or select a best checkpoint.
Released baseline checkpoints retain the authors' selection history; this is
a comparison of complete systems, not a controlled same-backbone experiment.

Full AmsterTime evaluation uses all 1231 historical queries and all 1231
contemporary database images. Ground truth comes from `image_pairs.csv`.
Canonical evaluation IDs are dataset-relative image paths, because original
archive/source IDs can repeat. OSG identity checks still use the original IDs.
The database is encoded in filename order, followed by queries in filename
order, without shuffle or dropping the final batch. Combined batch boundaries
are retained because CricaVPR correlates images within an inference batch.

Default search is exact squared L2 on native descriptors; this retains PCA
magnitudes used by the official evaluation pipelines. Unit-normalized
descriptors have the same L2/cosine ordering in exact arithmetic. Setting
`metric: cosine` normalizes all descriptors, including PCA outputs; record that
change when comparing published numbers. Ties use lexical database image IDs.
Invalid/zero query descriptors count as misses, and invalid database descriptors
are excluded from candidates; both counts are reported. Every query is retained
in the denominator, including an empty-graph COSMIC query.

```powershell
# Smoke inference: eight queries against the COMPLETE database.
python -B -m comparison.run --methods cosmic --max-queries 8 --device cpu

# Full benchmark after supplying trained checkpoints.
python -B -m comparison.run --methods cosmic mixvpr boq_resnet50 cricavpr fol
```

`max_queries` never reduces the database. Reports label reduced query runs as
smoke tests, not full benchmark results. Recall values are fractions in files
and percentages in the terminal.

## Checkpoints and author repositories

Update paths in the comparison YAML, or place repos/weights at its suggested
locations. Weights are not bundled. COSMIC requires a completed TrainEngine
checkpoint, such as `outputs/gsv/checkpoints/best.pt`; the experiment config
must match the checkpoint's model and image/graph preprocessing. You may point
`methods.cosmic.config` to the training YAML: the comparison dataset root is
selected independently. A raw/untrained COSMIC state dict is rejected.

Official sources:

- [MixVPR](https://github.com/amaralibey/MixVPR): ResNet50, 4096-D, 320x320.
  The adapter uses its official backbone/aggregator modules directly, without
  importing the Lightning training program.
- [BoQ](https://github.com/amaralibey/Bag-of-Queries): ResNet50, 16384-D,
  384x384, or DINOv2 ViT-B/14, 12288-D, 322x322. Checkpoint architecture follows
  the author's `hubconf.py`; preprocessing follows its README.
- [CricaVPR](https://github.com/Lu-Feng/CricaVPR): ViT-B/14, 224x224, inference
  batch 16 for AmsterTime. Native output is 10752-D. Evaluating the reported
  4096-D variant requires an independently fitted PCA transform, supplied below.
- [FoL](https://github.com/chenshunpeng/FoL): global retrieval, ViT-L/14 by
  default, 504x504. `params.backbone: vitb` selects its base variant with matching
  weights. Uses the official `test=True` forward. Its re-ranking uses mutual
  nearest-neighbor local matching and counts cosine similarities above
  `params.local_threshold` (default 0.7), following `local_sim` in
  [the author's test script](https://github.com/chenshunpeng/FoL/blob/main/test_FoL.py).
  Matching is implemented in device-neutral PyTorch so the runner does not
  import the upstream CUDA/Linux-specific test script. It does not add FoL++
  geometric verification. Re-ranking stages are available for COSMIC and FoL.

For example, clone the repos when preparing baseline assets:

```powershell
git clone https://github.com/amaralibey/MixVPR.git comparison/external/MixVPR
git clone https://github.com/amaralibey/Bag-of-Queries.git comparison/external/Bag-of-Queries
git clone https://github.com/Lu-Feng/CricaVPR.git comparison/external/CricaVPR
git clone https://github.com/chenshunpeng/FoL.git comparison/external/FoL
```

Download the GSV-trained weights linked by those repos and set the YAML paths.
The adapters load complete tensor weights with `weights_only=True` and strict
key/shape matching. Training checkpoints with `state_dict`/`model_state_dict`,
raw state dicts, and a leading `module.` prefix are supported for baselines.
Each method runs separately so upstream `backbone`, `network`, and other import
names cannot collide with another method.

Baseline inference uses the existing PyTorch/torchvision installation. Upstream
CricaVPR/DINO and FoL modules may need their own dependencies, such as timm,
einops, or matplotlib; consult the pinned author repo. BoQ-DINO/FoL may download
DINO source on first use, stored under `.cache/comparison_hub`, but skip redundant
DINO initialization weights because the complete VPR checkpoint replaces them.
For local/offline source, set `params.dinov2_repo` to a local
`facebookresearch/dinov2` checkout with its dependencies installed. The dry run
does not check third-party imports.

Optional PCA files use tensor data rather than a pickled sklearn estimator:

```python
torch.save({
    "mean": torch.as_tensor(pca.mean_, dtype=torch.float32),
    "components": torch.as_tensor(pca.components_, dtype=torch.float32),
}, "comparison/weights/cricavpr_pca.pt")
```

If the fitted estimator uses whitening, bake that scaling into each component
row (divide by the square root of its explained variance). Fit PCA on training
data, never on AmsterTime. The adapter applies `(descriptor - mean) @ components.T`.
Set `methods.cricavpr.pca` to that file. Native and PCA variants must be labeled
with their actual dimensions in the result table.

## Re-ranking and timing

For COSMIC, `stages: [retrieval, rerank]` encodes each observation once, retains the necessary
visual/graph features on CPU, and reorders the retrieved top `rerank_top_k`
candidates using alignment, verification and raw scene logits. Ground truth
is not passed to candidate generation or scoring. Ties preserve retrieval
order. Candidates without verified matches follow scored candidates; if all
candidates lack matches, the retrieval order is unchanged. Candidate recall
and the number of pairs without a pair score are reported. Pick K on validation
and freeze it before test. This implementation verifies one candidate pair at
a time; it supports CPU and bounds device memory but is not a batched latency
optimization.

FoL caches its selected local features and reorders the same retrieved top K by
the number of mutual matches above the configured cosine threshold. Zero-match
ties retain retrieval order. Both adapters verify one candidate pair at a time;
the cache size and actual timings are reported. Large FoL local caches can
require several GiB of CPU RAM on the full dataset.

The saved-OSG loader validates source image/graph identities and applies the
training graph filters. It does not invoke OvSGTR or modify source graphs.
OSG generation time is excluded from these measurements. Report it separately
for online COSMIC latency. Encoding milliseconds per image measure batched
throughput, not single-query response time. JSON also records preprocessing,
encoding, search and re-ranking time, descriptor/cache bytes, and peak CUDA
allocated memory when running on CUDA.

Each timestamped output directory contains `resolved_config.yaml`,
`manifest.json`, per-method logs/results, per-stage predictions, `results.json`
and `results.csv`. Checkpoint/PCA SHA256 hashes, source repo commit when
available, preprocessing, descriptor dimensions, batch size and benchmark
fingerprints make runs traceable. Failures are recorded rather than converted
into zero recall, and the runner returns nonzero if any selected method failed.
