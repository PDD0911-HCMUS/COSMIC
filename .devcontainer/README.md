# COSMIC: build and run on ginger

| Location | Role |
| --- | --- |
| csMachine | Pull code and run scripts/sync_to_radish.sh |
| radish | Store source, datasets, model cache, logs and results |
| ginger | Build cosmic:gpu-dev and run COSMIC containers |
| ginger's Docker storage | Store the image, build cache and container filesystem |
| /workspace/app in the container | Read/write the shared COSMIC source |

All Docker commands below run on ginger and operate on COSMIC's own image.
The container uses ginger's CPU, RAM and GPU. Editing /workspace/app writes
back to radish. GSV training and Pitts30k validation run through main.py.

## 1. Copy source from csMachine

After pushing Windows changes, on csMachine:

    cd ~/ThisPC-DuyPC/COSMIC
    git pull --ff-only
    bash scripts/sync_to_radish.sh

The script uses cp to update /radish/phamd/duypd-proj/COSMIC. It checks required
files and preserves datasets, weights, cache, outputs and private .env files.
Git remains on csMachine.

    /radish/phamd/duypd-proj/
      COSMIC/
        main.py
        Dockerfile.gpu
        .cache/
        data/OvSGTR/weights/bert-base-uncased/
        outputs/logs/
        outputs/gsv/
      SG-Retrieval/
        Datasets/
          GSV_CITITES/{Dataframes,Images,osg}/
          Pitts30k/{images,metadata,osg}/

Datasets stay in SG-Retrieval/Datasets and are mounted directly into Docker.
Source copying does not create dataset symlinks.

## 2. Build on ginger

Docker Engine, Docker Compose v2 and NVIDIA Container Toolkit must be available
on ginger. The default Dockerfile uses Python 3.10, PyTorch 2.7.1, torchvision
0.22.1 and CUDA 11.8 wheels with NVIDIA driver 535.

On ginger:

    cd /radish/phamd/duypd-proj/COSMIC
    bash scripts/lab.sh build

Docker reads Dockerfile.gpu and the requirement files from radish. Image and
build cache stay in ginger's Docker storage. Source and data are bind-mounted
at runtime. The image is cosmic:gpu-dev; other members' images are not used.

## 3. Run on ginger

Precomputed GSV/Pitts30k OSGs and the BERT/DINO assets must be present on radish.
BERT belongs in data/OvSGTR/weights/bert-base-uncased and the existing DINO cache
in .cache/visual_backbones. These assets are excluded from Git/source copying.

Check once after setup:

    CUDA_VISIBLE_DEVICES=3 bash scripts/lab.sh check

Then train:

    CUDA_VISIBLE_DEVICES=3 bash scripts/lab.sh train --detach

Replace 3 with the GPU for your run. All GPUs are exposed to the container;
CUDA_VISIBLE_DEVICES selects the process's GPU, which appears as cuda:0.
The current training pipeline uses one GPU per run. Training automatically
calls the configured Pitts30k validation.

Follow the container ID printed by train:

    bash scripts/lab.sh logs CONTAINER_ID

Logs are also written to COSMIC/outputs/logs on radish. Log filenames use Japan
time and a unique suffix. The logging pipeline preserves Python's exit status.
Results, training_history.json and checkpoints stay under COSMIC/outputs/gsv.

Standalone validation and completed-epoch resume:

    CUDA_VISIBLE_DEVICES=3 bash scripts/lab.sh valid
    CUDA_VISIBLE_DEVICES=3 bash scripts/lab.sh train --detach --resume outputs/gsv/checkpoints/last.pt

A short trial still performs full validation:

    CUDA_VISIBLE_DEVICES=3 bash scripts/lab.sh train --epochs 1 --max-train-batches 2 --output-dir outputs/lab_smoke

## Code and dependency updates

For code changes, finish the current experiment, copy updated source from
csMachine, then start a new train/valid process. Reuse the built image.

For Dockerfile or dependency changes, copy source, run lab.sh build on ginger,
then start a new job. Recreate an idle development container with lab.sh restart.

For an interactive container:

    bash scripts/lab.sh up
    bash scripts/lab.sh shell

It opens /workspace/app. In that terminal, select CUDA_VISIBLE_DEVICES when
starting Python. VS Code Remote SSH to ginger can also open the shared COSMIC
directory with Dev Containers: Reopen in Container.

## Mounts and settings

.devcontainer/docker-compose.yml handles both build and runtime:

| Host directory on radish | Container path | Access |
| --- | --- | --- |
| COSMIC | /workspace/app | Read/write |
| SG-Retrieval/Datasets | /workspace/app/Datasets | Read-only |
| COSMIC/.cache/huggingface | /workspace/.cache/huggingface | Read/write |
| COSMIC/.cache | /workspace/app/.cache | Read/write |
| COSMIC/outputs | /workspace/app/outputs | Read/write |
| COSMIC/data/OvSGTR/weights | /workspace/app/data/OvSGTR/weights | Read-only |

Additional data mounts preserve Dataset/GSV_CITITES and Pitts30k from the current
training YAML. Defaults follow the Compose file's parent and sibling dataset
directory; .devcontainer/.env is optional for path/offline settings.
Existing .env path settings continue to work. The image tag is fixed to
cosmic:gpu-dev; old COSMIC_IMAGE and COSMIC_TORCH_* overrides are no longer used.
Dependency versions live in Dockerfile.gpu and requirements-docker.txt.

COSMIC currently runs train/validation experiments through main.py. PostgreSQL,
API endpoints and frontend port forwarding belong to the RAIR-VF application
flow and are not part of this training entry point.

## References

- [Docker bind mounts](https://docs.docker.com/engine/storage/bind-mounts/)
- [Compose GPU support](https://docs.docker.com/compose/how-tos/gpu-support/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
