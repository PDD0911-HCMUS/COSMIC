# COSMIC: source on radish, training on ginger

| Location | Role |
| --- | --- |
| Windows | Edit code, commit and push |
| csMachine: ~/ThisPC-DuyPC/COSMIC | Git working copy; fetch/pull and build the environment image |
| /radish/phamd/duypd-proj/COSMIC | Shared source, pretrained assets, image archive and outputs |
| /radish/phamd/duypd-proj/SG-Retrieval/Datasets | Existing datasets, mounted directly into Docker |
| ginger | Load the image; select the GPU when launching training or validation |

Ginger already mounts the SMB radish share. The container reads source directly
from that mount. Git operations happen on csMachine. The environment image
contains Python/dependencies; source and datasets are runtime bind mounts.

The default environment is Python 3.10, PyTorch 2.7.1, torchvision 0.22.1 and
CUDA 11.8 wheels. This preserves the tested PyTorch API and uses backward
compatibility with ginger's NVIDIA driver 535.230.02. The driver-reported
"CUDA Version: 12.2" describes driver support; no host CUDA toolkit installation
is needed for this training image.

Ginger requires Docker Engine, Docker Compose v2 and NVIDIA Container Toolkit.
The runtime check verifies both CUDA discovery and a CUDA matrix multiplication.
The image is for training/validation with existing OSGs; the Windows
GroundingDINO extension is not part of the image.

## 1. Copy source on csMachine

After pushing the Windows changes, on csMachine:

    cd ~/ThisPC-DuyPC/COSMIC
    git pull --ff-only
    bash scripts/sync_radish.sh

The source is the project containing scripts/sync_radish.sh, normally
~/ThisPC-DuyPC/COSMIC. The destination defaults to
/radish/phamd/duypd-proj/COSMIC, alongside SG-Retrieval.
The script checks required source files, creates the destination, and uses cp
to overwrite source files with the current working copy. It checks the required
files again after copying. An optional first argument overrides the destination.

Datasets, weights, caches, outputs, .git and private environment files are
excluded. The small OvSGTR assets_manifest.json is copied with the source.
Existing runtime files and obsolete source files remain at the destination.
Copying uses regular files and does not follow source symlinks or preserve
timestamps/ownership. The script does not require rsync, Git, or a dataset link.

Synchronize before starting a new run. Keep the source unchanged for the duration
of a running experiment; fetch newer code into the csMachine working copy while
the current shared source is still in use.

Expected shared layout:

    /radish/phamd/duypd-proj/
      COSMIC/
        main.py
        config.yaml
        .devcontainer/.env
        data/OvSGTR/weights/bert-base-uncased/
        .cache/visual_backbones/
        outputs/gsv/
        outputs/docker/cosmic-training-image.tar
      SG-Retrieval/
        Datasets/
          GSV_CITITES/{Dataframes,Images,osg}/
          Pitts30k/{images,metadata,osg}/

Data remains in SG-Retrieval/Datasets. Docker reads that directory directly,
so source copying and training do not depend on symlink support on the SMB mount.
On ginger, verify the source and data paths before starting Docker:

    cd /radish/phamd/duypd-proj/COSMIC
    ls main.py config.yaml
    ls ../SG-Retrieval/Datasets/GSV_CITITES ../SG-Retrieval/Datasets/Pitts30k

## 2. Set the local paths

On csMachine, in the shared source directory:

    cd /radish/phamd/duypd-proj/COSMIC
    cp -n .devcontainer/devcontainer.env.example .devcontainer/.env
    nano .devcontainer/.env

The container exposes all GPUs on ginger through count: all and
NVIDIA_VISIBLE_DEVICES=all. GPU selection happens per command using
CUDA_VISIBLE_DEVICES, rather than through the environment file.
COSMIC_GPU_ID from an older .env is no longer used.

For example, CUDA_VISIBLE_DEVICES=3 selects GPU 3 for that process; PyTorch sees
it as cuda:0, so main.py continues to use --device cuda. A GPU UUID from
nvidia-smi -L is also accepted. CUDA_DEVICE_ORDER=PCI_BUS_ID makes numeric
selection follow PCI bus ordering.
See [NVIDIA's CUDA environment variables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html).

scripts/lab.sh forwards a CUDA_VISIBLE_DEVICES value set on the command line
into the check/train/valid container. With the variable unset, all GPUs remain
visible to CUDA. An explicitly empty value hides all GPUs.
The current training pipeline uses one GPU per run; making multiple GPUs
visible does not distribute training across them.

Set COSMIC_SOURCE_PATH=/radish/phamd/duypd-proj/COSMIC and
COSMIC_DATASETS_PATH=/radish/phamd/duypd-proj/SG-Retrieval/Datasets, including
in an existing .devcontainer/.env. When COSMIC_DATASETS_PATH is omitted, Docker
defaults to COSMIC_SOURCE_PATH/../SG-Retrieval/Datasets.
Optional cache/output/weight path overrides also refer to paths accessible on
ginger. Dataset mounts
are read-only; source, cache and output mounts are writable. Missing source/data/
weight directories cause startup to fail rather than creating empty datasets.

Inside Docker, source is mounted at /workspace/app and data at
/workspace/app/Datasets. Dataset mounts are read-only.

config.yaml retains Dataset/GSV_CITITES, Pitts30k and outputs/gsv. Additional bind
mounts provide those paths, so no Windows/server-specific config duplicate is
required. Existing absolute roots must instead be changed to these container
paths or explicitly mounted into the container.

## 3. Transfer pretrained encoders once

These assets are excluded from Git. Copy the existing BERT directory and DINO
cache from Windows. In Windows PowerShell, using the existing SSH alias:

    $labSource = "/radish/phamd/duypd-proj/COSMIC"
    ssh csMachine "mkdir -p '$labSource/data/OvSGTR/weights' '$labSource/.cache'"
    scp -r "F:\COSMIC\data\OvSGTR\weights\bert-base-uncased" "csMachine:${labSource}/data/OvSGTR/weights/"
    scp -r "F:\COSMIC\.cache\visual_backbones" "csMachine:${labSource}/.cache/"

The default COSMIC_OFFLINE=1 makes the encoders use these local assets.
The full OvSGTR checkpoint and compiled extension are not required to train
from precomputed graphs. The runtime check loads the configured encoders on CPU
to verify both their assets and dependencies.

## 4. Build/export on csMachine

On csMachine, from the shared source directory:

    bash scripts/lab.sh build
    bash scripts/lab.sh export-image

Build uses docker-compose.build.yml. .dockerignore allows only the Dockerfile
and requirement files into the build context. No GPU is needed during build.
The archive is saved under outputs/docker on radish, so ginger can read it
without another network copy or registry.

Export writes Docker's stdout stream to cosmic-training-image.tar.part, then
renames the completed, nonempty file to cosmic-training-image.tar. This avoids
Docker's --output permission changes on CIFS. The script shows the file size
and prints Image saved only after checking that the final archive exists.
Failed exports leave a .part file for inspection.

After Image saved, verify the archive on ginger before loading it:

    ls -lh outputs/docker/cosmic-training-image.tar

If ginger cannot see it, compare the share mounted at the archive directory on
both machines:

    findmnt -T /radish/phamd/duypd-proj/COSMIC/outputs/docker -o TARGET,SOURCE,FSTYPE

The scripts explicitly use the local default Docker context. They do not change
your global context. COSMIC_DOCKER_CONTEXT can select an existing context when
deliberately controlling another daemon.

## 5. Load/check/run on ginger

On ginger:

    cd /radish/phamd/duypd-proj/COSMIC
    bash scripts/lab.sh load-image
    CUDA_VISIBLE_DEVICES=3 bash scripts/lab.sh check
    CUDA_VISIBLE_DEVICES=3 bash scripts/lab.sh train --detach

These commands use GPU 3 as an example; set the index or UUID for your run.

Runtime uses docker-compose.yml with pull_policy: never and no build definition.
The Docker image and container storage are on ginger. Source/data/weights/cache/
outputs stay on radish. Checkpoint saving therefore persists after a container
exits or is removed.

The train command calls main.py, which builds GSV and Pitts30k loaders before
the model and optimizer. It runs GSV training and the configured periodic
Pitts30k validation in the same GPU container. main.py performs the full source
graph inventory checks; check_lab_runtime.py checks manifests and encoder assets.

Detached training prints a container ID and continues after the SSH session
closes. Follow its logs:

    bash scripts/lab.sh logs CONTAINER_ID

After it exits, inspect its exit code with docker inspect:

    docker --context default inspect --format '{{.State.Status}} exit={{.State.ExitCode}}' CONTAINER_ID

A short foreground run uses the same pipeline, including final full validation:

    CUDA_VISIBLE_DEVICES=3 bash scripts/lab.sh train --epochs 1 --max-train-batches 2 --output-dir outputs/lab_smoke

Resume a completed-epoch checkpoint using the original training settings:

    CUDA_VISIBLE_DEVICES=3 bash scripts/lab.sh train --detach --resume outputs/gsv/checkpoints/last.pt

Resume starts after the last completed epoch. Interrupted partial epochs are
rerun. Checkpoint/config compatibility rules remain enforced by TrainEngine.

Standalone validation of best.pt:

    CUDA_VISIBLE_DEVICES=3 bash scripts/lab.sh valid

This calls main.py --mode evaluate with configs/pitts30k.yaml and the checkpoint.
Keep output.root under /workspace/app/outputs to use the persistent output mount.
Existing .env overrides can select another shared output directory.

## Optional VS Code terminal

Connect to ginger with Remote - SSH, open the shared COSMIC directory, then
choose Dev Containers: Reopen in Container. .devcontainer/.env provides the
Compose variables, and the image must already have been loaded on ginger.

The development container also exposes all GPUs. In its terminal, run:

    nvidia-smi
    CUDA_VISIBLE_DEVICES=3 python -u -B main.py --config config.yaml --mode train --device cuda

This selects the GPU for that Python process while keeping the container's
device access available for subsequent commands.

The container has a Python terminal and no HTTP server. No application port
is required for train/validation. shutdownAction: none keeps the development
container alive when VS Code closes. For long training runs, use the detached
train command above.

## References

- [PyTorch 2.7.1 wheels](https://pytorch.org/get-started/previous-versions/#v271)
- [NVIDIA CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [Compose GPU devices](https://docs.docker.com/compose/how-tos/gpu-support/)
