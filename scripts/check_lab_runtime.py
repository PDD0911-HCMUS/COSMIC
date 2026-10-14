"""Check the shared mounts, pretrained encoders and GPU before a lab run."""

import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from config import load_config
from model.model import build_model


def check_manifest(root, osg_root, name):
    path = (osg_root or root / "osg") / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name} OSG manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("complete_dataset") is not True
            or not manifest.get("total_images")
            or manifest.get("completed_images") != manifest["total_images"]):
        raise ValueError(f"Incomplete {name} OSG export: {path}")
    print(f"{name}: {root} | {manifest['completed_images']:,} exported OSGs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--cpu", action="store_true", help="Check assets on a host without using a GPU.")
    args = parser.parse_args()
    path = Path(args.config)
    config = load_config(path if path.is_absolute() else ROOT / path)

    print(f"Source: {ROOT}")
    revision = ROOT / ".cosmic-revision"
    if revision.is_file():
        print(f"Source revision: {revision.read_text(encoding='utf-8').strip()}")
    print(f"PyTorch: {torch.__version__} | CUDA runtime: {torch.version.cuda}")
    if not args.cpu:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Check ginger's NVIDIA Container Toolkit and GPU selection.")
        visible_count = torch.cuda.device_count()
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '(unset: all GPUs)')}")
        print(f"Visible CUDA devices: {visible_count}")
        for index in range(visible_count):
            print(f"  cuda:{index}: {torch.cuda.get_device_name(index)}")
        probe = torch.eye(8, device="cuda")
        if not torch.equal(probe @ probe, probe):
            raise RuntimeError("CUDA matrix multiplication check failed.")
        torch.cuda.synchronize()
        print(f"CUDA kernel OK on cuda:{torch.cuda.current_device()}")

    if config.data.dataset != "gsv" or config.validation.data is None or config.validation.data.dataset != "pitts30k":
        raise ValueError("This lab check expects GSV training and Pitts30k validation.")
    required = [
        config.data.root / "Dataframes",
        config.data.root / "Images",
        config.validation.data.root / "images/val",
        config.validation.data.root / "metadata/pitts30k_val.mat",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset path: {path}. Check the radish bind mounts.")
    check_manifest(config.data.root, config.data.osg_root, "GSV")
    check_manifest(config.validation.data.root, config.validation.data.osg_root, "Pitts30k-val")

    # Check the actual configured encoders without allocating the full model on GPU.
    # Offline Compose settings make missing assets fail rather than downloading.
    model = build_model(config.model, device="cpu")
    print(f"Local encoders loaded | trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    config.output.root.mkdir(parents=True, exist_ok=True)
    with TemporaryFile(dir=config.output.root) as stream:
        stream.write(b"COSMIC output write check")
        stream.flush()
    print(f"Writable output: {config.output.root}")
    print("Lab checks passed. main.py additionally checks all required source OSG files before training.")


if __name__ == "__main__":
    main()
