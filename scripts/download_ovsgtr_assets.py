"""Download the author's ECCV OvD+R Swin-T checkpoint and BERT assets."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "data" / "OvSGTR" / "weights"
os.environ["HF_HOME"] = str(ROOT / ".cache" / "huggingface")
os.environ["HF_HUB_CACHE"] = str(ROOT / ".cache" / "huggingface" / "hub")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import HfApi, hf_hub_download


def main():
    api = HfApi()
    specs = {
        "JosephZ/OvSGTR": (["vg-ovdr-swint.pth"], ASSETS),
        "google-bert/bert-base-uncased": (
            ["config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json", "vocab.txt"],
            ASSETS / "bert-base-uncased",
        ),
    }
    manifest = {}
    manifest_path = ASSETS / "assets_manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for repo_id, (filenames, destination) in specs.items():
        revision = previous.get(repo_id, {}).get("revision") or api.model_info(repo_id).sha
        manifest[repo_id] = {"revision": revision, "files": filenames}
        for filename in filenames:
            if (destination / filename).is_file():
                print(f"Using local asset: {destination / filename}", flush=True)
                continue
            print(f"Downloading {repo_id}@{revision}: {filename}", flush=True)
            hf_hub_download(repo_id, filename, revision=revision, local_dir=destination)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Assets ready: {ASSETS}", flush=True)


if __name__ == "__main__":
    main()
