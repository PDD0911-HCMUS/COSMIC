"""Evaluate exactly one method in an isolated Python process."""

import argparse
from pathlib import Path
import random

import numpy as np
import torch

from comparison.adapters.base import provenance
from comparison.config import load_comparison_config
from comparison.datasets import build_observation_loader, read_benchmark
from comparison.evaluator import evaluate
from comparison.registry import build_adapter
from data.osg_storage import write_json_atomic
from runtime import resolve_device


def run_method(config, name, benchmark=None):
    spec = config.methods[name]
    device = resolve_device(config.device)
    torch.set_num_threads(config.cpu_threads)
    random.seed(config.seed)
    np.random.seed(config.seed % 2**32)
    torch.manual_seed(config.seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.reset_peak_memory_stats(device)
    torch.hub.set_dir(str(config.hub_dir))
    benchmark = benchmark or read_benchmark(config)
    adapter = build_adapter(spec, device)
    batch_size = spec.batch_size or (16 if spec.adapter == "cricavpr" else config.batch_size)
    loader = build_observation_loader(config, benchmark, adapter, batch_size)
    stages = evaluate(adapter, loader, benchmark, config)
    return {"method": name, "adapter": spec.adapter, "status": "ok", "protocol": benchmark.protocol,
            "metadata": provenance(adapter), "batch_size": batch_size, "device": device,
            "seed": config.seed, "torch_version": str(torch.__version__), "stages": stages,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.startswith("cuda") else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args(argv)
    config = load_comparison_config(args.config)
    result = run_method(config, args.method)
    predictions = {}
    for stage, values in result["stages"].items():
        predictions[stage] = values.pop("predictions")
    for stage, values in predictions.items():
        path = Path(args.result).with_name(f"{args.method}_{stage}_predictions.json")
        write_json_atomic(path, values)
        result["stages"][stage]["predictions_file"] = str(path)
    write_json_atomic(Path(args.result), result)


if __name__ == "__main__":
    main()
