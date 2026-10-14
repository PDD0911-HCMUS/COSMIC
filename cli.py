"""Command-line options for data preparation, training and evaluation."""

import argparse
from dataclasses import replace
from pathlib import Path

from config import DEFAULT_CONFIG_PATH
from data.config import PROJECT_ROOT
from registry import get_dataset


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="COSMIC: prepare source OSGs, train, or evaluate.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--mode", choices=("prepare", "train", "evaluate"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--osg-root", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--images-per-place", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int, help="Total epochs, including completed epochs on resume.")
    parser.add_argument("--max-train-batches", type=int, help="Limit batches per epoch for a short run.")
    parser.add_argument("--resume", type=Path, help="Resume training or evaluate a completed checkpoint.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prepare-data", action="store_true", help="Export source OSGs; alias for --mode prepare.")
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--overwrite-osg", action="store_true", default=None)
    parser.add_argument("--save-object-threshold", type=float)
    parser.add_argument("--save-relation-threshold", type=float)
    parser.add_argument("--graph-format", choices=("json", "json_gzip"))
    return parser.parse_args(argv)


def apply_overrides(config, args):
    mode = args.mode or config.mode
    if args.prepare_data:
        if args.mode is not None and args.mode != "prepare":
            raise ValueError("--prepare-data conflicts with --mode")
        mode = "prepare"
    data = {key: value for key, value in (
        ("root", args.data_root), ("osg_root", args.osg_root), ("batch_size", args.batch_size),
        ("num_workers", args.num_workers), ("images_per_place", args.images_per_place),
    ) if value is not None}
    if mode == "train":
        data["load_osg"] = True
    output = {}
    if args.output_dir is not None:
        output["root"] = args.output_dir.expanduser().resolve()
    exporter = {key: value for key, value in (
        ("overwrite", args.overwrite_osg), ("object_threshold", args.save_object_threshold),
        ("relation_threshold", args.save_relation_threshold), ("graph_format", args.graph_format),
    ) if value is not None}
    adapter = get_dataset(config.data.dataset)
    if args.max_images is not None and args.max_pairs is not None:
        raise ValueError("Use only one export limit")
    for name in ("max_images", "max_pairs"):
        value = getattr(args, name)
        if value is not None:
            if name != adapter.export_limit:
                raise ValueError(f"Use --{adapter.export_limit.replace('_', '-')} for {adapter.name}")
            exporter["max_items"] = value
    if exporter and mode != "prepare":
        raise ValueError("Export options require --prepare-data or --mode prepare")
    training = {key: value for key, value in (
        ("epochs", args.epochs), ("max_batches_per_epoch", args.max_train_batches),
    ) if value is not None}
    if training and mode != "train":
        raise ValueError("--epochs and --max-train-batches require mode=train")
    save = {}
    if args.resume is not None:
        if mode not in ("train", "evaluate"):
            raise ValueError("--resume requires mode=train or mode=evaluate")
        path = args.resume.expanduser()
        save["resume"] = path if path.is_absolute() else PROJECT_ROOT / path
    experiment = {key: value for key, value in (
        ("mode", mode), ("seed", args.seed), ("device", args.device),
    ) if value is not None}
    return replace(config, data=replace(config.data, **data), output=replace(config.output, **output),
                   training=replace(config.training, **training), save=replace(config.save, **save),
                   export=replace(config.export, **exporter), **experiment)
