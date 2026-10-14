"""COSMIC entry point: configure, create data loaders, then initialize and run."""

from collections.abc import Mapping
from functools import partial
import math

import torch

from cli import apply_overrides, parse_args
from config import load_config
from data.description import describe_loader
from data.validation import build_validation_inputs
from model.criteria import build_criteria
from model.model import build_model
from registry import build_data_loader, get_dataset
from runners.evaluation import load_evaluation_checkpoint
from runners.prepare import prepare_data
from runtime import prepare_run
from train_engine import TrainEngine, validate_training_request
from valid_engine import ValidEngine


def learning_rate_factor(epoch, *, training):
    """Linear warmup followed by epoch-based cosine decay, configured in YAML."""
    warmup = training.scheduler.warmup_epochs
    if epoch < warmup:
        return (epoch + 1) / warmup
    duration = max(1, training.epochs - warmup - 1)
    progress = min(max((epoch - warmup) / duration, 0), 1)
    minimum = training.scheduler.min_lr / training.optimizer.lr
    return minimum + (1 - minimum) * (1 + math.cos(math.pi * progress)) / 2


def main(argv=None, *, validation_inputs=None):
    """Create data, model, criteria and optimization components before engines.

    validation_inputs contains query_loader, database_loader and positives for
    ValidEngine. Omitted inputs are built from validation.data through the registry.
    """
    args = parse_args(argv)
    config = apply_overrides(load_config(args.config), args)
    adapter = get_dataset(config.data.dataset)

    if validation_inputs is not None:
        if config.mode not in ("train", "evaluate"):
            raise ValueError("validation_inputs requires mode=train or mode=evaluate")
        if not isinstance(validation_inputs, Mapping):
            raise ValueError("validation_inputs must contain validation loaders and ground truth")
    if config.mode == "train":
        validate_training_request(config)
    evaluation_state = load_evaluation_checkpoint(config) if config.mode == "evaluate" else None

    # Build all training/validation data before initializing models or engines.
    train_loader = None
    if config.mode == "train":
        train_loader = build_data_loader(config.data, require_complete_osg=True)
        describe_loader(train_loader, dataset_name=f"train ({adapter.name})")
    if validation_inputs is None and config.mode in ("train", "evaluate"):
        validation_inputs = build_validation_inputs(config)
    if validation_inputs is not None:
        for name in ("query_loader", "database_loader", "pair_loader"):
            loader = validation_inputs.get(name)
            if loader is not None:
                describe_loader(loader, dataset_name=f"validation {name.removesuffix('_loader')}")

    context = prepare_run(config, action={"validation_supplied": validation_inputs is not None})
    print(f"Experiment: {config.name} | mode={config.mode} | dataset={adapter.name} | device={context.device}")
    print(f"Output directory: {context.directory}")

    if config.mode == "prepare":
        prepare_data(config, context, adapter)
        return
    if config.mode == "evaluate":
        model = build_model(config.model, context.device)
        model.load_state_dict(evaluation_state, strict=True)
        return ValidEngine(config, context).run(model, **validation_inputs)

    if config.mode == "train":
        model = build_model(config.model, context.device)
        criteria = build_criteria(config.model.criteria, context.device)

        # Construct optimization components here; the engine only uses them.
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError("Training requires at least one trainable parameter")
        settings = config.training.optimizer
        learning_rate = settings.lr
        optimizer_factory = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam, "sgd": torch.optim.SGD}
        optimizer = optimizer_factory[settings.name](parameters, lr=learning_rate, weight_decay=settings.weight_decay)
        scheduler = None
        if config.training.scheduler.name == "cosine":
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=partial(learning_rate_factor, training=config.training))
        scaler = torch.amp.GradScaler("cuda", enabled=config.training.precision == "amp_fp16")

        validator = ValidEngine(config, context)
        validate = validator.as_callback(**validation_inputs) if validation_inputs is not None else None
        trainer = TrainEngine(model, config, context, criteria=criteria,
                              optimizer=optimizer, scheduler=scheduler, scaler=scaler)
        return trainer.fit(train_loader, validate=validate)


if __name__ == "__main__":
    main()
