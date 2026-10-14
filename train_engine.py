"""Train COSMIC on paired observations; validation is supplied by a separate engine."""

from collections.abc import Mapping
from itertools import islice
import math
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch

from config import config_to_dict
from data.osg_storage import write_json_atomic
from model.criteria import COSMICCriteria
from registry import get_dataset


def validate_training_request(config):
    """Check the training protocol before constructing data/models or writing output."""
    if config.mode != "train":
        raise ValueError("TrainEngine requires mode=train")
    adapter = get_dataset(config.data.dataset)
    if adapter.evaluation_only:
        raise ValueError(f"{adapter.name} is evaluation-only; use GSV for training")
    if not config.data.load_osg:
        raise ValueError("COSMIC training requires data.load_osg=true and exported source OSGs")
    if config.save.resume is not None and not Path(config.save.resume).is_file():
        raise FileNotFoundError(f"Training checkpoint not found: {config.save.resume}")


class TrainEngine:
    """Run optimization/checkpoints with components supplied by the caller.

    fit(loader, validate=None) optionally accepts validate(model, epoch), returning
    a mapping of validation metrics. Epoch numbers exposed to callers are 1-based.
    No validation dataset is chosen here. Without a callback, best selection is
    disabled. Checkpoints resume at completed epoch boundaries, not mid-epoch.
    main creates the model on the target device, criteria, optimizer, scheduler
    and scaler. This engine does not construct those components.
    """

    def __init__(self, model, config, context, *, criteria: COSMICCriteria,
                 optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler,
                 scheduler=None):
        validate_training_request(config)
        self.config, self.context = config, context
        self.device = torch.device(context.device)
        precision = config.training.precision
        if precision == "amp_fp16" and self.device.type != "cuda":
            raise ValueError("amp_fp16 training requires CUDA; use fp32 or amp_bf16 on CPU")
        if precision == "amp_bf16" and self.device.type == "cuda":
            with torch.cuda.device(self.device):
                if not torch.cuda.is_bf16_supported():
                    raise ValueError("This CUDA device does not support amp_bf16")
        self.model = model
        if not isinstance(criteria, COSMICCriteria):
            raise TypeError("TrainEngine criteria must be a COSMICCriteria instance")
        if criteria.config != config.model.criteria:
            raise ValueError("Criteria settings must match config.model.criteria")
        self.criteria = criteria
        self.parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not self.parameters:
            raise ValueError("Training requires at least one trainable parameter")
        if any(parameter.device.type != self.device.type or
               (self.device.index is not None and parameter.device.index != self.device.index)
               for parameter in model.parameters()):
            raise ValueError("Move the model to the training device before creating its optimizer")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("TrainEngine requires a caller-created torch optimizer")
        optimized = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        if (len(optimized) != len(self.parameters)
                or {id(parameter) for parameter in optimized} != {id(parameter) for parameter in self.parameters}):
            raise ValueError("Optimizer must contain exactly the model's trainable parameters")
        if (scheduler is None) != (config.training.scheduler.name == "none"):
            raise ValueError("Supplied scheduler must match config.training.scheduler")
        if scheduler is not None and scheduler.optimizer is not optimizer:
            raise ValueError("Scheduler must use the supplied optimizer")
        if not isinstance(scaler, torch.amp.GradScaler) or scaler.is_enabled() != (precision == "amp_fp16"):
            raise ValueError("Supplied GradScaler must match config.training.precision")
        self.optimizer, self.scheduler, self.scaler = optimizer, scheduler, scaler
        self.autocast_dtype = torch.float16 if precision == "amp_fp16" else torch.bfloat16
        self.start_epoch, self.global_step = 0, 0
        self.best_metric, self.history = None, []

    def _batch_losses(self, batch):
        if not isinstance(batch, Mapping) or not all(side in batch for side in ("a", "b", "label")):
            raise ValueError("Training expects an A/B loader batch with pair labels")
        a, b = self.model(batch["a"]), self.model(batch["b"])
        logits = None
        if self.config.model.criteria.verification_weight > 0:
            alignment = self.model.align(a, b, compute_hard=True)
            verification = self.model.verify(a, b, alignment)
            logits = self.model.score(verification)["scene_logits"]
        return self.criteria(a, b, batch, scene_logits=logits)

    def _optimizer_step(self, pair_count):
        # Normalize by actual pairs in this window, including a short final window.
        self.scaler.unscale_(self.optimizer)
        for parameter in self.parameters:
            if parameter.grad is not None:
                parameter.grad.div_(pair_count)
        norm = torch.nn.utils.clip_grad_norm_(
            self.parameters, self.config.training.grad_clip_norm or math.inf,
            error_if_nonfinite=not self.scaler.is_enabled())
        old_scale = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        skipped = self.scaler.is_enabled() and self.scaler.get_scale() < old_scale
        if not skipped:
            self.global_step += 1
        self.optimizer.zero_grad(set_to_none=True)
        return float(norm), int(skipped)

    def train_one_epoch(self, loader, epoch):
        """Return pair-weighted loss means and counts, without retaining batch graphs."""
        if type(epoch) is not int or epoch <= 0:
            raise ValueError("Training epoch must be a positive integer (1-based)")
        length = len(loader)
        if length <= 0:
            raise ValueError("Training loader has no batches")
        limit = self.config.training.max_batches_per_epoch
        batches = min(length, limit) if limit is not None else length
        for sampler in (getattr(loader, "batch_sampler", None), getattr(loader, "sampler", None)):
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch - 1)
        generator = getattr(loader, "generator", None)
        if generator is not None:
            generator.manual_seed(self.config.seed + epoch - 1)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss_keys = ("loss", "retrieval_loss", "verification_loss",
                     "weighted_retrieval_loss", "weighted_verification_loss")
        totals = {key: 0.0 for key in loss_keys}
        counts = {key: 0 for key in ("valid_retrieval_images", "retrieval_anchors", "verification_pairs")}
        pairs = positives = window_pairs = skipped_steps = updates = processed = unsupervised_batches = 0
        last_grad_norm = None
        started = time.perf_counter()
        for step, batch in enumerate(islice(loader, batches), 1):
            with torch.autocast(self.device.type, dtype=self.autocast_dtype,
                                enabled=self.config.training.precision != "fp32"):
                losses = self._batch_losses(batch)
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}, batch {step}; training stopped")
            count = len(batch["label"])
            if losses["retrieval_anchors"] or losses["verification_pairs"]:
                if not losses["loss"].requires_grad:
                    raise ValueError("The enabled losses have no path to trainable model parameters")
                self.scaler.scale(losses["loss"] * count).backward()
                window_pairs += count
            else:
                # Retrieval-only batches can have no usable graphs/positive anchors.
                unsupervised_batches += 1
            pairs += count
            positives += int(batch["label"].sum())
            for key in loss_keys:
                totals[key] += float(losses[key].detach()) * count
            for key in counts:
                counts[key] += losses[key]
            if step % self.config.training.accumulation_steps == 0 or step == batches:
                if window_pairs:
                    last_grad_norm, skipped = self._optimizer_step(window_pairs)
                    skipped_steps += skipped
                    updates += 1 - skipped
                window_pairs = 0
            processed = step
            if step == 1 or step % self.config.training.log_every == 0 or step == batches:
                print(f"Train epoch {epoch}/{self.config.training.epochs} | batch {step}/{batches} | "
                      f"loss={totals['loss']/pairs:.5f} | retrieval={totals['retrieval_loss']/pairs:.5f} | "
                      f"verification={totals['verification_loss']/pairs:.5f} | "
                      f"lr={self.optimizer.param_groups[0]['lr']:.3g} | "
                      f"no-supervision batches={unsupervised_batches}", flush=True)
            # Avoid retaining the preceding batch's attention/verification graph.
            del losses, batch
        if processed != batches:
            self.optimizer.zero_grad(set_to_none=True)
            raise ValueError("Training loader yielded fewer batches than len(loader)")
        return {**{key: value / pairs for key, value in totals.items()}, **counts,
                "batches": processed, "pairs": pairs, "positive_pairs": positives,
                "negative_pairs": pairs - positives, "optimizer_steps": updates,
                "unsupervised_batches": unsupervised_batches,
                "skipped_optimizer_steps": skipped_steps,
                "last_grad_norm": last_grad_norm if last_grad_norm is None or math.isfinite(last_grad_norm) else None,
                "lr": self.optimizer.param_groups[0]["lr"], "seconds": time.perf_counter() - started}

    def fit(self, loader, *, validate=None):
        """Train configured epochs; optionally select best using independent validation."""
        if len(loader) <= 0:
            raise ValueError("Training loader has no batches")
        if validate is not None and not callable(validate):
            raise ValueError("validate must be callable: validate(model, epoch) -> metrics")
        if self.config.save.resume is not None:
            self._resume(loader)
        if self.start_epoch >= self.config.training.epochs:
            print(f"Checkpoint already completed {self.start_epoch} epochs; no training remains.", flush=True)
            return self.history
        directory = Path(self.context.directory)
        if (self.config.save.enabled and self.config.save.resume is None
                and (Path(self.context.checkpoint_dir) / "last.pt").exists()):
            raise FileExistsError("This output already has last.pt; use save.resume/--resume "
                                  "or choose another output directory")
        directory.mkdir(parents=True, exist_ok=True)
        if validate is None:
            print("No validation callback: training losses are logged; best-checkpoint selection is disabled.", flush=True)
        print(f"Trainable parameters: {sum(p.numel() for p in self.parameters):,} | "
              f"precision={self.config.training.precision} | "
              f"accumulation={self.config.training.accumulation_steps}", flush=True)
        for completed in range(self.start_epoch, self.config.training.epochs):
            epoch = completed + 1
            metrics = self.train_one_epoch(loader, epoch)
            validation = None
            improved = False
            if validate is not None and (epoch % self.config.training.validate_every == 0
                                        or epoch == self.config.training.epochs):
                self.model.eval()
                with torch.no_grad():
                    result = validate(self.model, epoch)
                if not isinstance(result, Mapping):
                    raise ValueError("Validation must return a mapping of scalar metrics")
                validation = {key: float(value) for key, value in result.items()}
                if any(not math.isfinite(value) for value in validation.values()):
                    raise ValueError("Validation metrics must be finite")
                if self.config.save.enabled and self.config.save.save_best:
                    monitor = self.config.save.monitor
                    if monitor not in validation:
                        raise ValueError(f"Validation did not return checkpoint metric {monitor!r}")
                    value = validation[monitor]
                    improved = self.best_metric is None or (
                        value < self.best_metric if self.config.save.mode == "min" else value > self.best_metric)
                    if improved:
                        self.best_metric = value
            if self.scheduler is not None and metrics["optimizer_steps"] > 0:
                self.scheduler.step()
            self.history.append({"epoch": epoch, "global_step": self.global_step,
                                 "train": metrics, "validation": validation})
            write_json_atomic(directory / "training_history.json", {"epochs": self.history})
            if self.config.save.enabled:
                self._save(epoch, loader, improved)
        return self.history

    def _rng_state(self, loader):
        state = np.random.get_state()
        generator = getattr(loader, "generator", None)
        return {"python": random.getstate(), "numpy": (state[0], state[1].tolist(), *state[2:]),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None,
                "loader": generator.get_state() if generator is not None else None}

    def _save(self, epoch, loader, improved):
        directory = Path(self.context.checkpoint_dir).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        payload = {"format_version": 1, "epoch": epoch, "global_step": self.global_step,
                   "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                   "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                   "scaler": self.scaler.state_dict(), "best_metric": self.best_metric,
                   "history": self.history, "config": config_to_dict(self.config),
                   "rng": self._rng_state(loader)}
        self._atomic_checkpoint(payload, directory / "last.pt")
        if epoch % self.config.save.save_every == 0 or epoch == self.config.training.epochs:
            self._atomic_checkpoint(payload, directory / f"epoch_{epoch:08d}.pt")
            archives = sorted(path for path in directory.glob("epoch_????????.pt")
                              if path.stem[6:].isdigit() and path.is_file())
            for path in archives[:-self.config.save.keep_last]:
                if path.resolve().parent != directory:
                    raise ValueError(f"Checkpoint retention path escapes the checkpoint directory: {path}")
                path.unlink()
        if improved:
            self._atomic_checkpoint(payload, directory / "best.pt")
        print(f"Checkpoint: {directory / 'last.pt'} (epoch {epoch})", flush=True)

    @staticmethod
    def _atomic_checkpoint(payload, path):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _resume(self, loader):
        payload = torch.load(self.config.save.resume, map_location="cpu", weights_only=True)
        required = {"format_version", "epoch", "global_step", "model", "optimizer", "scheduler",
                    "scaler", "best_metric", "history", "config", "rng"}
        if not isinstance(payload, dict) or not required.issubset(payload) or payload["format_version"] != 1:
            raise ValueError("Resume requires a full TrainEngine checkpoint, not encoder-only weights")
        current, previous = config_to_dict(self.config), payload["config"]
        for section in ("model", "data", "seed"):
            if previous[section] != current[section]:
                raise ValueError(f"Resume requires matching {section} settings")
        old_training, new_training = dict(previous["training"]), dict(current["training"])
        for key in ("log_every", "validate_every"):
            old_training.pop(key, None)
            new_training.pop(key, None)
        if self.scheduler is None:
            old_training.pop("epochs", None)
            new_training.pop("epochs", None)
        if old_training != new_training:
            raise ValueError("Resume requires matching optimization settings and the cosine epoch horizon")
        for key in ("monitor", "mode"):
            if previous["save"][key] != current["save"][key]:
                raise ValueError(f"Resume requires matching save.{key}")
        self.model.load_state_dict(payload["model"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(payload["scheduler"])
        self.scaler.load_state_dict(payload["scaler"])
        self.start_epoch, self.global_step = payload["epoch"], payload["global_step"]
        self.best_metric, self.history = payload["best_metric"], payload["history"]
        rng = payload["rng"]
        random.setstate(rng["python"])
        state = rng["numpy"]
        np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))
        torch.set_rng_state(rng["torch"])
        if self.device.type == "cuda" and rng["cuda"] is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])
        generator = getattr(loader, "generator", None)
        if generator is not None and rng["loader"] is not None:
            generator.set_state(rng["loader"])
        print(f"Resumed {self.config.save.resume}: completed epoch {self.start_epoch}, "
              f"optimizer step {self.global_step}", flush=True)
