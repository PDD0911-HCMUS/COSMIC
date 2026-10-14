"""Dataset-independent COSMIC validation and aggregate JSON/CSV table reports."""

from collections.abc import Mapping
import csv
import math
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
from torch.nn import functional as F

from config import config_to_dict
from data.osg_storage import write_json_atomic


def compute_pair_metrics(logits, labels, threshold=0.5):
    """Binary metrics using sigmoid >= threshold and exact tie-aware logit AUROC.

    Undefined metrics return None, which is reported as a dash and omitted from
    TrainEngine's scalar callback. BCE includes every pair, including no matches.
    """
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Pair threshold must be finite and in [0,1]")
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 1 or not len(logits)
            or not logits.is_floating_point() or not torch.isfinite(logits).all()):
        raise ValueError("Pair metrics require nonempty finite logits [N]")
    if (not isinstance(labels, torch.Tensor) or labels.shape != logits.shape
            or not torch.isfinite(labels).all() or not ((labels == 0) | (labels == 1)).all()):
        raise ValueError("Pair labels must be binary [N]")
    logits, target = logits.detach().cpu().double(), labels.detach().cpu().bool()
    prediction = logits.sigmoid() >= threshold
    tp, fp = int((prediction & target).sum()), int((prediction & ~target).sum())
    fn, tn = int((~prediction & target).sum()), int((~prediction & ~target).sum())
    positives, negatives = tp + fn, tn + fp
    auc = None
    if positives and negatives:
        order = torch.argsort(logits, stable=True)
        _, groups, counts = torch.unique_consecutive(logits[order], return_inverse=True, return_counts=True)
        group_positive = torch.zeros(len(counts), dtype=torch.float64)
        group_positive.scatter_add_(0, groups, target[order].double())
        group_negative = counts.double() - group_positive
        earlier_negative = group_negative.cumsum(0) - group_negative
        auc = float((group_positive * (earlier_negative + .5 * group_negative)).sum() / (positives * negatives))
    return {"pair/pairs": len(target), "pair/positive_pairs": positives, "pair/negative_pairs": negatives,
            "pair/tp": tp, "pair/fp": fp, "pair/fn": fn, "pair/tn": tn,
            "pair/precision": tp / (tp + fp) if tp + fp else None,
            "pair/recall": tp / positives if positives else None,
            "pair/f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "pair/auroc": auc, "pair/threshold": float(threshold),
            "pair/bce": float(F.binary_cross_entropy_with_logits(logits, target.double()))}


def _id_sets(mapping, name):
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{name} must map query image_id to database image IDs")
    result = {}
    for key, values in mapping.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} query IDs must be nonempty strings")
        if not isinstance(values, (list, tuple, set, frozenset)):
            raise ValueError(f"{name}[{key!r}] must be a sequence/set of database image IDs")
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"{name} database IDs must be nonempty strings")
        result[key] = set(values)
    return result


def _validate_loaders(*, query_loader=None, database_loader=None, positives=None,
                      ignore=None, pair_loader=None):
    retrieval = any(value is not None for value in (query_loader, database_loader, positives, ignore))
    if retrieval and any(value is None for value in (query_loader, database_loader, positives)):
        raise ValueError("Retrieval requires query_loader, database_loader and positives together")
    if not retrieval and pair_loader is None:
        raise ValueError("Supply retrieval loaders/ground truth or pair_loader")
    return retrieval


class ValidEngine:
    """Evaluate explicitly supplied loaders; no dataset or split is selected here.

    Retrieval loaders yield single observations with image/osg/image_id. Positive
    and optional ignore mappings use image IDs, independent of loader order.
    Pair loaders yield existing a/b/label batches. run() returns flat scalar
    metrics compatible with TrainEngine; as_callback() binds reusable loaders.
    """

    def __init__(self, config, context):
        self.settings = config.validation
        self.directory = Path(context.directory) / self.settings.directory
        self.device = torch.device(context.device)

    def _progress(self, name, step, loader):
        total = len(loader) if hasattr(loader, "__len__") else None
        if step == 1 or step % self.settings.log_every == 0 or step == total:
            print(f"Valid {name}: batch {step}" + (f"/{total}" if total is not None else ""), flush=True)

    def _encode(self, model, loader, name):
        ids, seen, descriptors, masks = [], set(), [], []
        for step, observation in enumerate(loader, 1):
            if not isinstance(observation, Mapping) or "image_id" not in observation:
                raise ValueError(f"{name} loader must yield one observation with image_id, image and osg")
            batch_ids = observation["image_id"]
            if (not isinstance(batch_ids, (list, tuple)) or not batch_ids
                    or any(not isinstance(value, str) or not value for value in batch_ids)):
                raise ValueError(f"{name} image_id must be a nonempty list of strings")
            if len(set(batch_ids)) != len(batch_ids) or seen.intersection(batch_ids):
                raise ValueError(f"Duplicate image_id in {name} loader")
            grounding = model(observation)["grounding"]
            descriptor = grounding["fused_descriptor"]
            if (not isinstance(descriptor, torch.Tensor) or descriptor.ndim != 2
                    or descriptor.shape[0] != len(batch_ids) or descriptor.shape[1] <= 0
                    or not descriptor.is_floating_point() or not torch.isfinite(descriptor).all()):
                raise ValueError(f"{name} requires finite fused_descriptor [B,D] matching image IDs")
            descriptor = descriptor.detach().cpu().float()
            if not torch.isfinite(descriptor).all():
                raise ValueError(f"{name} descriptors cannot be represented as float32")
            norm = descriptor.double().norm(dim=1)
            mask = grounding.get("graph_mask")
            if mask is None:
                mask = torch.ones(len(batch_ids), dtype=torch.bool)
            if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != (len(batch_ids),):
                raise ValueError(f"{name} graph_mask must be bool [B]")
            mask = mask.detach().cpu() & (norm > 1e-6)
            descriptors.append((descriptor.double() / norm.clamp_min(1e-6)[:, None]).float())
            masks.append(mask)
            ids.extend(batch_ids)
            seen.update(batch_ids)
            self._progress(name, step, loader)
            del grounding, descriptor, mask, observation
        if not ids:
            raise ValueError(f"{name} loader is empty")
        # Lexical ID ordering gives deterministic ranking ties even with shuffled loaders.
        order = sorted(range(len(ids)), key=ids.__getitem__)
        return [ids[index] for index in order], torch.cat(descriptors)[order], torch.cat(masks)[order]

    def _retrieval(self, model, query_loader, database_loader, positives, ignore):
        query_ids, queries, valid_queries = self._encode(model, query_loader, "queries")
        database_ids, database, valid_database = self._encode(model, database_loader, "database")
        if queries.shape[1] != database.shape[1]:
            raise ValueError("Query/database descriptor dimensions must match")
        if set(positives) != set(query_ids):
            raise ValueError("positives must contain exactly one entry per query image_id (use [] for no positives)")
        if set(ignore) - set(query_ids):
            raise ValueError("ignore contains unknown query IDs")
        database_index = {value: index for index, value in enumerate(database_ids)}
        positive_indices, ignored_indices = [], []
        for query_id in query_ids:
            positive, excluded = positives[query_id], ignore.get(query_id, set())
            if (positive | excluded) - database_index.keys():
                raise ValueError(f"Ground truth refers to unknown database IDs for query {query_id!r}")
            if positive & excluded:
                raise ValueError(f"Positive and ignored database IDs overlap for query {query_id!r}")
            positive_indices.append({database_index[value] for value in positive})
            ignored_indices.append(torch.tensor(sorted(database_index[value] for value in excluded), dtype=torch.long))
        evaluated = [index for index, values in enumerate(positive_indices) if values]
        if not evaluated:
            raise ValueError("Retrieval needs at least one query with ground-truth positives")
        maximum = min(max(self.settings.recall_k), len(database))
        hits = {k: 0 for k in self.settings.recall_k}
        # Store descriptors on CPU; bound the score matrix by both chunk sizes.
        for start in range(0, len(queries), self.settings.query_chunk_size):
            query = queries[start:start + self.settings.query_chunk_size]
            best_scores = torch.empty(len(query), 0)
            best_indices = torch.empty(len(query), 0, dtype=torch.long)
            for db_start in range(0, len(database), self.settings.database_chunk_size):
                db_end = min(db_start + self.settings.database_chunk_size, len(database))
                scores = query @ database[db_start:db_end].t()
                scores[:, ~valid_database[db_start:db_end]] = -torch.inf
                scores[~valid_queries[start:start + len(query)]] = -torch.inf
                for row in range(len(query)):
                    excluded = ignored_indices[start + row]
                    local = excluded[(excluded >= db_start) & (excluded < db_end)] - db_start
                    scores[row, local] = -torch.inf
                indices = torch.arange(db_start, db_end).expand(len(query), -1)
                candidates = torch.cat((best_scores, scores), dim=1)
                candidate_indices = torch.cat((best_indices, indices), dim=1)
                # Previous ties are already sorted; all new IDs follow previous IDs.
                order = torch.argsort(candidates, dim=1, descending=True, stable=True)[:, :maximum]
                best_scores = candidates.gather(1, order)
                best_indices = candidate_indices.gather(1, order)
            best_indices = best_indices.masked_fill(~torch.isfinite(best_scores), -1)
            for row in range(len(query)):
                target = positive_indices[start + row]
                if target:
                    ranked = best_indices[row].tolist()
                    for k in hits:
                        hits[k] += bool(target.intersection(ranked[:k]))
        return {**{f"recall@{k}": count / len(evaluated) for k, count in hits.items()},
                "retrieval/queries": len(queries), "retrieval/evaluated_queries": len(evaluated),
                "retrieval/empty_gt_queries": len(queries) - len(evaluated),
                "retrieval/valid_queries": int(valid_queries.sum()),
                "retrieval/database": len(database), "retrieval/valid_database": int(valid_database.sum())}

    def _pairs(self, model, loader):
        logits, labels = [], []
        no_matches, availability_known = 0, True
        for step, batch in enumerate(loader, 1):
            if not isinstance(batch, Mapping) or not all(key in batch for key in ("a", "b", "label")):
                raise ValueError("Pair loader must yield a/b/label batches")
            a, b = model(batch["a"]), model(batch["b"])
            alignment = model.align(a, b, compute_hard=True)
            verification = model.verify(a, b, alignment)
            logit = model.score(verification)["scene_logits"]
            if isinstance(logit, torch.Tensor) and logit.ndim == 2 and logit.shape[1] == 1:
                logit = logit.squeeze(1)
            # Validate each batch before concatenation so shape errors stay local.
            compute_pair_metrics(logit, batch["label"], self.settings.pair_threshold)
            logits.append(logit.detach().cpu())
            labels.append(batch["label"].detach().cpu())
            available = verification.get("has_verified_matches")
            if available is None:
                availability_known = False
            elif (not isinstance(available, torch.Tensor) or available.dtype != torch.bool
                  or available.shape != logit.shape):
                raise ValueError("has_verified_matches must be bool [B]")
            else:
                no_matches += int((~available).sum())
            self._progress("pairs", step, loader)
            del a, b, alignment, verification, logit, batch
        if not logits:
            raise ValueError("Pair loader is empty")
        metrics = compute_pair_metrics(torch.cat(logits), torch.cat(labels), self.settings.pair_threshold)
        metrics["pair/no_verified_matches"] = no_matches if availability_known else None
        return metrics

    def run(self, model, *, query_loader=None, database_loader=None, positives=None,
            ignore=None, pair_loader=None, epoch=None):
        """Validate one/both tasks and return numeric metrics (all rates in [0,1])."""
        if epoch is not None and (type(epoch) is not int or epoch <= 0):
            raise ValueError("Validation epoch must be a positive integer or None")
        retrieval = _validate_loaders(query_loader=query_loader, database_loader=database_loader,
                                     positives=positives, ignore=ignore, pair_loader=pair_loader)
        positives = _id_sets(positives, "positives") if retrieval else None
        ignore = _id_sets(ignore, "ignore") if ignore is not None else {}
        modes = [(module, module.training) for module in model.modules()]
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
               torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None)
        metrics = {}
        try:
            model.eval()
            with torch.no_grad(), torch.autocast(self.device.type, enabled=False):
                if retrieval:
                    metrics.update(self._retrieval(model, query_loader, database_loader, positives, ignore))
                if pair_loader is not None:
                    metrics.update(self._pairs(model, pair_loader))
        finally:
            for module, mode in modes:
                module.training = mode
            random.setstate(rng[0])
            np.random.set_state(rng[1])
            torch.set_rng_state(rng[2])
            if rng[3] is not None:
                torch.cuda.set_rng_state_all(rng[3])
        report = {"epoch": epoch, "settings": config_to_dict(self.settings), "metrics": metrics,
                  "notes": ["Rates are fractions in JSON/CSV; the terminal displays percentages.",
                            "Queries with nonempty ground truth stay in the Recall denominator even with invalid/empty graph descriptors.",
                            "Empty-ground-truth queries are excluded and counted; invalid database descriptors are excluded from candidates.",
                            "Rank ties use lexical database image_id; self-exclusion is supplied explicitly through ignore."]}
        if metrics.get("pair/auroc") is None and pair_loader is not None:
            report["notes"].append("Pair data contains only one label class: AUROC is undefined; classification validation requires both classes.")
        self._report(report)
        return {key: value for key, value in metrics.items() if value is not None}

    def as_callback(self, **inputs):
        """Bind loaders/ground truth for TrainEngine.fit(..., validate=callback)."""
        if "epoch" in inputs:
            raise ValueError("TrainEngine supplies epoch to the validation callback")
        _validate_loaders(**inputs)
        # Freeze ground truth containers so an epoch cannot consume/change them.
        for name in ("positives", "ignore"):
            if inputs.get(name) is not None:
                inputs[name] = _id_sets(inputs[name], name)
        def validate(model, epoch):
            return self.run(model, epoch=epoch, **inputs)
        return validate

    @staticmethod
    def _table(title, headers, row):
        widths = [max(len(str(header)), len(str(value))) for header, value in zip(headers, row)]
        border = "+-" + "-+-".join("-" * width for width in widths) + "-+"
        def line(values):
            return "| " + " | ".join(str(value).rjust(width) for value, width in zip(values, widths)) + " |"
        print(f"\n{title}\n{border}\n{line(headers)}\n{border}\n{line(row)}\n{border}", flush=True)

    def _report(self, report):
        metrics, epoch = report["metrics"], report["epoch"]
        tables = []
        if "retrieval/queries" in metrics:
            keys = ["retrieval/queries", "retrieval/evaluated_queries", "retrieval/database"]
            recalls = [f"recall@{k}" for k in self.settings.recall_k]
            self._table("Retrieval validation (%)", ["Epoch", "Queries", "Eval Q", "Database", *[f"R@{k}" for k in self.settings.recall_k]],
                        [epoch or "eval", *[metrics[key] for key in keys], *[f"{metrics[key]*100:.2f}" for key in recalls]])
            tables.append(("retrieval", {"epoch": epoch, **{key: metrics[key] for key in [*keys, *recalls,
                           "retrieval/empty_gt_queries", "retrieval/valid_queries", "retrieval/valid_database"]}}))
        if "pair/pairs" in metrics:
            rates = [metrics[f"pair/{key}"] for key in ("precision", "recall", "f1", "auroc")]
            self._table("Pair verification validation (%)", ["Epoch", "Pairs", "Pos", "Neg", "Threshold", "Precision", "Recall", "F1", "AUROC", "BCE"],
                        [epoch or "eval", *[metrics[f"pair/{key}"] for key in ("pairs", "positive_pairs", "negative_pairs", "threshold")],
                         *[f"{value*100:.2f}" if value is not None else "-" for value in rates], f"{metrics['pair/bce']:.5f}"])
            tables.append(("pairs", {"epoch": epoch, **{key: value for key, value in metrics.items() if key.startswith("pair/")}}))
            if metrics["pair/auroc"] is None:
                print(report["notes"][-1], flush=True)
        if self.settings.save_reports:
            self.directory.mkdir(parents=True, exist_ok=True)
            stem = f"epoch_{epoch:08d}" if epoch is not None else "evaluation"
            write_json_atomic(self.directory / f"{stem}.json", report)
            for name, row in tables:
                self._csv(self.directory / f"{stem}_{name}.csv", row)
            print(f"Validation report: {self.directory / (stem + '.json')}", flush=True)

    @staticmethod
    def _csv(path, row):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent,
                                             suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
