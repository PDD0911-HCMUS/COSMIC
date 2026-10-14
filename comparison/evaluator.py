"""Shared feature validation, exact retrieval, Recall@K and candidate-only re-ranking."""

from dataclasses import dataclass
import math
from time import perf_counter

import torch


@dataclass
class Encoded:
    ids: list[str]
    descriptors: torch.Tensor
    valid: torch.Tensor
    features: dict
    timing: dict


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(item) for item in value)
    return 0


def validate_descriptors(descriptor, mask, count, metric):
    if (not isinstance(descriptor, torch.Tensor) or descriptor.ndim != 2
            or descriptor.shape[0] != count or descriptor.shape[1] <= 0
            or not descriptor.is_floating_point() or not torch.isfinite(descriptor).all()):
        raise ValueError("Adapter must return finite floating descriptors [B,D]")
    descriptor = descriptor.detach().cpu().float()
    if not torch.isfinite(descriptor).all():
        raise ValueError("Descriptors cannot be represented as float32")
    norm = descriptor.double().norm(dim=1)
    if mask is None:
        mask = torch.ones(count, dtype=torch.bool)
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != (count,):
        raise ValueError("Descriptor validity must be bool [B]")
    valid = mask.detach().cpu() & (norm > 1e-6)
    if metric == "cosine":
        descriptor = (descriptor.double() / norm.clamp_min(1e-6)[:, None]).float()
    return descriptor, valid


def encode_observations(adapter, loader, config, *, retain_features=False):
    ids, seen, descriptors, masks, features = [], set(), [], [], {}
    load_seconds = forward_seconds = 0.0
    iterator = iter(loader)
    step = 0
    with torch.inference_mode():
        while True:
            start = perf_counter()
            try:
                observation = next(iterator)
            except StopIteration:
                break
            load_seconds += perf_counter() - start
            batch_ids = observation["image_id"]
            if (not batch_ids or any(not isinstance(value, str) or not value for value in batch_ids)
                    or len(set(batch_ids)) != len(batch_ids) or seen.intersection(batch_ids)):
                raise ValueError("Inference image IDs must be nonempty and globally unique")
            synchronize(adapter.device)
            start = perf_counter()
            descriptor, mask, cached = adapter.encode(observation, retain_features=retain_features)
            synchronize(adapter.device)
            forward_seconds += perf_counter() - start
            descriptor, valid = validate_descriptors(descriptor, mask, len(batch_ids), config.metric)
            if descriptors and descriptor.shape[1] != descriptors[0].shape[1]:
                raise ValueError("Descriptor dimension changed between inference batches")
            if retain_features:
                if not isinstance(cached, list) or len(cached) != len(batch_ids):
                    raise ValueError("Re-ranking requires one cached feature mapping per image")
                features.update(zip(batch_ids, cached))
            ids.extend(batch_ids)
            seen.update(batch_ids)
            descriptors.append(descriptor)
            masks.append(valid)
            step += 1
            if step == 1 or step % config.log_every == 0 or step == len(loader):
                print(f"Encode: batch {step}/{len(loader)} | images={len(ids)}", flush=True)
            del observation, descriptor, cached
    if not ids:
        raise ValueError("Inference loader is empty")
    return Encoded(ids, torch.cat(descriptors), torch.cat(masks), features,
                   {"loading_preprocessing_seconds": load_seconds, "encoding_seconds": forward_seconds,
                    "mean_batch_encode_ms_per_image": forward_seconds * 1000 / len(ids),
                    "feature_cache_bytes": tensor_bytes(features)})


def select_images(encoded, ids):
    index = {value: i for i, value in enumerate(encoded.ids)}
    if len(index) != len(encoded.ids) or not set(ids) <= index.keys():
        raise ValueError("Encoded image IDs differ from the benchmark manifest")
    rows = torch.tensor([index[value] for value in ids], dtype=torch.long)
    return encoded.descriptors[rows], encoded.valid[rows]


def rank_descriptors(queries, database, valid_queries, valid_database, *, top_k,
                     query_chunk_size=128, database_chunk_size=4096, metric="l2"):
    """Exact CPU search with bounded score matrices; DB rows must be in tie order.

    Zero/invalid descriptors do not produce candidates. Invalid queries remain
    in the Recall denominator. L2 preserves native/PCA descriptor magnitudes;
    cosine expects descriptors normalized by validate_descriptors.
    """
    if (queries.ndim != 2 or database.ndim != 2 or not len(queries) or not len(database)
            or queries.shape[1] != database.shape[1] or queries.shape[1] <= 0
            or not torch.isfinite(queries).all() or not torch.isfinite(database).all()):
        raise ValueError("Queries/database must be nonempty finite [N,D] with matching dimensions")
    for value, count in ((valid_queries, len(queries)), (valid_database, len(database))):
        if value.dtype != torch.bool or value.shape != (count,):
            raise ValueError("Validity masks must be bool [N]")
    for value in (top_k, query_chunk_size, database_chunk_size):
        if type(value) is not int or value <= 0:
            raise ValueError("Search sizes must be positive integers")
    if metric not in ("l2", "cosine"):
        raise ValueError("Unknown retrieval metric")
    maximum = min(top_k, len(database))
    rankings = []
    for start in range(0, len(queries), query_chunk_size):
        query = queries[start:start + query_chunk_size]
        best_scores = torch.empty(len(query), 0)
        best_indices = torch.empty(len(query), 0, dtype=torch.long)
        for db_start in range(0, len(database), database_chunk_size):
            chunk = database[db_start:db_start + database_chunk_size]
            scores = query @ chunk.t()
            if metric == "l2":
                scores = 2 * scores - query.square().sum(1)[:, None] - chunk.square().sum(1)[None, :]
            if not torch.isfinite(scores).all():
                raise ValueError("Retrieval scores overflowed; check descriptor magnitudes")
            scores[:, ~valid_database[db_start:db_start + len(chunk)]] = -torch.inf
            scores[~valid_queries[start:start + len(query)]] = -torch.inf
            indices = torch.arange(db_start, db_start + len(chunk)).expand(len(query), -1)
            candidates = torch.cat((best_scores, scores), dim=1)
            candidate_indices = torch.cat((best_indices, indices), dim=1)
            order = torch.argsort(candidates, dim=1, descending=True, stable=True)[:, :maximum]
            best_scores = candidates.gather(1, order)
            best_indices = candidate_indices.gather(1, order)
        rankings.append(best_indices.masked_fill(~torch.isfinite(best_scores), -1))
    return torch.cat(rankings)


def compute_recalls(rankings, query_ids, database_ids, positives, recall_k):
    if (rankings.ndim != 2 or rankings.shape[0] != len(query_ids) or rankings.dtype != torch.long
            or len(set(query_ids)) != len(query_ids) or len(set(database_ids)) != len(database_ids)
            or set(positives) != set(query_ids) or not query_ids or not database_ids):
        raise ValueError("Rankings and unique image IDs must match ground truth")
    if ((rankings < -1) | (rankings >= len(database_ids))).any():
        raise ValueError("Ranking contains out-of-range database indices")
    if any(type(k) is not int or k <= 0 for k in recall_k):
        raise ValueError("Recall K must be positive integers")
    if rankings.shape[1] < min(max(recall_k), len(database_ids)):
        raise ValueError("Ranking does not cover requested Recall K")
    database_index = {value: i for i, value in enumerate(database_ids)}
    hits = {k: 0 for k in recall_k}
    for query_id, row in zip(query_ids, rankings.tolist()):
        values = positives[query_id]
        if not isinstance(values, (list, tuple, set)) or not values or set(values) - database_index.keys():
            raise ValueError("Every benchmark query must have known database ground truth")
        target = {database_index[value] for value in values}
        for k in hits:
            hits[k] += bool(target.intersection(row[:k]))
    return {f"recall@{k}": value / len(query_ids) for k, value in hits.items()}


def rerank_candidates(adapter, rankings, query_ids, database_ids, cache, top_k, log_every=20):
    """Only reorder retrieved candidates; ties preserve retrieval order.

    No pair score => candidate follows scored candidates. If no
    candidate can be verified, preserve the complete original retrieval order.
    Ground truth is deliberately not an argument.
    """
    result = rankings.clone()
    evaluated = unavailable = 0
    synchronize(adapter.device)
    start = perf_counter()
    with torch.inference_mode():
        for row, query_id in enumerate(query_ids):
            candidates = [i for i in rankings[row, :top_k].tolist() if i >= 0]
            scores = []
            for index in candidates:
                score = adapter.pair_score(cache[query_id], cache[database_ids[index]])
                if score is None:
                    unavailable += 1
                elif not math.isfinite(score):
                    raise ValueError("Re-ranking scores must be finite or None")
                scores.append(-math.inf if score is None else score)
                evaluated += 1
            order = sorted(range(len(candidates)), key=lambda i: -scores[i])
            result[row, :len(candidates)] = torch.tensor([candidates[i] for i in order], dtype=torch.long)
            if row == 0 or (row + 1) % log_every == 0 or row + 1 == len(query_ids):
                print(f"Re-rank: query {row+1}/{len(query_ids)} | pairs={evaluated}", flush=True)
    synchronize(adapter.device)
    return result, {"reranking_seconds": perf_counter() - start, "rerank_top_k": top_k,
                    "candidate_pairs": evaluated, "pairs_without_pair_score": unavailable,
                    "rerank_ties": "original retrieval order",
                    "unscored_policy": "after scored candidates; retain original order for ties"}


def evaluate(adapter, loader, benchmark, config):
    retain = "rerank" in adapter.spec.stages
    encoded = encode_observations(adapter, loader, config, retain_features=retain)
    if set(encoded.ids) != set(benchmark.query_ids) | set(benchmark.database_ids):
        raise ValueError("Inference omitted or added benchmark images")
    query_ids = list(benchmark.query_ids)
    database_ids = sorted(benchmark.database_ids)
    queries, valid_queries = select_images(encoded, query_ids)
    database, valid_database = select_images(encoded, database_ids)
    search_k = max(max(config.recall_k), adapter.spec.rerank_top_k if retain else 0)
    start = perf_counter()
    rankings = rank_descriptors(queries, database, valid_queries, valid_database, top_k=search_k,
                               query_chunk_size=config.query_chunk_size,
                               database_chunk_size=config.database_chunk_size, metric=config.metric)
    timing = {**encoded.timing, "ranking_seconds": perf_counter() - start,
              "descriptor_bytes": tensor_bytes(encoded.descriptors)}
    counts = {"queries": len(query_ids), "database": len(database_ids),
              "valid_queries": int(valid_queries.sum()), "valid_database": int(valid_database.sum()),
              "descriptor_dimension": queries.shape[1]}
    retrieval = compute_recalls(rankings, query_ids, database_ids, benchmark.positives, config.recall_k)
    stages = {}
    if "retrieval" in adapter.spec.stages:
        stages["retrieval"] = {"metrics": retrieval, "timing": dict(timing), "counts": counts,
                                "predictions": prediction_records(rankings, query_ids, database_ids)}
    if retain:
        reranked, rerank_timing = rerank_candidates(adapter, rankings, query_ids, database_ids,
                                                   encoded.features, adapter.spec.rerank_top_k, config.log_every)
        metrics = compute_recalls(reranked, query_ids, database_ids, benchmark.positives, config.recall_k)
        stages["rerank"] = {"metrics": metrics, "retrieval_metrics": retrieval,
                            "candidate_recall": compute_recalls(rankings, query_ids, database_ids,
                                                               benchmark.positives, [search_k])[f"recall@{search_k}"],
                            "timing": {**timing, **rerank_timing}, "counts": counts,
                            "predictions": prediction_records(reranked, query_ids, database_ids)}
    return stages


def prediction_records(rankings, query_ids, database_ids):
    return [{"query_id": query_id, "database_ids": [database_ids[i] for i in row if i >= 0]}
            for query_id, row in zip(query_ids, rankings.tolist())]
