"""Export reusable per-image AmsterTime OSGs and a CSV-derived pair index."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from data.amstertime_loader import AmsterTimeDataset, DEFAULT_DATA_ROOT
from data.osg_storage import (
    file_sha256, graph_file_candidates, graph_file_path, read_json,
    validate_graph, write_json_atomic,
)
from data.osg_builder import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, REPO, OvSGTRBuilder


def generation_fingerprint(object_threshold, relation_threshold, device):
    """Identify the actual model, inference code and source export settings."""
    source_paths = [REPO / "source_manifest.json", REPO / "config" / "vocabulary.json",
                    REPO / "weights" / "assets_manifest.json"]
    source_paths += sorted((REPO / "GroundingDINO" / "groundingdino").rglob("*.py"))
    source_paths += sorted((REPO / "util").glob("*.py"))
    generation = {
        "export_version": 1,
        "config_sha256": file_sha256(DEFAULT_CONFIG),
        "checkpoint_sha256": file_sha256(DEFAULT_CHECKPOINT),
        "adapter_sha256": file_sha256(Path(__file__).with_name("osg_builder.py")),
        "sources_sha256": {path.relative_to(REPO).as_posix(): file_sha256(path) for path in source_paths},
        "object_threshold": object_threshold, "relation_threshold": relation_threshold,
        "max_relations": None, "device": device,
    }
    identifier = hashlib.sha256(json.dumps(generation, sort_keys=True).encode()).hexdigest()
    return generation, identifier


def reusable_graph(paths, image_id, relative_image, generation_id, image_hash):
    """Reuse only graphs with matching image and inference provenance."""
    for path in paths:
        if not path.is_file():
            continue
        try:
            candidate = read_json(path)
            validate_graph(candidate)
            if (candidate["image_id"] == image_id and candidate["image_path"] == relative_image
                    and candidate["provenance"].get("generation_id") == generation_id
                    and candidate["provenance"].get("image_sha256") == image_hash):
                return candidate, path
        except (OSError, ValueError, KeyError, TypeError, OverflowError):
            pass  # Invalid exports must be regenerated, never silently reused.
    return None, None


def prepare_amstertime_osg(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    osg_root: str | Path | None = None,
    max_pairs: int | None = None,
    overwrite: bool = False,
    object_threshold: float = 0.0,
    relation_threshold: float = 0.0,
    device: str = "cuda",
    graph_format: str = "json_gzip",
) -> dict:
    """Export source graphs once; resume unchanged, valid graphs on later runs.

    Default export keeps all valid objects after upstream NMS/top-K and the
    best predicate for every directed pair. No additional node/edge budget is
    applied here. Optional confidence floors are recorded, not training limits.
    Only the pair index receives labels; inference sees one image at a time.
    """
    if max_pairs is not None and max_pairs <= 0:
        raise ValueError("max_pairs must be positive or None")
    if not 0 <= object_threshold <= 1 or not 0 <= relation_threshold <= 1:
        raise ValueError("Export confidence thresholds must be in [0, 1]")
    graph_file_path(Path("graph.json"), graph_format)  # Validate before side effects.
    dataset = AmsterTimeDataset(root=data_root, load_osg=False)
    root = dataset.root
    osg_root = root / "osg" if osg_root is None else Path(osg_root).expanduser().resolve()
    pairs = dataset.pairs[:max_pairs]
    # Fingerprint the actual checkpoint, config and inference sources, not just
    # filenames. Changing assets or adapter code invalidates previous exports.
    generation, generation_id = generation_fingerprint(object_threshold, relation_threshold, device)

    records, images = [], {}
    for pair_index, pair in enumerate(pairs):
        record = {"pair_index": pair_index, "label": 1}
        for side, path, image_id in (("a", pair.path_a, pair.image_id_a),
                                     ("b", pair.path_b, pair.image_id_b)):
            relative_image = path.relative_to(root).as_posix()
            relative_osg = graph_file_path(Path(relative_image).with_suffix(".json"), graph_format).as_posix()
            previous = images.get(relative_image)
            if previous is not None and previous[1] != image_id:
                raise ValueError(f"Conflicting CSV IDs for {relative_image}")
            images[relative_image] = (path, image_id, relative_osg)
            record[side] = {"image_id": image_id, "image_path": relative_image, "osg_path": relative_osg}
        records.append(record)

    print(f"Export: {len(pairs)}/{len(dataset)} pairs, {len(images)} unique images -> {osg_root}", flush=True)
    builder = None
    written = reused = converted = 0
    for position, (relative_image, (path, image_id, relative_osg)) in enumerate(images.items(), start=1):
        output_path = osg_root / relative_osg
        image_hash = file_sha256(path)
        graph, existing_path = None, None
        if not overwrite:
            graph, existing_path = reusable_graph(
                graph_file_candidates(osg_root / Path(relative_image).with_suffix(".json"), graph_format),
                image_id, relative_image, generation_id, image_hash)
        if graph is None:
            if builder is None:
                builder = OvSGTRBuilder(device=device, object_threshold=object_threshold,
                                       relation_threshold=relation_threshold, max_relations=None)
            graph = builder.build(path, image_id=image_id)
            graph["image_path"] = relative_image
            graph["provenance"].update(generation_id=generation_id, image_sha256=image_hash)
            validate_graph(graph)
            write_json_atomic(output_path, graph)
            written += 1
            action = "saved"
        else:
            reused += 1
            action = "reused"
            if existing_path != output_path:
                write_json_atomic(output_path, graph)
                converted += 1
                action = "converted"
        print(f"[{position}/{len(images)}] {action} {relative_image}: "
              f"{len(graph['objects'])} objects, {len(graph['relations'])} relations", flush=True)

    index_path = osg_root / "index.json"
    index = {
        "schema_version": 1,
        "dataset": "AmsterTime",
        "image_path_base": "dataset_root",
        "osg_path_base": "index_directory",
        "source_csv": "image_pairs.csv",
        "source_csv_sha256": file_sha256(root / "image_pairs.csv"),
        "generation_id": generation_id,
        "generation": generation,
        "graph_format": graph_format,
        "total_dataset_pairs": len(dataset),
        "exported_pairs": len(pairs),
        "unique_images": len(images),
        "complete_dataset": len(pairs) == len(dataset),
        "pairs": records,
    }
    write_json_atomic(index_path, index)
    print(f"Done: {written} saved, {reused} reused ({converted} converted). Pair index: {index_path}", flush=True)
    return {"written": written, "reused": reused, "converted": converted, "index_path": index_path}


def prepare_gsv_osg(data_root, osg_root=None, max_images=None, overwrite=False,
                    object_threshold=0.0, relation_threshold=0.0, device="cuda",
                    graph_format="json_gzip"):
    """Resume per-image GSV export; manifest paths follow city/image filenames.

    Partial exports are useful for pilots. The loader verifies each requested
    source graph and fails if absent, never substituting an empty graph.
    """
    from data.gsv_cities_loader import GSVCitiesDataset

    if max_images is not None and (type(max_images) is not int or max_images <= 0):
        raise ValueError("max_images must be a positive integer or None")
    if not 0 <= object_threshold <= 1 or not 0 <= relation_threshold <= 1:
        raise ValueError("Export confidence thresholds must be in [0, 1]")
    graph_file_path(Path("graph.json"), graph_format)
    dataset = GSVCitiesDataset(root=data_root, load_osg=False)
    output_root = dataset.root / "osg" if osg_root is None else Path(osg_root).expanduser().resolve()
    generation, generation_id = generation_fingerprint(object_threshold, relation_threshold, device)
    selected = dataset.images[:max_images]
    manifest_path = output_root / "manifest.json"
    manifest = {"schema_version": 1, "dataset": "GSV-Cities",
                "source_csv_sha256": dataset.metadata_hashes(),
                "generation": generation, "generation_id": generation_id,
                "total_images": len(dataset), "requested_images": len(selected),
                "completed_images": 0, "complete_dataset": False,
                "image_path_base": "dataset_root", "graph_format": graph_format,
                "graph_path_pattern": "{city}/{image_stem}.json" + (".gz" if graph_format == "json_gzip" else "")}
    # Persist provenance before inference so interruption remains resumable.
    write_json_atomic(manifest_path, manifest)
    builder = None
    written = reused = converted = 0
    print(f"Export GSV-Cities: {len(selected)}/{len(dataset)} images -> {output_root}", flush=True)
    for position, entry in enumerate(selected, start=1):
        path = dataset.root / entry.relative_path
        output_path = graph_file_path(output_root / entry.graph_path, graph_format)
        image_hash = file_sha256(path)
        graph, existing_path = None, None
        if not overwrite:
            graph, existing_path = reusable_graph(
                graph_file_candidates(output_root / entry.graph_path, graph_format),
                entry.relative_path, entry.relative_path, generation_id, image_hash)
        if graph is None:
            if builder is None:
                builder = OvSGTRBuilder(device=device, object_threshold=object_threshold,
                                       relation_threshold=relation_threshold, max_relations=None)
            graph = builder.build(path, image_id=entry.relative_path)
            graph["image_path"] = entry.relative_path
            graph["provenance"].update(generation_id=generation_id, image_sha256=image_hash)
            validate_graph(graph)
            write_json_atomic(output_path, graph)
            written += 1
            action = "saved"
        else:
            reused += 1
            action = "reused"
            if existing_path != output_path:
                write_json_atomic(output_path, graph)
                converted += 1
                action = "converted"
        print(f"[{position}/{len(selected)}] {action} {entry.relative_path}: "
              f"{len(graph['objects'])} objects, {len(graph['relations'])} relations", flush=True)
    manifest.update(completed_images=len(selected), complete_dataset=len(selected) == len(dataset))
    write_json_atomic(manifest_path, manifest)
    print(f"Done: {written} saved, {reused} reused ({converted} converted). Manifest: {manifest_path}", flush=True)
    return {"written": written, "reused": reused, "converted": converted, "manifest_path": manifest_path}
