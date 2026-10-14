"""Resumable, lossless per-image OvSGTR exports for Pitts30k-val."""

import math
from pathlib import Path

from data.osg_storage import file_sha256, graph_file_candidates, graph_file_path, validate_graph, write_json_atomic
from data.pitts30k_metadata import DEFAULT_ROOT, read_pitts30k_metadata
from data.prepare_data import generation_fingerprint, reusable_graph
from data.osg_builder import OvSGTRBuilder


def prepare_pitts30k_osg(data_root=DEFAULT_ROOT, osg_root=None, max_images=None, overwrite=False,
                         object_threshold=0.0, relation_threshold=0.0, device="cuda", graph_format="json_gzip"):
    if max_images is not None and (type(max_images) is not int or max_images <= 0):
        raise ValueError("max_images must be a positive integer or None")
    if any(type(value) not in (int,float) or not math.isfinite(value) or not 0 <= value <= 1
           for value in (object_threshold,relation_threshold)):
        raise ValueError("Export thresholds must be finite and in [0,1]")
    graph_file_path(Path("graph.json"),graph_format)
    metadata = read_pitts30k_metadata(data_root)
    output_root = metadata.root / "osg" if osg_root is None else Path(osg_root).expanduser().resolve()
    generation, generation_id = generation_fingerprint(object_threshold,relation_threshold,device)
    selected = metadata.images[:max_images]
    manifest_path = output_root / "manifest.json"
    manifest = {"schema_version":1, "dataset":"Pitts30k", "split":"val",
                "source_mat":"metadata/pitts30k_val.mat", "source_mat_sha256":metadata.source_sha256,
                "image_path_base":"dataset_root", "generation":generation, "generation_id":generation_id,
                "graph_format":graph_format, "total_images":len(metadata.images),
                "database_images":len(metadata.database), "query_images":len(metadata.queries),
                "requested_images":len(selected), "completed_images":0, "complete_dataset":False}
    write_json_atomic(manifest_path,manifest)
    written = reused = converted = 0
    builder = None
    print(f"Export Pitts30k-val: {len(selected)}/{len(metadata.images)} images -> {output_root}",flush=True)
    for position,entry in enumerate(selected,1):
        path = metadata.root / entry.relative_path
        output_path = graph_file_path(output_root / entry.graph_path,graph_format)
        image_hash = file_sha256(path)
        graph = existing = None
        if not overwrite:
            graph,existing = reusable_graph(graph_file_candidates(output_root / entry.graph_path,graph_format),
                                            entry.image_id,entry.relative_path,generation_id,image_hash)
        if graph is None:
            if builder is None:
                builder = OvSGTRBuilder(device=device,object_threshold=object_threshold,
                                       relation_threshold=relation_threshold,max_relations=None)
            graph = builder.build(path,image_id=entry.image_id)
            graph["image_path"] = entry.relative_path
            graph["provenance"].update(generation_id=generation_id,image_sha256=image_hash)
            validate_graph(graph)
            write_json_atomic(output_path,graph)
            written += 1
            action = "saved"
        else:
            reused += 1
            action = "reused"
            if existing != output_path:
                write_json_atomic(output_path,graph)
                converted += 1
                action = "converted"
        manifest["completed_images"] = position
        if position % 100 == 0:
            write_json_atomic(manifest_path,manifest)
        print(f"[{position}/{len(selected)}] {action} {entry.relative_path}: "
              f"{len(graph['objects'])} objects, {len(graph['relations'])} relations",flush=True)
    manifest["complete_dataset"] = len(selected) == len(metadata.images)
    write_json_atomic(manifest_path,manifest)
    print(f"Done: {written} saved, {reused} reused ({converted} converted). Manifest: {manifest_path}",flush=True)
    return {"written":written,"reused":reused,"converted":converted,"manifest_path":manifest_path}
