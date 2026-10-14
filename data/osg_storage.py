"""Validate source OSGs and read/write lossless JSON or gzip JSON safely."""

from __future__ import annotations

import hashlib
import gzip
import io
import json
import math
import os
from pathlib import Path
import tempfile
import zlib


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: dict) -> None:
    """Replace only after a complete JSON is on disk; keep old data on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            if path.suffix == ".gz":
                # Close gzip before fsync/replace so its checksum and footer are
                # committed too. Exclude temporary filenames and timestamps.
                with gzip.GzipFile(fileobj=stream, mode="wb", filename="", mtime=0,
                                   compresslevel=9) as compressed:
                    with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                        json.dump(value, text, separators=(",", ":"),
                                  ensure_ascii=False, allow_nan=False)
                        text.write("\n")
            else:
                stream.write((json.dumps(value, indent=2, ensure_ascii=False,
                                         allow_nan=False) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict:
    """Read plain or gzip JSON by extension; never round source coordinates."""
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    except (EOFError, zlib.error) as exc:
        raise ValueError(f"Incomplete or corrupt gzip JSON: {path}") from exc


def graph_file_path(json_path: Path, graph_format: str) -> Path:
    if graph_format not in ("json", "json_gzip"):
        raise ValueError("graph_format must be json or json_gzip")
    return json_path.with_suffix(".json.gz") if graph_format == "json_gzip" else json_path


def graph_file_candidates(json_path: Path, graph_format: str):
    """Prefer the requested format, with legacy fallback for resumable exports."""
    preferred = graph_file_path(json_path, graph_format)
    alternate = graph_file_path(json_path, "json" if graph_format == "json_gzip" else "json_gzip")
    return preferred, alternate


def validate_graph(graph: dict) -> None:
    """Reject incomplete graphs, invalid boxes and dangling relation endpoints."""
    if graph["schema_version"] != 1 or graph["bbox_format"] != "xyxy_pixels":
        raise ValueError("Unsupported OSG schema or bbox format")
    width, height = graph["image_size"]["width"], graph["image_size"]["height"]
    if width <= 0 or height <= 0:
        raise ValueError("Invalid original image dimensions")
    objects = graph["objects"]
    if [obj["id"] for obj in objects] != list(range(len(objects))):
        raise ValueError("Object IDs must be contiguous and unique")
    for obj in objects:
        if not obj["label"] or not 0 <= obj["confidence"] <= 1:
            raise ValueError("Invalid object label or confidence")
        x1, y1, x2, y2 = obj["bbox"]
        if not all(math.isfinite(value) for value in obj["bbox"]):
            raise ValueError("Nonfinite bounding box")
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("Bounding box outside original image")
    for relation in graph["relations"]:
        subject, target = relation["subject"], relation["object"]
        if not (0 <= subject < len(objects) and 0 <= target < len(objects)) or subject == target:
            raise ValueError("Invalid relation endpoints")
        if not relation["predicate"] or not 0 <= relation["confidence"] <= 1:
            raise ValueError("Invalid predicate or confidence")
        expected = relation["confidence"] * objects[subject]["confidence"] * objects[target]["confidence"]
        if not math.isclose(relation["triplet_score"], expected, rel_tol=1e-5, abs_tol=1e-7):
            raise ValueError("Invalid triplet score")


