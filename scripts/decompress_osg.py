"""Expand OSGs losslessly and switch their manifest/index to plain JSON.

Run on a host with write access to the datasets, not the read-only training
container. Only the Python standard library is needed; no models are loaded.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import gzip
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import time


CHUNK_SIZE = 1024 * 1024


def check_graph(path: Path, generation_id: str | None) -> None:
    """Validate graph contents/provenance without loading torch or encoders."""
    from data.osg_storage import read_json, validate_graph

    try:
        graph = read_json(path)
        validate_graph(graph)
        if generation_id is not None and graph["provenance"]["generation_id"] != generation_id:
            raise ValueError("Generation fingerprint differs from the dataset manifest")
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        raise ValueError(f"Invalid OSG {path}: {error}") from error


def expand_file(source: Path, *, fast: bool = False, repair: bool = False,
                generation_id: str | None = None) -> tuple[str, int]:
    """Verify copies, or repair invalid JSONs from validated gzip originals."""
    target = source.with_suffix("")
    # One target lookup instead of separate symlink/existence queries on SMB.
    try:
        target_stat = target.lstat()
    except FileNotFoundError:
        target_stat = None
    if source.is_symlink() or (target_stat is not None and stat.S_ISLNK(target_stat.st_mode)):
        raise ValueError(f"Refusing a symbolic-link graph: {source} / {target}")
    if repair and target_stat is not None:
        try:
            check_graph(target, generation_id)
        except ValueError as error:
            print(f"Repairing {target}: {error}", flush=True)
        else:
            return "verified", target_stat.st_size
    elif target_stat is not None:
        size = 0
        with gzip.open(source, "rb") as compressed, target.open("rb") as existing:
            while chunk := compressed.read(CHUNK_SIZE):
                if existing.read(len(chunk)) != chunk:
                    raise ValueError(f"Existing JSON differs from its gzip source: {target}")
                size += len(chunk)
            if existing.read(1):
                raise ValueError(f"Existing JSON differs from its gzip source: {target}")
        return "verified", size

    temporary = None
    try:
        with gzip.open(source, "rb") as compressed:
            with tempfile.NamedTemporaryFile(
                dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False
            ) as output:
                temporary = Path(output.name)
                size = 0
                while chunk := compressed.read(CHUNK_SIZE):
                    output.write(chunk)
                    size += len(chunk)
                # Reading through EOF also checks gzip CRC/length before publication.
                output.flush()
                if not fast:
                    os.fsync(output.fileno())
        if repair:
            # Never replace an invalid copy with invalid source graph content.
            try:
                check_graph(temporary, generation_id)
            except ValueError as error:
                raise ValueError(f"Cannot restore {target} from {source}: {error}") from error
        os.replace(temporary, target)
        temporary = None  # The rename consumed it; avoid a failed SMB unlink.
        return ("repaired" if target_stat is not None else "written"), size
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def discover(root: Path, *, include_plain: bool = False) -> list[Path]:
    sources = []
    last_message = time.monotonic()

    def scan_error(error):
        raise error

    for directory, _, filenames in os.walk(root, followlinks=False, onerror=scan_error):
        sources.extend(Path(directory) / name for name in filenames if name.endswith(".json.gz"))
        if include_plain:
            names = set(filenames)
            for name in filenames:
                if not name.endswith(".json") or name + ".gz" in names:
                    continue
                if Path(directory) == root and name in ("manifest.json", "index.json"):
                    continue
                # Validate plain-only copies too; gzip is needed only if repair is required.
                sources.append(Path(directory) / (name + ".gz"))
        if time.monotonic() - last_message >= 10:
            print(f"Scanning: {len(sources):,} graph candidates found", flush=True)
            last_message = time.monotonic()
    return sorted(sources)


def convert_files(sources: list[Path], workers: int, *, fast: bool = False,
                  repair: bool = False, generation_id: str | None = None) -> None:
    """Bound queued work so a 500k-file dataset does not create 500k futures."""
    total = len(sources)
    written = verified = repaired = size = done_count = 0
    started = last_message = time.monotonic()
    iterator = iter(sources)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {}

        def submit_one():
            source = next(iterator, None)
            if source is not None:
                future = executor.submit(expand_file, source, fast=fast, repair=repair,
                                         generation_id=generation_id)
                pending[future] = source

        for _ in range(min(total, workers * 2)):
            submit_one()
        try:
            while pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    source = pending.pop(future)
                    try:
                        action, byte_count = future.result()
                    except Exception as error:
                        raise ValueError(f"OSG conversion failed for {source}: {error}") from error
                    written += action == "written"
                    verified += action == "verified"
                    repaired += action == "repaired"
                    size += byte_count
                    done_count += 1
                    now = time.monotonic()
                    if done_count == 1 or done_count % 1000 == 0 or now - last_message >= 10 or done_count == total:
                        elapsed = max(now - started, 1e-9)
                        rate = done_count / elapsed
                        eta_hours = (total - done_count) / rate / 3600
                        print(
                            f"[{done_count:,}/{total:,}] new={written:,} verified={verified:,} repaired={repaired:,} "
                            f"JSON={size / 1024**3:.2f} GiB "
                            f"elapsed={elapsed / 60:.1f} min | "
                            f"avg={rate:.1f} files/s ETA={eta_hours:.1f} h",
                            flush=True,
                        )
                        last_message = now
                    submit_one()
        except BaseException:
            for future in pending:
                future.cancel()
            raise


def read_metadata(root: Path) -> tuple[Path, bytes, dict]:
    paths = [path for path in (root / "manifest.json", root / "index.json") if path.is_file()]
    if len(paths) != 1:
        raise ValueError(f"Pass an OSG root containing one manifest.json or index.json: {root}")
    path = paths[0]
    if path.is_symlink():
        raise ValueError(f"Refusing a symbolic-link manifest/index: {path}")
    original = path.read_bytes()
    metadata = json.loads(original)
    if metadata.get("schema_version") != 1 or metadata.get("graph_format") not in ("json", "json_gzip"):
        raise ValueError(f"Unsupported OSG manifest/index: {path}")
    return path, original, metadata


def atomic_bytes(path: Path, content: bytes) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def activate_json(root: Path, path: Path, original: bytes, metadata: dict) -> None:
    """Change storage metadata only; preserve graph provenance and completion."""
    previous = json.dumps(metadata, sort_keys=True)
    if path.name == "index.json":
        # AmsterTime indexes contain explicit per-image paths.
        for pair in metadata["pairs"]:
            for side in ("a", "b"):
                entry = pair[side]
                relative = entry["osg_path"]
                if relative.endswith(".json.gz"):
                    relative = relative[:-3]
                target = (root / relative).resolve()
                try:
                    target.relative_to(root)
                except ValueError:
                    raise ValueError(f"Outside-root plain graph in index: {target}") from None
                if not target.is_file():
                    raise ValueError(f"Missing/outside-root plain graph in index: {target}")
                entry["osg_path"] = relative
    pattern = metadata.get("graph_path_pattern")
    if isinstance(pattern, str) and pattern.endswith(".json.gz"):
        metadata["graph_path_pattern"] = pattern[:-3]
    metadata["graph_format"] = "json"
    if path.read_bytes() != original:
        raise ValueError(f"Manifest/index changed during extraction; leaving it untouched: {path}")
    if json.dumps(metadata, sort_keys=True) != previous:
        backup = path.with_name(path.name + ".before_decompress.bak")
        if not backup.exists():
            atomic_bytes(backup, original)
        content = (json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        atomic_bytes(path, content)
        print(f"Activated plain JSON: {path} (original metadata: {backup})", flush=True)
    else:
        print(f"Manifest/index already selects plain JSON: {path}", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("osg_roots", type=Path, nargs="+", help="Dataset osg directories, each with a manifest/index")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent extraction workers (default: 4)")
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip per-graph fsync on NAS; atomic rename/CRC checks remain, but power-loss durability is deferred",
    )
    parser.add_argument(
        "--repair", action="store_true",
        help="Validate existing JSON graphs; replace invalid ones from validated gzip originals",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    # Script execution sets sys.path to scripts/, not the project root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    roots = list(dict.fromkeys(path.expanduser().resolve() for path in args.osg_roots))
    try:
        # Reject wrong paths before starting a long conversion of another dataset.
        jobs = [(root, *read_metadata(root)) for root in roots]
        if args.fast:
            print(
                "Fast mode: per-graph fsync disabled; files are closed before atomic rename. "
                "Power-loss durability is deferred; gzip originals are kept.",
                flush=True,
            )
        if args.repair:
            print("Repair mode: validate plain graphs/provenance; restore invalid copies from gzip.", flush=True)
        for root, metadata_path, original, metadata in jobs:
            print(f"\nOSG root: {root}\nScanning graphs...", flush=True)
            sources = discover(root, include_plain=args.repair)
            if not sources and metadata["graph_format"] == "json_gzip":
                raise ValueError(f"No .json.gz files found; cannot verify this gzip export: {root}")
            kind = "graph candidates" if args.repair else "gzip graphs"
            print(f"Found {len(sources):,} {kind} | workers={args.workers}", flush=True)
            convert_files(sources, args.workers, fast=args.fast, repair=args.repair,
                          generation_id=metadata.get("generation_id"))
            activate_json(root, metadata_path, original, metadata)
        print("\nDone. Gzip originals kept. New loader instances prefer plain JSON.", flush=True)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted. Originals and completed JSONs kept; rerun the same command to resume.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"\nExtraction failed: {error}\nGzip originals kept. This OSG root was not activated.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
