"""Resume official Pittsburgh archives and retain only the Pitts30k validation split."""

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile
import time

from PIL import Image
import requests

from data.osg_storage import file_sha256, write_json_atomic
from data.loader_utils import resolve_under
from data.pitts30k_metadata import DEFAULT_ROOT, OFFICIAL_COUNTS, OFFICIAL_MAT_SHA256, read_pitts30k_metadata


SOURCE = "https://data.ciirc.cvut.cz/public/projects/2015netVLAD/Pittsburgh250k/"
CHECKSUMS = {"000.tar": "3edec2fac461f4a740a21d509ad6e30e",
             "001.tar": "c97e5fffd53a2387f1f57b3ec3cae5cb",
             "002.tar": "afab887a76f8c4b5c017bd51ea105f5c",
             "003.tar": "f217320114299249ea6ebbd71eeb76de",
             "queries_real.tar": "5611c4d601c92f6c7dd702fc2716914a"}


def md5(path):
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url, destination, expected_md5=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if expected_md5 is not None and md5(destination) != expected_md5:
            raise ValueError(f"Existing archive checksum differs from the official checksum: {destination}")
        return destination
    partial = destination.with_name(destination.name + ".part")
    for attempt in range(3):
        offset = partial.stat().st_size if partial.is_file() else 0
        try:
            with requests.get(url, headers={"Range": f"bytes={offset}-", "Accept-Encoding": "identity"},
                              stream=True, timeout=(20, 60)) as response:
                if response.status_code == 416 and offset and expected_md5 is not None and md5(partial) == expected_md5:
                    partial.replace(destination)
                    return destination
                response.raise_for_status()
                if offset and response.status_code == 206:
                    if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("Server resumed the download at a different byte offset")
                    mode = "ab"
                else:
                    offset, mode = 0, "wb"
                total = offset + int(response.headers.get("Content-Length", 0))
                completed, reported = offset, time.monotonic()
                print(f"Download {destination.name}: resume={offset / 2**20:.1f} MiB; total={total / 2**20:.1f} MiB", flush=True)
                with partial.open(mode) as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
                        completed += len(chunk)
                        if time.monotonic() - reported >= 20:
                            print(f"{destination.name}: {completed / 2**20:.1f}/{total / 2**20:.1f} MiB", flush=True)
                            reported = time.monotonic()
                    stream.flush()
                    os.fsync(stream.fileno())
                if total and completed != total:
                    raise OSError("Incomplete HTTP response; rerun to resume")
            break
        except (requests.RequestException, OSError):
            if attempt == 2:
                raise
            time.sleep(2 ** (attempt + 1))
    if expected_md5 is not None and md5(partial) != expected_md5:
        raise ValueError(f"Downloaded archive failed official MD5 validation: {partial}")
    partial.replace(destination)
    print(f"Downloaded and checked: {destination.name}", flush=True)
    return destination


def write_member(bundle, member, destination):
    if not member.isfile() or member.size <= 0 or member.size > 10 * 1024 * 1024:
        raise ValueError(f"Unexpected archive member: {member.name}")
    with bundle.extractfile(member) as stream:
        contents = stream.read()
    if len(contents) != member.size:
        raise ValueError(f"Incomplete archive member: {member.name}")
    digest = hashlib.sha256(contents).hexdigest()
    if destination.exists():
        if file_sha256(destination) != digest:
            raise ValueError(f"Existing image/metadata differs from official archive: {destination}")
        return digest
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        if destination.suffix.lower() == ".jpg":
            with Image.open(temporary) as image:
                image.verify()
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return digest


def prepare_metadata(root):
    archive = download_file(SOURCE + "netvlad_v100_datasets.tar.gz", root / "metadata/netvlad_v100_datasets.tar.gz")
    with tarfile.open(archive, "r:gz") as bundle:
        members = [member for member in bundle if PurePosixPath(member.name).name == "pitts30k_val.mat" and member.isfile()]
        if len(members) != 1:
            raise ValueError("Expected exactly one pitts30k_val.mat in the official metadata archive")
        digest = write_member(bundle, members[0], root / "metadata/pitts30k_val.mat")
    if digest != OFFICIAL_MAT_SHA256:
        raise ValueError("The official validation split has changed; inspect metadata before proceeding")
    metadata = read_pitts30k_metadata(root, check_images=False)
    if (len(metadata.database), len(metadata.queries)) != OFFICIAL_COUNTS:
        raise ValueError("Expected 10000 database and 7608 query images")
    return metadata


def extract_selected(archive, root, entries):
    wanted = {entry.source_path: entry for entry in entries}
    found, hashes = set(), {}
    with tarfile.open(archive, "r:") as bundle:
        for member in bundle:
            if not member.isfile():
                continue
            parts = PurePosixPath(member.name).parts
            if ".." in parts:
                raise ValueError("Archive contains a parent-relative member")
            # Archive prefixes differ between the original database/query bundles.
            source = "/".join(parts[-2:])
            if entries[0].role == "queries":
                source = "queries_real/" + source
            entry = wanted.get(source)
            if entry is None:
                continue
            if source in found:
                raise ValueError(f"Duplicate required image in archive: {source}")
            path = resolve_under(root, entry.relative_path)
            hashes[entry.image_id] = write_member(bundle, member, path)
            found.add(source)
    if found != wanted.keys():
        raise ValueError(f"Archive misses {len(wanted.keys() - found)} required validation images")
    print(f"Extracted/verified {len(found)} validation images from {archive.name}", flush=True)
    return hashes


def prepare_download(root=DEFAULT_ROOT):
    root = Path(root).expanduser().resolve()
    metadata = prepare_metadata(root)
    grouped = {"queries_real.tar": list(metadata.queries)}
    for entry in metadata.database:
        name = PurePosixPath(entry.source_path).parts[0] + ".tar"
        grouped.setdefault(name, []).append(entry)
    manifest_path = root / "metadata/download_manifest.json"
    manifest = {"schema_version": 1, "dataset": "Pitts30k", "split": "val", "source": SOURCE,
                "checksum_source": SOURCE + "md5sums.txt", "source_mat_sha256": metadata.source_sha256,
                "database_images": len(metadata.database), "query_images": len(metadata.queries),
                "complete": False, "archives": {}, "image_sha256": {}}
    write_json_atomic(manifest_path, manifest)
    for name, entries in grouped.items():
        if name not in CHECKSUMS:
            raise ValueError(f"No verified official checksum for archive {name}")
        archive = download_file(SOURCE + name, root / "downloads" / name, CHECKSUMS[name])
        manifest["image_sha256"].update(extract_selected(archive, root, entries))
        manifest["archives"][name] = {"md5": CHECKSUMS[name], "bytes": archive.stat().st_size,
                                      "retained_images": len(entries)}
        write_json_atomic(manifest_path, manifest)
    read_pitts30k_metadata(root)  # Every required image must exist before marking complete.
    from data.pitts30k_loader import prepare_validation_metadata
    validation_manifest = prepare_validation_metadata(root)
    manifest["validation_manifest"] = "metadata/validation_manifest.json"
    manifest["complete"] = True
    write_json_atomic(manifest_path, manifest)
    print(f"Pitts30k-val images and ground truth ready: {validation_manifest['database']} database / "
          f"{validation_manifest['queries']} queries. Root: {root}", flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    prepare_download(args.root)
