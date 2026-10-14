"""Canonical validation observations from the official NetVLAD MATLAB split."""

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
from scipy.io import loadmat

from data.loader_utils import resolve_under
from data.osg_storage import file_sha256


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "Pitts30k"
OFFICIAL_COUNTS = (10000, 7608)
OFFICIAL_MAT_SHA256 = "abf6e578d4d474a474e0bb26c64ed85d45f4cc314da619950ecc9d1e9da11222"


@dataclass(frozen=True)
class PittsImage:
    source_path: str
    relative_path: str
    role: str
    utm: tuple[float, float]

    @property
    def image_id(self):
        return self.relative_path

    @property
    def graph_path(self):
        return Path(self.relative_path).relative_to("images").with_suffix(".json")


@dataclass(frozen=True)
class PittsMetadata:
    root: Path
    database: tuple[PittsImage, ...]
    queries: tuple[PittsImage, ...]
    radius: float
    source_sha256: str

    @property
    def images(self):
        return self.database + self.queries


def read_pitts30k_metadata(root=DEFAULT_ROOT, *, check_images=True):
    root = Path(root).expanduser().resolve()
    path = root / "metadata/pitts30k_val.mat"
    if not path.is_file():
        raise FileNotFoundError(f"Missing official split: {path}. Run python -m data.download_pitts30k first.")
    raw = loadmat(path)["dbStruct"].item()
    if len(raw) != 10 or raw[0].item() != "val":
        raise ValueError("Expected the official NetVLAD validation dbStruct")
    counts = tuple(int(raw[index].item()) for index in (5, 6))
    radius = float(raw[7].item())
    if radius != 25.0 or float(raw[8].item()) != radius ** 2:
        raise ValueError("Pitts30k validation must use the official 25-meter positive radius")
    groups = []
    for role, names_index, coords_index, count in (("database", 1, 2, counts[0]), ("queries", 3, 4, counts[1])):
        names = [item[0].item() for item in raw[names_index]]
        coordinates = np.asarray(raw[coords_index], dtype=np.float64).T
        if (not names or len(names) != count or len(set(names)) != count
                or coordinates.shape != (count, 2) or not np.isfinite(coordinates).all()):
            raise ValueError(f"Invalid {role} image names or UTM coordinates in {path}")
        records = []
        for name, utm in zip(names, coordinates):
            relative = PurePosixPath(name)
            if (not isinstance(name, str) or "\\" in name or relative.is_absolute()
                    or len(relative.parts) != 2 or ".." in relative.parts
                    or not relative.parts[0].isdigit() or relative.suffix.lower() != ".jpg"):
                raise ValueError(f"Invalid original Pittsburgh image path: {name!r}")
            image_path = f"images/val/{role}/{relative.as_posix()}"
            resolved = resolve_under(root, image_path)
            if check_images and not resolved.is_file():
                raise FileNotFoundError(f"Missing validation image: {resolved}. Resume the download.")
            source_path = relative.as_posix() if role == "database" else f"queries_real/{relative.as_posix()}"
            records.append(PittsImage(source_path, image_path, role, tuple(float(x) for x in utm)))
        groups.append(tuple(records))
    return PittsMetadata(root, *groups, radius, file_sha256(path))
