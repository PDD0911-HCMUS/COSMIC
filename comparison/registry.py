"""Lazy method registry: adding an adapter does not change the runner."""

from importlib import import_module


METHOD_REGISTRY = {
    "cosmic": ("comparison.adapters.cosmic", "CosmicAdapter"),
    "mixvpr": ("comparison.adapters.mixvpr", "MixVPRAdapter"),
    "boq": ("comparison.adapters.boq", "BoQAdapter"),
    "cricavpr": ("comparison.adapters.cricavpr", "CricaVPRAdapter"),
    "fol": ("comparison.adapters.fol", "FoLAdapter"),
}
METHOD_NAMES = list(METHOD_REGISTRY)


def check_method(name):
    if name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method adapter {name!r}. Available: {', '.join(METHOD_NAMES)}")


def asset_problems(spec):
    check_method(spec.adapter)
    problems = []
    for field in ("checkpoint", "config") if spec.adapter == "cosmic" else ("checkpoint", "repo"):
        path = getattr(spec, field)
        exists = path is not None and (path.is_dir() if field == "repo" else path.is_file())
        if not exists:
            problems.append(f"{field}: {path or 'not configured'}")
    if spec.pca is not None and not spec.pca.is_file():
        problems.append(f"pca: {spec.pca}")
    return problems


def build_adapter(spec, device):
    problems = asset_problems(spec)
    if problems:
        raise FileNotFoundError("Missing comparison assets: " + "; ".join(problems))
    module, name = METHOD_REGISTRY[spec.adapter]
    return getattr(import_module(module), name)(spec, device)
