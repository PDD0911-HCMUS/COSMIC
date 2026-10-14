"""Build validation inputs through the dataset registry, independently of training pairs."""


def build_validation_inputs(config):
    from registry import get_dataset
    data = config.validation.data
    if data is None:
        if config.mode != "evaluate":
            return None
        data = config.data
    adapter = get_dataset(data.dataset)
    if adapter.validation_factory is None:
        raise ValueError(f"Dataset {adapter.name!r} has no retrieval validation factory")
    return adapter.validation_factory(data)
