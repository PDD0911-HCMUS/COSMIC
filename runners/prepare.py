"""Dispatch resumable OSG export through the registered dataset adapter."""


def prepare_data(config, context, adapter):
    return adapter.exporter(
        data_root=config.data.root, osg_root=config.data.osg_root,
        **{adapter.export_limit: config.export.max_items},
        overwrite=config.export.overwrite, object_threshold=config.export.object_threshold,
        relation_threshold=config.export.relation_threshold, device=context.device,
        graph_format=config.export.graph_format,
    )
