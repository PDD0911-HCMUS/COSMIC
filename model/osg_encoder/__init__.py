"""BERT-based object, predicate and triplet encoding for COSMIC."""

from model.osg_encoder.encoder import OSGEncoder, build_osg_encoder

__all__ = ["OSGEncoder", "build_osg_encoder"]
