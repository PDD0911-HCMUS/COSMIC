"""COSMIC model components."""

from model.config import AlignmentConfig, CriteriaConfig, GroundingConfig, ModelConfig, OSGEncoderConfig, SceneSimilarityConfig, VerificationConfig, VisualEncoderConfig, VISUAL_BACKBONES
from model.visual_encoder import VisualEncoder, build_visual_encoder
from model.osg_encoder import OSGEncoder, build_osg_encoder
from model.model import COSMIC, build_model
from model.grounding import ImageOSGGrounding
from model.cross_osg_alignment import CrossOSGAlignment
from model.image_graph_verification import ImageGraphCrossVerification
from model.criteria import COSMICCriteria, SupConLoss, build_criteria
from model.global_scene_similarity import GlobalSceneSimilarity

__all__ = ["ModelConfig", "VisualEncoderConfig", "VISUAL_BACKBONES",
           "VisualEncoder", "build_visual_encoder", "OSGEncoderConfig", "OSGEncoder", "build_osg_encoder",
           "COSMIC", "build_model", "GroundingConfig", "ImageOSGGrounding",
           "AlignmentConfig", "CrossOSGAlignment", "VerificationConfig", "ImageGraphCrossVerification",
           "CriteriaConfig", "COSMICCriteria", "SupConLoss", "build_criteria", "SceneSimilarityConfig", "GlobalSceneSimilarity"]
