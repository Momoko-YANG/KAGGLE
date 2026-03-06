"""
Rule Violation Detection Package
A machine learning system for detecting rule violations in text.
"""

version = "0.1.0"
author = "Momoko Yang"
email = "yangmy1215@gmail.com"

# Package-level imports for convenience
from src.data.loader import DataLoader
from src.data.preprocessor import TextPreprocessor
from src.features.centroids import CentroidBuilder
from src.inference.predictor import ViolationPredictor
from src.models.embedding_model import EmbeddingModel
from src.models.trainer import ModelTrainer

__all__ = [
    "DataLoader",
    "TextPreprocessor",
    "EmbeddingModel",
    "ModelTrainer",
    "CentroidBuilder",
    "ViolationPredictor",
]
