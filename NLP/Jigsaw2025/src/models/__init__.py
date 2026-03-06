"""
Model definition and training modules.
"""

from .embedding_model import EmbeddingModel
from .trainer import ModelTrainer

__all__ = [
    "EmbeddingModel",
    "ModelTrainer",
]
