"""
Tests for prediction functionality.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import TextPreprocessor
from src.inference.predictor import ViolationPredictor


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    df = pd.DataFrame({"row_id": [1, 2, 3], "body": ["Body 1", "Body 2", "Body 3"], "rule": ["Rule A", "Rule A", "Rule B"]})

    # Create mock embeddings
    text_to_embedding = {
        "Body 1": np.array([0.1, 0.2, 0.3]),
        "Body 2": np.array([0.2, 0.3, 0.4]),
        "Body 3": np.array([0.3, 0.4, 0.5]),
    }

    # Create mock centroids
    rule_centroids = {
        "Rule A": {
            "positive": np.array([0.15, 0.25, 0.35]),
            "negative": np.array([0.5, 0.6, 0.7]),
            "pos_count": 5,
            "neg_count": 5,
            "rule_embedding": np.array([0.1, 0.1, 0.1]),
        },
        "Rule B": {
            "positive": np.array([0.4, 0.5, 0.6]),
            "negative": np.array([0.1, 0.2, 0.3]),
            "pos_count": 3,
            "neg_count": 3,
            "rule_embedding": np.array([0.2, 0.2, 0.2]),
        },
    }

    return df, text_to_embedding, rule_centroids


class TestViolationPredictor:
    """Test suite for ViolationPredictor class."""

    def test_predictor_initialization(self):
        """Test predictor initialization."""
        predictor = ViolationPredictor(distance_metric="euclidean")
        assert predictor.distance_metric == "euclidean"

    def test_predict_success(self, sample_data):
        """Test successful prediction."""
        df, text_to_embedding, rule_centroids = sample_data
        preprocessor = TextPreprocessor()

        predictor = ViolationPredictor()
        row_ids, predictions = predictor.predict(df, text_to_embedding, rule_centroids, preprocessor)

        assert len(row_ids) == 3
        assert len(predictions) == 3
        assert isinstance(predictions, np.ndarray)

    def test_predict_missing_rule(self, sample_data):
        """Test prediction with missing rule in centroids."""
        df, text_to_embedding, rule_centroids = sample_data
        preprocessor = TextPreprocessor()

        # Remove one rule from centroids
        del rule_centroids["Rule B"]

        predictor = ViolationPredictor()
        row_ids, predictions = predictor.predict(df, text_to_embedding, rule_centroids, preprocessor)

        # Should only predict for Rule A (2 samples)
        assert len(row_ids) == 2

    def test_predict_missing_embedding(self, sample_data):
        """Test prediction with missing embeddings."""
        df, text_to_embedding, rule_centroids = sample_data
        preprocessor = TextPreprocessor()

        # Remove one embedding
        del text_to_embedding["Body 2"]

        predictor = ViolationPredictor()
        row_ids, predictions = predictor.predict(df, text_to_embedding, rule_centroids, preprocessor)

        # Should skip the sample without embedding
        assert len(row_ids) == 2

    def test_predict_output_shape(self, sample_data):
        """Test prediction output shapes."""
        df, text_to_embedding, rule_centroids = sample_data
        preprocessor = TextPreprocessor()

        predictor = ViolationPredictor()
        row_ids, predictions = predictor.predict(df, text_to_embedding, rule_centroids, preprocessor)

        assert len(row_ids) == len(predictions)
        assert predictions.ndim == 1  # 1D array
