import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ViolationPredictor:
    """Predict rule violations using centroid distance."""

    def __init__(self, distance_metric: str = "euclidean"):
        self.distance_metric = distance_metric

    def predict(
        self, df: pd.DataFrame, text_to_embedding: Dict[str, np.ndarray], rule_centroids: Dict, text_preprocessor
    ) -> Tuple[list, np.ndarray]:
        """Make predictions on test set."""
        logger.info("Making predictions...")

        row_ids = []
        predictions = []

        for rule in df["rule"].unique():
            if rule not in rule_centroids:
                continue

            rule_data = df[df["rule"] == rule]
            pos_centroid = rule_centroids[rule]["positive"]
            neg_centroid = rule_centroids[rule]["negative"]

            valid_embeddings = []
            valid_row_ids = []

            for _, row in rule_data.iterrows():
                body = text_preprocessor.clean_text(row["body"])
                if body in text_to_embedding:
                    valid_embeddings.append(text_to_embedding[body])
                    valid_row_ids.append(row["row_id"])

            if not valid_embeddings:
                continue

            query_embs = np.array(valid_embeddings)

            # Compute distances
            pos_distances = np.linalg.norm(query_embs - pos_centroid, axis=1)
            neg_distances = np.linalg.norm(query_embs - neg_centroid, axis=1)

            # Score: closer to positive = higher violation
            rule_preds = neg_distances - pos_distances

            row_ids.extend(valid_row_ids)
            predictions.extend(rule_preds)

        logger.info(f"Made predictions for {len(predictions)} examples")
        return row_ids, np.array(predictions)
