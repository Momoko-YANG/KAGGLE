import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CentroidBuilder:
    """Build centroids from examples."""

    @staticmethod
    def build_rule_centroids(
        df: pd.DataFrame, text_to_embedding: Dict[str, np.ndarray], rule_embeddings: Dict[str, np.ndarray], text_preprocessor
    ) -> Dict:
        """Create centroids for each rule."""
        logger.info("Building rule centroids...")

        rule_centroids = {}

        for rule in df["rule"].unique():
            rule_data = df[df["rule"] == rule]

            # Collect positive/negative embeddings
            pos_embeddings = []
            neg_embeddings = []

            for _, row in rule_data.iterrows():
                # Positive examples (violating)
                for col in ["positive_example_1", "positive_example_2"]:
                    if pd.notna(row[col]):
                        clean = text_preprocessor.clean_text(row[col])
                        if clean in text_to_embedding:
                            pos_embeddings.append(text_to_embedding[clean])

                # Negative examples (compliant)
                for col in ["negative_example_1", "negative_example_2"]:
                    if pd.notna(row[col]):
                        clean = text_preprocessor.clean_text(row[col])
                        if clean in text_to_embedding:
                            neg_embeddings.append(text_to_embedding[clean])

            if pos_embeddings and neg_embeddings:
                pos_arr = np.array(pos_embeddings)
                neg_arr = np.array(neg_embeddings)

                # Compute and normalize centroids
                pos_centroid = pos_arr.mean(axis=0)
                neg_centroid = neg_arr.mean(axis=0)

                pos_centroid /= np.linalg.norm(pos_centroid)
                neg_centroid /= np.linalg.norm(neg_centroid)

                rule_centroids[rule] = {
                    "positive": pos_centroid,
                    "negative": neg_centroid,
                    "pos_count": len(pos_embeddings),
                    "neg_count": len(neg_embeddings),
                    "rule_embedding": rule_embeddings[rule],
                }

        logger.info(f"Created centroids for {len(rule_centroids)} rules")
        return rule_centroids
