import logging
import random
from typing import List, Tuple

import pandas as pd
from datasets import Dataset

logger = logging.getLogger(__name__)


class TripletDatasetCreator:
    """Create triplet datasets for training."""

    def __init__(self, training_config):
        self.config = training_config

    def create_triplet_dataset(self, df: pd.DataFrame) -> Dataset:
        """Create triplet dataset from dataframe."""
        triplets = []

        for _, row in df.iterrows():
            rule = row["rule"]

            # Collect positive and negative examples
            positives = [row["positive_example_1"], row["positive_example_2"]]
            negatives = [row["negative_example_1"], row["negative_example_2"]]

            # Remove NaN values
            positives = [p for p in positives if pd.notna(p)]
            negatives = [n for n in negatives if pd.notna(n)]

            # Create triplets: (anchor, positive, negative)
            for _ in range(self.config.augmentation_factor):
                if positives and negatives:
                    anchor = random.choice(positives)
                    positive = random.choice(positives)
                    negative = random.choice(negatives)

                    triplets.append({"anchor": anchor, "positive": positive, "negative": negative})

        # Subsample if needed
        if self.config.subsample_fraction < 1.0:
            sample_size = int(len(triplets) * self.config.subsample_fraction)
            triplets = random.sample(triplets, sample_size)

        logger.info(f"Created {len(triplets)} triplets")
        return Dataset.from_list(triplets)
