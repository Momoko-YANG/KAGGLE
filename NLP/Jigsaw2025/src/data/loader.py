import logging
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and validate data."""

    @staticmethod
    def load_test_data(file_path: str) -> pd.DataFrame:
        """Load test data from file."""
        logger.info(f"Loading test data from {file_path}...")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} test examples with {df['rule'].nunique()} unique rules")
        return df

    @staticmethod
    def validate_data(df: pd.DataFrame) -> bool:
        """Validate data."""
        required_cols = [
            "row_id",
            "body",
            "rule",
            "positive_example_1",
            "positive_example_2",
            "negative_example_1",
            "negative_example_2",
        ]

        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False

        return True
