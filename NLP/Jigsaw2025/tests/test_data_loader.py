"""
Tests for data loading functionality.
"""

import os
import tempfile

import pandas as pd
import pytest

from src.data.loader import DataLoader


@pytest.fixture
def sample_csv_file():
    """Create a temporary CSV file for testing."""
    data = {
        "row_id": [1, 2, 3],
        "body": ["Test body 1", "Test body 2", "Test body 3"],
        "rule": ["Rule A", "Rule B", "Rule A"],
        "positive_example_1": ["Pos 1", "Pos 2", "Pos 3"],
        "positive_example_2": ["Pos 4", "Pos 5", "Pos 6"],
        "negative_example_1": ["Neg 1", "Neg 2", "Neg 3"],
        "negative_example_2": ["Neg 4", "Neg 5", "Neg 6"],
    }
    df = pd.DataFrame(data)

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


class TestDataLoader:
    """Test suite for DataLoader class."""

    def test_load_test_data_success(self, sample_csv_file):
        """Test successful data loading."""
        loader = DataLoader()
        df = loader.load_test_data(sample_csv_file)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "body" in df.columns
        assert "rule" in df.columns

    def test_load_test_data_file_not_found(self):
        """Test loading non-existent file."""
        loader = DataLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_test_data("non_existent_file.csv")

    def test_validate_data_success(self, sample_csv_file):
        """Test data validation with valid data."""
        loader = DataLoader()
        df = loader.load_test_data(sample_csv_file)

        assert loader.validate_data(df) is True

    def test_validate_data_missing_columns(self):
        """Test data validation with missing columns."""
        loader = DataLoader()
        df = pd.DataFrame({"row_id": [1, 2], "body": ["a", "b"]})

        assert loader.validate_data(df) is False

    def test_unique_rules_count(self, sample_csv_file):
        """Test counting unique rules."""
        loader = DataLoader()
        df = loader.load_test_data(sample_csv_file)

        unique_rules = df["rule"].nunique()
        assert unique_rules == 2  # Rule A and Rule B
