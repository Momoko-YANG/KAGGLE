"""
Tests for text preprocessing functionality.
"""

import pandas as pd
import pytest

from src.data.preprocessor import TextPreprocessor


class TestTextPreprocessor:
    """Test suite for TextPreprocessor class."""

    def test_clean_text_with_url(self):
        """Test URL cleaning."""
        preprocessor = TextPreprocessor()
        text = "Check out https://www.example.com/path/to/page for more info"
        cleaned = preprocessor.clean_text(text)

        assert "<url>:" in cleaned
        assert "example.com" in cleaned
        assert "www." not in cleaned

    def test_clean_text_without_url(self):
        """Test cleaning text without URLs."""
        preprocessor = TextPreprocessor()
        text = "This is a normal text without URLs"
        cleaned = preprocessor.clean_text(text)

        assert cleaned == text

    def test_clean_text_empty(self):
        """Test cleaning empty text."""
        preprocessor = TextPreprocessor()
        assert preprocessor.clean_text(None) == ""
        assert preprocessor.clean_text("") == ""

    def test_clean_text_multiple_urls(self):
        """Test cleaning text with multiple URLs."""
        preprocessor = TextPreprocessor()
        text = "Visit https://example.com and https://test.org/page"
        cleaned = preprocessor.clean_text(text)

        assert cleaned.count("<url>:") == 2

    def test_collect_unique_texts(self):
        """Test collecting unique texts from dataframe."""
        preprocessor = TextPreprocessor()

        data = {
            "body": ["Body 1", "Body 2", "Body 1"],
            "positive_example_1": ["Pos 1", "Pos 2", None],
            "positive_example_2": ["Pos 3", None, None],
            "negative_example_1": ["Neg 1", "Neg 1", "Neg 2"],
            "negative_example_2": [None, "Neg 3", None],
        }
        df = pd.DataFrame(data)

        unique_texts = preprocessor.collect_unique_texts(df)

        # Should have unique texts only
        assert len(unique_texts) == len(set(unique_texts))
        assert len(unique_texts) <= 8  # Maximum possible unique texts

    def test_collect_unique_texts_with_nan(self):
        """Test collecting texts with NaN values."""
        preprocessor = TextPreprocessor()

        data = {
            "body": ["Body 1", None, "Body 2"],
            "positive_example_1": [None, None, None],
            "positive_example_2": ["Pos 1", None, "Pos 2"],
            "negative_example_1": ["Neg 1", None, None],
            "negative_example_2": [None, None, None],
        }
        df = pd.DataFrame(data)

        unique_texts = preprocessor.collect_unique_texts(df)

        # Should only have non-None texts
        assert all(text is not None and text != "" for text in unique_texts)
