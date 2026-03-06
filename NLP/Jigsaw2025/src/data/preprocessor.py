import re
from typing import List, Optional, Set
from urllib.parse import urlparse

import pandas as pd


class TextPreprocessor:
    """Preprocess text data."""

    @staticmethod
    def clean_text(text: Optional[str]) -> str:
        """Replace URLs with standardized format."""
        if not text:
            return ""

        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'

        def replace_url(match):
            url = match.group(0)
            try:
                parsed = urlparse(url)
                domain = parsed.netloc.lower()
                if domain.startswith("www."):
                    domain = domain[4:]

                path_parts = [p for p in parsed.path.split("/") if p]
                if path_parts:
                    important_path = "/".join(path_parts[:2])
                    return f"<url>: ({domain}/{important_path})"
                else:
                    return f"<url>: ({domain})"
            except:
                return "<url>: (unknown)"

        return re.sub(url_pattern, replace_url, str(text))

    @staticmethod
    def collect_unique_texts(df: pd.DataFrame) -> List[str]:
        """Collect all unique texts from dataframe."""
        all_texts = set()

        # Add bodies
        for body in df["body"]:
            if pd.notna(body):
                all_texts.add(TextPreprocessor.clean_text(body))

        # Add positive and negative examples
        example_cols = ["positive_example_1", "positive_example_2", "negative_example_1", "negative_example_2"]
        for col in example_cols:
            for example in df[col]:
                if pd.notna(example):
                    all_texts.add(TextPreprocessor.clean_text(example))

        return list(all_texts)
