"""
Embedding utilities for text and rule features.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.data.preprocessor import TextPreprocessor
    from src.models.embedding_model import EmbeddingModel


class EmbeddingGenerator:
    """Generate and cache embeddings for arbitrary text collections."""

    def __init__(self, model: "EmbeddingModel", batch_size: int = 64, normalize: bool = True):
        if model is None:
            raise ValueError("An initialized EmbeddingModel instance is required.")

        self.model = model
        self.batch_size = batch_size
        self.normalize = normalize

    def build_text_embeddings(
        self,
        texts: Sequence[str],
        existing_embeddings: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """Encode a sequence of texts, avoiding redundant computation."""
        embedding_store = dict(existing_embeddings or {})
        unique_texts = self._deduplicate(texts, skip=set(embedding_store))

        if not unique_texts:
            return embedding_store

        logger.info("Encoding %d unique texts", len(unique_texts))
        embeddings = self.model.encode(
            list(unique_texts),
            batch_size=self.batch_size,
            normalize=self.normalize,
        )

        for text, emb in zip(unique_texts, embeddings):
            embedding_store[text] = np.asarray(emb)

        return embedding_store

    def build_dataframe_embeddings(
        self,
        df: pd.DataFrame,
        text_preprocessor: "TextPreprocessor",
        existing_embeddings: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Create embeddings for all texts referenced in a dataframe and the associated rules."""
        if df.empty:
            return dict(existing_embeddings or {}), {}

        texts = text_preprocessor.collect_unique_texts(df)
        text_embeddings = self.build_text_embeddings(texts, existing_embeddings)

        rule_lookup = {}
        cleaned_rules: List[str] = []

        for rule in df["rule"].dropna().unique():
            cleaned_rule = text_preprocessor.clean_text(rule)
            rule_lookup[rule] = cleaned_rule
            cleaned_rules.append(cleaned_rule)

        cleaned_rules = self._deduplicate(cleaned_rules)
        rule_text_embeddings = self.build_text_embeddings(cleaned_rules, text_embeddings)

        rule_embeddings: Dict[str, np.ndarray] = {}
        for rule, cleaned_rule in rule_lookup.items():
            if cleaned_rule in rule_text_embeddings:
                rule_embeddings[rule] = rule_text_embeddings[cleaned_rule]

        return text_embeddings, rule_embeddings

    @staticmethod
    def _deduplicate(items: Iterable[str], skip: Optional[set] = None) -> List[str]:
        """Return unique, truthy strings while respecting an optional skip set."""
        seen = set(skip or ())
        unique_items: List[str] = []

        for item in items:
            if not item:
                continue
            if item in seen:
                continue
            seen.add(item)
            unique_items.append(item)

        return unique_items
