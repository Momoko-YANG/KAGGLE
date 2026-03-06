"""Feature engineering helpers."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import ProjectConfig


def _rename_columns(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    renamed = df.copy()
    renamed.columns = [f"{c}__{suffix}" for c in renamed.columns]
    return renamed


def build_feature_matrix(train_df: pd.DataFrame, config: ProjectConfig) -> tuple[pd.DataFrame, list[str]]:
    """Create a wide feature matrix indexed by ``date_id``."""

    df = train_df.sort_values("date_id").set_index("date_id")
    numeric_cols = df.columns

    # Base numeric values
    base = df.astype("float32")
    feature_frames = [_rename_columns(base, "value")]

    log_values = np.log(base.clip(lower=config.log_epsilon))
    feature_frames.append(_rename_columns(log_values.diff().fillna(0.0), "log_diff"))
    pct_change = base.pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    feature_frames.append(_rename_columns(pct_change, "pct_change"))

    for window in config.features.windows:
        rolling_mean = base.rolling(window=window, min_periods=1).mean()
        rolling_std = base.rolling(window=window, min_periods=1).std().fillna(0.0)
        ewma = base.ewm(span=max(2, min(window, config.features.ewma_span)), adjust=False).mean()
        feature_frames.append(_rename_columns(rolling_mean, f"roll_mean_{window}"))
        feature_frames.append(_rename_columns(rolling_std, f"roll_std_{window}"))
        feature_frames.append(_rename_columns(ewma, f"ewm_{window}"))

    features = pd.concat(feature_frames, axis=1).astype("float32")
    features = features.reset_index()
    features = features.fillna(0.0)

    feature_cols = [c for c in features.columns if c != "date_id"]
    if config.features.top_features and len(feature_cols) > config.features.top_features:
        variances = features[feature_cols].var().sort_values(ascending=False)
        selected = variances.head(config.features.top_features).index.tolist()
        features = features[["date_id"] + selected]
        feature_cols = selected
    return features, feature_cols
