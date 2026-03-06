"""Missing value handling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def _is_discontinued_pattern(series: pd.Series, min_tail_ratio: float = 0.1) -> bool:
    if series.isna().all():
        return True
    last_valid = series.last_valid_index()
    if last_valid is None:
        return True
    loc = series.index.get_loc(last_valid)
    if loc == len(series) - 1:
        return False
    tail = series.iloc[loc + 1 :]
    if len(tail) < max(3, int(len(series) * min_tail_ratio)):
        return False
    return tail.isna().all()


def detect_discontinued_columns(
    df: pd.DataFrame,
    *,
    threshold: float = 0.8,
    exclude: Sequence[str] = ("date_id",),
) -> list[str]:
    candidates: list[str] = []
    for col in df.columns:
        if col in exclude:
            continue
        share = df[col].isna().mean()
        if share >= threshold and _is_discontinued_pattern(df[col]):
            candidates.append(col)
    return candidates


def apply_discontinued_strategy(
    df: pd.DataFrame,
    discontinued: Sequence[str],
    *,
    strategy: str = "fill_last",
) -> tuple[pd.DataFrame, list[str]]:
    if not discontinued:
        return df.copy(), []
    augmented: list[str] = []
    result = df.copy()
    if strategy == "fill_last":
        for col in discontinued:
            if result[col].notna().any():
                last_value = result[col].ffill().iloc[-1]
                result[col] = result[col].ffill().fillna(last_value if pd.notna(last_value) else 0.0)
            else:
                result[col] = 0.0
    elif strategy == "indicator":
        for col in discontinued:
            indicator = f"{col}_is_active"
            result[indicator] = result[col].notna().astype(np.int8)
            result[col] = result[col].ffill().fillna(0.0)
            augmented.append(indicator)
    elif strategy == "fill_zero":
        for col in discontinued:
            result[col] = result[col].fillna(0.0)
    elif strategy == "drop":
        result = result.drop(columns=list(discontinued))
    else:
        raise ValueError(f"Unknown discontinued strategy: {strategy}")
    return result, augmented


def fill_regular_missing(
    df: pd.DataFrame,
    *,
    strategy: str = "forward_only",
    exclude: Sequence[str] = ("date_id",),
) -> pd.DataFrame:
    result = df.copy()
    cols = [c for c in df.columns if c not in exclude]
    if strategy == "forward_only":
        result[cols] = result[cols].ffill()
    elif strategy == "forward_backward":
        result[cols] = result[cols].ffill().bfill()
    elif strategy == "zero":
        result[cols] = result[cols].fillna(0.0)
    else:
        raise ValueError(f"Unknown missing strategy: {strategy}")
    result[cols] = result[cols].fillna(0.0)
    return result
