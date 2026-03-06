"""Data loading utilities for the Mitsui MLE project."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected file at {path} but it does not exist")
    return pd.read_csv(path, low_memory=False, **kwargs)


def load_train(path: Path) -> pd.DataFrame:
    df = _read_csv(path)
    if "date_id" not in df.columns:
        raise ValueError("train.csv must contain a 'date_id' column")
    df["date_id"] = df["date_id"].astype("int32")
    feature_cols = [c for c in df.columns if c != "date_id"]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    return df.sort_values("date_id").reset_index(drop=True)


def load_labels(path: Path, filler: float = -999.0) -> pd.DataFrame:
    df = _read_csv(path)
    if "date_id" not in df.columns:
        raise ValueError("train_labels.csv must contain a 'date_id' column")
    df = df.replace(filler, pd.NA)
    df["date_id"] = df["date_id"].astype("int32")
    return df.sort_values("date_id").reset_index(drop=True)


def load_pairs(path: Path) -> pd.DataFrame:
    df = _read_csv(path)
    required = {"target", "lag", "pair"}
    if not required <= set(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise ValueError(f"target_pairs.csv missing required columns: {missing}")
    df["target"] = df["target"].astype(str)
    df["pair"] = df["pair"].astype(str)
    df["lag"] = df["lag"].astype(int)
    return df[sorted(required)].copy()
