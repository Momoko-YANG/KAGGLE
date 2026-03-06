"""Model training utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from sklearn.base import BaseEstimator, clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import ProjectConfig


class ConstantProbabilityModel(BaseEstimator):
    """Fallback probability model returning a constant value."""

    def __init__(self, value: float):
        self.value = float(value)

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        value = float(np.clip(self.value, 0.0, 1.0))
        return np.column_stack([1 - value, np.full(len(X), value)])


@dataclass(slots=True)
class ModelArtifacts:
    """Container for all fitted models."""

    feature_cols: list[str]
    logistic_models: list[BaseEstimator]
    rank_models: list[LGBMRanker]
    logistic_oof: np.ndarray
    rank_oof: np.ndarray
    feature_importances: Optional[pd.DataFrame] = None


def _build_logistic_pipeline(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=False)),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=seed,
                ),
            ),
        ]
    )


def train_logistic_models(
    X: pd.DataFrame,
    y: pd.Series,
    config: ProjectConfig,
) -> tuple[list[Pipeline], np.ndarray, float]:
    folds = config.training.folds
    seed = config.training.seed
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    models: list[Pipeline] = []
    oof = np.zeros(len(X), dtype=float)
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X), start=1):
        model = _build_logistic_pipeline(seed + fold)
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        model.fit(X_train, y_train)
        oof[valid_idx] = model.predict_proba(X_valid)[:, 1]
        models.append(model)
    score = roc_auc_score(y, oof) if len(np.unique(y)) > 1 else float("nan")
    return models, oof, score


def train_rank_models(
    X: pd.DataFrame,
    y: pd.Series,
    groups: Sequence[int],
    config: ProjectConfig,
) -> tuple[list[LGBMRanker], np.ndarray]:
    folds = config.training.folds
    seed = config.training.seed
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    models: list[LGBMRanker] = []
    oof = np.zeros(len(X), dtype=float)
    cumulative_groups = np.array(groups, dtype=int)
    cum_offsets = np.concatenate([[0], np.cumsum(cumulative_groups)])

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X), start=1):
        model = LGBMRanker(**config.training.lgbm_params, random_state=seed + fold)
        train_groups = _groups_from_index(train_idx, cum_offsets)
        valid_groups = _groups_from_index(valid_idx, cum_offsets)
        model.fit(
            X.iloc[train_idx],
            y.iloc[train_idx],
            group=train_groups,
            eval_set=[(X.iloc[valid_idx], y.iloc[valid_idx])],
            eval_group=[valid_groups],
            eval_at=[3, 5, 10],
            verbose=False,
        )
        oof[valid_idx] = model.predict(X.iloc[valid_idx])
        models.append(model)
    return models, oof


def _groups_from_index(index: np.ndarray, offsets: np.ndarray) -> list[int]:
    counts: list[int] = []
    start = 0
    for left, right in zip(offsets[:-1], offsets[1:]):
        mask = (index >= left) & (index < right)
        count = int(mask.sum())
        if count:
            counts.append(count)
    if not counts:
        raise ValueError("Unable to infer group sizes for ranking model")
    return counts
