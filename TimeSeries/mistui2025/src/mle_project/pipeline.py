"""High level training and inference pipeline."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .config import ProjectConfig
from . import data as data_utils
from . import features as feature_utils
from . import preprocessing
from .models import (
    ConstantProbabilityModel,
    ModelArtifacts,
    train_logistic_models,
    train_rank_models,
)


@dataclass(slots=True)
class TrainingResult:
    artifacts: ModelArtifacts
    metrics: Dict[str, Any]
    feature_matrix: pd.DataFrame


class MitsuiMLEPipeline:
    """End-to-end pipeline orchestrating data preparation and model training."""

    def __init__(self, config: ProjectConfig):
        self.config = config
        self._feature_matrix: Optional[pd.DataFrame] = None
        self._feature_cols: list[str] = []
        self._artifacts: Optional[ModelArtifacts] = None
        self._metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def prepare_data(self) -> pd.DataFrame:
        train_path = self.config.resolve_path(self.config.train_file)
        labels_path = self.config.resolve_path(self.config.labels_file)
        pairs_path = self.config.resolve_path(self.config.pairs_file)

        train_df = data_utils.load_train(train_path)
        labels_df = data_utils.load_labels(labels_path)
        _pairs_df = data_utils.load_pairs(pairs_path)

        discontinued = preprocessing.detect_discontinued_columns(
            train_df, threshold=self.config.missing.discontinued_threshold
        )
        self._metadata['discontinued_columns'] = discontinued
        train_df, added_cols = preprocessing.apply_discontinued_strategy(
            train_df,
            discontinued,
            strategy=self.config.missing.discontinued_strategy,
        )
        if added_cols:
            self._metadata["discontinued_indicators"] = added_cols
        train_df = preprocessing.fill_regular_missing(
            train_df, strategy=self.config.missing.regular_strategy
        )

        feature_matrix, feature_cols = feature_utils.build_feature_matrix(
            train_df, self.config
        )
        merged = feature_matrix.merge(labels_df, on="date_id", how="left")
        merged = merged.dropna(axis=0, how="any")
        merged = merged.sort_values("date_id").reset_index(drop=True)

        self._feature_matrix = merged
        self._feature_cols = feature_cols
        return merged

    def transform_raw_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw daily features into the engineered feature space."""

        discontinued = self._metadata.get('discontinued_columns')
        if discontinued is None:
            discontinued = preprocessing.detect_discontinued_columns(
                raw_df, threshold=self.config.missing.discontinued_threshold
            )
        processed, added_cols = preprocessing.apply_discontinued_strategy(
            raw_df, discontinued, strategy=self.config.missing.discontinued_strategy
        )
        if added_cols:
            self._metadata.setdefault('discontinued_indicators', added_cols)
        processed = preprocessing.fill_regular_missing(
            processed, strategy=self.config.missing.regular_strategy
        )
        features, _ = feature_utils.build_feature_matrix(processed, self.config)
        if self._artifacts is not None:
            required = ['date_id'] + self._artifacts.feature_cols
            for col in self._artifacts.feature_cols:
                if col not in features.columns:
                    features[col] = 0.0
            missing = [c for c in required if c not in features.columns]
            if missing:
                raise KeyError(f'Missing engineered features after transformation: {missing}')
            features = features[required]
        return features

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, target: str) -> TrainingResult:
        if self._feature_matrix is None:
            self.prepare_data()
        assert self._feature_matrix is not None

        if target not in self._feature_matrix.columns:
            raise KeyError(f"Target '{target}' not present in the labels")

        df = self._feature_matrix.dropna(subset=[target]).copy()
        feature_cols = self._feature_cols
        X = df[feature_cols]
        y_rank = df[target]
        # Binary target for the classification model
        threshold = float(np.nanmedian(y_rank))
        if np.isnan(threshold):
            threshold = 0.0
        y_clf = (y_rank > threshold).astype(int)

        groups = df.groupby("date_id").size().tolist()

        if y_clf.nunique() < 2:
            constant = float(y_clf.iloc[0]) if len(y_clf) else 0.0
            logistic_models = [ConstantProbabilityModel(constant)]
            logistic_oof = np.full(len(df), constant, dtype=float)
            auc = float('nan')
        else:
            logistic_models, logistic_oof, auc = train_logistic_models(X, y_clf, self.config)
        rank_models, rank_oof = train_rank_models(X, y_rank, groups, self.config)

        importances = self._collect_feature_importance(rank_models, feature_cols)

        artifacts = ModelArtifacts(
            feature_cols=feature_cols,
            logistic_models=logistic_models,
            rank_models=rank_models,
            logistic_oof=logistic_oof,
            rank_oof=rank_oof,
            feature_importances=importances,
        )
        metrics = {
            "logistic_auc": auc,
            "rank_oof_mean": float(rank_oof.mean()),
            "target": target,
        }
        self._artifacts = artifacts
        self._metadata.update(metrics)
        return TrainingResult(artifacts=artifacts, metrics=metrics, feature_matrix=df)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_artifacts(self, path: Optional[Path] = None) -> Path:
        if self._artifacts is None:
            raise RuntimeError("No artifacts to save. Train the pipeline first.")
        bundle_path = path or self.config.bundle_path
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "artifacts": self._artifacts,
            "metadata": self._metadata,
        }
        with open(bundle_path, "wb") as f:
            pickle.dump(payload, f)
        return bundle_path

    def load_artifacts(self, path: Optional[Path] = None) -> ModelArtifacts:
        bundle_path = path or self.config.bundle_path
        with open(bundle_path, "rb") as f:
            payload = pickle.load(f)
        self._artifacts = payload["artifacts"]
        self._metadata = payload.get("metadata", {})
        return self._artifacts

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        if self._artifacts is None:
            raise RuntimeError("Load or train artifacts before prediction")
        missing = [c for c in self._artifacts.feature_cols if c not in features.columns]
        if missing:
            raise KeyError(f"Missing required features: {missing[:5]} ...")
        X = features[self._artifacts.feature_cols]
        preds_lr = np.column_stack(
            [model.predict_proba(X)[:, 1] for model in self._artifacts.logistic_models]
        ).mean(axis=1)
        preds_rank = np.column_stack(
            [model.predict(X) for model in self._artifacts.rank_models]
        ).mean(axis=1)
        return pd.DataFrame(
            {
                "prediction_logistic": preds_lr,
                "prediction_rank": preds_rank,
            }
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _collect_feature_importance(
        models: list, feature_cols: list[str]
    ) -> Optional[pd.DataFrame]:
        gains = []
        for model in models:
            booster = getattr(model, "booster_", None)
            if booster is None:
                continue
            gain = booster.feature_importance(importance_type="gain")
            gains.append(gain)
        if not gains:
            return None
        arr = np.vstack(gains)
        df = pd.DataFrame(arr, columns=feature_cols)
        df.loc["mean"] = df.mean(axis=0)
        return df
