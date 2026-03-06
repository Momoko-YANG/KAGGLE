"""Configuration dataclasses for the Mitsui MLE project."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Tuple


@dataclass(slots=True)
class MissingValuePolicy:
    """Configuration for the missing value handling."""

    discontinued_threshold: float = 0.80
    discontinued_strategy: str = "fill_last"  # fill_last|indicator|fill_zero|drop
    regular_strategy: str = "forward_only"    # forward_only|forward_backward|zero


@dataclass(slots=True)
class FeatureConfig:
    """Configuration for feature engineering."""

    windows: Tuple[int, ...] = (5, 10, 20, 60, 120, 252)
    ewma_span: int = 10
    corr_window: int = 20
    top_features: int = 64


@dataclass(slots=True)
class TrainingConfig:
    """Training related configuration."""

    folds: int = 3
    seed: int = 42
    lgbm_params: dict[str, float | int | str] = field(
        default_factory=lambda: {
            "objective": "lambdarank",
            "metric": "ndcg",
            "boosting_type": "gbdt",
            "num_leaves": 63,
            "learning_rate": 0.03,
            "n_estimators": 600,
            "min_child_samples": 60,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        }
    )


@dataclass(slots=True)
class ProjectConfig:
    """Top level configuration for the MLE project."""

    input_dir: Path = Path(".")
    train_file: str = "train.csv"
    labels_file: str = "train_labels.csv"
    pairs_file: str = "target_pairs.csv"
    bundle_path: Path = Path("artifacts/model_bundle.pkl")
    log_epsilon: float = 1e-8
    missing: MissingValuePolicy = field(default_factory=MissingValuePolicy)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def resolve_path(self, relative: str) -> Path:
        """Resolve a path within the configured input directory."""

        return (self.input_dir / relative).resolve()
