"""Command line entry-point for the Mitsui MLE project."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from src.mle_project import MitsuiMLEPipeline, ProjectConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mitsui MLE project helper")
    parser.add_argument("mode", choices={"train", "predict"}, help="Pipeline mode")
    parser.add_argument("--input-dir", dest="input_dir", default=".", help="Data directory")
    parser.add_argument(
        "--bundle", dest="bundle", default="artifacts/model_bundle.pkl", help="Model bundle path"
    )
    parser.add_argument("--target", dest="target", default="target", help="Target column name")
    parser.add_argument(
        "--features", dest="features", default=None, help="Path to raw features for prediction"
    )
    parser.add_argument(
        "--output", dest="output", default="predictions.csv", help="Prediction output path"
    )
    return parser


def run_train(args: argparse.Namespace) -> None:
    config = ProjectConfig(input_dir=Path(args.input_dir), bundle_path=Path(args.bundle))
    pipeline = MitsuiMLEPipeline(config)
    pipeline.prepare_data()
    result = pipeline.train(args.target)
    pipeline.save_artifacts()
    print(f"Training complete. Logistic AUC: {result.metrics['logistic_auc']:.4f}")
    print(f"Model bundle saved to {config.bundle_path}")


def run_predict(args: argparse.Namespace) -> None:
    config = ProjectConfig(input_dir=Path(args.input_dir), bundle_path=Path(args.bundle))
    pipeline = MitsuiMLEPipeline(config)
    pipeline.load_artifacts()
    if args.features is None:
        raw_path = config.resolve_path("test.csv")
    else:
        raw_path = Path(args.features)
    raw_df = pd.read_csv(raw_path, low_memory=False)
    if "date_id" not in raw_df.columns:
        raise ValueError("Prediction features must include a 'date_id' column")
    features = pipeline.transform_raw_features(raw_df)
    preds = pipeline.predict(features)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preds.insert(0, "date_id", features["date_id"].values)
    preds.to_csv(output_path, index=False)
    print(f"Predictions written to {output_path}")


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode == "train":
        run_train(args)
    elif args.mode == "predict":
        run_predict(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
