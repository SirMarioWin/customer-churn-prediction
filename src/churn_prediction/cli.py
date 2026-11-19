"""Command-line interface for the churn prediction baseline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from churn_prediction.pipeline import train_and_evaluate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="churn-predict",
        description="Train and evaluate a reproducible customer-churn baseline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train", help="fit a model and report holdout metrics")
    train_parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="path to the Telco churn CSV (for example, data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv)",
    )
    train_parser.add_argument(
        "--target", default="Churn", help="binary target column (default: Churn)"
    )
    train_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="directory for model and evaluation artifacts",
    )
    train_parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="holdout fraction, strictly between 0 and 0.5 (default: 0.2)",
    )
    train_parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="seed for the stratified split and model (default: 42)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        try:
            metrics = train_and_evaluate(
                data_path=args.data,
                output_dir=args.output_dir,
                target=args.target,
                test_size=args.test_size,
                random_state=args.random_state,
            )
        except (FileNotFoundError, OSError, ValueError, pd.errors.ParserError) as error:
            parser.error(str(error))
        summary = {
            key: metrics[key]
            for key in (
                "model",
                "rows",
                "features_used",
                "train_rows",
                "test_rows",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "average_precision",
            )
        }
        summary["artifacts"] = str(args.output_dir)
        print(json.dumps(summary, indent=2))
        return 0

    parser.error("No command selected.")
    return 2
