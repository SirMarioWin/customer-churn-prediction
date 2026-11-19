from __future__ import annotations

import json

import pandas as pd
import pytest

from churn_prediction.cli import main
from churn_prediction.pipeline import prepare_features, train_and_evaluate


def make_dataset(rows: int = 60) -> pd.DataFrame:
    """Build a small Telco-shaped fixture with numeric strings and blank charges."""
    return pd.DataFrame(
        {
            "customerID": [f"customer-{index:04d}" for index in range(rows)],
            "tenure": [str(index % 72) for index in range(rows)],
            "MonthlyCharges": [f"{35 + (index % 60) * 0.75:.2f}" for index in range(rows)],
            "TotalCharges": [" " if index == 0 else f"{index * 42.5:.2f}" for index in range(rows)],
            "InternetService": ["Fiber optic" if index % 2 else "DSL" for index in range(rows)],
            "Churn": ["Yes" if index % 3 == 0 else "No" for index in range(rows)],
        }
    )


def test_prepare_features_encodes_target_and_drops_identifier() -> None:
    frame = make_dataset(12)

    features, labels, target_column = prepare_features(frame, target="churn")

    assert target_column == "Churn"
    assert labels.tolist() == [1 if index % 3 == 0 else 0 for index in range(12)]
    assert "customerID" not in features.columns
    assert pd.api.types.is_numeric_dtype(features["TotalCharges"])
    assert pd.isna(features.loc[0, "TotalCharges"])
    assert frame.loc[0, "TotalCharges"] == " "  # preparation does not mutate caller data


def test_prepare_features_rejects_unknown_target_labels() -> None:
    frame = make_dataset(12)
    frame.loc[0, "Churn"] = "maybe"

    with pytest.raises(ValueError, match="unrecognized values"):
        prepare_features(frame)


def test_prepare_features_rejects_missing_target_column() -> None:
    frame = make_dataset(12).drop(columns="Churn")

    with pytest.raises(ValueError, match="was not found"):
        prepare_features(frame)


def test_train_and_evaluate_writes_metrics_model_and_private_predictions(tmp_path) -> None:
    csv_path = tmp_path / "telco.csv"
    output_dir = tmp_path / "artifacts"
    make_dataset().to_csv(csv_path, index=False)

    metrics = train_and_evaluate(csv_path, output_dir=output_dir)

    assert metrics["rows"] == 60
    assert metrics["test_rows"] == 12
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert len(metrics["confusion_matrix"]) == 2
    assert all(len(row) == 2 for row in metrics["confusion_matrix"])
    assert sum(sum(row) for row in metrics["confusion_matrix"]) == metrics["test_rows"]
    assert (output_dir / "model.joblib").is_file()
    assert json.loads((output_dir / "metrics.json").read_text()) == metrics

    predictions = pd.read_csv(output_dir / "test_predictions.csv")
    assert list(predictions.columns) == [
        "actual_churn",
        "predicted_churn",
        "churn_probability",
    ]
    assert len(predictions) == metrics["test_rows"]
    assert predictions["churn_probability"].between(0, 1).all()


def test_cli_trains_and_prints_summary(tmp_path, capsys) -> None:
    csv_path = tmp_path / "telco.csv"
    make_dataset().to_csv(csv_path, index=False)
    output_dir = tmp_path / "cli-output"

    exit_code = main(["train", "--data", str(csv_path), "--output-dir", str(output_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert '"roc_auc"' in captured.out
    assert (output_dir / "metrics.json").exists()


def test_training_rejects_single_class_target(tmp_path) -> None:
    csv_path = tmp_path / "one-class.csv"
    frame = make_dataset()
    frame["Churn"] = "No"
    frame.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="both churn classes"):
        train_and_evaluate(csv_path, output_dir=tmp_path / "artifacts")
