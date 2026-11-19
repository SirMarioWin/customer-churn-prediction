"""Data validation, preprocessing, training, and evaluation for churn prediction."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

IDENTIFIER_COLUMNS = {"id", "customerid", "customer_id"}
TARGET_LABELS = {
    "yes": 1,
    "no": 0,
    "true": 1,
    "false": 0,
    "1": 1,
    "0": 0,
}


def _resolve_column(frame: pd.DataFrame, requested: str) -> str:
    """Resolve a column name without case sensitivity, requiring one match."""
    matches = [column for column in frame.columns if str(column).casefold() == requested.casefold()]
    if not matches:
        raise ValueError(f"Target column {requested!r} was not found in the CSV.")
    if len(matches) > 1:
        raise ValueError(f"Target column {requested!r} is ambiguous (case-insensitive match).")
    return matches[0]


def _encode_target(values: pd.Series, target_name: str) -> pd.Series:
    if values.isna().any():
        raise ValueError(f"Target column {target_name!r} contains missing values.")

    normalized = values.astype("string").str.strip().str.casefold()
    encoded = normalized.map(TARGET_LABELS)
    if encoded.isna().any():
        bad_values = sorted(normalized[encoded.isna()].unique().tolist())
        raise ValueError(
            f"Target column {target_name!r} must contain yes/no or 1/0 labels; "
            f"unrecognized values: {bad_values[:10]}"
        )
    return encoded.astype("int64")


def prepare_features(
    frame: pd.DataFrame, target: str = "Churn"
) -> tuple[pd.DataFrame, pd.Series, str]:
    """Validate a dataset and prepare features while excluding target and ID fields.

    Numeric-looking string columns (including Telco ``TotalCharges`` values with
    whitespace) are converted to numeric. Empty strings become missing values
    and are imputed inside the model pipeline, so the same preprocessing is
    applied consistently at fit and prediction time.
    """
    if frame.empty:
        raise ValueError("The input CSV contains no rows.")
    if not frame.columns.is_unique:
        raise ValueError("The input CSV contains duplicate column names.")

    target_column = _resolve_column(frame, target)
    y = _encode_target(frame[target_column], target_column)
    X = frame.drop(columns=[target_column]).copy()

    id_columns = [
        column for column in X.columns if str(column).strip().casefold() in IDENTIFIER_COLUMNS
    ]
    X = X.drop(columns=id_columns)

    for column in X.columns:
        values = X[column]
        if pd.api.types.is_bool_dtype(values.dtype):
            X[column] = values.astype("int64")
        elif (
            pd.api.types.is_object_dtype(values.dtype)
            or pd.api.types.is_string_dtype(values.dtype)
            or isinstance(values.dtype, pd.CategoricalDtype)
        ):
            text = values.astype("string").str.strip()
            text = text.mask(text.eq(""), pd.NA)
            present_count = int(text.notna().sum())
            numeric = pd.to_numeric(text, errors="coerce")
            if present_count and int(numeric.notna().sum()) == present_count:
                X[column] = numeric
            else:
                X[column] = text.astype(object).where(text.notna(), np.nan)

    X = X.dropna(axis=1, how="all")
    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain after removing the target and IDs.")

    return X, y, target_column


def _build_model(features: pd.DataFrame, random_state: int) -> Pipeline:
    numeric_columns = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]
    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1_000,
        random_state=random_state,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def train_and_evaluate(
    data_path: str | Path,
    output_dir: str | Path = "outputs",
    target: str = "Churn",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train a reproducible baseline and write metrics, predictions, and model.

    The saved test predictions contain only actual/predicted labels and churn
    probabilities; source identifiers and feature values are not exported.
    """
    if not 0.0 < test_size < 0.5:
        raise ValueError("test_size must be greater than 0 and less than 0.5.")
    if random_state < 0:
        raise ValueError("random_state must be a non-negative integer.")

    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {data_path}")
    frame = pd.read_csv(data_path, low_memory=False)
    features, labels, target_column = prepare_features(frame, target=target)

    class_counts = labels.value_counts()
    if len(class_counts) != 2:
        raise ValueError("The target must contain both churn classes (yes and no).")
    if int(class_counts.min()) < 2:
        raise ValueError("Each churn class needs at least two examples for a stratified split.")
    test_rows = math.ceil(len(labels) * test_size)
    if test_rows < 2 or len(labels) - test_rows < 2:
        raise ValueError("The dataset is too small for the requested train/test split.")

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    model = _build_model(X_train, random_state=random_state)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test).astype(int)
    churn_class_index = list(model.classes_).index(1)
    churn_probability = model.predict_proba(X_test)[:, churn_class_index]

    metrics: dict[str, Any] = {
        "model": "logistic_regression",
        "target_column": target_column,
        "rows": len(frame),
        "features_used": int(features.shape[1]),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "test_size": test_size,
        "random_state": random_state,
        "class_weight": "balanced",
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, churn_probability)),
        "average_precision": float(average_precision_score(y_test, churn_probability)),
        "confusion_matrix_labels": ["no_churn", "churn"],
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["no_churn", "churn"],
            output_dict=True,
            zero_division=0,
        ),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.joblib")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2, sort_keys=True)
        metrics_file.write("\n")
    pd.DataFrame(
        {
            "actual_churn": y_test.to_numpy(dtype=int),
            "predicted_churn": predictions,
            "churn_probability": churn_probability,
        }
    ).to_csv(output_dir / "test_predictions.csv", index=False)

    return metrics
