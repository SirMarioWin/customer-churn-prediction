# Customer Churn Prediction

A small, reproducible **binary-classification baseline** for the IBM Telco Customer Churn dataset. The repository contains the training and evaluation code, tests, and setup instructions; it does **not** include the source dataset or claim that this baseline is ready for production use.

## What it does

- Reads a local Telco churn CSV and validates its binary churn target.
- Removes common customer identifier columns (`customerID`, `customer_id`, or `id`) to avoid learning from row identifiers.
- Converts numeric-looking text (including whitespace in `TotalCharges`), imputes missing values, scales numeric features, and one-hot encodes categorical features inside a scikit-learn pipeline.
- Fits a class-weighted logistic-regression baseline with a deterministic, stratified holdout split.
- Writes a serialized model, holdout metrics, and predictions without exporting customer IDs or feature values.

The reported holdout scores are a starting point, not evidence of business impact. The split is random rather than time-based, and no tuning, calibration, fairness review, or prospective validation is performed. Do not use these scores alone to make customer-level decisions.

## Requirements and setup

Python 3.10 or newer is required. From the repository root, create and activate a virtual environment, then install the project and development tools:

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Obtain the data

The input dataset is not committed because it is a third-party dataset and may have separate access and use terms. Obtain the **IBM Telco Customer Churn** CSV from the [Kaggle dataset page](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), review its current license/terms, and place `WA_Fn-UseC_-Telco-Customer-Churn.csv` at:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The `data/raw/*.csv` ignore rule keeps local source data out of Git. Keep the original file unchanged; use a separate copy for any custom preprocessing. The default target column is `Churn` and accepts `Yes`/`No` or `1`/`0` labels (case and surrounding whitespace do not matter). The canonical Telco file includes columns such as `customerID`, `tenure`, `MonthlyCharges`, and `TotalCharges`; common ID columns are excluded automatically.

## Train and evaluate

Install the package as above, then run:

```bash
churn-predict train \
  --data data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv \
  --output-dir outputs
```

Equivalent module invocation:

```bash
python -m churn_prediction.cli train \
  --data data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Options include `--target`, `--test-size` (strictly between 0 and 0.5), and `--random-state`. The default split is 80% training / 20% test with seed 42. The CLI prints key scores and writes:

- `outputs/model.joblib` — fitted preprocessing and logistic-regression pipeline.
- `outputs/metrics.json` — accuracy, precision, recall, F1, ROC-AUC, average precision, classification report, confusion matrix, and run metadata.
- `outputs/test_predictions.csv` — actual and predicted churn labels plus predicted churn probability; no customer IDs or raw feature values.

Treat `model.joblib` as a trusted-code artifact: joblib uses Python pickle serialization, so **never load a model file from an untrusted source**. Generated output is ignored by Git.

## Tests and code checks

```bash
pytest
ruff check src tests
```

The tests use generated, synthetic records and do not require the Telco dataset. GitHub Actions runs both checks on Python 3.10 and 3.12 for pushes and pull requests.

## Project layout

```text
src/churn_prediction/   validation, preprocessing, training, evaluation, and CLI
tests/                  unit and end-to-end tests using synthetic data
data/raw/               local, untracked input CSV (download separately)
outputs/                generated model and evaluation artifacts (untracked)
```
