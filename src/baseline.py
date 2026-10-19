"""Logistic regression baseline on the application table.

Run it with ``python run.py baseline`` or ``python -m src.baseline``. It trains on the train
split only, scores the validation split, and writes reports/metrics.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import load_split  # noqa: F401  (you will use these in run_baseline)
from src.metrics import evaluate  # noqa: F401
from src.utils import DEFAULT_CONFIG, load_config, set_seed  # noqa: F401

# PROVIDED
DAYS_EMPLOYED_SENTINEL = 365243
METRICS_KEYS = (
    "model", "roc_auc", "pr_auc", "ks", "n_train", "n_valid", "valid_default_rate", "seed",
)
MODEL_NAME = "logistic_regression"


# TODO(student)
def clean_application(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the known DAYS_EMPLOYED data problem and return a cleaned copy.

    In DAYS_EMPLOYED, the value ``DAYS_EMPLOYED_SENTINEL`` (365243, about 1,000 years) is
    a placeholder, not a real number of days.

    Args:
        df: Application rows. TARGET may or may not be present.

    Returns:
        A new DataFrame with the same rows where
        - DAYS_EMPLOYED holds NaN wherever it held the sentinel, and
        - a new integer column DAYS_EMPLOYED_ANOM is 1 where the sentinel was and 0 elsewhere.
        Every other column, including TARGET when present, is unchanged.

    Hints:
        - Do not change the DataFrame you were given. The tests check this.
        - Build the indicator before you overwrite the sentinel, or you lose the information.
        - Use the module constant, not a typed-in number.
        - It must work on frames without TARGET too.
    """
    cleaned = df.copy()
    anomaly = cleaned["DAYS_EMPLOYED"].eq(DAYS_EMPLOYED_SENTINEL)
    cleaned["DAYS_EMPLOYED_ANOM"] = anomaly.astype("int64")
    cleaned.loc[anomaly, "DAYS_EMPLOYED"] = float("nan")
    return cleaned


# TODO(student)
def get_feature_columns(df: pd.DataFrame, config: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Decide which columns go into the model, split into numeric and categorical lists.

    Args:
        df: Application rows (after ``clean_application``).
        config: The loaded config. ``config["id_col"]`` and ``config["target_col"]`` name
            the id and target columns.

    Returns:
        ``(numeric_cols, categorical_cols)``: two lists of column names, in the order the
        columns appear in ``df``. Together they hold every column except the id and the target.
        Numeric means numbers or booleans; everything else (text, category) is categorical.

    Hints:
        - Which two columns must never be features, and why?
        - pandas has helpers in ``pd.api.types`` that check a column's dtype.
        - With pandas 3, text columns have dtype ``str``, not ``object``. Test for "numeric",
          and treat everything else as categorical.
    """
    id_col = config["id_col"]
    target_col = config["target_col"]
    numeric_cols = []
    categorical_cols = []
    for column in df.columns:
        if column in (id_col, target_col):
            continue
        is_numeric = pd.api.types.is_numeric_dtype(df[column].dtype)
        is_boolean = pd.api.types.is_bool_dtype(df[column].dtype)
        if is_numeric or is_boolean:
            numeric_cols.append(column)
        else:
            categorical_cols.append(column)
    return numeric_cols, categorical_cols


# TODO(student)
def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """Build the (unfitted) preprocessing step for the model.

    Args:
        numeric_cols: Columns to treat as numbers.
        categorical_cols: Columns to treat as categories.

    Returns:
        A ColumnTransformer that
        - fills missing numeric values with the column median and then standardizes them, and
        - fills missing categorical values with the most frequent value and then one-hot
          encodes them, ignoring categories it did not see during fitting.
        Its output must contain no missing values.

    Hints:
        - Chain the two steps for each column type with a Pipeline.
        - Validation data can contain a category that never appears in training.
        - Return the transformer unfitted; fitting happens later, on training data only.
    """
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric, numeric_cols),
        ("categorical", categorical, categorical_cols),
    ])


# TODO(student)
def build_model(config: dict[str, Any], preprocessor: ColumnTransformer) -> Pipeline:
    """Build the full (unfitted) model: preprocessing followed by logistic regression.

    Args:
        config: The loaded config. Read the model settings from ``config["baseline"]``
            and the random seed from ``config["seed"]``.
        preprocessor: The output of ``build_preprocessor``.

    Returns:
        A Pipeline whose last step is a LogisticRegression. It must support
        ``fit`` and ``predict_proba``.

    Hints:
        - Take every setting from the config, so a config change needs no code change.
        - Give each step a short, readable name.
    """
    settings = config["baseline"]
    classifier = LogisticRegression(
        max_iter=int(settings["max_iter"]),
        class_weight=settings.get("class_weight"),
        random_state=int(config["seed"]),
    )
    return Pipeline([("preprocessor", preprocessor), ("model", classifier)])


# TODO(student)
def write_metrics(metrics: dict[str, float], extra: dict[str, Any], path: str | Path) -> None:
    """Write the run's results to a JSON file.

    Args:
        metrics: The output of ``src.metrics.evaluate``.
        extra: The other values to record: model, n_train, n_valid, valid_default_rate, seed.
        path: Where to write the JSON file.

    The file holds one JSON object with every key in ``METRICS_KEYS``.

    Raises:
        ValueError: if any key in ``METRICS_KEYS`` is missing from metrics and extra combined.

    Hints:
        - The folder may not exist yet on a fresh clone.
        - Indent the JSON so a person can read it.
        - NumPy numbers are not JSON-serializable; make sure the values are plain Python types.
    """
    combined = {**metrics, **extra}
    missing = set(METRICS_KEYS) - set(combined)
    if missing:
        raise ValueError(f"Missing metrics keys: {sorted(missing)}")
    output = {key: combined[key] for key in METRICS_KEYS}
    for key, value in output.items():
        if hasattr(value, "item"):
            output[key] = value.item()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n")


# TODO(student)
def run_baseline(config: dict[str, Any]) -> dict[str, Any]:
    """Run the whole baseline: load, clean, train, evaluate, and save the results.

    Steps:
        1. Seed the random generators with ``config["seed"]``.
        2. Load the fixed train and validation frames (``load_split``).
        3. Clean both with ``clean_application``.
        4. Pick the feature columns from the training frame (``get_feature_columns``).
        5. Build the model (``build_preprocessor`` + ``build_model``) and fit it on the
           training rows only.
        6. Score the validation rows with ``predict_proba`` (the probability of TARGET = 1).
        7. Compute the metrics with ``src.metrics.evaluate``.
        8. Save them with ``write_metrics`` to ``config["paths"]["metrics_file"]``, adding
           model (``MODEL_NAME``), n_train, n_valid, valid_default_rate, and seed.

    Args:
        config: The loaded config.

    Returns:
        A dict with every key in ``METRICS_KEYS`` (the same values written to the file).

    Hints:
        - Nothing may be fitted on the validation rows: not the imputer, the scaler, the
          encoder, or the model.
        - ``predict_proba`` returns one column per class. Which one do you need?
        - The model must see the same columns, in the same order, at fit and at predict time.
        - Never pass TARGET in as a feature.
    """
    set_seed(int(config["seed"]))
    train, valid = load_split(config)
    train = clean_application(train)
    valid = clean_application(valid)
    numeric_cols, categorical_cols = get_feature_columns(train, config)
    feature_cols = numeric_cols + categorical_cols
    model = build_model(config, build_preprocessor(numeric_cols, categorical_cols))
    model.fit(train[feature_cols], train[config["target_col"]])
    scores = model.predict_proba(valid[feature_cols])[:, 1]
    metrics = evaluate(valid[config["target_col"]], scores)
    extra = {
        "model": MODEL_NAME,
        "n_train": int(len(train)),
        "n_valid": int(len(valid)),
        "valid_default_rate": float(valid[config["target_col"]].mean()),
        "seed": int(config["seed"]),
    }
    write_metrics(metrics, extra, config["paths"]["metrics_file"])
    return {**metrics, **extra}

# PROVIDED
def main() -> None:
    """Command-line entry point: ``python -m src.baseline [--config PATH]``."""
    parser = argparse.ArgumentParser(description="Train and evaluate the baseline model.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    results = run_baseline(config)
    print(json.dumps(results, indent=2))
    print(f"Wrote {config['paths']['metrics_file']}")


if __name__ == "__main__":
    main()
