"""Validation and evaluation helpers for the Streamlit application."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class PreparedUpload:
    """Validated representation of an uploaded CSV file."""

    features: pd.DataFrame
    target: pd.Series | None
    identifiers: pd.Series | None
    original: pd.DataFrame
    warnings: list[str]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_target(target: pd.Series) -> pd.Series:
    """Accept Yes/No, 1/0, or boolean target values and return integers."""
    mapping = {
        "yes": 1,
        "no": 0,
        "1": 1,
        "0": 0,
        "1.0": 1,
        "0.0": 0,
        "true": 1,
        "false": 0,
        "churn": 1,
        "no churn": 0,
    }

    normalized = target.map(
        lambda value: mapping.get(str(value).strip().lower())
        if pd.notna(value)
        else np.nan
    )

    invalid_mask = normalized.isna()
    if invalid_mask.any():
        invalid_values = sorted(
            str(value) for value in target.loc[invalid_mask].dropna().unique().tolist()
        )
        raise ValueError(
            "The Churn column must contain Yes/No or 1/0 values. "
            f"Invalid values: {invalid_values[:10]}"
        )

    return normalized.astype(int)


def prepare_uploaded_data(
    frame: pd.DataFrame,
    metadata: dict[str, Any],
) -> PreparedUpload:
    """Validate schema, normalize types, and preserve optional IDs/targets."""
    if frame is None or frame.empty:
        raise ValueError("The CSV file is empty.")

    working = frame.copy()
    working.columns = [str(column).strip() for column in working.columns]

    schema = metadata["schema"]
    expected_features = list(schema["feature_order"])
    numeric_features = set(schema["numeric_features"])
    categorical_features = set(schema["categorical_features"])
    id_column = schema["id_column"]
    target_column = schema["target_column"]

    identifiers = None
    if id_column in working.columns:
        identifiers = working[id_column].astype(str).reset_index(drop=True)

    target = None
    if target_column in working.columns:
        target = normalize_target(working[target_column]).reset_index(drop=True)

    available_feature_columns = [
        column for column in working.columns if column not in {id_column, target_column}
    ]
    missing_features = sorted(set(expected_features).difference(available_feature_columns))
    if missing_features:
        raise ValueError(
            "The uploaded CSV is missing required feature columns: "
            + ", ".join(missing_features)
        )

    extra_columns = sorted(set(available_feature_columns).difference(expected_features))
    warnings: list[str] = []
    if extra_columns:
        warnings.append(
            "Extra columns were ignored: " + ", ".join(extra_columns)
        )

    features = working.loc[:, expected_features].copy()

    for column in expected_features:
        if column in numeric_features:
            before_missing = int(features[column].isna().sum())
            features[column] = pd.to_numeric(features[column], errors="coerce")
            after_missing = int(features[column].isna().sum())
            introduced = after_missing - before_missing
            if introduced > 0:
                warnings.append(
                    f"{introduced} value(s) in {column} could not be converted to a "
                    "number and will be median-imputed by the pipeline."
                )
        elif column in categorical_features:
            features[column] = features[column].map(
                lambda value: str(value).strip() if pd.notna(value) else np.nan
            )

    known_categories = schema.get("categorical_values", {})
    unknown_messages = []
    for column in categorical_features:
        allowed = set(str(value) for value in known_categories.get(column, []))
        observed = set(features[column].dropna().astype(str).unique())
        unknown = sorted(observed.difference(allowed))
        if unknown:
            unknown_messages.append(f"{column}: {unknown[:5]}")
    if unknown_messages:
        warnings.append(
            "Unknown category values were found. The one-hot encoder will ignore them: "
            + "; ".join(unknown_messages)
        )

    return PreparedUpload(
        features=features.reset_index(drop=True),
        target=target,
        identifiers=identifiers,
        original=working.reset_index(drop=True),
        warnings=warnings,
    )


def predictions_from_probability(
    probability: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Apply a user-selected probability threshold to binary probabilities."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1.")
    return (np.asarray(probability) >= threshold).astype(int)


def calculate_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float | None]:
    """Calculate the assignment metrics and handle single-class uploads."""
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    probability_array = np.asarray(probability)

    auc: float | None
    if len(np.unique(y_true_array)) >= 2:
        auc = float(roc_auc_score(y_true_array, probability_array))
    else:
        auc = None

    return {
        "Accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "AUC": auc,
        "Precision": float(
            precision_score(y_true_array, y_pred_array, pos_label=1, zero_division=0)
        ),
        "Recall": float(
            recall_score(y_true_array, y_pred_array, pos_label=1, zero_division=0)
        ),
        "F1": float(
            f1_score(y_true_array, y_pred_array, pos_label=1, zero_division=0)
        ),
        "MCC": float(matthews_corrcoef(y_true_array, y_pred_array)),
    }


def confusion_matrix_frame(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(
        matrix,
        index=["Actual No Churn", "Actual Churn"],
        columns=["Predicted No Churn", "Predicted Churn"],
    )


def classification_report_frame(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["No Churn", "Churn"],
        output_dict=True,
        zero_division=0,
    )
    report_frame = pd.DataFrame(report).transpose()
    return report_frame.loc[:, ["precision", "recall", "f1-score", "support"]]


def build_prediction_output(
    prepared: PreparedUpload,
    prediction: np.ndarray,
    probability: np.ndarray,
) -> pd.DataFrame:
    """Return a downloadable table that preserves row identity and actual labels."""
    output = pd.DataFrame(index=prepared.features.index)

    if prepared.identifiers is not None:
        output["customerID"] = prepared.identifiers
    else:
        output["row_number"] = np.arange(1, len(prepared.features) + 1)

    if prepared.target is not None:
        output["Actual_Churn"] = prepared.target.map({0: "No", 1: "Yes"})

    output["Predicted_Churn"] = pd.Series(prediction).map({0: "No", 1: "Yes"})
    output["Churn_Probability"] = np.asarray(probability)
    output["No_Churn_Probability"] = 1.0 - np.asarray(probability)
    return output


def model_filename(metadata: dict[str, Any], model_name: str) -> str:
    files = metadata["models"]["files"]
    if model_name not in files:
        raise KeyError(f"Unknown model: {model_name}")
    return str(files[model_name])
