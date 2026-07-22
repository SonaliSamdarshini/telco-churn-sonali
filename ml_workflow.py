"""Reusable training workflow for the Telco Customer Churn assignment.

The notebook and command-line training script both call the functions in this
module. Keeping the logic in one place prevents the notebook, saved models, and
Streamlit application from silently using different preprocessing rules.
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
POSITIVE_CLASS = 1
POSITIVE_LABEL = "Yes"
NEGATIVE_LABEL = "No"
DATASET_SOURCE = "https://www.kaggle.com/datasets/blastchar/telco-customer-churn"

NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

MODEL_ORDER = [
    "Logistic Regression",
    "Decision Tree",
    "kNN",
    "Naive Bayes",
    "Random Forest",
]

MODEL_FILENAMES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
}


def load_telco_dataset(csv_path: Path) -> pd.DataFrame:
    """Load the source CSV and perform schema-level checks."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    frame.columns = [str(column).strip() for column in frame.columns]

    required = {ID_COLUMN, TARGET_COLUMN, "TotalCharges", *NUMERIC_FEATURES}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    if frame.empty:
        raise ValueError("The dataset is empty.")

    return frame


def clean_and_split_columns(
    raw_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Return model features, encoded target, IDs, and a cleaned audit frame."""
    frame = raw_frame.copy()

    for column in frame.select_dtypes(include=["object", "string"]).columns:
        frame[column] = frame[column].map(
            lambda value: str(value).strip() if pd.notna(value) else np.nan
        )

    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")

    allowed_targets = {NEGATIVE_LABEL, POSITIVE_LABEL}
    observed_targets = set(frame[TARGET_COLUMN].dropna().unique())
    if not observed_targets.issubset(allowed_targets):
        raise ValueError(
            "Unexpected target labels found: "
            f"{sorted(observed_targets.difference(allowed_targets))}"
        )

    target = frame[TARGET_COLUMN].map({NEGATIVE_LABEL: 0, POSITIVE_LABEL: 1})
    if target.isna().any():
        raise ValueError("The target column contains missing or unmapped values.")

    identifiers = frame[ID_COLUMN].astype(str)
    features = frame.drop(columns=[ID_COLUMN, TARGET_COLUMN])

    categorical_features = [
        column for column in features.columns if column not in NUMERIC_FEATURES
    ]
    for column in categorical_features:
        features[column] = features[column].map(
            lambda value: str(value).strip() if pd.notna(value) else np.nan
        )

    return features, target.astype(int), identifiers, frame


def split_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    identifiers: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Create one fixed stratified 80/20 split shared by every model."""
    train_index, test_index = train_test_split(
        np.arange(len(features)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    x_train = features.iloc[train_index].reset_index(drop=True)
    x_test = features.iloc[test_index].reset_index(drop=True)
    y_train = target.iloc[train_index].reset_index(drop=True)
    y_test = target.iloc[test_index].reset_index(drop=True)
    id_train = identifiers.iloc[train_index].reset_index(drop=True)
    id_test = identifiers.iloc[test_index].reset_index(drop=True)

    return x_train, x_test, y_train, y_test, id_train, id_test


def build_preprocessor(
    feature_columns: list[str],
    *,
    scale_numeric: bool,
) -> ColumnTransformer:
    """Build a fresh preprocessing graph for one model pipeline."""
    numeric_features = [
        column for column in NUMERIC_FEATURES if column in feature_columns
    ]
    categorical_features = [
        column for column in feature_columns if column not in numeric_features
    ]

    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model_pipelines(feature_columns: list[str]) -> dict[str, Pipeline]:
    """Create independent preprocessing-plus-estimator pipelines."""
    return {
        "Logistic Regression": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(feature_columns, scale_numeric=True),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        C=2.0,
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(feature_columns, scale_numeric=False),
                ),
                (
                    "classifier",
                    DecisionTreeClassifier(
                        max_depth=5,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "kNN": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(feature_columns, scale_numeric=True),
                ),
                (
                    "classifier",
                    KNeighborsClassifier(
                        n_neighbors=31,
                        weights="uniform",
                    ),
                ),
            ]
        ),
        "Naive Bayes": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(feature_columns, scale_numeric=True),
                ),
                ("classifier", GaussianNB(var_smoothing=1e-9)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(feature_columns, scale_numeric=False),
                ),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=8,
                        min_samples_leaf=3,
                        max_features="sqrt",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def calculate_binary_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_probability: np.ndarray,
) -> dict[str, float]:
    """Calculate the six metrics required by the assignment."""
    if len(np.unique(y_true)) >= 2:
        auc = float(roc_auc_score(y_true, y_probability))
    else:
        auc = float("nan")

    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": auc,
        "Precision": float(
            precision_score(y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0)
        ),
        "Recall": float(
            recall_score(y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0)
        ),
        "F1": float(
            f1_score(y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0)
        ),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }


def cross_validate_models(
    pipelines: dict[str, Pipeline],
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.DataFrame:
    """Measure training stability with stratified five-fold cross-validation."""
    scoring = {
        "Accuracy": make_scorer(accuracy_score),
        "AUC": "roc_auc",
        "Precision": make_scorer(
            precision_score,
            pos_label=POSITIVE_CLASS,
            zero_division=0,
        ),
        "Recall": make_scorer(
            recall_score,
            pos_label=POSITIVE_CLASS,
            zero_division=0,
        ),
        "F1": make_scorer(
            f1_score,
            pos_label=POSITIVE_CLASS,
            zero_division=0,
        ),
        "MCC": make_scorer(matthews_corrcoef),
    }
    folds = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    rows: list[dict[str, float | str]] = []
    for model_name in MODEL_ORDER:
        scores = cross_validate(
            pipelines[model_name],
            x_train,
            y_train,
            cv=folds,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
        )
        row: dict[str, float | str] = {"Model": model_name}
        for metric_name in scoring:
            values = scores[f"test_{metric_name}"]
            row[f"{metric_name}_Mean"] = float(np.mean(values))
            row[f"{metric_name}_Std"] = float(np.std(values, ddof=1))
        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_and_save_models(
    pipelines: dict[str, Pipeline],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    model_directory: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit each model, evaluate the untouched test set, and save pipelines."""
    model_directory.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, float | str]] = []
    evaluation_details: dict[str, Any] = {}

    for model_name in MODEL_ORDER:
        pipeline = pipelines[model_name]
        pipeline.fit(x_train, y_train)

        predicted = pipeline.predict(x_test)
        probability = pipeline.predict_proba(x_test)[:, POSITIVE_CLASS]
        metrics = calculate_binary_metrics(y_test, predicted, probability)

        metric_rows.append({"Model": model_name, **metrics})
        evaluation_details[model_name] = {
            "metrics": metrics,
            "confusion_matrix": confusion_matrix(
                y_test,
                predicted,
                labels=[0, 1],
            ).tolist(),
            "classification_report": classification_report(
                y_test,
                predicted,
                labels=[0, 1],
                target_names=["No Churn", "Churn"],
                output_dict=True,
                zero_division=0,
            ),
        }

        destination = model_directory / MODEL_FILENAMES[model_name]
        joblib.dump(pipeline, destination, compress=3)

        reloaded = joblib.load(destination)
        reloaded_prediction = reloaded.predict(x_test.iloc[:5])
        if not np.array_equal(predicted[:5], reloaded_prediction):
            raise RuntimeError(f"Reload verification failed for {model_name}.")

    return pd.DataFrame(metric_rows), evaluation_details


def export_test_data(
    x_test: pd.DataFrame,
    y_test: pd.Series,
    test_identifiers: pd.Series,
    output_path: Path,
) -> str:
    """Save the exact held-out records used for assignment evaluation."""
    test_frame = x_test.copy()
    test_frame.insert(0, ID_COLUMN, test_identifiers.astype(str))
    test_frame[TARGET_COLUMN] = y_test.map({0: NEGATIVE_LABEL, 1: POSITIVE_LABEL})
    test_frame.to_csv(output_path, index=False)

    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    return digest


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _build_observations(metrics_frame: pd.DataFrame) -> dict[str, str]:
    by_model = metrics_frame.set_index("Model")
    best_auc = by_model["AUC"].idxmax()
    best_recall = by_model["Recall"].idxmax()
    best_precision = by_model["Precision"].idxmax()

    ranking_columns = ["Accuracy", "AUC", "F1", "MCC"]
    mean_rank = by_model[ranking_columns].rank(ascending=False).mean(axis=1)
    overall_winner = str(mean_rank.idxmin())

    observations = {
        "Logistic Regression": (
            "Strong balanced baseline with the highest combined rank across "
            "accuracy, F1, and MCC. Its coefficients are also easier to explain "
            "than tree ensembles."
        ),
        "Decision Tree": (
            "Competitive and interpretable, but a single tree is more sensitive "
            "to the chosen depth and can be less stable than an ensemble."
        ),
        "kNN": (
            "Scaling is essential because prediction is distance based. It gives "
            "competitive F1 but stores the training observations and is slower at "
            "prediction time than parametric models."
        ),
        "Naive Bayes": (
            f"Produces the highest recall ({by_model.loc['Naive Bayes', 'Recall']:.4f}) "
            "but many more false positives, which lowers precision and accuracy."
        ),
        "Random Forest": (
            f"Provides the best AUC ({by_model.loc['Random Forest', 'AUC']:.4f}) "
            "and strong precision while reducing the variance of a single tree."
        ),
        "Overall Winner": (
            f"{overall_winner}. It has the best average rank over Accuracy, AUC, "
            "F1, and MCC. Random Forest remains attractive when ranking quality "
            "measured by AUC is the main priority."
        ),
        "Best AUC Model": str(best_auc),
        "Best Recall Model": str(best_recall),
        "Best Precision Model": str(best_precision),
    }
    return observations


def build_metadata(
    cleaned_frame: pd.DataFrame,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    metrics_frame: pd.DataFrame,
    test_sha256: str,
) -> dict[str, Any]:
    """Create schema and provenance information used by the app."""
    categorical_features = [
        column for column in x_train.columns if column not in NUMERIC_FEATURES
    ]

    category_values = {
        column: sorted(
            str(value)
            for value in x_train[column].dropna().astype(str).unique().tolist()
        )
        for column in categorical_features
    }

    numeric_summary = {}
    for column in NUMERIC_FEATURES:
        numeric_series = pd.to_numeric(x_train[column], errors="coerce")
        numeric_summary[column] = {
            "min": float(numeric_series.min()),
            "max": float(numeric_series.max()),
            "median": float(numeric_series.median()),
        }

    target_counts = cleaned_frame[TARGET_COLUMN].value_counts().to_dict()
    observations = _build_observations(metrics_frame)

    return {
        "project": {
            "title": "Customer Retention Radar",
            "student_name": "Sonali Samdarshini",
            "bits_id": "2025AC05254",
            "assignment": "Machine Learning Assignment 2",
        },
        "dataset": {
            "name": "Telco Customer Churn",
            "source_url": DATASET_SOURCE,
            "raw_rows": int(len(cleaned_frame)),
            "raw_columns": int(cleaned_frame.shape[1]),
            "model_feature_count": int(x_train.shape[1]),
            "target_counts": {str(key): int(value) for key, value in target_counts.items()},
            "blank_total_charges": int(cleaned_frame["TotalCharges"].isna().sum()),
            "duplicate_rows": int(cleaned_frame.duplicated().sum()),
        },
        "split": {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "training_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "training_positive_rate": float(y_train.mean()),
            "test_positive_rate": float(y_test.mean()),
            "test_data_sha256": test_sha256,
        },
        "schema": {
            "id_column": ID_COLUMN,
            "target_column": TARGET_COLUMN,
            "feature_order": x_train.columns.tolist(),
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": categorical_features,
            "categorical_values": category_values,
            "numeric_summary": numeric_summary,
            "target_mapping": {NEGATIVE_LABEL: 0, POSITIVE_LABEL: 1},
        },
        "models": {
            "order": MODEL_ORDER,
            "files": MODEL_FILENAMES,
            "default_threshold": 0.50,
            "positive_class": POSITIVE_CLASS,
        },
        "observations": observations,
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    }


def run_training(project_root: Path, *, run_cross_validation: bool = True) -> dict[str, Any]:
    """Execute the full reproducible training workflow and write all artifacts."""
    project_root = project_root.resolve()
    data_path = project_root / "data" / "telco_churn.csv"
    model_directory = project_root / "model"
    metrics_path = project_root / "metrics.csv"
    cv_metrics_path = project_root / "cv_metrics.csv"
    test_data_path = project_root / "test_data.csv"

    raw_frame = load_telco_dataset(data_path)
    features, target, identifiers, cleaned_frame = clean_and_split_columns(raw_frame)
    x_train, x_test, y_train, y_test, _, id_test = split_dataset(
        features,
        target,
        identifiers,
    )

    test_sha256 = export_test_data(x_test, y_test, id_test, test_data_path)
    pipelines = build_model_pipelines(x_train.columns.tolist())

    if run_cross_validation:
        cv_metrics = cross_validate_models(pipelines, x_train, y_train)
        cv_metrics.to_csv(cv_metrics_path, index=False)
    else:
        cv_metrics = pd.DataFrame()
        if cv_metrics_path.exists():
            cv_metrics_path.unlink()

    metrics_frame, evaluation_details = evaluate_and_save_models(
        pipelines,
        x_train,
        y_train,
        x_test,
        y_test,
        model_directory,
    )
    metrics_frame.to_csv(metrics_path, index=False)

    metadata = build_metadata(
        cleaned_frame,
        x_train,
        x_test,
        y_train,
        y_test,
        metrics_frame,
        test_sha256,
    )

    metadata_path = model_directory / "metadata.json"
    evaluation_path = model_directory / "evaluation_details.json"
    metadata_path.write_text(
        json.dumps(_json_safe(metadata), indent=2),
        encoding="utf-8",
    )
    evaluation_path.write_text(
        json.dumps(_json_safe(evaluation_details), indent=2),
        encoding="utf-8",
    )

    return {
        "metrics": metrics_frame,
        "cv_metrics": cv_metrics,
        "metadata": metadata,
        "evaluation_details": evaluation_details,
        "test_data_path": test_data_path,
        "model_directory": model_directory,
    }
