from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from utils import (
    build_prediction_output,
    calculate_metrics,
    model_filename,
    predictions_from_probability,
    prepare_uploaded_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model"


@pytest.fixture(scope="session")
def metadata() -> dict:
    return json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def test_frame() -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / "test_data.csv")


def test_required_repository_artifacts_exist(metadata: dict) -> None:
    required_paths = [
        PROJECT_ROOT / "app.py",
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "test_data.csv",
        PROJECT_ROOT / "metrics.csv",
        PROJECT_ROOT / "cv_metrics.csv",
        PROJECT_ROOT / "notebooks" / "telco_churn_experiments.ipynb",
        MODEL_DIR / "metadata.json",
        MODEL_DIR / "evaluation_details.json",
    ]
    required_paths.extend(
        MODEL_DIR / filename for filename in metadata["models"]["files"].values()
    )

    missing = [str(path) for path in required_paths if not path.exists()]
    assert not missing, f"Missing required artifacts: {missing}"


def test_test_data_hash_matches_metadata(metadata: dict) -> None:
    path = PROJECT_ROOT / "test_data.csv"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == metadata["split"]["test_data_sha256"]


def test_upload_schema_is_valid(metadata: dict, test_frame: pd.DataFrame) -> None:
    prepared = prepare_uploaded_data(test_frame, metadata)
    assert prepared.features.shape == (metadata["split"]["test_rows"], 19)
    assert prepared.target is not None
    assert set(prepared.target.unique()) == {0, 1}
    assert prepared.identifiers is not None
    assert prepared.warnings == []


def test_all_models_reproduce_saved_metrics(
    metadata: dict,
    test_frame: pd.DataFrame,
) -> None:
    prepared = prepare_uploaded_data(test_frame, metadata)
    assert prepared.target is not None

    expected_table = pd.read_csv(PROJECT_ROOT / "metrics.csv").set_index("Model")

    for model_name in metadata["models"]["order"]:
        pipeline = joblib.load(MODEL_DIR / model_filename(metadata, model_name))
        probability = pipeline.predict_proba(prepared.features)[:, 1]
        prediction = predictions_from_probability(probability, 0.50)
        observed = calculate_metrics(prepared.target, prediction, probability)

        for metric_name in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]:
            assert observed[metric_name] is not None
            assert np.isclose(
                observed[metric_name],
                expected_table.loc[model_name, metric_name],
                rtol=0,
                atol=1e-12,
            ), f"{model_name} {metric_name} mismatch"


def test_unlabelled_upload_produces_predictions(
    metadata: dict,
    test_frame: pd.DataFrame,
) -> None:
    unlabelled = test_frame.drop(columns=[metadata["schema"]["target_column"]])
    prepared = prepare_uploaded_data(unlabelled, metadata)
    assert prepared.target is None

    model_name = metadata["models"]["order"][0]
    pipeline = joblib.load(MODEL_DIR / model_filename(metadata, model_name))
    probability = pipeline.predict_proba(prepared.features)[:, 1]
    prediction = predictions_from_probability(probability, 0.50)
    output = build_prediction_output(prepared, prediction, probability)

    assert len(output) == len(unlabelled)
    assert "Predicted_Churn" in output.columns
    assert "Churn_Probability" in output.columns
    assert "Actual_Churn" not in output.columns


def test_missing_feature_is_rejected(metadata: dict, test_frame: pd.DataFrame) -> None:
    broken = test_frame.drop(columns=[metadata["schema"]["feature_order"][0]])
    with pytest.raises(ValueError, match="missing required feature columns"):
        prepare_uploaded_data(broken, metadata)


def test_unknown_category_is_warned_not_crashed(
    metadata: dict,
    test_frame: pd.DataFrame,
) -> None:
    changed = test_frame.head(3).copy()
    changed.loc[0, "PaymentMethod"] = "A new payment method"
    prepared = prepare_uploaded_data(changed, metadata)
    assert any("Unknown category values" in warning for warning in prepared.warnings)
