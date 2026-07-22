"""Interactive Streamlit application for Telco Customer Churn classification."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay

from utils import (
    build_prediction_output,
    calculate_metrics,
    classification_report_frame,
    confusion_matrix_frame,
    load_json,
    model_filename,
    predictions_from_probability,
    prepare_uploaded_data,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
METADATA_PATH = MODEL_DIR / "metadata.json"
METRICS_PATH = BASE_DIR / "metrics.csv"
TEST_DATA_PATH = BASE_DIR / "test_data.csv"

st.set_page_config(
    page_title="Customer Retention Radar",
    page_icon="\U0001F4CA",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f6fbff 0%, #ffffff 32%);
    }
    .hero-card {
        background: linear-gradient(120deg, #073b4c 0%, #0b6e75 56%, #118ab2 100%);
        padding: 1.5rem 1.7rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 10px 28px rgba(7, 59, 76, 0.18);
    }
    .hero-card h1 {
        margin: 0 0 0.25rem 0;
        font-size: 2.15rem;
    }
    .hero-card p {
        margin: 0.25rem 0;
        opacity: 0.94;
    }
    .small-card {
        border: 1px solid #d7e8ee;
        border-radius: 14px;
        padding: 0.9rem 1rem;
        background: #ffffff;
        min-height: 112px;
    }
    .small-card h4 {
        color: #073b4c;
        margin: 0 0 0.35rem 0;
    }
    .small-card p {
        color: #365663;
        margin: 0;
        font-size: 0.94rem;
    }
    .risk-high {
        border-left: 6px solid #d1495b;
        background: #fff4f5;
        padding: 0.8rem 1rem;
        border-radius: 8px;
    }
    .risk-medium {
        border-left: 6px solid #f4a261;
        background: #fff8ef;
        padding: 0.8rem 1rem;
        border-radius: 8px;
    }
    .risk-low {
        border-left: 6px solid #2a9d8f;
        background: #effaf8;
        padding: 0.8rem 1rem;
        border-radius: 8px;
    }
    div[data-testid="stMetric"] {
        border: 1px solid #d7e8ee;
        background: white;
        padding: 0.75rem;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    return load_json(METADATA_PATH)


@st.cache_data(show_spinner=False)
def load_metrics_table() -> pd.DataFrame:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Required file not found: {METRICS_PATH}")
    return pd.read_csv(METRICS_PATH)


@st.cache_data(show_spinner=False)
def load_sample_test_data() -> pd.DataFrame:
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(f"Required file not found: {TEST_DATA_PATH}")
    return pd.read_csv(TEST_DATA_PATH)


@st.cache_resource(show_spinner=False)
def load_model(path_text: str):
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"Saved model not found: {path}")
    return joblib.load(path)


def format_score(value: float | None) -> str:
    return "N/A" if value is None or pd.isna(value) else f"{value:.4f}"


def category_options(
    metadata: dict,
    column: str,
    preferred_order: list[str],
) -> list[str]:
    available = [
        str(value)
        for value in metadata["schema"]["categorical_values"].get(column, [])
    ]
    ordered = [value for value in preferred_order if value in available]
    ordered.extend(value for value in available if value not in ordered)
    return ordered


def render_metric_cards(scores: dict[str, float | None]) -> None:
    first_row = st.columns(3)
    second_row = st.columns(3)
    for container, metric_name in zip(
        first_row + second_row,
        ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
    ):
        container.metric(metric_name, format_score(scores[metric_name]))


def render_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray) -> None:
    matrix = confusion_matrix_frame(y_true, y_pred)
    figure, axis = plt.subplots(figsize=(5.0, 3.8))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix.to_numpy(),
        display_labels=["No Churn", "Churn"],
    )
    display.plot(ax=axis, cmap="Blues", colorbar=False, values_format="d")
    axis.set_title("Confusion Matrix")
    figure.tight_layout()
    st.pyplot(figure, use_container_width=False)
    plt.close(figure)


def evaluate_one_model(model_name: str, prepared, threshold: float) -> dict:
    path = MODEL_DIR / model_filename(metadata, model_name)
    model = load_model(str(path))
    probability = model.predict_proba(prepared.features)[:, 1]
    prediction = predictions_from_probability(probability, threshold)

    result = {
        "model": model,
        "probability": probability,
        "prediction": prediction,
        "metrics": None,
    }
    if prepared.target is not None:
        result["metrics"] = calculate_metrics(
            prepared.target,
            prediction,
            probability,
        )
    return result


try:
    metadata = load_metadata()
    metrics_table = load_metrics_table()
except Exception as exc:
    st.error(
        "The application could not load its required artifacts. "
        "Run `python -m model.train_models` from the repository root first."
    )
    st.exception(exc)
    st.stop()

project = metadata["project"]
dataset = metadata["dataset"]
split = metadata["split"]
model_names = list(metadata["models"]["order"])

st.markdown(
    f"""
    <div class="hero-card">
        <h1>Customer Retention Radar</h1>
        <p>Interactive Telco Customer Churn classification dashboard</p>
        <p><strong>{project['student_name']}</strong> | BITS ID: {project['bits_id']}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Control Panel")
    selected_model = st.selectbox(
        "Select a classification model",
        options=model_names,
        index=0,
        help="The selected saved pipeline is used for evaluation and prediction.",
    )
    threshold = st.slider(
        "Churn probability threshold",
        min_value=0.10,
        max_value=0.90,
        value=float(metadata["models"]["default_threshold"]),
        step=0.01,
        help=(
            "A customer is labelled as churn when the predicted probability is "
            "greater than or equal to this value. AUC does not depend on this threshold."
        ),
    )
    st.divider()
    st.caption("Bundled evaluation data")
    try:
        sample_bytes = TEST_DATA_PATH.read_bytes()
        st.download_button(
            "Download test_data.csv",
            data=sample_bytes,
            file_name="test_data.csv",
            mime="text/csv",
            use_container_width=True,
        )
    except OSError:
        st.warning("The sample test file is unavailable.")
    st.caption(
        "This application loads already-trained pipelines. It never retrains a model "
        "from uploaded test data."
    )

overview_tab, evaluation_tab, single_tab, method_tab = st.tabs(
    [
        "Model Dashboard",
        "Evaluate CSV",
        "Single Customer",
        "Methodology",
    ]
)

with overview_tab:
    fact_columns = st.columns(4)
    fact_columns[0].metric("Dataset rows", f"{dataset['raw_rows']:,}")
    fact_columns[1].metric("Input features", dataset["model_feature_count"])
    fact_columns[2].metric("Held-out test rows", f"{split['test_rows']:,}")
    fact_columns[3].metric("Models compared", len(model_names))

    st.subheader("Assignment metric comparison")
    st.caption(
        "These values were calculated once on the fixed, untouched 20 percent test split "
        "using a 0.50 decision threshold."
    )
    display_metrics = metrics_table.copy()
    numeric_columns = [
        column for column in display_metrics.columns if column != "Model"
    ]
    display_metrics[numeric_columns] = display_metrics[numeric_columns].round(4)
    st.dataframe(display_metrics, use_container_width=True, hide_index=True)

    chart_frame = metrics_table.set_index("Model")[["Accuracy", "AUC", "F1", "MCC"]]
    st.bar_chart(chart_frame, use_container_width=True)

    st.subheader("What the results mean")
    observations = metadata["observations"]
    card_columns = st.columns(2)
    for index, model_name in enumerate(model_names):
        with card_columns[index % 2]:
            st.markdown(
                f"""
                <div class="small-card">
                    <h4>{model_name}</h4>
                    <p>{observations[model_name]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")

    st.success(observations["Overall Winner"])

    with st.expander("How to read the six metrics"):
        st.markdown(
            """
            - **Accuracy:** fraction of all records classified correctly.
            - **AUC:** ability to rank churners above non-churners across all thresholds.
            - **Precision:** among predicted churners, the fraction that truly churned.
            - **Recall:** among actual churners, the fraction the model detected.
            - **F1:** harmonic balance of precision and recall.
            - **MCC:** balanced correlation between actual and predicted classes; useful with class imbalance.
            """
        )

with evaluation_tab:
    st.subheader("Evaluate labelled test data or predict an unlabelled CSV")
    st.write(
        "Use the exact bundled test split or upload another CSV with the same 19 feature "
        "columns. The optional `customerID` and `Churn` columns are handled automatically."
    )

    source_mode = st.radio(
        "Choose the evaluation data source",
        options=["Bundled test_data.csv", "Upload a CSV"],
        horizontal=True,
    )

    input_frame = None
    if source_mode == "Bundled test_data.csv":
        try:
            input_frame = load_sample_test_data()
            st.info(
                f"Loaded the assignment test set containing {len(input_frame):,} rows."
            )
        except Exception as exc:
            st.error(str(exc))
    else:
        uploaded_file = st.file_uploader(
            "Upload CSV test data",
            type=["csv"],
            help="Do not upload the complete training dataset to the deployed app.",
        )
        if uploaded_file is not None:
            try:
                input_frame = pd.read_csv(uploaded_file)
            except Exception as exc:
                st.error(f"The CSV could not be read: {exc}")

    if input_frame is not None:
        try:
            prepared = prepare_uploaded_data(input_frame, metadata)
        except ValueError as exc:
            st.error(str(exc))
        else:
            for warning in prepared.warnings:
                st.warning(warning)

            with st.expander("Preview validated input", expanded=False):
                st.dataframe(prepared.original.head(25), use_container_width=True)
                st.caption(
                    f"Rows: {len(prepared.features):,} | Model features: "
                    f"{prepared.features.shape[1]}"
                )

            selected_result = evaluate_one_model(
                selected_model,
                prepared,
                threshold,
            )

            st.markdown(
                f"### Results for {selected_model} at threshold {threshold:.2f}"
            )

            if prepared.target is not None:
                render_metric_cards(selected_result["metrics"])
                if selected_result["metrics"]["AUC"] is None:
                    st.warning(
                        "AUC is unavailable because the uploaded target contains only one class."
                    )

                left_panel, right_panel = st.columns([1, 1.25])
                with left_panel:
                    render_confusion_matrix(
                        prepared.target,
                        selected_result["prediction"],
                    )
                with right_panel:
                    st.markdown("#### Classification report")
                    report = classification_report_frame(
                        prepared.target,
                        selected_result["prediction"],
                    )
                    st.dataframe(report.round(4), use_container_width=True)
            else:
                st.info(
                    "No `Churn` column was found. Predictions and probabilities are "
                    "available, but evaluation metrics cannot be calculated without labels."
                )

            prediction_output = build_prediction_output(
                prepared,
                selected_result["prediction"],
                selected_result["probability"],
            )
            st.markdown("#### Prediction output")
            prediction_preview = prediction_output.head(50).copy()
            prediction_preview["Churn_Probability"] = prediction_preview[
                "Churn_Probability"
            ].round(4)
            prediction_preview["No_Churn_Probability"] = prediction_preview[
                "No_Churn_Probability"
            ].round(4)
            st.dataframe(
                prediction_preview,
                use_container_width=True,
            )
            st.download_button(
                "Download predictions as CSV",
                data=prediction_output.to_csv(index=False).encode("utf-8"),
                file_name=(
                    selected_model.lower().replace(" ", "_")
                    + "_predictions.csv"
                ),
                mime="text/csv",
            )

            if prepared.target is not None:
                compare_all = st.checkbox(
                    "Compare all five saved models on this uploaded data",
                    value=False,
                )
                if compare_all:
                    comparison_rows = []
                    with st.spinner("Evaluating all saved pipelines..."):
                        for model_name in model_names:
                            result = evaluate_one_model(
                                model_name,
                                prepared,
                                threshold,
                            )
                            comparison_rows.append(
                                {"Model": model_name, **result["metrics"]}
                            )
                    live_comparison = pd.DataFrame(comparison_rows)
                    live_numeric = [
                        column
                        for column in live_comparison.columns
                        if column != "Model"
                    ]
                    live_comparison[live_numeric] = live_comparison[
                        live_numeric
                    ].round(4)
                    st.markdown(
                        f"#### Live all-model comparison at threshold {threshold:.2f}"
                    )
                    st.dataframe(
                        live_comparison,
                        use_container_width=True,
                        hide_index=True,
                    )

with single_tab:
    st.subheader("Single customer churn simulator")
    st.write(
        "Enter one customer's profile. The selected saved pipeline applies the same "
        "imputation, scaling, and one-hot encoding used during training."
    )

    numeric_summary = metadata["schema"]["numeric_summary"]

    yes_no = ["No", "Yes"]
    internet_addon_order = ["No", "Yes", "No internet service"]

    with st.form("single_customer_form"):
        st.markdown("#### Customer profile")
        profile_columns = st.columns(4)
        gender = profile_columns[0].selectbox(
            "Gender",
            category_options(metadata, "gender", ["Female", "Male"]),
        )
        senior_label = profile_columns[1].selectbox(
            "Senior citizen",
            yes_no,
        )
        partner = profile_columns[2].selectbox(
            "Partner",
            category_options(metadata, "Partner", yes_no),
        )
        dependents = profile_columns[3].selectbox(
            "Dependents",
            category_options(metadata, "Dependents", yes_no),
        )

        st.markdown("#### Account and billing")
        account_columns = st.columns(3)
        tenure = account_columns[0].slider(
            "Tenure in months",
            min_value=int(numeric_summary["tenure"]["min"]),
            max_value=int(numeric_summary["tenure"]["max"]),
            value=int(round(numeric_summary["tenure"]["median"])),
        )
        contract = account_columns[1].selectbox(
            "Contract",
            category_options(
                metadata,
                "Contract",
                ["Month-to-month", "One year", "Two year"],
            ),
        )
        paperless = account_columns[2].selectbox(
            "Paperless billing",
            category_options(metadata, "PaperlessBilling", yes_no),
        )

        billing_columns = st.columns(3)
        payment = billing_columns[0].selectbox(
            "Payment method",
            category_options(
                metadata,
                "PaymentMethod",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            ),
        )
        monthly_charges = billing_columns[1].number_input(
            "Monthly charges",
            min_value=float(numeric_summary["MonthlyCharges"]["min"]),
            max_value=float(numeric_summary["MonthlyCharges"]["max"]),
            value=float(numeric_summary["MonthlyCharges"]["median"]),
            step=0.05,
            format="%.2f",
        )
        total_charges = billing_columns[2].number_input(
            "Total charges",
            min_value=0.0,
            max_value=float(numeric_summary["TotalCharges"]["max"]),
            value=float(numeric_summary["TotalCharges"]["median"]),
            step=1.0,
            format="%.2f",
        )

        st.markdown("#### Services")
        service_columns_1 = st.columns(4)
        phone_service = service_columns_1[0].selectbox(
            "Phone service",
            category_options(metadata, "PhoneService", yes_no),
        )
        multiple_lines = service_columns_1[1].selectbox(
            "Multiple lines",
            category_options(
                metadata,
                "MultipleLines",
                ["No", "Yes", "No phone service"],
            ),
        )
        internet_service = service_columns_1[2].selectbox(
            "Internet service",
            category_options(
                metadata,
                "InternetService",
                ["DSL", "Fiber optic", "No"],
            ),
        )
        online_security = service_columns_1[3].selectbox(
            "Online security",
            category_options(metadata, "OnlineSecurity", internet_addon_order),
        )

        service_columns_2 = st.columns(3)
        online_backup = service_columns_2[0].selectbox(
            "Online backup",
            category_options(metadata, "OnlineBackup", internet_addon_order),
        )
        device_protection = service_columns_2[1].selectbox(
            "Device protection",
            category_options(metadata, "DeviceProtection", internet_addon_order),
        )
        tech_support = service_columns_2[2].selectbox(
            "Tech support",
            category_options(metadata, "TechSupport", internet_addon_order),
        )

        service_columns_3 = st.columns(2)
        streaming_tv = service_columns_3[0].selectbox(
            "Streaming TV",
            category_options(metadata, "StreamingTV", internet_addon_order),
        )
        streaming_movies = service_columns_3[1].selectbox(
            "Streaming movies",
            category_options(metadata, "StreamingMovies", internet_addon_order),
        )

        submitted = st.form_submit_button(
            "Predict churn risk",
            use_container_width=True,
        )

    if submitted:
        record = {
            "gender": gender,
            "SeniorCitizen": 1 if senior_label == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }
        single_frame = pd.DataFrame([record]).loc[
            :, metadata["schema"]["feature_order"]
        ]
        single_prepared = prepare_uploaded_data(single_frame, metadata)
        single_result = evaluate_one_model(
            selected_model,
            single_prepared,
            threshold,
        )
        probability = float(single_result["probability"][0])
        predicted_label = "Yes" if single_result["prediction"][0] == 1 else "No"

        result_columns = st.columns([1, 1, 2])
        result_columns[0].metric("Predicted churn", predicted_label)
        result_columns[1].metric("Churn probability", f"{probability:.2%}")
        result_columns[2].progress(
            int(round(probability * 100)),
            text=f"Estimated churn probability: {probability:.2%}",
        )

        if probability >= 0.65:
            risk_class = "risk-high"
            risk_text = "High risk: this profile may deserve proactive retention attention."
        elif probability >= 0.35:
            risk_class = "risk-medium"
            risk_text = "Moderate risk: review service experience and contract options."
        else:
            risk_class = "risk-low"
            risk_text = "Lower risk under the selected model and current inputs."

        st.markdown(
            f'<div class="{risk_class}"><strong>{risk_text}</strong></div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "This is an educational prediction from historical data. It is not a causal "
            "explanation and should not be used as the sole basis for a customer decision."
        )

with method_tab:
    st.subheader("End-to-end engineering workflow")
    st.markdown(
        """
        1. Load the public Telco Customer Churn CSV and verify its schema.
        2. Drop `customerID` from model inputs and convert blank `TotalCharges` values to missing numeric values.
        3. Encode the target as `No = 0` and `Yes = 1`.
        4. Create one fixed stratified 80/20 split. The 20 percent test data is saved as `test_data.csv`.
        5. Fit preprocessing only on training folds: median numeric imputation, most-frequent categorical imputation, one-hot encoding, and model-specific scaling.
        6. Measure training stability with five-fold stratified cross-validation.
        7. Fit five complete pipelines on the training partition and evaluate the untouched test partition.
        8. Save each full pipeline with Joblib. The app only loads these artifacts; it never fits on uploaded test records.
        """
    )

    model_design = pd.DataFrame(
        [
            ["Logistic Regression", "Scaled numeric + one-hot", "C=2.0, max_iter=2000"],
            ["Decision Tree", "Imputed numeric + one-hot", "max_depth=5"],
            ["kNN", "Scaled numeric + one-hot", "n_neighbors=31"],
            ["Gaussian Naive Bayes", "Scaled numeric + one-hot", "var_smoothing=1e-9"],
            ["Random Forest", "Imputed numeric + one-hot", "300 trees, max_depth=8, min_leaf=3"],
        ],
        columns=["Model", "Preprocessing", "Key settings"],
    )
    st.dataframe(model_design, use_container_width=True, hide_index=True)

    st.markdown("#### Rebuild and run locally")
    st.code(
        """python -m model.train_models
python -m streamlit run app.py""",
        language="bash",
    )

    environment = metadata["environment"]
    st.caption(
        "Saved artifacts were trained with Python "
        f"{environment['python']}, pandas {environment['pandas']}, NumPy "
        f"{environment['numpy']}, and scikit-learn {environment['scikit_learn']}."
    )

    with st.expander("Important limitations"):
        st.markdown(
            """
            - The dataset is observational and does not prove that any feature causes churn.
            - Customer behaviour and pricing can change after the historical data was collected.
            - The positive class is less frequent, so accuracy alone can hide missed churners.
            - Unknown categorical values are ignored by the one-hot encoder rather than learned.
            - The default threshold of 0.50 is not a business-optimized retention policy.
            """
        )
