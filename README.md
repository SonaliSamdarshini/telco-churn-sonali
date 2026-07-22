# Machine Learning Assignment 2 - Telco Customer Churn

**Student:** Sonali Samdarshini  
**BITS ID:** 2025AC05254  
**Programme:** M.Tech (AIML / DSE)  
**Assignment:** Machine Learning Assignment 2  
**Streamlit application name:** Customer Retention Radar

## Submission links

- **GitHub repository:** `REPLACE_WITH_SONALI_GITHUB_REPOSITORY_URL`
- **Live Streamlit app:** `REPLACE_WITH_SONALI_STREAMLIT_APP_URL`

Replace both placeholders after the repository is pushed and the application is deployed.

## A. Problem statement

The objective is to build an end-to-end binary classification system that predicts whether a telecommunications customer will churn. The project implements all five classifiers explicitly listed in the assignment, evaluates them on one common held-out test set, compares six required metrics, saves complete preprocessing-plus-model pipelines, and exposes the results through an interactive Streamlit application.

The practical learning goals are:

1. Clean mixed numeric and categorical data.
2. prevent preprocessing leakage through scikit-learn pipelines.
3. train several classification algorithms on the same split.
4. compare Accuracy, AUC, Precision, Recall, F1, and MCC.
5. save and reload trained pipelines.
6. validate uploaded CSV files in a web interface.
7. deploy the application on Streamlit Community Cloud.

## B. Dataset description

### Dataset source

Kaggle: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

### Dataset shape

- Raw instances: **7,043**
- Raw columns: **21**
- Identifier column: `customerID`
- Target column: `Churn`
- Model input features after removing ID and target: **19**
- Classification type: **Binary**
- Target mapping: `No = 0`, `Yes = 1`

### Target distribution

| Churn class | Count | Percentage |
|---|---:|---:|
| No | 5,174 | 73.46% |
| Yes | 1,869 | 26.54% |

The target is moderately imbalanced, so accuracy is not interpreted alone. Recall, F1, AUC, and MCC are also used.

### Feature groups

Numeric features:

- `SeniorCitizen`
- `tenure`
- `MonthlyCharges`
- `TotalCharges`

The remaining 15 model features are categorical service, account, and demographic variables.

### Data quality and preprocessing

- `customerID` is retained only for traceability in exported predictions and excluded from training.
- `TotalCharges` contains 11 blank strings. They are stripped, converted to numeric, and represented as missing values.
- Numeric missing values are median-imputed inside each pipeline.
- Categorical missing values are most-frequent-imputed inside each pipeline.
- Categorical features are one-hot encoded with `handle_unknown="ignore"`.
- Numeric features are standardized for Logistic Regression, kNN, and Gaussian Naive Bayes.
- Numeric scaling is not applied to Decision Tree and Random Forest because tree splits are not distance based.
- A fresh preprocessor is created for every model. No mutable preprocessor object is shared across pipelines.

### Train-test split

- Split: **80% training / 20% testing**
- Strategy: stratified by `Churn`
- Random state: `42`
- Training rows: **5,634**
- Test rows: **1,409**
- Training churn rate: **26.54%**
- Test churn rate: **26.54%**

The exact held-out rows are saved as root-level `test_data.csv`. They are not used to fit any model. The file SHA256 generated during training is:

```text
2e2634913de349f17ddae5151155d54856aa637e6b0875f274cc590252d8a8ac
```

## C. GitHub repository link

`REPLACE_WITH_SONALI_GITHUB_REPOSITORY_URL`

The repository contains the source dataset, executed notebook, reusable training code, saved model pipelines, test data, metrics, Streamlit application, dependency files, tests, and detailed execution instructions.

## D. Models used

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors Classifier
4. Gaussian Naive Bayes Classifier
5. Random Forest Classifier (Ensemble)

### Model settings

| Model | Preprocessing | Main settings |
|---|---|---|
| Logistic Regression | Median imputation, standard scaling, one-hot encoding | `C=2.0`, `max_iter=2000` |
| Decision Tree | Median imputation, one-hot encoding | `max_depth=5` |
| kNN | Median imputation, standard scaling, one-hot encoding | `n_neighbors=31`, uniform weights |
| Gaussian Naive Bayes | Median imputation, standard scaling, one-hot encoding | `var_smoothing=1e-9` |
| Random Forest | Median imputation, one-hot encoding | 300 trees, `max_depth=8`, `min_samples_leaf=3` |

### Evaluation methodology

All five models use the same fixed training and test partitions. Preprocessing is fitted only inside the training pipeline. Five-fold stratified cross-validation is performed on the training partition as a stability check. The required comparison table below is calculated only on the untouched held-out test data using the default probability threshold of 0.50.

For binary classification:

- AUC uses the predicted probability of class `1` (`Churn = Yes`).
- Precision, Recall, and F1 treat class `1` as the positive class.
- `zero_division=0` is used for safe metric calculation.
- MCC is included because it remains informative when class frequencies are unequal.

## Required model comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8077 | 0.8418 | 0.6604 | 0.5668 | 0.6101 | 0.4859 |
| Decision Tree | 0.7984 | 0.8303 | 0.6347 | 0.5668 | 0.5989 | 0.4662 |
| kNN | 0.7935 | 0.8285 | 0.6131 | 0.6016 | 0.6073 | 0.4672 |
| Naive Bayes | 0.6948 | 0.8074 | 0.4589 | 0.8369 | 0.5928 | 0.4245 |
| Random Forest (Ensemble) | 0.8070 | 0.8444 | 0.6809 | 0.5134 | 0.5854 | 0.4706 |

The unrounded values are available in `metrics.csv`.

## Training cross-validation summary

The following values are five-fold means on the training partition. Standard deviations are available in `cv_metrics.csv`.

| Model | Accuracy mean | AUC mean | Precision mean | Recall mean | F1 mean | MCC mean |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8026 | 0.8462 | 0.6536 | 0.5452 | 0.5938 | 0.4688 |
| Decision Tree | 0.7914 | 0.8289 | 0.6345 | 0.5130 | 0.5649 | 0.4359 |
| kNN | 0.7888 | 0.8330 | 0.6061 | 0.5833 | 0.5944 | 0.4519 |
| Naive Bayes | 0.6967 | 0.8212 | 0.4612 | 0.8482 | 0.5974 | 0.4331 |
| Random Forest | 0.8051 | 0.8475 | 0.6781 | 0.5084 | 0.5807 | 0.4654 |

## Model observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | It provides the best combined balance in the held-out comparison: the highest Accuracy, F1, and MCC, while maintaining a strong AUC. It is also more explainable than the ensemble. Its churn recall is moderate rather than high, so it still misses some churners at the 0.50 threshold. |
| Decision Tree | The depth-limited tree is interpretable and competitive, but its AUC, F1, and MCC are below Logistic Regression. A single tree is more sensitive to depth and sample changes than Random Forest. |
| kNN | Scaling is essential because kNN is distance based. With 31 neighbours it produces the second-highest F1 and the highest recall among Logistic Regression, Decision Tree, kNN, and Random Forest. It stores training examples and is slower at prediction time than a parametric model. |
| Naive Bayes | It has the highest recall (0.8369), so it detects most actual churners. However, precision is only 0.4589, producing many false positives and the lowest accuracy. It may be useful only when missing a churner is much more costly than contacting a non-churner. |
| Random Forest (Ensemble) | It has the best AUC (0.8444) and best precision (0.6809), showing strong ranking quality and fewer false churn alerts. Its recall is lower, so at the default threshold it misses more churners than Logistic Regression, kNN, or Naive Bayes. |
| Overall Winner | **Logistic Regression** is selected because it has the best average rank over Accuracy, AUC, F1, and MCC. Random Forest is a close alternative when AUC and precision are the primary priorities. |

## Streamlit application

The customized UI is named **Customer Retention Radar** and includes:

- model selection dropdown for all five saved pipelines.
- probability threshold slider.
- bundled `test_data.csv` or user CSV upload.
- schema validation and informative error messages.
- support for optional `customerID` and `Churn` columns.
- live Accuracy, AUC, Precision, Recall, F1, and MCC.
- confusion matrix and classification report.
- live comparison of all five models on uploaded labelled data.
- prediction table and downloadable CSV.
- single-customer interactive churn simulator.
- model comparison dashboard and methodology explanation.
- cached loading of saved pipelines.
- no retraining from uploaded test data.

## Repository structure

```text
sonali_telco_churn_complete/
|-- app.py
|-- ml_workflow.py
|-- utils.py
|-- README.md
|-- EXECUTION_GUIDE.md
|-- SUBMISSION_CHECKLIST.md
|-- requirements.txt
|-- requirements-dev.txt
|-- metrics.csv
|-- cv_metrics.csv
|-- test_data.csv
|-- data/
|   `-- telco_churn.csv
|-- model/
|   |-- __init__.py
|   |-- train_models.py
|   |-- logistic_regression.joblib
|   |-- decision_tree.joblib
|   |-- knn.joblib
|   |-- naive_bayes.joblib
|   |-- random_forest.joblib
|   |-- metadata.json
|   `-- evaluation_details.json
|-- notebooks/
|   `-- telco_churn_experiments.ipynb
|-- tests/
|   `-- test_project.py
|-- docs/
|   `-- ORIGINAL_PROJECT_REVIEW.md
`-- .streamlit/
    `-- config.toml
```

## How to run locally

The saved artifacts were generated with Python 3.13. Use the same Python and dependency versions locally and in Streamlit Community Cloud.

### 1. Open a terminal in the repository root

```bash
cd /path/to/sonali_telco_churn_complete
```

### 2. Create and activate a virtual environment

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

### 3. Install application and notebook dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

### 4. Rebuild all model artifacts

```bash
python -m model.train_models
```

This command creates or refreshes:

- `metrics.csv`
- `cv_metrics.csv`
- `test_data.csv`
- five `.joblib` pipelines
- `model/metadata.json`
- `model/evaluation_details.json`

### 5. Open the executed notebook

```bash
jupyter lab
```

Open `notebooks/telco_churn_experiments.ipynb`, select the project virtual environment as the kernel, and run all cells from top to bottom.

### 6. Start the Streamlit application

```bash
python -m streamlit run app.py
```

Open the local URL shown by Streamlit, normally `http://localhost:8501`.

### 7. Run automated checks

```bash
python -m pytest -q
```

## Streamlit Community Cloud deployment

1. Push this repository to Sonali's GitHub account.
2. Sign in to Streamlit Community Cloud with the same GitHub account.
3. Create a new app.
4. Select the repository and `main` branch.
5. Set the entrypoint to `app.py`.
6. In advanced settings, select Python 3.13 to match the saved model environment.
7. Deploy and inspect the build logs.
8. Test every model with the bundled `test_data.csv`.
9. Copy the final `https://...streamlit.app` URL into this README and the submission PDF.

No secrets are required for this project.

## BITS Virtual Lab execution

The assignment requires proof of execution in BITS Virtual Lab. In Sonali's own lab session:

```bash
git clone REPLACE_WITH_SONALI_GITHUB_REPOSITORY_URL
cd REPOSITORY_NAME
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m model.train_models
python -m pytest -q
python -m streamlit run app.py
```

Capture one readable screenshot that visibly includes the BITS Virtual Lab environment, successful execution, and model or Streamlit output.

## Test data description

`test_data.csv` contains exactly the 1,409 held-out observations used for the reported metrics. It contains:

- `customerID` for row traceability.
- all 19 model input features.
- the actual `Churn` label as `Yes` or `No`.

The Streamlit application accepts this file directly. It also accepts unlabelled data with the same 19 features, in which case predictions are shown but evaluation metrics are not available.

## Limitations

- This is an educational assignment, not a production retention system.
- The data is observational and cannot establish causal relationships.
- Historical customer patterns may not represent future customers.
- Unknown category values are ignored by the one-hot encoder.
- A 0.50 threshold is a technical default, not a business-optimized policy.
- Model fairness and cost-sensitive decision analysis are outside the assignment scope.
- Saved scikit-learn objects should be loaded with the pinned dependency versions in `requirements.txt`.

## Reproducibility note

The notebook and command-line trainer use the same functions in `ml_workflow.py`. This design prevents duplicated preprocessing code and reduces the risk that notebook results differ from the models served by Streamlit.
