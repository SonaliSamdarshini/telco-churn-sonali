"""Train and save all five classification pipelines.

Run from the repository root:

    python -m model.train_models

Use --skip-cv only for a quick local rebuild. The submitted artifacts should be
created without --skip-cv so that cv_metrics.csv is available.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_workflow import run_training


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Telco Customer Churn models and save deployment artifacts."
    )
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Skip five-fold cross-validation for a faster development run.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    artifacts = run_training(
        PROJECT_ROOT,
        run_cross_validation=not arguments.skip_cv,
    )

    print("\nTraining completed successfully.\n")
    print(artifacts["metrics"].round(4).to_string(index=False))
    print(f"\nTest data: {artifacts['test_data_path']}")
    print(f"Model directory: {artifacts['model_directory']}")


if __name__ == "__main__":
    main()
