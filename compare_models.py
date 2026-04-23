"""
compare_models.py — Production CLI for Telecom Churn Model Comparison

Refactored from Module 5 Week B Integration Task.
Compares 6 model configurations using stratified cross-validation,
produces PR curves and calibration plots, and persists the best model.

Usage:
    python compare_models.py --data-path data/telecom_churn.csv
    python compare_models.py --data-path data/telecom_churn.csv --dry-run
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.calibration import CalibrationDisplay
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay, average_precision_score,
    precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "tenure", "monthly_charges", "total_charges",
    "num_support_calls", "senior_citizen",
    "has_partner", "has_dependents", "contract_months"
]

REQUIRED_COLUMNS = NUMERIC_FEATURES + ["churned"]


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(level=logging.INFO):
    """Configure root logger to write to stdout with timestamp."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare 6 ML model configurations on a telecom churn dataset "
            "using stratified cross-validation and save all results to disk."
        )
    )

    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to the input dataset CSV file (must contain the expected columns)."
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Directory where all results, plots, and logs are saved (default: ./output)."
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of stratified cross-validation folds (default: 5)."
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate data and print pipeline configuration "
            "without training any models."
        )
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------

def load_data(data_path):
    """Load dataset from CSV.

    Args:
        data_path: Path string to the CSV file.

    Returns:
        pandas DataFrame, or raises SystemExit if file is not found.
    """
    logger = logging.getLogger(__name__)

    if not os.path.exists(data_path):
        logger.error("Data file not found: %s", data_path)
        sys.exit(1)

    logger.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path)
    logger.info(
        "Loaded dataset — shape: %d rows × %d columns",
        df.shape[0], df.shape[1]
    )
    return df


# ---------------------------------------------------------------------------
# validate_data
# ---------------------------------------------------------------------------

def validate_data(df):
    """Validate that the DataFrame has the expected columns and no critical issues.

    Args:
        df: pandas DataFrame loaded from CSV.

    Returns:
        True if validation passes, raises SystemExit otherwise.
    """
    logger = logging.getLogger(__name__)

    # Check required columns
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        logger.error(
            "Validation failed — missing required columns: %s", missing
        )
        sys.exit(1)

    logger.info("All required columns present: %s", REQUIRED_COLUMNS)

    # Report shape
    logger.info("Dataset shape: %d rows, %d columns", df.shape[0], df.shape[1])

    # Class distribution
    class_counts = df["churned"].value_counts()
    churn_rate = df["churned"].mean()
    logger.info(
        "Class distribution — churned=1: %d (%.1f%%), churned=0: %d (%.1f%%)",
        class_counts.get(1, 0), churn_rate * 100,
        class_counts.get(0, 0), (1 - churn_rate) * 100
    )

    # Warn about missing values
    null_counts = df[REQUIRED_COLUMNS].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        logger.warning(
            "Missing values detected in columns: %s",
            cols_with_nulls.to_dict()
        )
    else:
        logger.info("No missing values detected in required columns.")

    return True


# ---------------------------------------------------------------------------
# define_models
# ---------------------------------------------------------------------------

def define_models(random_seed=42):
    """Define 6 model configurations for comparison.

    Returns:
        Dict of {name: sklearn Pipeline}.
    """
    models = {
        "Dummy": Pipeline([
            ("scaler", "passthrough"),
            ("model", DummyClassifier(strategy="most_frequent", random_state=random_seed))
        ]),
        "LR_default": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=random_seed))
        ]),
        "LR_balanced": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=random_seed
            ))
        ]),
        "DT_depth5": Pipeline([
            ("scaler", "passthrough"),
            ("model", DecisionTreeClassifier(max_depth=5, random_state=random_seed))
        ]),
        "RF_default": Pipeline([
            ("scaler", "passthrough"),
            ("model", RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=random_seed
            ))
        ]),
        "RF_balanced": Pipeline([
            ("scaler", "passthrough"),
            ("model", RandomForestClassifier(
                n_estimators=100, max_depth=10,
                class_weight="balanced", random_state=random_seed
            ))
        ]),
    }
    return models


# ---------------------------------------------------------------------------
# train_and_evaluate
# ---------------------------------------------------------------------------

def train_and_evaluate(models, X_train, y_train, n_folds=5, random_seed=42):
    """Run stratified cross-validation on all models.

    Args:
        models: Dict of {name: Pipeline}.
        X_train: Training features DataFrame.
        y_train: Training target Series.
        n_folds: Number of CV folds.
        random_seed: Random seed for StratifiedKFold.

    Returns:
        DataFrame with mean ± std for each metric per model.
    """
    logger = logging.getLogger(__name__)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    logger.info(
        "Starting %d-fold stratified cross-validation on %d models ...",
        n_folds, len(models)
    )

    rows = []

    for name, pipeline in models.items():
        logger.info("  Evaluating model: %s", name)
        fold_scores = {
            "accuracy": [], "precision": [],
            "recall": [], "f1": [], "pr_auc": []
        }

        for fold_num, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            pipeline.fit(X_tr, y_tr)
            y_pred = pipeline.predict(X_val)
            y_proba = pipeline.predict_proba(X_val)[:, 1]

            fold_scores["accuracy"].append(accuracy_score(y_val, y_pred))
            fold_scores["precision"].append(
                precision_score(y_val, y_pred, zero_division=0)
            )
            fold_scores["recall"].append(
                recall_score(y_val, y_pred, zero_division=0)
            )
            fold_scores["f1"].append(
                f1_score(y_val, y_pred, zero_division=0)
            )
            fold_scores["pr_auc"].append(
                average_precision_score(y_val, y_proba)
            )

            logger.debug(
                "    Fold %d — F1: %.3f  PR-AUC: %.3f",
                fold_num,
                fold_scores["f1"][-1],
                fold_scores["pr_auc"][-1]
            )

        rows.append({
            "model":          name,
            "accuracy_mean":  np.mean(fold_scores["accuracy"]),
            "accuracy_std":   np.std(fold_scores["accuracy"]),
            "precision_mean": np.mean(fold_scores["precision"]),
            "precision_std":  np.std(fold_scores["precision"]),
            "recall_mean":    np.mean(fold_scores["recall"]),
            "recall_std":     np.std(fold_scores["recall"]),
            "f1_mean":        np.mean(fold_scores["f1"]),
            "f1_std":         np.std(fold_scores["f1"]),
            "pr_auc_mean":    np.mean(fold_scores["pr_auc"]),
            "pr_auc_std":     np.std(fold_scores["pr_auc"]),
        })

        logger.info(
            "    %s — PR-AUC: %.3f ± %.3f  |  F1: %.3f ± %.3f",
            name,
            rows[-1]["pr_auc_mean"], rows[-1]["pr_auc_std"],
            rows[-1]["f1_mean"],     rows[-1]["f1_std"]
        )

    logger.info("Cross-validation complete.")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# save_results
# ---------------------------------------------------------------------------

def save_results(results_df, fitted_models, X_test, y_test, output_dir):
    """Save all results: CSV table, plots, best model, and experiment log.

    Args:
        results_df: DataFrame from train_and_evaluate().
        fitted_models: Dict of {name: fitted Pipeline}.
        X_test: Test features.
        y_test: Test labels.
        output_dir: Root directory for all output files.
    """
    logger = logging.getLogger(__name__)
    os.makedirs(output_dir, exist_ok=True)

    # --- Comparison table ---
    table_path = os.path.join(output_dir, "comparison_table.csv")
    results_df.to_csv(table_path, index=False)
    logger.info("Comparison table saved → %s", table_path)

    # --- PR curves (top 3) ---
    pr_aucs = {
        name: average_precision_score(
            y_test, pipeline.predict_proba(X_test)[:, 1]
        )
        for name, pipeline in fitted_models.items()
    }
    top3 = sorted(pr_aucs, key=pr_aucs.get, reverse=True)[:3]

    pr_path = os.path.join(output_dir, "pr_curves.png")
    fig, ax = plt.subplots(figsize=(8, 6))
    for name in top3:
        PrecisionRecallDisplay.from_estimator(
            fitted_models[name], X_test, y_test,
            name=f"{name} (AP={pr_aucs[name]:.3f})", ax=ax
        )
    ax.set_title("Precision-Recall Curves — Top 3 Models")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(pr_path, dpi=150)
    plt.close(fig)
    logger.info("PR curves saved → %s", pr_path)

    # --- Calibration plot (top 3) ---
    cal_path = os.path.join(output_dir, "calibration.png")
    fig, ax = plt.subplots(figsize=(8, 6))
    for name in top3:
        CalibrationDisplay.from_estimator(
            fitted_models[name], X_test, y_test,
            n_bins=10, name=name, ax=ax
        )
    ax.set_title("Calibration Curves — Top 3 Models")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(cal_path, dpi=150)
    plt.close(fig)
    logger.info("Calibration plot saved → %s", cal_path)

    # --- Best model ---
    best_name = results_df.sort_values(
        "pr_auc_mean", ascending=False
    ).iloc[0]["model"]
    logger.info("Best model by PR-AUC: %s", best_name)
    model_path = os.path.join(output_dir, "best_model.joblib")
    dump(fitted_models[best_name], model_path)
    logger.info("Best model saved → %s", model_path)

    # --- Experiment log ---
    log_path = os.path.join(output_dir, "experiment_log.csv")
    timestamp = datetime.now().isoformat()
    log_df = pd.DataFrame({
        "model_name": results_df["model"],
        "accuracy":   results_df["accuracy_mean"],
        "precision":  results_df["precision_mean"],
        "recall":     results_df["recall_mean"],
        "f1":         results_df["f1_mean"],
        "pr_auc":     results_df["pr_auc_mean"],
        "timestamp":  timestamp,
    })
    if os.path.exists(log_path):
        log_df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        log_df.to_csv(log_path, index=False)
    logger.info("Experiment log saved → %s", log_path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Model Comparison Pipeline — starting")
    logger.info("  data-path  : %s", args.data_path)
    logger.info("  output-dir : %s", args.output_dir)
    logger.info("  n-folds    : %d", args.n_folds)
    logger.info("  random-seed: %d", args.random_seed)
    logger.info("  dry-run    : %s", args.dry_run)
    logger.info("=" * 60)

    # Step 1: Load
    df = load_data(args.data_path)

    # Step 2: Validate
    validate_data(df)

    # --dry-run: stop here after printing config
    if args.dry_run:
        models = define_models(args.random_seed)
        logger.info("--- DRY RUN: pipeline configuration ---")
        logger.info("Models to compare : %s", list(models.keys()))
        logger.info("CV folds          : %d", args.n_folds)
        logger.info("Random seed       : %d", args.random_seed)
        logger.info("Output directory  : %s", args.output_dir)
        logger.info("No models were trained. Exiting dry run.")
        sys.exit(0)

    # Step 3: Prepare features and split
    X = df[NUMERIC_FEATURES]
    y = df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=args.random_seed
    )
    logger.info(
        "Train/test split — train: %d rows, test: %d rows, "
        "train churn rate: %.2f%%",
        len(X_train), len(X_test), y_train.mean() * 100
    )

    # Step 4: Define models
    models = define_models(args.random_seed)
    logger.info(
        "%d model configurations defined: %s",
        len(models), list(models.keys())
    )

    # Step 5: Cross-validation
    results_df = train_and_evaluate(
        models, X_train, y_train,
        n_folds=args.n_folds,
        random_seed=args.random_seed
    )

    logger.info("\n=== Model Comparison Table ===")
    logger.info("\n%s", results_df.to_string(index=False))

    # Step 6: Fit all models on full training set
    logger.info("Fitting all models on full training set ...")
    fitted_models = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline
        logger.debug("  Fitted: %s", name)

    # Step 7: Save all results
    os.makedirs(args.output_dir, exist_ok=True)
    save_results(results_df, fitted_models, X_test, y_test, args.output_dir)

    logger.info("=" * 60)
    logger.info("Pipeline complete. All results saved to: %s", args.output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()