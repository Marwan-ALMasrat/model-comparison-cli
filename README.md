# model-comparison-cli

A production-quality command-line tool that compares 6 machine learning model configurations on a telecom churn dataset using stratified cross-validation, and saves all results, plots, and the best model to disk.

---

## Installation

**Requirements:** Python 3.8+

Install dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib joblib
```

---

## Usage

```bash
python compare_models.py --data-path <path-to-csv> [options]
```

### Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--data-path` | Yes | — | Path to the input dataset CSV file |
| `--output-dir` | No | `./output` | Directory where all results and plots are saved |
| `--n-folds` | No | `5` | Number of cross-validation folds |
| `--random-seed` | No | `42` | Random seed for reproducibility |
| `--dry-run` | No | `False` | Validate data and print config without training |

---

## Example Commands

**Normal run:**
```bash
python compare_models.py --data-path data/telecom_churn.csv --output-dir ./output
```

**Dry run (validate data and print config only, no training):**
```bash
python compare_models.py --data-path data/telecom_churn.csv --dry-run
```

**Custom folds and seed:**
```bash
python compare_models.py --data-path data/telecom_churn.csv --n-folds 10 --random-seed 123 --output-dir ./results
```

---

## Input Data

The CSV file must contain the following columns:

| Column | Description |
|---|---|
| `tenure` | Number of months the customer has been with the company |
| `monthly_charges` | Monthly charge amount |
| `total_charges` | Total charges to date |
| `num_support_calls` | Number of support calls made |
| `senior_citizen` | Whether the customer is a senior citizen (0/1) |
| `has_partner` | Whether the customer has a partner (0/1) |
| `has_dependents` | Whether the customer has dependents (0/1) |
| `contract_months` | Contract length in months |
| `churned` | Target variable — whether the customer churned (0/1) |

---

## Output Files

All files are saved to `--output-dir` (default: `./output`):

| File | Description |
|---|---|
| `comparison_table.csv` | CV results with mean ± std for all metrics |
| `pr_curves.png` | Precision-Recall curves for the top 3 models |
| `calibration.png` | Calibration curves for the top 3 models |
| `best_model.joblib` | The best model serialized with joblib |
| `experiment_log.csv` | Timestamped log of all experiment runs |

---

## Models Compared

| Model | Description |
|---|---|
| `Dummy` | Baseline — always predicts the majority class |
| `LR_default` | Logistic Regression with default settings |
| `LR_balanced` | Logistic Regression with balanced class weights |
| `DT_depth5` | Decision Tree with max depth 5 |
| `RF_default` | Random Forest (100 trees, max depth 10) |
| `RF_balanced` | Random Forest with balanced class weights |

---

## Exit Codes

| Code | Meaning |
|---|---|
| `0` | Success (or clean dry run) |
| `1` | Data file not found or validation failed |

---

## Project Structure

```
model-comparison-cli/
├── compare_models.py   # Main CLI script
├── README.md           # This file
├── data/
│   └── telecom_churn.csv
└── output/             # Generated after running the script
    ├── comparison_table.csv
    ├── pr_curves.png
    ├── calibration.png
    ├── best_model.joblib
    └── experiment_log.csv
```