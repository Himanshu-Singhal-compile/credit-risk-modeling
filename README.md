# Credit Risk Modeling

An exploratory credit-delinquency classification project using Give Me Some Credit, with feature engineering, model comparison, Optuna experiments, SHAP analysis, and a separate reproducible baseline.

[![CI](https://github.com/Himanshu-Singhal-compile/credit-risk-modeling/actions/workflows/ci.yml/badge.svg)](https://github.com/Himanshu-Singhal-compile/credit-risk-modeling/actions/workflows/ci.yml)


## Environment

Use Python 3.11 in an isolated environment. The version ranges in `requirements.txt`
describe the intended environment; they are not a fully resolved lockfile.

```sh
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyter lab
```

Launch Jupyter from the repository root and select a notebook in `notebooks/`.
Data defaults to `data/`; set `PROJECT_DATA_DIR` to override it. Generated artifacts
are ignored by Git. Clear notebook outputs before committing.

## Data setup

Download the [Give Me Some Credit competition data](https://www.kaggle.com/competitions/GiveMeSomeCredit/data) after reviewing its access and reuse terms. Put `cs-training.csv` and `cs-test.csv` in `data/`. The supplied folder contained the test CSV and sample submission, but **not the labeled training CSV**; neither raw data nor the data dictionary is redistributed here.

The target is `SeriousDlqin2yrs`. The ten predictors cover credit utilization, age, past-due history, debt ratio, income, loans, and dependents. The original CSV index is not a predictor.

## Start with the baseline

```sh
python scripts/train_baseline.py --input data/cs-training.csv --output-dir artifacts
```

The baseline uses fixed domain cleaning, a stratified 70/30 split, training-only median imputation, and a class-weighted random forest. It writes `artifacts/metrics.json` (ROC AUC, average precision, per-class report, and split sizes) and `artifacts/baseline.joblib`. It does not tune on the holdout. Reuse requires applying `clean` from the training script before prediction; load only trusted model artifacts.

## Notebook workflow

Open `notebooks/credit_risk_analysis.ipynb` to explore missingness, outliers, class imbalance, delinquency features, income ratios, SMOTE, logistic regression, random forests, gradient boosting, XGBoost, Optuna, and SHAP. Notebook training and explanation can take substantial time and memory; SHAP evaluates the full held-out set in the original workflow.

The exploratory notebook estimates some clipping/imputation statistics before splitting and uses the test partition inside the Optuna objective. Consequently, its tuned scores are exploratory and are **not an unbiased final evaluation**. Use the baseline for a clean holdout or refactor tuning into nested training/validation splits before reporting new benchmark claims. The baseline and notebook intentionally have different preprocessing/model choices. No historical accuracy or AUC is presented as a freshly reproduced result.

## Checks and scope

```sh
python scripts/validate_notebooks.py
python -m unittest discover -s tests -v
```

CI validates notebook syntax and exercises baseline training on synthetic data; it does not download competition data or reproduce full notebook training. This is a research/portfolio experiment, without a deployed lending service, fairness evaluation, or calibrated approval policy.


