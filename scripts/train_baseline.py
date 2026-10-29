"""Train a reproducible baseline with preprocessing fitted only on training data."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

FEATURES = ['RevolvingUtilizationOfUnsecuredLines', 'age',
    'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
    'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
    'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfDependents']


def clean(frame):
    """Apply fixed domain rules, without estimating statistics from the data."""
    frame = frame[FEATURES].copy()
    frame['age'] = frame['age'].replace(0, np.nan)
    for col in [FEATURES[2], FEATURES[6], FEATURES[8]]:
        frame[col] = frame[col].replace({96: np.nan, 98: np.nan})
    return frame.replace([np.inf, -np.inf], np.nan)


def train(input_path, output_dir, trees=100):
    frame = pd.read_csv(input_path)
    target = frame['SeriousDlqin2yrs']
    if target.isna().any() or set(target.unique()) != {0, 1}:
        raise ValueError('Expected both binary target classes 0 and 1, without missing labels')
    X_train, X_test, y_train, y_test = train_test_split(
        clean(frame), target, test_size=0.3, random_state=42, stratify=target)
    model = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('classifier', RandomForestClassifier(n_estimators=trees, max_depth=10,
            min_samples_leaf=5, class_weight='balanced', random_state=42, n_jobs=-1))])
    model.fit(X_train, y_train)
    probability = model.predict_proba(X_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    metrics = {'roc_auc': roc_auc_score(y_test, probability),
        'average_precision': average_precision_score(y_test, probability),
        'classification_report': classification_report(y_test, prediction, output_dict=True, zero_division=0),
        'train_rows': len(X_train), 'test_rows': len(X_test), 'seed': 42,
        'note': 'One stratified holdout; no tuning on this holdout.'}
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / 'baseline.joblib')
    (output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('data/cs-training.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts'))
    args = parser.parse_args()
    print(json.dumps(train(args.input, args.output_dir), indent=2))
