"""Synthetic smoke test for the supported baseline and domain cleaning."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('baseline', Path(__file__).resolve().parents[1] / 'scripts/train_baseline.py')
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)

class BaselineTests(unittest.TestCase):
    def test_cleaning_and_training(self):
        rng = np.random.default_rng(42)
        frame = pd.DataFrame(rng.uniform(1, 10, size=(200, 10)), columns=baseline.FEATURES)
        frame['SeriousDlqin2yrs'] = [0, 1] * 100
        frame.loc[0, 'age'] = 0
        frame.loc[0, 'NumberOfTimes90DaysLate'] = 98
        cleaned = baseline.clean(frame)
        self.assertTrue(pd.isna(cleaned.loc[0, 'age']))
        self.assertTrue(pd.isna(cleaned.loc[0, 'NumberOfTimes90DaysLate']))
        self.assertEqual(len(cleaned.columns), 10)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            frame.to_csv(path / 'train.csv', index=False)
            metrics = baseline.train(path / 'train.csv', path / 'result', trees=10)
            self.assertEqual(metrics['train_rows'], 140)
            self.assertEqual(metrics['test_rows'], 60)
            self.assertTrue((path / 'result/baseline.joblib').exists())
            self.assertTrue((path / 'result/metrics.json').exists())
            self.assertTrue(0 <= metrics['roc_auc'] <= 1)

if __name__ == '__main__':
    unittest.main()
