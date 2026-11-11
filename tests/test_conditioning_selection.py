"""Guard against test-dependent model selection."""
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from wwdc_repro.norm.validate_conditioning import select, score


class ValidationSelectionTest(unittest.TestCase):
    def test_test_failure_does_not_change_choice(self):
        rows = pd.DataFrame([
            dict(Length=5, val_r2=.7, test_r2=.99, selectable=True),
            dict(Length=9, val_r2=.8, test_r2=np.nan, selectable=True),
        ])
        self.assertEqual(select(rows, 29, selection="validation").Length, 9)
        self.assertEqual(select(rows, 5, selection="validation").Length, 5)
        rows['test_r2'] = [np.nan, -.5]
        self.assertEqual(select(rows, 29, selection="validation").Length, 9)

    def test_eligibility_ties_and_empty_pool(self):
        rows = pd.DataFrame([
            dict(Length=5, val_r2=.9, selectable=False),
            dict(Length=9, val_r2=.8, selectable=True),
            dict(Length=7, val_r2=.8, selectable=True),
        ])
        self.assertEqual(select(rows, 29, selection="validation").Length, 7)
        self.assertIsNone(select(rows, 5, selection="validation"))

    def test_one_invalid_prediction_fails_whole_split(self):
        x = np.array([[1.], [0.], [2.]])
        y = np.array([1., 0., .5])
        self.assertTrue(np.isnan(score('1 / X1', x, y)))
        self.assertAlmostEqual(score('X1', x, x[:, 0]), 1.)


if __name__ == '__main__':
    unittest.main()
