"""Regression checks for separation of validation selection and test evaluation."""
import unittest
from unittest.mock import patch
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from wwdc_repro.redshift.frontier import select_by_budget, evaluate_selected, curves, score_fronts, front_path


class SelectionTests(unittest.TestCase):
    def test_display_fallback_uses_best_lower_budget_without_changing_selection(self):
        from wwdc_repro.redshift.figures import frontier_display

        selected = pd.DataFrame({
            "arm": ["orig"] * 6, "seed": [42] * 5 + [43],
            "budget": [5, 6, 7, 8, 9, 8], "chosen_length": [5, 6, 7, 8, 9, 8],
            "test_r2": [.8, .7, np.nan, np.inf, .95, np.nan],
        })
        original = selected.copy(deep=True)
        display = frontier_display(selected)
        pd.testing.assert_frame_equal(selected, original)
        for index in (2, 3):
            self.assertEqual(display.loc[index, "display_r2"], .8)
            self.assertEqual(display.loc[index, "fallback_budget"], 5)
            self.assertTrue(display.loc[index, "display_fallback"])
        self.assertTrue(np.isnan(display.loc[5, "display_r2"]))
        self.assertFalse(display.loc[5, "display_fallback"])
        self.assertEqual(display.loc[1, "display_r2"], .7)

    def test_recovered_training_score_overrides_ambiguous_flag_without_test_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = front_path(root, "orig", 42)
            path.parent.mkdir()
            pd.DataFrame([dict(Seed=42, Length=1, Equation="X1", Train_R2=-np.inf,
                               Test_R2=-np.inf)]).to_csv(path, index=False)
            recovery = {"candidates": [dict(arm="orig", seed=42, length=1,
                equation="X1", native_train_r2=1.,
                front_sha256=hashlib.sha256(path.read_bytes()).hexdigest())]}
            # Test matrices are intentionally absent: eligibility must not access them.
            data = {"X_train": np.array([[1.], [2.]]), "y_train": np.array([1., 2.]),
                    "X_val": np.array([[3.], [4.]]), "y_val": np.array([3., 4.])}
            with patch("wwdc_repro.redshift.frontier.ARMS", ("orig",)), \
                 patch("wwdc_repro.redshift.frontier.SEEDS", (42,)), \
                 patch("wwdc_repro.redshift.frontier.matrices", return_value=data), \
                 patch("wwdc_repro.redshift.frontier.importlib.resources.files") as resources:
                resources.return_value.joinpath.return_value.read_text.return_value = json.dumps(recovery)
                scores = score_fronts({}, root)
                self.assertTrue(scores.iloc[0].selectable)
                self.assertEqual(scores.iloc[0].reference_train_r2, 1.)
                recovery["candidates"][0]["front_sha256"] = "wrong"
                resources.return_value.joinpath.return_value.read_text.return_value = json.dumps(recovery)
                with self.assertRaisesRegex(ValueError, "does not match"):
                    score_fronts({}, root)

    def test_test_failure_does_not_replace_validation_winner(self):
        candidates = pd.DataFrame([
            dict(arm="orig", seed=42, length=5, equation="X1", validation_r2=.7,
                 selectable=True, source="fixture"),
            dict(arm="orig", seed=42, length=6, equation="log(X1)", validation_r2=.9,
                 selectable=True, source="fixture"),
        ])
        selected = select_by_budget(candidates, selection="validation")
        # No test scores exist when the choice is made.
        self.assertNotIn("test_r2", selected)
        with patch("wwdc_repro.redshift.frontier.matrices", return_value={
            "X_test": np.array([[-1.], [-2.]]), "y_test": np.array([-1., -2.])
        }):
            evaluated = evaluate_selected({}, selected)
        winner = evaluated[evaluated.budget == 6].iloc[0]
        self.assertEqual(winner.equation, "log(X1)")
        self.assertEqual(winner.test_status, "evaluation_failure")
        self.assertTrue(np.isnan(winner.test_r2))
        self.assertEqual(evaluated[evaluated.budget == 5].iloc[0].test_r2, 1.)

    def test_failed_seed_is_not_silently_dropped_from_mean(self):
        selected = pd.DataFrame({"arm": ["orig"] * 3, "budget": [6] * 3,
                                 "test_r2": [.8, .9, np.nan]})
        row = curves(selected).iloc[0]
        self.assertTrue(np.isnan(row["mean"]))
        self.assertEqual(row.evaluation_failures, 1)
        self.assertEqual(row.n, 3)


if __name__ == "__main__":
    unittest.main()
