"""Check the anatomy curves against SymPy series and hand-worked Taylor examples.

Run from reproduction/:
    MPLCONFIGDIR=/tmp/flower-mpl .venv/bin/python -m unittest discover \
        -s tests -p test_anatomy_taylor.py -v

Scaling every latent by epsilon makes series(..., epsilon, 0, 2) retain
only terms of total latent degree zero and one. Redshift is not expanded.
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import sympy as sp

from wwdc_repro.redshift.paper_panels import anatomy
from wwdc_repro.redshift.frontier import SEEDS, front_path


def series_first_order(expression, latents):
    epsilon = sp.Symbol("epsilon", real=True)
    scaled = expression.subs({x: epsilon*x for x in latents}, simultaneous=True)
    return sp.series(scaled, epsilon, 0, 2).removeO().subs(epsilon, 1).expand()


class TaylorExamples(unittest.TestCase):
    def test_hand_worked_examples(self):
        x1, x2 = sp.symbols("X1 X2", real=True)
        z = sp.Symbol("z", positive=True)
        examples = [
            ("linear", 3 + z + 2*x1 - x2, 3 + z + 2*x1 - x2),
            ("quadratic and mixed terms", 3 + z + 2*x1 - x2 + x1*x2 + x1**2,
             3 + z + 2*x1 - x2),
            ("latent exponent", 5 + sp.log(1+7*z)*(3*z)**(2*x1-x2),
             5 + sp.log(1+7*z)*(1 + (2*x1-x2)*sp.log(3*z))),
        ]
        for name, expression, expected in examples:
            with self.subTest(name=name):
                zero = {x1: 0, x2: 0}
                derivative_form = expression.subs(zero) + sum(
                    sp.diff(expression, x).subs(zero)*x for x in (x1, x2))
                series_form = series_first_order(expression, (x1, x2))
                self.assertEqual(sp.simplify(derivative_form-expected), 0)
                self.assertEqual(sp.simplify(series_form-expected), 0)
                print(f"\n{name}:\n  f = {expression}\n  first order (verified by series) = {expected}")

    def test_production_anatomy_against_series_and_error_scaling(self):
        root = Path(__file__).resolve().parents[1]
        rows = []
        for seed in SEEDS:
            front = pd.read_csv(front_path(root / "artifacts/fronts", "cond+z", seed))
            equation = front.loc[front.Length == 24, "Equation"].item()
            rows.append({"seed": seed, "equation": equation})
        selected = pd.DataFrame(rows)
        # Synthetic inputs isolate the curve calculation; no catalogue is needed.
        z_sample = np.linspace(.02, .3, 20)
        x = np.zeros((len(z_sample), 11))
        x[:, -1] = z_sample
        data = {"X_test": x, "y_test": np.full(len(x), 10.)}
        def equation_for_seed(frame, seed):
            return frame.loc[frame.seed == seed, "equation"].item()
        with tempfile.TemporaryDirectory() as directory:
            outdir = Path(directory)
            with patch("wwdc_repro.redshift.paper_panels.matrices", return_value=data), \
                 patch("matplotlib.figure.Figure.savefig"):
                anatomy({}, selected, outdir, equation_for_seed)
            curves = pd.read_csv(outdir / "equation_anatomy_by_seed.csv")
            diagnostics = json.loads((outdir / "equation_anatomy_by_seed.json").read_text())
        latents = sp.symbols("X1:11", real=True)
        z = sp.Symbol("X11", positive=True)
        symbols = (*latents, z)
        names = {x.name: x for x in symbols}
        names.update(exp=sp.exp, log=sp.log, abs=sp.Abs, Abs=sp.Abs)
        for report in diagnostics:
            seed = report["seed"]
            with self.subTest(seed=seed):
                # Rational coefficients avoid symbolic rounding during expansion.
                expression = sp.sympify(report["equation"].replace("^", "**"),
                                        locals=names, rational=True)
                taylor = series_first_order(expression, latents)
                full = sp.lambdify(symbols, expression, "numpy")
                # Independently check the series against f(0,z) + gradient(0,z) dot X.
                at_zero = dict.fromkeys(latents, 0)
                derivative_form = expression.subs(at_zero) + sum(
                    sp.diff(expression, symbol).subs(at_zero)*symbol for symbol in latents)
                derivative_approximation = sp.lambdify(symbols, derivative_form, "numpy")
                approximation = sp.lambdify(symbols, taylor, "numpy")
                group = curves[curves.seed == seed]
                grid = group.z.to_numpy()
                weights = np.array(report["unit_direction"])
                for score in (-2, -1, 0, 1, 2):
                    values = approximation(*(score*weights), grid)
                    np.testing.assert_allclose(values, group[f"s_{score:+d}"],
                                               rtol=1e-12, atol=1e-12)
                # Compare at arbitrary latent points, not only along the score axis.
                latent_values = np.random.default_rng(42).normal(size=(10, len(grid)))
                np.testing.assert_allclose(
                    approximation(*latent_values, grid),
                    derivative_approximation(*latent_values, grid),
                    rtol=1e-12, atol=1e-12)
                projected = weights @ latent_values
                np.testing.assert_allclose(
                    approximation(*latent_values, grid),
                    group.baseline.to_numpy() + group.A.to_numpy()*projected,
                    rtol=1e-12, atol=1e-12)
                errors = []
                for h in (1., .5, .25):
                    values = h*weights
                    errors.append(float(np.max(np.abs(
                        full(*values, grid)-approximation(*values, grid)))))
                print(f"\nSeed {seed}: all five production curves match SymPy series."
                      f"\n  max error at s=1, 0.5, 0.25: {errors}")
                if seed == 42:
                    self.assertLess(max(errors), 1e-12)
                else:
                    # A first-order remainder is O(s^2): halving s quarters error.
                    ratios = np.array(errors[:-1])/errors[1:]
                    np.testing.assert_allclose(ratios, 4., rtol=.1)
                    print(f"  error reduction factors: {ratios}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
