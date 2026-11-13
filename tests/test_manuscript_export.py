"""Check physical-unit conversion and reject changes to the stated equation family."""
import unittest

import numpy as np

from tools.export_manuscript import norm_family_coefficients
from wwdc_repro.shared.equations import evaluate


class NormFamilyTests(unittest.TestCase):
    equation = '2 + 3*exp(0.4*X11) + 5*log(1+6*X12) - 0.1*X5 + 0.2*X6 + 0.3*X9'

    def test_unscaling_preserves_predictions(self):
        c = norm_family_coefficients(self.equation, 1., 2.)
        self.assertAlmostEqual(c['C'], 2.)
        self.assertAlmostEqual(c['A'], 3*np.exp(-.2))
        self.assertAlmostEqual(c['b'], .2)
        rng = np.random.default_rng(42)
        x = rng.normal(size=(100, 12))
        x[:, 11] = rng.uniform(.02, .3, 100)
        ell = 2*x[:, 10]+1
        physical = (c['C'] + c['A']*np.exp(c['b']*ell)
                    + c['D']*np.log1p(c['k']*x[:, 11])
                    + sum(c[f'a_{i}']*x[:, i-1] for i in (5, 6, 9)))
        np.testing.assert_allclose(physical, evaluate(self.equation, x), atol=1e-12)

    def test_extra_latent_or_coupled_terms_require_review(self):
        for extra in (' + X1', ' + X5*X6', ' + X5*X11'):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                norm_family_coefficients(self.equation + extra, 1., 2.)

    def test_invalid_scaling_is_rejected(self):
        for std in (0., -1., np.nan):
            with self.subTest(std=std), self.assertRaises(ValueError):
                norm_family_coefficients(self.equation, 1., std)
