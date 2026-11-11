"""Independent checks of exact latent expansion and pinned-redshift coverage."""
import unittest

import numpy as np

from wwdc_repro.redshift.diagnostic_tables import first_order_expression, coverage_values
from wwdc_repro.shared.equations import evaluate


class DiagnosticTablesTest(unittest.TestCase):
    def test_series_keeps_redshift_and_discards_mixed_latent_terms(self):
        x = np.array([[.2, -.3, .05], [-.1, .7, .2]])
        expression = '3 + log(X3) * (2*X1 + X2) + X1*X2 + X2**2'
        first = evaluate(first_order_expression(expression, 3), x)
        expected = 3 + np.log(x[:, 2])*(2*x[:, 0]+x[:, 1])
        np.testing.assert_allclose(first, expected, atol=1e-12)

    def test_power_series_matches_analytic_derivative(self):
        x = np.array([[.2, -.3, .05], [-.1, .7, .2]])
        expression = '1 + log(1+100*X3)*(3*X3)**(2*X1-X2)'
        actual = evaluate(first_order_expression(expression, 3), x)
        expected = 1 + np.log1p(100*x[:, 2])*(1+np.log(3*x[:, 2])*(2*x[:, 0]-x[:, 1]))
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_coverage_uses_unit_score_amplitude_and_inclusive_bounds(self):
        x = np.array([[-2., .02], [-1., .1], [0., .15], [1., .2], [2., .3]])
        result, curves = coverage_values('10 + X2 + 2*X1', x)
        self.assertEqual(result['count_1A'], 3)
        self.assertEqual(result['count_2A'], 5)
        np.testing.assert_allclose(curves['amplitude'], 2)
        np.testing.assert_allclose(curves['baseline'], 10+np.array(curves['z']))


if __name__ == '__main__':
    unittest.main()
