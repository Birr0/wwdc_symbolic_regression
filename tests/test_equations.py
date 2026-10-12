"""Numerical contract for the exported-expression evaluator."""
import unittest
import numpy as np
from wwdc_repro.equations import evaluate, ExpressionError, EvaluationInputError
from wwdc_repro.frontier import score_expression_details


class EquationTests(unittest.TestCase):
    def test_arithmetic_constants_and_feature_order(self):
        x = np.array([[2., 3.], [4., 5.]])
        np.testing.assert_array_equal(evaluate("2", x), [2., 2.])
        np.testing.assert_array_equal(evaluate("X1^2+X2/2-1", x), [4.5, 17.5])
        np.testing.assert_allclose(evaluate("exp(log(abs(X1)))", x), x[:, 0])

    def test_nonfinite_domains_and_overflow_fail_without_dropping_rows(self):
        x = np.array([[-1.], [0.], [1.]])
        for expr in ("log(X1)", "log(abs(X1))", "exp(1000*X1)", "X1^0.5"):
            with self.subTest(expr=expr):
                score, reason = score_expression_details(expr, x, np.arange(3.))
                self.assertTrue(np.isnan(score))
                self.assertEqual(reason, "nonfinite_prediction")

    def test_historical_simplification_is_preserved(self):
        x = np.array([[-1.], [0.], [1.]])
        np.testing.assert_array_equal(evaluate("X1/X1", x), np.ones(3))
        np.testing.assert_array_equal(evaluate("log(X1)-log(X1)", x), np.zeros(3))

    def test_unsupported_expressions_fail_before_numeric_evaluation(self):
        x = np.ones((3, 1))
        for expr in ("unknown(X1)", "X2", "X2-X2", "[X1, X1]", "X1[0]", "X1.foo", "log(X1,2)", "", "X1+"):
            with self.subTest(expr=expr), self.assertRaises(ExpressionError):
                evaluate(expr, x)
        score, reason = score_expression_details("unknown(X1)", x, np.arange(3.))
        self.assertTrue(np.isnan(score))
        self.assertTrue(reason.startswith("invalid_expression:"))

    def test_invalid_arrays_have_distinct_failure_reason(self):
        for x in (np.ones(3), np.empty((0, 1)), np.full((3, 1), np.nan), np.ones((3, 1), dtype=complex)):
            with self.subTest(shape=x.shape), self.assertRaises(EvaluationInputError):
                evaluate("X1", x)
        score, reason = score_expression_details("X1", np.ones((3, 1)), np.ones((3, 1)))
        self.assertTrue(np.isnan(score))
        self.assertTrue(reason.startswith("invalid_input:"))


if __name__ == "__main__":
    unittest.main()
