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


class StellarMassExportTests(unittest.TestCase):
    def test_only_latex_is_copied_to_manuscript(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from tools.export_manuscript import export_stellar_mass_tables
        from wwdc_repro.redshift import stellar_mass_mlp as mlp

        for grid in (True, False):
            with self.subTest(grid=grid), TemporaryDirectory() as tmp:
                root = Path(tmp)
                run, paper, analysis = root/'run', root/'paper/tables', root/'outputs/tables'
                (run/'orig').mkdir(parents=True)
                if grid:
                    (run/'orig/trial_metrics.csv').write_text('fixture')
                table = 'stellar_mass_mlp_grid.tex' if grid else 'stellar_mass_mlp.tex'
                def render(inputs, destination):
                    destination.mkdir(parents=True)
                    for name in (table, 'stellar_mass_mlp_numbers.tex', 'scores.csv', 'provenance.json'):
                        (destination/name).write_text(name)
                with patch.object(mlp, 'verified_results', return_value={'results':[{'arm':'orig'}]}), \
                     patch.object(mlp, 'export_grid', side_effect=render), \
                     patch.object(mlp, 'export_results', side_effect=render):
                    export_stellar_mass_tables([run], paper, analysis)
                self.assertEqual({p.name for p in paper.iterdir()}, {table, 'stellar_mass_mlp_numbers.tex'})
                self.assertTrue((analysis/'scores.csv').is_file())
                self.assertTrue((analysis/'provenance.json').is_file())
