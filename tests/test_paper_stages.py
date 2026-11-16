"""Stage boundaries, fresh search coverage, and verified reporting prerequisites."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from wwdc_repro.workflows import paper_stages as stages


class PaperStageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.config = self.home / 'config.json'
        self.config.write_text(json.dumps({'data_root': str(self.home / 'data'),
                                          'cmd_cache': str(self.home / 'cache.parquet'),
                                          'norm_generation': str(self.home / 'spectra')}))
        self.paths = stages.load_config(self.config)
        self.out = self.home / 'out'

    def test_equations_plan_fits_51_unique_searches_and_both_mlp_comparisons(self):
        steps = stages.build_steps('equations', self.paths, self.out)
        searches = [s for s in steps if s['search']]
        self.assertEqual(len(searches), 51)
        self.assertEqual(len({s['name'] for s in searches}), 51)
        self.assertEqual(sum(s['name'].startswith('fit-redshift-') for s in searches), 18)
        joint = [s for s in searches if s['name'].startswith('fit-znorm-')]
        self.assertEqual(len(joint), 3)
        for step in joint:
            self.assertIn(str(self.paths['joint_embeddings']), step['command'])
            self.assertIn('--append-norm', step['command'])
        names = [s['name'] for s in steps]
        self.assertLess(names.index('prepare-controls'), names.index('prepare-fibre'))
        self.assertIn('fit-norm-probes', names)
        self.assertIn('fit-fibre-mlp', names)
        self.assertFalse(any('artifacts/fronts' in arg or 'archive' == arg for s in steps for arg in s['command']))

    def test_fresh_front_checks_keep_sample_checks_without_archived_counts(self):
        import pandas as pd
        from importlib.resources import files
        from wwdc_repro.redshift.cli import check_reference
        reference = json.loads(files('wwdc_repro.resources').joinpath('reference.json').read_text())
        samples = pd.DataFrame([dict(values, split=split) for split, values in reference['samples'].items()])
        checks = check_reference(samples, pd.DataFrame(), pd.DataFrame(), fresh_fronts=True)
        self.assertEqual(len(checks), 9)
        self.assertTrue(all(row['passed'] for row in checks))
        samples.loc[0, 'rows'] += 1
        checks = check_reference(samples, pd.DataFrame(), pd.DataFrame(), fresh_fronts=True)
        self.assertFalse(all(row['passed'] for row in checks))

    def test_stellar_mlp_can_run_without_symbolic_searches_and_exports(self):
        self.paths['stellar_mlp_spectra'] = self.home / 'spectral-cache'
        steps = stages.build_steps('mlp', self.paths, self.out, config=self.config)
        self.assertEqual(len(steps), 2)
        self.assertIn('fit', steps[0]['command'])
        self.assertIn('--spectra-dir', steps[0]['command'])
        self.assertIn(str(self.config), steps[0]['command'])
        self.assertIn('export', steps[1]['command'])
        self.assertTrue(steps[0]['managed'])
        self.assertFalse(any(s['search'] for s in steps))
        with patch.object(stages, 'source_files', return_value=[]), patch.object(stages, 'run_steps') as run:
            stages.main(['mlp', '--check', '--config', str(self.config), '--outdir', str(self.out)])
            run.assert_not_called()

    def test_stellar_mlp_prepares_cache_before_fitting_when_raw_data_supplied(self):
        self.paths['stellar_mlp_data_root'] = self.home/'raw-data'
        steps = stages.build_steps('mlp',self.paths,self.out,config=self.config)
        self.assertEqual([s['name'] for s in steps],
                         ['prepare-stellar-spectra','fit-stellar-mlp','mlp-stellar-mlp'])
        self.assertIn(str(self.home/'raw-data'),steps[0]['command'])
        cache = str(self.out/'work/stellar_mass_spectra')
        self.assertIn(cache,steps[0]['command'])
        self.assertIn(cache,steps[1]['command'])
        self.assertTrue(steps[0]['managed'])

    def test_raw_flux_mode_reaches_preparation_and_fitting(self):
        self.paths['stellar_mlp_data_root'] = self.home/'raw-data'
        steps = stages.build_steps('mlp',self.paths,self.out,config=self.config,spectrum_mode='raw')
        self.assertIn('--raw-flux',steps[0]['command'])
        self.assertIn('--spectrum-mode',steps[1]['command'])
        self.assertIn('raw',steps[1]['command'])

    def test_reporting_only_reads_current_run_and_never_fits(self):
        for stage in ('plots', 'tables'):
            steps = stages.build_steps(stage, self.paths, self.out)
            for step in steps:
                self.assertFalse(step['search'])
                self.assertNotIn('fit', step['command'])
                self.assertFalse(any('artifacts/' in arg for arg in step['command']))
            self.assertIn(str(self.out / 'work/redshift/analysis/selected.csv'), steps[0]['command'])
        tables = stages.build_steps('tables', self.paths, self.out)
        self.assertIn('diagnostic-tables', tables[0]['command'])
        self.assertNotIn('cmd', tables[0]['command'])

    def test_missing_equation_completion_stops_reporting(self):
        with patch.object(stages, 'source_files', return_value=[]), patch.object(stages, 'run_steps') as run:
            with self.assertRaisesRegex(FileNotFoundError, 'Run equations first'):
                stages.main(['tables', '--config', str(self.config), '--outdir', str(self.out)])
            run.assert_not_called()
            self.assertFalse(self.out.exists())

    def test_check_prints_plan_without_creating_outputs(self):
        with patch.object(stages, 'source_files', return_value=[]), patch('builtins.print'), patch.object(stages, 'run_steps') as run:
            stages.main(['equations', '--check', '--config', str(self.config), '--outdir', str(self.out)])
            run.assert_not_called()
            self.assertFalse(self.out.exists())

    def test_receipt_prevents_silent_use_of_changed_product(self):
        work = self.out / 'work/conditioning'
        step = dict(name='fixture', search=False, command=['fixture'], cwd=str(work),
                    products=['result.csv'], owned=[])
        def create(*args, **kwargs):
            (work / 'result.csv').write_text('result')
            return subprocess.CompletedProcess([], 0)
        with patch.object(stages.subprocess, 'run', side_effect=create) as run:
            stages.run_steps([step], self.out, {'stage': 'tables'}, {})
            stages.run_steps([step], self.out, {'stage': 'tables'}, {})
            self.assertEqual(run.call_count, 1)
            (work / 'result.csv').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'Missing/changed run output'):
                stages.run_steps([step], self.out, {'stage': 'tables'}, {})
