"""Exercise search completion, interruptions, corruption and competing writers."""
import json
from pathlib import Path
import tempfile
import unittest
from wwdc_repro.shared.search_runs import search_run


class SearchRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'arm_seed42'
        self.config = {'seed': 42, 'inputs': {'matrix': 'hash'}}
        self.outputs = ('front.csv',)

    def run_context(self, **kwargs):
        return search_run(self.path, self.config, self.outputs, **kwargs)

    def finish(self):
        with self.run_context() as needed:
            self.assertTrue(needed)
            (self.path / 'front.csv').write_text('equation\nX1\n')

    def test_completed_run_skips_only_if_settings_and_outputs_match(self):
        self.finish()
        with self.run_context() as needed:
            self.assertFalse(needed)
        self.config['seed'] = 43
        with self.assertRaisesRegex(ValueError, 'Inputs/settings differ'):
            with self.run_context(restart_incomplete=True):
                self.fail('must not run')
        self.config['seed'] = 42
        (self.path / 'front.csv').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'missing or changed'):
            with self.run_context():
                self.fail('must not run')

    def test_interruption_requires_explicit_restart_and_preserves_attempt(self):
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            with self.run_context():
                (self.path / 'partial.txt').write_text('keep')
                raise RuntimeError('interrupted')
        self.assertFalse((self.path / 'search_complete.json').exists())
        with self.assertRaisesRegex(FileExistsError, '--restart-incomplete'):
            with self.run_context():
                self.fail('must not run')
        with self.run_context(restart_incomplete=True):
            (self.path / 'front.csv').write_text('new result')
        attempts = list((self.path.parent / 'incomplete_attempts').iterdir())
        self.assertEqual((attempts[0] / 'partial.txt').read_text(), 'keep')
        self.assertTrue((self.path / 'search_complete.json').exists())

    def test_missing_output_never_marks_complete(self):
        with self.assertRaises(FileNotFoundError):
            with self.run_context():
                pass
        self.assertFalse((self.path / 'search_complete.json').exists())

    def test_second_writer_cannot_restart_active_search(self):
        with self.run_context():
            with self.assertRaisesRegex(RuntimeError, 'already running'):
                with self.run_context(restart_incomplete=True):
                    self.fail('must not run')
            (self.path / 'front.csv').write_text('result')

    def test_old_manifest_alone_is_not_verified_completion(self):
        self.path.mkdir()
        (self.path / 'manifest.json').write_text(json.dumps({'seed': 42}))
        with self.assertRaises(FileExistsError):
            with self.run_context():
                self.fail('must not run')


if __name__ == '__main__':
    unittest.main()
