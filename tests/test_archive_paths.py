"""Archive import preserves bytes and provenance while using stable output paths."""
import json
from pathlib import Path
import tempfile
import unittest
from wwdc_repro.shared.paths import ARCHIVE_PATHS, working_path
from wwdc_repro.shared.utils import sha256
from wwdc_repro.workflows.conditioning_suite import export, ROOTS


class ArchivePathTests(unittest.TestCase):
    def test_export_supports_legacy_and_stable_roots(self):
        for legacy in (True, False):
            with self.subTest(legacy=legacy), tempfile.TemporaryDirectory() as temp:
                source, work = Path(temp) / 'archive', Path(temp) / 'work'
                for root in ROOTS:
                    folder = source / (ARCHIVE_PATHS['directories'][root] if legacy else root)
                    folder.mkdir(parents=True)
                    (folder / 'matrices.npz').write_bytes(b'unchanged matrix bytes')
                if legacy:
                    # Older working trees may also contain a scripts-only topic folder.
                    (source / 'fibre_mass').mkdir()
                    (source / 'fibre_mass/launcher.py').write_text('# archived launcher')
                cache = source / 'plot_cmd_latent_results/vagc_join.parquet'
                cache.parent.mkdir()
                cache.write_bytes(b'catalogue')
                export(source, work)
                records = json.loads((work / 'conditioning_inputs.json').read_text())['files']
                self.assertEqual(len(records), 4)
                for record in records:
                    original = source / record['source_path']
                    copied = work / record['path']
                    self.assertEqual(original.read_bytes(), copied.read_bytes())
                    self.assertEqual(sha256(copied), record['sha256'])
                for root in ROOTS:
                    self.assertTrue((work / root / 'matrices.npz').is_file())
                with self.assertRaises(FileExistsError):
                    export(source, work)

    def test_nested_historical_paths_and_stable_paths(self):
        old = ARCHIVE_PATHS['mlp_references']['fibre_mass']
        self.assertEqual(working_path(old), Path('fibre_mass/archived_mlp/summary.csv'))
        self.assertEqual(working_path('norm_analysis/results.csv'), Path('norm_analysis/results.csv'))


if __name__ == '__main__':
    unittest.main()
