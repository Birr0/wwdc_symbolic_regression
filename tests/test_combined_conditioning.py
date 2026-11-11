"""Check recovery integrity and the training/validation/test boundary."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

import wwdc_repro.norm.validate_conditioning as workflow
from wwdc_repro.shared.utils import sha256


class CombinedConditioningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)

    def recovery_fixture(self):
        rows = []
        for arm in ('norm_cond', 'norm_cond+z'):
            source = self.out / f'{arm}.csv'
            source.write_text('original front\n')
            rows.append(dict(arm=arm, seed=42, row=0, source=str(source),
                             Length=5, equation='0 * X1', train_r2_stored=-1.))
            directory = self.out / f'rerun_{arm}_seed42'
            directory.mkdir()
            (directory / 'manifest.json').write_text(json.dumps({
                'exact_front_match': True, 'original_sha256': sha256(source)}))
            pd.DataFrame([dict(Length=5, Equation_p60='X1', Train_R2=1.)]).to_csv(
                directory / 'native_front.csv', index=False)
        return pd.DataFrame(rows)

    def test_recovery_is_checked_and_applied_before_audit(self):
        rows = self.recovery_fixture()
        recovered, manifests = workflow.apply_recoveries(rows, self.out)
        self.assertEqual(len(manifests), 2)
        self.assertTrue(recovered.recovered_export.all())
        self.assertEqual(recovered.equation.tolist(), ['X1', 'X1'])
        self.assertEqual(rows.equation.tolist(), ['0 * X1', '0 * X1'])
        x = np.arange(6.).reshape(-1, 1)
        # No test arrays: auditing must never require them.
        for arm in rows.arm:
            np.savez(self.out / f'{arm}_matrices.npz', X_train=x, y_train=x[:, 0],
                     X_val=x, y_val=x[:, 0])
        audited = workflow.audit_candidates(recovered, self.out)
        self.assertTrue(audited.selectable.all())
        np.testing.assert_allclose(audited.train_r2, 1.)
        Path(rows.source.iloc[0]).write_text('changed front\n')
        with self.assertRaisesRegex(ValueError, 'checksum differs'):
            workflow.apply_recoveries(rows, self.out)

    def test_fresh_precise_exports_do_not_require_recovery_files(self):
        rows = pd.DataFrame([
            dict(arm=arm, seed=42, row=0, source='fresh.csv', Length=5,
                 equation='X1', train_r2_stored=1., export_precision=60)
            for arm in ('norm_cond', 'norm_cond+z')
        ])
        rows.to_csv(self.out / 'candidates.csv', index=False)
        loaded = workflow.load_candidates(self.out)
        result, manifests = workflow.apply_recoveries(loaded, self.out)
        self.assertEqual(manifests, [])
        self.assertFalse(result.recovered_export.any())
        pd.testing.assert_frame_equal(result.drop(columns='recovered_export'), rows)

    def test_recovery_rejects_unverified_front_or_wrong_lengths(self):
        rows = self.recovery_fixture()
        directory = self.out / 'rerun_norm_cond_seed42'
        path = directory / 'manifest.json'
        manifest = json.loads(path.read_text())
        manifest['exact_front_match'] = False
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'did not reproduce'):
            workflow.apply_recoveries(rows, self.out)
        manifest['exact_front_match'] = True
        path.write_text(json.dumps(manifest))
        pd.DataFrame([dict(Length=9, Equation_p60='X1', Train_R2=1.)]).to_csv(
            directory / 'native_front.csv', index=False)
        with self.assertRaisesRegex(ValueError, 'length differs'):
            workflow.apply_recoveries(rows, self.out)

    def test_archived_scores_are_discarded_and_new_metadata_takes_precedence(self):
        rows = self.recovery_fixture()
        rows['selectable'] = False
        rows['val_r2'] = -999.
        rows['test_r2'] = 1.
        rows.to_csv(self.out / 'candidate_audit.csv', index=False)
        loaded = workflow.load_candidates(self.out)
        self.assertFalse({'selectable', 'val_r2', 'test_r2'} & set(loaded.columns))
        loaded['equation'] = '2 * X1'
        loaded.to_csv(self.out / 'candidates.csv', index=False)
        self.assertEqual(workflow.load_candidates(self.out).equation.tolist(), ['2 * X1'] * 2)

    def test_run_freezes_choices_before_test_and_keeps_failed_winner(self):
        rows = self.recovery_fixture()
        rows.to_csv(self.out / 'candidate_audit.csv', index=False)
        x = np.arange(6.).reshape(-1, 1)
        for arm in rows.arm:
            np.savez(self.out / f'{arm}_matrices.npz', X_train=x, y_train=x[:, 0],
                     X_val=x, y_val=x[:, 0], X_test=x, y_test=x[:, 0])
        for flow in ('norm', 'znorm', 'z'):
            np.save(self.out / f'{flow}_test_ids.npy', np.arange(6))

        def failed_test_score(*args):
            frozen = pd.read_csv(self.out / 'selection_frozen_before_test.csv')
            self.assertNotIn('test_r2', frozen)
            self.assertEqual(frozen.equation.unique().tolist(), ['X1'])
            return np.nan

        with patch.object(workflow, 'score', side_effect=failed_test_score) as score:
            with patch.object(workflow, 'write_outputs'):
                result = workflow.run(self.out)
        self.assertGreater(score.call_count, 0)
        self.assertEqual(len(result), 50)
        self.assertTrue(result.test_r2.isna().all())
        self.assertTrue((result.test_status == 'evaluation_failure').all())
        self.assertTrue((result.equation == 'X1').all())


if __name__ == '__main__':
    unittest.main()
