"""Shared eligibility policy, including the explicit full-training option."""
import unittest
import numpy as np
import pandas as pd
from wwdc_repro.shared.selection import audit_candidate, select_candidate, report_front


class SharedSelectionTests(unittest.TestCase):
    def test_training_default_requires_no_validation_and_ignores_its_failure(self):
        train = np.array([[1.], [2.]])
        data = {'X_train': train, 'y_train': np.array([1., .5])}
        result = audit_candidate('1/X1', data, np.arange(2), 1.)
        self.assertTrue(result['selectable'])
        self.assertEqual(result['selection'], 'train')
        data.update(X_val=np.zeros((2, 1)), y_val=np.array([1., 2.]))
        self.assertTrue(audit_candidate('1/X1', data, np.arange(2), 1.)['selectable'])
        self.assertFalse(audit_candidate('1/X1', data, np.arange(2), 1.,
                                         selection='validation')['selectable'])

    def test_training_default_and_optional_validation_choose_different_winners(self):
        rows = pd.DataFrame({'length': [5, 6], 'train_r2': [.9, .8],
                             'validation_r2': [.6, .95], 'test_r2': [np.nan, 1.],
                             'selectable': [True, True], 'equation': ['train', 'val']})
        self.assertEqual(select_candidate(rows, 6).equation, 'train')
        self.assertEqual(select_candidate(rows, 6, selection='validation').equation, 'val')
        rows['test_r2'] = [1., -100.]
        self.assertEqual(select_candidate(rows, 6).equation, 'train')
        self.assertEqual(select_candidate(rows, 6, selection='validation').equation, 'val')

    def test_report_front_retains_training_winner_when_test_fails(self):
        x = np.array([[1.], [2.], [4.]])
        data = {'X_train': x, 'y_train': 1 / x[:, 0],
                'X_test': np.array([[0.], [1.]]), 'y_test': np.array([0., 1.])}
        from wwdc_repro.shared.selection import score_expression
        audit = pd.DataFrame([
            {'seed': 42, 'Length': 5, 'equation': eq,
             'train_r2_stored': score_expression(eq, x, data['y_train']),
             'selectable': False, 'test_r2': 1.}
            for eq in ('1/X1', '0.5')
        ])
        result = report_front(audit, data, budgets=[5, 6])
        self.assertEqual(result.equation.tolist(), ['1/X1', '1/X1'])
        self.assertTrue(result.test_r2.isna().all())
        self.assertTrue((result.test_status == 'evaluation_failure').all())
        self.assertTrue((result.selection == 'train').all())

    def test_full_training_policy_is_explicit_and_never_needs_test_data(self):
        data = {'X_train': np.array([[1.], [2.], [0.]]),
                'y_train': np.array([1., .5, 0.]),
                'X_val': np.array([[1.], [2.]]), 'y_val': np.array([1., .5])}
        indices = np.array([0, 1])
        # Fit and validation rows are finite, but an unsampled training row fails.
        redshift = audit_candidate('1/X1', data, indices, 1.)
        conditioning = audit_candidate('1/X1', data, indices, 1., check_full_training=True, selection="validation")
        self.assertTrue(redshift['selectable'])
        self.assertFalse(conditioning['selectable'])
        self.assertFalse(conditioning['train_full_finite'])

    def test_reference_mismatch_rejects_candidate(self):
        x=np.array([[1.], [2.]])
        data={'X_train':x,'y_train':x[:,0],'X_val':x,'y_val':x[:,0]}
        self.assertFalse(audit_candidate('X1',data,np.arange(2),.5)['selectable'])

    def test_column_adapters_share_ties_and_ignore_test_scores(self):
        rows=pd.DataFrame({'length':[6,5,5],'validation_r2':[.8,.8,.8], 'train_r2':[.8,.8,.8],
                           'selectable':[True]*3,'equation':['long','first','second'],
                           'test_r2':[1.,np.nan,1.]})
        self.assertEqual(select_candidate(rows,6).equation,'first')
        legacy=rows.rename(columns={'length':'Length','validation_r2':'val_r2'})
        self.assertEqual(select_candidate(legacy,6,length_column='Length',validation_column='val_r2').equation,'first')
        self.assertIsNone(select_candidate(rows,4))


if __name__=='__main__':unittest.main()
