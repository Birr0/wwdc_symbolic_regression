"""Scientific contracts for the matched spectral/latent benchmark."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import torch

from wwdc_repro.redshift import stellar_mass_mlp as mlp
from wwdc_repro.shared.search_runs import search_run
from wwdc_repro.shared.utils import sha256


class StellarMassMLPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        torch.set_num_threads(1)

    def test_prepare_cache_uses_local_loader_and_explicit_data_root(self):
        from datasets import Dataset, DatasetDict
        raw = self.home/'raw'
        wave = (10 ** torch.arange(3.578, 3.97, 0.0001)).numpy()
        spectrum = dict(flux=np.full(len(wave), 2., dtype=np.float32),
                        ivar=np.ones(len(wave), dtype=np.float32),
                        **{'lambda':wave}, mask=np.zeros(len(wave), dtype=bool))
        splits = DatasetDict({s:Dataset.from_dict(dict(
            object_id=['101','202'], Z=[0.1,0.1], spectrum=[spectrum,spectrum]))
            for s in mlp.SPLITS})
        splits.save_to_disk(str(raw/'sdss/sdss_II_catalog'))
        frames = {s:pd.DataFrame(dict(specObjID=[202,101],lgm_tot_p50=[10.,11.]))
                  for s in mlp.SPLITS}
        paths = dict(data_root=self.home/'catalogues', embeddings=self.home/'embeddings')
        cache = self.home/'cache'
        with patch.object(mlp, 'sample_sources', return_value=[]), \
             patch.object(mlp, 'benchmark_frames', return_value=frames):
            mlp.prepare_cache(paths, raw, cache, threads=1)
            mlp.prepare_cache(paths, raw, cache, threads=1)
        values = mlp.read_spectra(cache, frames)
        for split in mlp.SPLITS:
            np.testing.assert_array_equal(values[split], np.ones((2,3921)))
        np.testing.assert_array_equal(np.load(cache/'wave_observed.npy'), wave)
        self.assertFalse((cache/'upstream_sdss.py').exists())
        raw_cache = self.home/'raw-cache'
        with patch.object(mlp, 'sample_sources', return_value=[]), \
             patch.object(mlp, 'benchmark_frames', return_value=frames):
            mlp.prepare_cache(paths, raw, raw_cache, threads=1, raw_flux=True)
        raw_values = mlp.read_spectra(raw_cache, frames, raw_flux=True)
        for split in mlp.SPLITS:
            np.testing.assert_array_equal(raw_values[split], np.full((2,3921),2.))
        with self.assertRaisesRegex(ValueError, 'normalisation'):
            mlp.read_spectra(raw_cache, frames)
        with self.assertRaisesRegex(ValueError, 'normalisation'):
            mlp.read_spectra(cache, frames, raw_flux=True)

    def test_preparation_matches_ids_instead_of_assuming_dataset_order(self):
        class Dataset:
            dataset = {'object_id':[b'202',b'101',b'303']}
            _wave_obs = torch.arange(3921)
            def __getitem__(self, index):
                identity = self.dataset['object_id'][index]
                return torch.full((3921,),float(identity)), None, identity
        frame = pd.DataFrame(dict(specObjID=[303,101],lgm_tot_p50=[11.,10.]))
        result = mlp.write_spectral_split(Dataset(),frame,self.home,'train')
        self.assertEqual(result['n'],2)
        values = np.load(self.home/'train_spectra.npy')
        np.testing.assert_array_equal(values[:,0],[303,101])
        pd.testing.assert_frame_equal(pd.read_csv(self.home/'train_rows.csv'),frame)
        missing = pd.DataFrame(dict(specObjID=[404],lgm_tot_p50=[10.]))
        with self.assertRaisesRegex(ValueError,'absent'):
            mlp.write_spectral_split(Dataset(),missing,self.home,'val')

    def test_preparation_rejects_nonfinite_flux_and_missing_dataset(self):
        class Dataset:
            dataset = {'object_id':['101']}
            _wave_obs = torch.arange(3921)
            def __getitem__(self, index):
                return torch.full((3921,),float('nan')), None, '101'
        frame = pd.DataFrame(dict(specObjID=[101],lgm_tot_p50=[10.]))
        with self.assertRaisesRegex(ValueError,'invalid prepared spectrum'):
            mlp.write_spectral_split(Dataset(),frame,self.home,'train')
        with self.assertRaisesRegex(FileNotFoundError,'Existing SDSS split'):
            mlp.spectral_dataset_sources(self.home)

    def test_generated_cache_receipt_rejects_modified_arrays(self):
        cache = self.home/'cache'
        frames = {s:pd.DataFrame(dict(specObjID=[i*10+1,i*10+2],lgm_tot_p50=[10.,11.]))
                  for i,s in enumerate(mlp.SPLITS)}
        products = [f'{s}_{suffix}' for s in mlp.SPLITS for suffix in ('spectra.npy','rows.csv')]
        metadata = {}
        with search_run(cache,{'fixture':True},products,completion_metadata=metadata):
            for split in mlp.SPLITS:
                frames[split].to_csv(cache/f'{split}_rows.csv',index=False)
                np.save(cache/f'{split}_spectra.npy',np.ones((2,3921),dtype=np.float32))
            metadata.update(samples={s:{'n':len(frames[s])} for s in mlp.SPLITS},
                            preprocessing='test fixture')
        record = json.loads((cache/'search_complete.json').read_text())
        self.assertEqual(record['metadata']['samples']['train']['n'],2)
        self.assertFalse((cache/'cache_manifest.json').exists())
        with search_run(cache,{'fixture':True},products,completion_metadata={}) as required:
            self.assertFalse(required)
        self.assertEqual(json.loads((cache/'search_complete.json').read_text()),record)
        mlp.read_spectra(cache,frames)
        np.save(cache/'test_spectra.npy',np.zeros((2,3921),dtype=np.float32))
        with self.assertRaisesRegex(ValueError,'cache missing/changed'):
            mlp.read_spectra(cache,frames)

    def test_spectral_cache_requires_exact_ids_order_targets_and_pixels(self):
        frames = {}
        for i, split in enumerate(mlp.SPLITS):
            frames[split] = pd.DataFrame(dict(specObjID=[10*i+1,10*i+2], lgm_tot_p50=[10.,11.]))
            frames[split].to_csv(self.home/f'{split}_rows.csv', index=False)
            np.save(self.home/f'{split}_spectra.npy', np.ones((2,3921), dtype=np.float32))
        self.assertEqual(mlp.read_spectra(self.home, frames)['test'].shape, (2,3921))
        frames['test'].iloc[::-1].to_csv(self.home/'test_rows.csv', index=False)
        with self.assertRaisesRegex(ValueError, 'IDs/order'):
            mlp.read_spectra(self.home, frames)
        frames['test'].to_csv(self.home/'test_rows.csv', index=False)
        altered = frames['test'].copy()
        altered['lgm_tot_p50'] += 1
        altered.to_csv(self.home/'test_rows.csv', index=False)
        with self.assertRaisesRegex(ValueError, 'targets'):
            mlp.read_spectra(self.home, frames)

    def test_checkpoint_and_selection_ignore_test_target_changes(self):
        rng = np.random.default_rng(17)
        arrays = {s:rng.normal(size=(n,3)).astype(np.float32)
                  for s,n in [('train',30),('val',12),('test',14)]}
        targets = {s:(10+v[:,0]-.5*v[:,1]).astype(np.float32) for s,v in arrays.items()}
        frames = {s:pd.DataFrame(dict(specObjID=np.arange(len(v))+i*100))
                  for i,(s,v) in enumerate(arrays.items())}
        kwargs = dict(arm='orig',seed=42,threads=1,max_epochs=4,patience=2,
                      widths=((8,4),(12,6)),rates=(.001,),train_indices=np.arange(19,-1,-1))
        first = mlp.fit_arm(arrays,targets,frames,self.home/'first',**kwargs)
        changed = dict(targets,test=targets['test']+100)
        second = mlp.fit_arm(arrays,changed,frames,self.home/'second',**kwargs)
        self.assertEqual(first['selected'],second['selected'])
        grid = pd.read_csv(self.home/'first/trial_metrics.csv')
        self.assertEqual(len(grid), 6)
        self.assertEqual(set(grid['split']), set(mlp.SPLITS))
        self.assertEqual(int(grid.selected.sum()), 3)
        self.assertTrue(grid[grid['split']=='train'].n.eq(20).all())
        saved_train = pd.read_csv(self.home/'first/train_predictions.csv')
        np.testing.assert_array_equal(saved_train.specObjID,frames['train'].specObjID.iloc[np.arange(19,-1,-1)])
        for trial in grid.trial.unique():
            self.assertTrue((self.home/'first'/f'{trial}.joblib').is_file())
        a = pd.read_csv(self.home/'first/test_predictions.csv')
        b = pd.read_csv(self.home/'second/test_predictions.csv')
        np.testing.assert_array_equal(a.prediction,b.prediction)
        checkpoint = joblib.load(self.home/'first/model.joblib')
        self.assertEqual(checkpoint['n_preprocessing_train'],30)
        self.assertEqual(checkpoint['n_fit_train'],20)
        pred = mlp.shared_mlp.predict_mlp(checkpoint, arrays['test'], threads=1)
        np.testing.assert_allclose(pred,a.prediction,atol=2e-6)

    def test_export_verifies_predictions_and_never_refits(self):
        run = self.home/'run'
        row = dict(arm='orig',seed=42,selected=dict(trial='fixture',hidden_1=8,hidden_2=4,
                   learning_rate=.001,best_epoch=2,epochs_run=4,validation_mse_standardised=.2),
                   metrics={s:dict(n=10,r2=.8,rmse_dex=.2,mae_dex=.1) for s in mlp.SPLITS})
        with search_run(run,{'fixture':True},['results.json','predictions.csv','orig/trial_metrics.csv']):
            (run/'results.json').write_text(json.dumps(dict(configuration={'seed':42},samples={},results=[row])))
            (run/'predictions.csv').write_text('truth,prediction\n10,10\n')
            (run/'orig').mkdir()
            pd.DataFrame([dict(arm='orig', seed=42, **row['selected'], selected=True,
                               split=split, **row['metrics'][split]) for split in mlp.SPLITS]).to_csv(
                                   run/'orig/trial_metrics.csv',index=False)
        mlp.export_results(run,self.home/'tables')
        mlp.export_grid([run,run],self.home/'tables')
        self.assertEqual(len(pd.read_csv(self.home/'tables/stellar_mass_mlp_grid.csv')),3)
        tex = (self.home/'tables/stellar_mass_mlp_grid.tex').read_text()
        self.assertIn('Test $R^2$',tex)
        self.assertNotIn('longtable',tex)
        self.assertNotIn(' & Validation & ',tex)
        self.assertIn('0.8000',(self.home/'tables/stellar_mass_mlp_numbers.tex').read_text())
        (run/'predictions.csv').write_text('modified')
        with self.assertRaisesRegex(ValueError,'missing/changed'):
            mlp.export_results(run,self.home/'tables')


if __name__ == '__main__':
    unittest.main()
