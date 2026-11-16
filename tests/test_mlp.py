"""Check external validation and training-only scaling for the sklearn MLP."""

from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
from sklearn.metrics import mean_squared_error
from wwdc_repro.shared.mlp import fit_mlp, predict_mlp, run_mlp_experiment


class MLPTests(unittest.TestCase):
    def test_external_validation_checkpoint_and_scaling(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(1100, 12))
        y = 3 + 2 * x[:, 0] - x[:, 1]
        xv = rng.normal(loc=2, size=(80, 12))
        yv = 3 + 2 * xv[:, 0] - xv[:, 1]
        checkpoint, history = fit_mlp(
            x, y, xv, yv, seed=42, max_epochs=8, patience=3
        )
        np.testing.assert_allclose(checkpoint['model']['scaler'].mean_, x.mean(0))
        np.testing.assert_allclose(checkpoint['target_scaler'].mean_, [y.mean()])
        self.assertEqual(checkpoint['best_epoch'],
                         int(history.loc[history.val_mse_dex.idxmin(), 'epoch']))
        prediction = predict_mlp(checkpoint, xv)
        np.testing.assert_allclose(mean_squared_error(yv, prediction),
                                   history.val_mse_dex.min())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'checkpoint.joblib'
            joblib.dump(checkpoint, path)
            np.testing.assert_array_equal(predict_mlp(joblib.load(path), xv), prediction)

    def test_spectral_scaling_options_and_saved_prediction(self):
        rng = np.random.default_rng(19)
        x = rng.normal(size=(300, 6)).astype(np.float32)
        xv = rng.normal(loc=20, size=(20, 6)).astype(np.float32)
        y, yv = 10+x[:,0], 10+xv[:,0]
        checkpoint, history = fit_mlp(x,y,xv,yv,seed=42,max_epochs=3,patience=2,
            asinh=True,threads=1,model_parameters=dict(hidden_layer_sizes=(8,4),
                batch_size=256,alpha=1e-4,learning_rate_init=1e-4))
        pipeline = checkpoint['model']
        q = float(np.median(np.abs(x)))
        self.assertEqual(pipeline['asinh'].kw_args['scale'],q)
        np.testing.assert_allclose(pipeline['scaler'].mean_,np.arcsinh(x/q).mean(0),atol=1e-7)
        self.assertEqual(pipeline['mlp'].hidden_layer_sizes,(8,4))
        self.assertEqual(pipeline['mlp'].alpha,1e-4)
        pred = predict_mlp(checkpoint,xv)
        np.testing.assert_allclose(mean_squared_error(yv,pred),history.val_mse_dex.min(),rtol=1e-5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'spectra.joblib'
            joblib.dump(checkpoint,path)
            np.testing.assert_array_equal(predict_mlp(joblib.load(path),xv),pred)
        self.assertEqual(pipeline['asinh'].kw_args['scale'],q)

    def test_full_training_scaling_precedes_model_subsample(self):
        from unittest.mock import patch
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        rng = np.random.default_rng(31)
        x = rng.normal(size=(80, 4))
        x[20:] += 10
        y = 5 + x[:,0]
        indices = np.arange(19, -1, -1)
        xv, yv = x[:8]+100, y[:8]+50
        for asinh in (False, True):
            calls = []
            original = MLPRegressor.partial_fit
            def record(model, features, targets):
                calls.append((features.copy(), targets.copy()))
                return original(model, features, targets)
            with patch.object(MLPRegressor, 'partial_fit', record):
                checkpoint, _ = fit_mlp(x,y,xv,yv,seed=42,max_epochs=1,
                    train_indices=indices,asinh=asinh,threads=1,
                    model_parameters=dict(hidden_layer_sizes=(4,2),batch_size=16))
            transformed = np.arcsinh(x/np.median(np.abs(x))) if asinh else x
            expected = StandardScaler().fit(transformed)
            np.testing.assert_allclose(checkpoint['model']['scaler'].mean_,expected.mean_)
            np.testing.assert_allclose(calls[0][0],expected.transform(transformed[indices]))
            np.testing.assert_allclose(calls[0][1],StandardScaler().fit(y[:,None]).transform(y[indices,None]).ravel())
            self.assertEqual(len(calls[0][0]),20)
            self.assertEqual(checkpoint['n_preprocessing_train'],80)
            self.assertEqual(checkpoint['n_fit_train'],20)
            if asinh:
                self.assertEqual(checkpoint['model']['asinh'].kw_args['scale'],float(np.median(np.abs(x))))

    def test_experiment_supports_both_input_widths(self):
        import contextlib
        import io
        import pandas as pd

        rng = np.random.default_rng(17)
        for width in (10, 12):
            with self.subTest(width=width), tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp)
                x = rng.normal(size=(10000, width))
                y = x[:, 0] + 2
                np.savez(base / 'norm_matrices.npz', X_train=x, y_train=y,
                         X_val=x[:30], y_val=y[:30], X_test=x[30:60], y_test=y[30:60])
                np.save(base / 'train_indices_seed42.npy', np.arange(10000))
                out = base / 'results'
                with contextlib.redirect_stdout(io.StringIO()):
                    summary = run_mlp_experiment(
                        base, out, arms=('norm',), n_features=width,
                        target='test', inputs='test', seeds=(42,), max_epochs=2,
                    )
                checkpoint = joblib.load(out / 'norm_seed42.joblib')
                self.assertEqual(checkpoint['model']['mlp'].n_features_in_, width)
                prediction = predict_mlp(checkpoint, x[30:60])
                np.testing.assert_array_equal(
                    prediction, np.load(out / 'norm_seed42_test_predictions.npy'))
                results = pd.read_csv(out / 'results.csv')
                self.assertEqual(results.train_rows.iloc[0], 10000)
                self.assertEqual(summary.arm.tolist(), ['norm'])


if __name__ == '__main__':
    unittest.main()
