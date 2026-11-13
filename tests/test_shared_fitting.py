"""Verify shared search inputs and the validation/test boundary without long searches."""

import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import wwdc_repro.shared.fitting as utils
from wwdc_repro.shared.conditioning_data import build_matrices
import wwdc_repro.norm.matched_conditioning_controls as controls
import wwdc_repro.workflows.run_total_mass as run_total_mass
import wwdc_repro.norm.run_normflow as run_normflow


class RecordingRegressor:
    """Capture fitter inputs; expose one simple equation for orchestration tests."""

    def __init__(self, **parameters):
        self.parameters = parameters
        self.model_ = types.SimpleNamespace(Length=5)
        self.pareto_front_ = [{"tree": self.model_, "model": "X1"}]

    def fit(self, x, y):
        self.x, self.y = x.copy(), y.copy()
        return self

    def get_model_string(self, tree, precision):
        return "X1"

    def get_params(self):
        return self.parameters

    def evaluate_model(self, tree, x):
        return x[:, 0]


def fake_operon():
    package = types.ModuleType("pyoperon")
    module = types.ModuleType("pyoperon.sklearn")
    module.SymbolicRegressor = RecordingRegressor
    return {"pyoperon": package, "pyoperon.sklearn": module}


def frames():
    rng = np.random.default_rng(12)
    result = {}
    for split, n in [("train", 10007), ("val", 9), ("test", 13)]:
        latents = rng.normal(size=(n, 10))
        target = rng.normal(10, 0.5, n)
        target[:2] = [np.nan, -9999]
        result[split] = pd.DataFrame(
            {
                "orig": list(latents),
                "cond": list(latents * 2 + 3),
                "uncond": list(latents - 2),
                "z_x": rng.uniform(0.02, 0.3, n),
                "lgm_tot_p50": target,
                "y": list(rng.normal(size=(n, 1))),
            }
        )
    return result


class SharedFittingTests(unittest.TestCase):
    def test_original_redshift_baseline_uses_original_latents(self):
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler

        data = frames()
        for frame in data.values():
            frame["cond"] = list(np.stack(frame.orig) ** 2)
        estimator = LinearRegression()
        result = utils.fit_model_fn(
            "lgm_tot_p50", "orig+z", estimator, data
        )
        raw = {
            split: np.column_stack((np.stack(frame.orig)[2:], frame.z_x.to_numpy()[2:]))
            for split, frame in data.items()
        }
        scaler = StandardScaler().fit(raw["train"])
        expected = LinearRegression().fit(
            scaler.transform(raw["train"]), data["train"].lgm_tot_p50.to_numpy()[2:]
        )
        np.testing.assert_allclose(estimator.coef_, expected.coef_)
        self.assertAlmostEqual(
            result["Test_R2"],
            expected.score(scaler.transform(raw["test"]), data["test"].lgm_tot_p50.to_numpy()[2:]),
        )

    def test_front_exports_precise_equation_and_keeps_train_score_when_test_fails(self):
        train = np.arange(4.).reshape(-1, 1)
        test = np.array([[np.nan]])
        model = RecordingRegressor()
        model.pareto_front_[0]['model'] = '0.000000 * X1'
        result = dict(Model=model, X_train=train, y_train=train[:, 0],
                      X_test=test, y_test=np.ones(1))
        with patch.object(run_total_mass, 'fit_sym_fn', return_value=result), \
             patch.object(run_total_mass, 'fit_model_fn', return_value={'Test_R2': 0., 'Test_MSE': 1.}):
            front, _ = run_total_mass.run_cell('mass', 'cond', 42, {})
        self.assertEqual(front.iloc[0].Train_R2, 1.)
        self.assertEqual(front.iloc[0].Train_MSE, 0.)
        self.assertEqual(front.iloc[0].Test_R2, -np.inf)
        self.assertEqual(front.iloc[0].Equation_p60, 'X1')

    def test_flux_targets_and_appended_features_with_partial_splits(self):
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(51)
        data = {}
        raw = {}
        for split in ("train", "test"):
            raw[split] = rng.normal(size=(8, 11))
            data[split] = pd.DataFrame({
                "cond": list(raw[split][:, None, :]),
                "z_x": np.linspace(0.03, 0.2, 8),
                "flux": [np.nan, -9999., 0., -1., 1., 10., 100., 1000.],
            })
        result = build_matrices(data, "flux", "cond+z")
        scaler = StandardScaler().fit(raw["train"][4:])
        for split in data:
            expected = np.column_stack((scaler.transform(raw[split][4:]),
                                        data[split].z_x.to_numpy()[4:]))
            np.testing.assert_array_equal(result[f"X_{split}"], expected)
            np.testing.assert_array_equal(result[f"y_{split}"], [0., 1., 2., 3.])
            self.assertEqual(np.stack(data[split].cond).shape, (8, 1, 11))

    def test_all_arms_match_historical_scaling_sampling_and_parameters(self):
        # Extract just the archived function and constants; do not alter or import
        # its unused plotting dependencies. Its constructor is replaced, not its
        # numerical preprocessing or sampling.
        source = ast.parse((ROOT / "experiments/historical/utils.py").read_text())
        function = next(
            n
            for n in source.body
            if isinstance(n, ast.FunctionDef) and n.name == "fit_sym_fn"
        )
        from sklearn.preprocessing import StandardScaler

        namespace = dict(
            np=np,
            StandardScaler=StandardScaler,
            SymbolicRegressor=RecordingRegressor,
            N=10000,
            symbols=utils.SEARCH_PARAMETERS["allowed_symbols"],
        )
        exec(
            compile(
                ast.Module(body=[function], type_ignores=[]),
                "<historical fitter>",
                "exec",
            ),
            namespace,
        )
        data = frames()
        with patch.dict(sys.modules, fake_operon()):
            for arm in ("z", "cond", "cond+z", "uncond", "orig", "orig+z"):
                for seed in (42, 43, 44):
                    with self.subTest(arm=arm, seed=seed):
                        historical_data = ({split: frame.assign(cond=frame.orig)
                                            for split, frame in data.items()}
                                           if arm == "orig+z" else data)
                        old = namespace["fit_sym_fn"](
                            "lgm_tot_p50", "cond+z" if arm == "orig+z" else arm,
                            historical_data, seed)
                        new = utils.fit_sym_fn("lgm_tot_p50", arm, data, seed)
                        mats = build_matrices(data, "lgm_tot_p50", arm)
                        idx = np.random.default_rng(seed).choice(
                            len(mats["y_train"]), 10000, replace=False
                        )
                        direct = utils.fit_sym_fn(
                            "lgm_tot_p50",
                            arm,
                            None,
                            seed,
                            matrices=mats,
                            train_indices=idx,
                        )
                        for key in ("X_train", "y_train", "X_test", "y_test"):
                            np.testing.assert_array_equal(new[key], old[key])
                            np.testing.assert_array_equal(direct[key], old[key])
                        self.assertEqual(
                            new["Model"].get_params(), old["Model"].get_params()
                        )
                        self.assertEqual(
                            direct["Model"].get_params(), old["Model"].get_params()
                        )

    def test_explicit_rows_and_overrides_are_preserved_and_bad_indices_rejected(self):
        x = np.arange(60.0).reshape(20, 3)
        mats = dict(X_train=x, y_train=x[:, 0], X_test=x[:3], y_test=x[:3, 0])
        with patch.dict(sys.modules, fake_operon()):
            result = utils.fit_sym_fn(
                "mass",
                "cond+z",
                None,
                43,
                matrices=mats,
                train_indices=np.array([9, 2, 17]),
                parameter_overrides={"max_length": 50},
            )
            np.testing.assert_array_equal(result["X_train"], x[[9, 2, 17]])
            self.assertEqual(result["Model"].get_params()["max_length"], 50)
            self.assertEqual(utils.search_parameters(43)["max_length"], 25)
            for idx in ([1, 1], [-1], [20], [1.5], []):
                with self.assertRaises(ValueError):
                    utils.fit_sym_fn(
                        "mass", "cond+z", None, 43, matrices=mats, train_indices=idx
                    )
            with self.assertRaises(ValueError):
                utils.search_parameters(43, {"random_state": 42})

    def test_prepared_fit_needs_no_previous_run_and_freezes_before_test(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(sys.modules, fake_operon()),
        ):
            out = Path(tmp)
            x = np.arange(30.0).reshape(10, 3)
            np.savez(
                out / "original_matrices.npz",
                X_train=x,
                y_train=x[:, 0],
                X_val=x + 1,
                y_val=x[:, 0] + 1,
                X_test=x + 100,
                y_test=x[:, 0] + 100,
            )
            np.save(out / "train_indices_seed42.npy", np.array([5, 1, 8]))
            (out / "train_ids_seed42.csv").write_text("row,specObjID\n5,5\n1,1\n8,8\n")
            real_score = controls.score

            def test_score(eq, inputs, targets):
                frozen = pd.read_csv(
                    out / "original_seed42/selection_frozen_before_test.csv"
                )
                self.assertNotIn("test_r2", frozen)
                return real_score(eq, inputs, targets)

            with (
                patch.object(controls, "OUT", out),
                patch.object(controls, "BASE", out / "no_prior_runs"),
                patch.object(controls, "score", side_effect=test_score),
                patch.object(
                    controls.importlib.metadata, "version", return_value="test"
                ),
            ):
                controls.fit("original", 42, prepared_indices=True)
            audit = pd.read_csv(out / "original_seed42/candidate_audit.csv")
            self.assertTrue(audit.selectable.all())
            self.assertFalse({"native_aic", "native_bic", "native_mdl"} & set(audit))
            chosen = pd.read_csv(out / "original_seed42/selected.csv")
            np.testing.assert_allclose(chosen.test_r2, 1)
            manifest = json.loads((out / "original_seed42/manifest.json").read_text())
            self.assertEqual(manifest["parameters"]["random_state"], 42)
            self.assertTrue(
                all(Path(name).suffix != ".py" for name in manifest["input_hashes"])
            )

    def test_origz_cli_builds_original_latents_and_standard_front_path(self):
        from sklearn.preprocessing import StandardScaler
        from wwdc_repro.redshift.frontier import front_path
        data = frames()
        # Avoid affine-related cond/orig latents that become identical after scaling.
        for frame in data.values():
            frame['cond'] = list(np.stack(frame.orig) ** 2)
        original_cond = {split: np.stack(frame.cond).copy() for split, frame in data.items()}
        clean = {split: frame.loc[frame.lgm_tot_p50.notna() & (frame.lgm_tot_p50 != -9999)]
                 for split, frame in data.items()}
        scaler = StandardScaler().fit(np.stack(clean['train'].orig))

        def cell(feature, arm, seed, merged, *, threads=8):
            self.assertEqual((arm, seed), ('orig+z', 42))
            mats = build_matrices(merged, feature, arm)
            for split, frame in clean.items():
                expected = np.column_stack((scaler.transform(np.stack(frame.orig)), frame.z_x))
                np.testing.assert_array_equal(mats[f'X_{split}'], expected)
                np.testing.assert_array_equal(np.stack(merged[split].cond), original_cond[split])
            return pd.DataFrame([{'Feature': feature, 'Equation': 'X11'}]), {'Feature': feature}

        with tempfile.TemporaryDirectory() as tmp:
            argv = ['run_total_mass', '--embed-type', 'orig+z', '--seed', '42', '--out', tmp, '--embeddings', tmp]
            for split in ('train', 'val', 'test'):
                (Path(tmp) / split).mkdir()
                (Path(tmp) / split / 'data.parquet').touch()
            with patch.object(sys, 'argv', argv), \
                 patch.object(run_total_mass, 'build_merged', return_value=data), \
                 patch.object(run_total_mass, 'input_hashes', return_value={}), \
                 patch.object(run_total_mass, 'run_cell', side_effect=cell) as fit:
                run_total_mass.main()
            fit.assert_called_once()
            self.assertTrue(front_path(Path(tmp), 'orig+z', 42).is_file())

    def test_joint_cli_appends_only_norm_and_keeps_distinct_front_name(self):
        data = frames()
        for frame in data.values():
            frame['y'] = list(np.column_stack((frame.z_x, np.arange(len(frame)) + 10.)))
        def cell(feature, arm, seed, merged, *, threads):
            for frame in merged.values():
                self.assertEqual(np.stack(frame.cond).shape[1], 11)
                np.testing.assert_array_equal(np.stack(frame.cond)[:, -1], np.stack(frame.y)[:, -1])
            return pd.DataFrame([{'Feature': feature, 'Equation': 'X1'}]), {'Feature': feature}
        with tempfile.TemporaryDirectory() as tmp:
            for split in ('train', 'val', 'test'):
                (Path(tmp) / split).mkdir()
                (Path(tmp) / split / 'data.parquet').touch()
            argv = ['run_total_mass', '--conditioning', 'znorm', '--append-norm',
                    '--embed-type', 'cond+z', '--seed', '42', '--embeddings', tmp, '--out', tmp]
            with patch.object(sys, 'argv', argv), patch.object(run_total_mass, 'input_hashes', return_value={}), \
                 patch.object(run_total_mass, 'build_merged', return_value=data), \
                 patch.object(run_total_mass, 'run_cell', side_effect=cell):
                run_total_mass.main()
            self.assertTrue((Path(tmp) / 'job_results_znormflow_condnorm_lgm_tot_p50_cond+z_seed42/pareto_fronts.csv').is_file())

    def test_norm_cli_keeps_output_names_without_target_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = frames()

            def cell(feature, arm, seed, merged, *, threads=8):
                self.assertEqual(feature, "lgm_tot_p50")
                self.assertEqual(arm, "cond+z")
                for split, frame in merged.items():
                    self.assertEqual(np.stack(frame.cond).shape[1], 11)
                    self.assertNotIn("normflow_condnorm_lgm_tot_p50", frame)
                    np.testing.assert_array_equal(
                        np.stack(frame.cond)[:, -1], np.stack(frame.y)[:, 0]
                    )
                return pd.DataFrame([{"Feature": feature, "Equation": "X1"}]), {
                    "Feature": feature
                }

            argv = [
                "run_normflow", "--embeddings", tmp,
                "--append-norm",
                "--embed-type",
                "cond+z",
                "--seed",
                "42",
                "--out",
                tmp,
            ]
            for split in ("train", "val", "test"):
                (Path(tmp) / split).mkdir(exist_ok=True)
                (Path(tmp) / split / "data.parquet").touch()
            with (
                patch.object(run_total_mass, "input_hashes", return_value={}),
                patch.object(sys, "argv", argv),
                patch.object(run_total_mass, "build_merged", return_value=data),
                patch.object(run_total_mass, "run_cell", side_effect=cell),
            ):
                run_normflow.main()
            output = (
                Path(tmp) / "job_results_normflow_condnorm_lgm_tot_p50_cond+z_seed42"
            )
            self.assertEqual(
                pd.read_csv(output / "pareto_fronts.csv").Feature.iloc[0],
                "normflow_condnorm_lgm_tot_p50",
            )
            for split in ("train", "val", "test"):
                (Path(tmp) / split).mkdir(exist_ok=True)
                (Path(tmp) / split / "data.parquet").touch()
            with (
                patch.object(run_total_mass, "input_hashes", return_value={}),
                patch.object(sys, "argv", argv),
                patch.object(run_total_mass, "build_merged") as load,
            ):
                run_normflow.main()
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
