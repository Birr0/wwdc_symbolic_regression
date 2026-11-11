"""Scientific data invariants, independent of external catalogues."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from wwdc_repro.shared.data import (
    clean_target, latent_array, load_sample, matrices, sample_summary,
    spectrum_ids, validate_split_ids,
)


def frames():
    result = {}
    for index, split in enumerate(("train", "val", "test")):
        x = np.array([np.arange(10), np.arange(10) + 2.]) + 100 * index
        result[split] = pd.DataFrame({
            "specObjID": [2 * index + 1, 2 * index + 2],
            "z_x": [.1, .2], "lgm_tot_p50": [10., 11.],
            **{name: list(x.copy()) for name in ("orig", "cond", "uncond")},
        })
    return result


class DataTests(unittest.TestCase):
    def test_ids_preserve_large_integer_precision_and_reject_embedded_digits(self):
        value = 1548176745079793664
        parsed = spectrum_ids(pd.Series([value, str(value), f"b'  {value}'", str(value).encode()]), "fixture")
        self.assertEqual(parsed.tolist(), [value] * 4)
        for invalid in ("object123", "123.0", "-123", "123junk", None, str(2**63)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                spectrum_ids(pd.Series([invalid]), "fixture")

    def test_missing_catalogue_never_invokes_downloader(self):
        with tempfile.TemporaryDirectory() as directory, patch("wwdc_repro.shared.data.fetch_sdss_specgals") as fetch:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "Missing local catalogue"):
                load_sample(root, root / "missing.fits", root)
            fetch.assert_not_called()

    def test_train_scaling_is_independent_of_held_out_data_and_redshift_is_raw(self):
        data = frames()
        baseline = matrices(data, "orig+z")
        np.testing.assert_array_equal(baseline["X_train"][:, :10], np.array([[-1.] * 10, [1.] * 10]))
        for split in ("val", "test"):
            data[split]["orig"] = [np.full(10, 1e6), np.full(10, -1e6)]
        changed = matrices(data, "orig+z")
        np.testing.assert_array_equal(changed["X_train"], baseline["X_train"])
        for split in data:
            np.testing.assert_array_equal(changed[f"X_{split}"][:, -1], data[split].z_x)

    def test_norm_is_scaled_with_training_latents_before_raw_redshift(self):
        data = frames()
        for split, values in zip(data, ([10., 14.], [18., 22.], [26., 30.])):
            data[split]["norm"] = values
        original = {split: np.stack(frame.cond) for split, frame in data.items()}
        baseline = matrices(data, "cond+z")
        result = matrices(data, "cond+z", scaled_columns=("norm",))
        for split, expected in zip(data, ([-1., 1.], [3., 5.], [7., 9.])):
            self.assertEqual(result[f"X_{split}"].shape, (2, 12))
            np.testing.assert_array_equal(result[f"X_{split}"][:, :10], baseline[f"X_{split}"][:, :10])
            np.testing.assert_array_equal(result[f"X_{split}"][:, 10], expected)
            np.testing.assert_array_equal(result[f"X_{split}"][:, 11], data[split].z_x)
            np.testing.assert_array_equal(np.stack(data[split].cond), original[split])
        data["val"]["norm"] = [1e6, -1e6]
        changed = matrices(data, "cond+z", scaled_columns=("norm",))
        np.testing.assert_array_equal(changed["X_train"], result["X_train"])
        without_z = matrices(data, "cond", scaled_columns=("norm",))
        np.testing.assert_array_equal(without_z["X_train"], result["X_train"][:, :-1])

    def test_extra_scaled_inputs_are_explicit_finite_scalars(self):
        data = frames()
        for frame in data.values():
            frame["norm"] = [10., 14.]
        for columns in ("norm", ("norm", "norm"), ("missing",)):
            with self.subTest(columns=columns), self.assertRaises(ValueError):
                matrices(data, "cond", scaled_columns=columns)
        with self.assertRaisesRegex(ValueError, "redshift-only"):
            matrices(data, "z", scaled_columns=("norm",))
        for invalid in ([np.nan, 1.], [[1., 2.], [3., 4.]]):
            data["val"]["norm"] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                matrices(data, "cond", scaled_columns=("norm",))

    def test_cleaning_preserves_missing_sentinel_rule_but_rejects_infinity(self):
        data = frames()
        data["train"].loc[0, "lgm_tot_p50"] = np.nan
        data["val"].loc[0, "lgm_tot_p50"] = -9999
        cleaned = clean_target(data)
        self.assertEqual(cleaned["train"].specObjID.tolist(), [2])
        self.assertEqual(cleaned["val"].specObjID.tolist(), [4])
        data["test"].loc[0, "lgm_tot_p50"] = np.inf
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            clean_target(data)

    def test_invalid_latents_redshift_and_arm_fail_clearly(self):
        data = frames()
        data["train"]["orig"] = [np.zeros(9), np.zeros(9)]
        with self.assertRaisesRegex(ValueError, "expected 10"):
            latent_array(data["train"], "orig", "train")
        data = frames()
        data["train"].at[0, "orig"] = np.full(10, np.nan)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            clean_target(data)
        data = frames()
        data["val"].loc[0, "z_x"] = np.inf
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            matrices(data, "z")
        with self.assertRaisesRegex(ValueError, "Unknown representation"):
            matrices(frames(), "typo")

    def test_row_order_hash_detects_reordering_and_split_overlap_is_rejected(self):
        data = frames()
        before = sample_summary(data)[0]
        data["train"] = data["train"].iloc[::-1]
        after = sample_summary(data)[0]
        self.assertEqual(before["id_sha256"], after["id_sha256"])
        self.assertNotEqual(before["ordered_id_sha256"], after["ordered_id_sha256"])
        data["test"].loc[0, "specObjID"] = 1
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_split_ids(data)


if __name__ == "__main__":
    unittest.main()
