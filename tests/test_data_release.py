"""Verify release profiles, checksums, path safety and publication locks."""

import json
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tools.data_release import copy_local_bundle, digest, rows_for, verify
from tools.release_data import upload


class DataReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.source = self.home / "source"
        self.source.mkdir()
        rows = []
        for component in ("analysis", "conditioning", "generation", "spectra", "models", "pretrained"):
            path = self.source / component / "fixture.bin"
            path.parent.mkdir()
            path.write_bytes(component.encode())
            rows.append(dict(path=str(path.relative_to(self.source)), component=component,
                             bytes=path.stat().st_size, sha256=digest(path)))
        self.manifest = {"files": rows, "repo_id": "example/paper"}
        (self.source / "release.json").write_text(json.dumps(self.manifest))
        (self.source / "README.md").write_text("Fixture")

    def test_saved_profile_omits_training_inputs(self):
        self.assertEqual({r["component"] for r in rows_for(self.manifest, "saved")},
                         {"analysis", "conditioning", "generation"})
        self.assertEqual(len(rows_for(self.manifest, "full")), 6)

    def test_checksum_change_and_unsafe_paths_fail(self):
        self.assertEqual(verify(self.source, self.manifest), 6)
        (self.source / "analysis/fixture.bin").write_bytes(b"changed!")
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            verify(self.source, self.manifest)
        self.manifest["files"][0]["path"] = "../outside"
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            rows_for(self.manifest, "full")

    def test_local_roundtrip_and_generated_paths(self):
        destination = self.home / "download"
        config = copy_local_bundle(self.source, destination, self.manifest, "saved")
        self.assertEqual(verify(destination, self.manifest, "saved"), 3)
        self.assertFalse((destination / "spectra").exists())
        paths = json.loads(config.read_text())
        self.assertEqual(paths["conditioning_archive"], str(destination / "conditioning"))
        self.assertEqual(paths["norm_generation"], str(destination / "generation"))
        config.write_text('{"custom": "keep"}')
        with self.assertRaisesRegex(ValueError, "different path configuration"):
            copy_local_bundle(self.source, destination, self.manifest, "saved")

    def test_duplicate_paths_are_rejected(self):
        self.manifest["files"].append(self.manifest["files"][0].copy())
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            rows_for(self.manifest, "full")

    def test_missing_auth_never_creates_repository(self):
        api = Mock()
        fake = SimpleNamespace(get_token=lambda: None, HfApi=api, hf_hub_download=Mock())
        with patch.dict(sys.modules, {"huggingface_hub": fake}):
            with self.assertRaisesRegex(ValueError, "credential"):
                upload(SimpleNamespace(folder=self.source, visibility="public", lock_out=self.home / "lock.json"))
        api.assert_not_called()

    def test_incomplete_upload_does_not_publish_download_lock(self):
        lock = self.home / "lock.json"
        lock.write_text('{"status":"pending"}')
        api = Mock()
        fake = SimpleNamespace(get_token=lambda: "fixture", HfApi=api,
                               hf_hub_download=Mock(return_value=str(self.source / "release.json")))
        with patch.dict(sys.modules, {"huggingface_hub": fake}):
            api.return_value.repo_info.side_effect = [SimpleNamespace(private=False), SimpleNamespace(sha="a" * 40)]
            api.return_value.list_repo_files.return_value = ["release.json"]
            with self.assertRaisesRegex(ValueError, "missing files"):
                upload(SimpleNamespace(folder=self.source, visibility="public", lock_out=lock))
        self.assertEqual(json.loads(lock.read_text())["status"], "pending")


if __name__ == "__main__":
    unittest.main()
