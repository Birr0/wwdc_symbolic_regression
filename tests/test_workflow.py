"""Check workflow input integrity, ordering, and failure reporting without real data."""

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("reproduce", Path(__file__).resolve().parents[1] / "reproduce.py")
workflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.config = self.home / "inputs.json"
        self.config.write_text(json.dumps({key: key for key in workflow.REQUIRED}))
        self.paths = workflow.load_inputs(self.config)
        self.out = self.home / "outputs"

    def prepare(self):
        for split in ("train", "val", "test"):
            folder = self.paths["embeddings"] / split
            folder.mkdir(parents=True)
            (folder / "000.parquet").write_bytes(b"fixture")
        for path in (self.paths["galspec"], self.paths["cmd_cache"], *[
            self.paths["specgals_home"] / f"SDSSspecgalsDR8_{i}.fit.gz" for i in (1, 2)
        ]):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
        pinned = self.paths["conditioning_archive"] / "matrix.npz"
        pinned.write_bytes(b"pinned fixture")
        manifest = self.home / "artifacts/conditioning/prepared_inputs.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({"files": [{"path": "matrix.npz", "bytes": pinned.stat().st_size,
                                                  "sha256": workflow.digest(pinned)}]}))
        suite = self.home / "experiments/conditioning/suite.py"
        suite.parent.mkdir(parents=True)
        suite.write_text("def check_code():\n    pass\n")
        return pinned

    def test_relative_paths_and_unknown_keys(self):
        self.assertEqual(self.paths["embeddings"], self.home / "embeddings")
        self.assertEqual(self.paths["cmd_cache"], self.home / "conditioning_archive/plot_cmd_latent_results/vagc_join.parquet")
        raw = json.loads(self.config.read_text())
        raw["embedings"] = "typo"
        self.config.write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError, "unknown.*embedings"):
            workflow.load_inputs(self.config)

    def test_check_verifies_archive_without_creating_output(self):
        pinned = self.prepare()
        with patch.object(workflow, "ROOT", self.home):
            self.assertEqual(workflow.check_inputs(self.paths, self.out), 1)
            self.assertFalse(self.out.exists())
            pinned.write_bytes(b"changed bytes!")
            with self.assertRaisesRegex(ValueError, "checksum differs"):
                workflow.check_inputs(self.paths, self.out)
            self.assertFalse(self.out.exists())

    def test_existing_outputs_and_input_overlap_are_rejected(self):
        self.prepare()
        self.out.mkdir()
        sentinel = self.out / "keep.txt"
        sentinel.write_text("keep")
        with patch.object(workflow, "ROOT", self.home):
            with self.assertRaisesRegex(ValueError, "new or empty"):
                workflow.check_inputs(self.paths, self.out)
            with self.assertRaisesRegex(ValueError, "separate from embeddings"):
                workflow.check_inputs(self.paths, self.paths["embeddings"] / "output")
        self.assertEqual(sentinel.read_text(), "keep")

    def test_plan_orders_dependencies_and_never_trains(self):
        steps = workflow.build_steps(self.paths, self.out)
        commands = [command for _, command in steps]
        frozen = next(i for i, command in enumerate(commands) if "finalize_conditioning_validation" in command)
        ablation = next(i for i, command in enumerate(commands) if "norm_input_ablation" in command)
        cmd_rows = next(i for i, command in enumerate(commands) if "plot_cmd_norm_s" in command)
        cmd_plot = next(i for i, command in enumerate(commands) if "plot_norm_cmd_full" in command)
        self.assertLess(frozen, ablation)
        self.assertLess(cmd_rows, cmd_plot)
        self.assertTrue(commands[-1][-3].endswith("check_outputs.py"))
        for command in commands:
            self.assertFalse(set(command) & {"fit", "train", "norm_conditioning_probe", "fibre_mass_mlp_control", "generate_norm_sequence"})
        self.paths["norm_generation"] = self.home / "saved spectra with spaces"
        command = workflow.build_steps(self.paths, self.out)[-1][1]
        self.assertEqual(command[command.index("--input") + 1], str(self.paths["norm_generation"]))

    def test_failure_stops_pipeline_and_records_log(self):
        steps = [("first", ["tool", "first"]), ("second", ["tool", "second"])]
        with patch.object(workflow.subprocess, "run", return_value=subprocess.CompletedProcess([], 9)) as run:
            with patch.object(workflow, "collect_results") as collect:
                with self.assertRaisesRegex(RuntimeError, "01.log"):
                    workflow.run_steps(steps, self.out, self.paths, self.config)
        self.assertEqual(run.call_count, 1)
        collect.assert_not_called()
        summary = json.loads((self.out / "summary.json").read_text())
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["steps"][0]["returncode"], 9)
        self.assertTrue((self.out / "logs/01.log").is_file())

    def test_zero_exit_without_expected_results_is_not_success(self):
        with patch.object(workflow.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)):
            with self.assertRaisesRegex(ValueError, "No redshift figure"):
                workflow.run_steps([("empty output", ["tool"])], self.out, self.paths, self.config)
        summary = json.loads((self.out / "summary.json").read_text())
        self.assertEqual(summary["status"], "failed")
        self.assertFalse((self.out / "RESULTS.md").exists())


if __name__ == "__main__":
    unittest.main()
