#!/usr/bin/env python3
"""Reproduce the paper's saved-equation analysis from one input-path file.

Run from any directory. No model training or new symbolic search is launched.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REQUIRED = ("embeddings", "galspec", "specgals_home", "conditioning_archive")
OPTIONAL = ("cmd_cache", "norm_generation")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_inputs(config):
    raw = json.loads(config.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Input configuration must be a JSON object")
    missing = [key for key in REQUIRED if not raw.get(key)]
    unknown = set(raw) - set(REQUIRED + OPTIONAL)
    if missing or unknown:
        raise ValueError(f"Configuration keys: missing {missing}; unknown {sorted(unknown)}")
    paths = {}
    for key, value in raw.items():
        if value is None and key in OPTIONAL:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a nonempty path string")
        path = Path(value).expanduser()
        paths[key] = (config.parent / path).resolve() if not path.is_absolute() else path.resolve()
    paths.setdefault("cmd_cache", paths["conditioning_archive"] / "plot_cmd_latent_results/vagc_join.parquet")
    return paths


def check_inputs(paths, outdir):
    """Check required paths and pinned prepared inputs before creating any output."""
    problems = []
    for split in ("train", "val", "test"):
        folder = paths["embeddings"] / split
        if not folder.is_dir() or not any(folder.glob("*.parquet")):
            problems.append(f"embeddings: no parquet shards in {folder}")
    for path in (paths["galspec"], paths["cmd_cache"], *[
        paths["specgals_home"] / f"SDSSspecgalsDR8_{i}.fit.gz" for i in (1, 2)
    ]):
        if not path.is_file():
            problems.append(f"Missing file: {path}")
    if "norm_generation" in paths:
        for name in ("generated_spectra.npz", "observed_spectra.npz"):
            if not (paths["norm_generation"] / name).is_file():
                problems.append(f"Missing generation arrays: {paths['norm_generation'] / name}")
    for name in ("embeddings", "conditioning_archive", "norm_generation"):
        if name in paths:
            source = paths[name]
            if outdir == source or source in outdir.parents or outdir in source.parents:
                problems.append(f"Output directory must be separate from {name}: {source}")
    if outdir.exists() and (not outdir.is_dir() or any(outdir.iterdir())):
        problems.append(f"Output directory must be new or empty: {outdir}")
    archive = paths["conditioning_archive"]
    expected = json.loads((ROOT / "artifacts/conditioning/prepared_inputs.json").read_text())["files"]
    for row in expected:
        path = archive / row["path"]
        if not path.is_file():
            problems.append(f"Missing prepared input: {path}")
        elif path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            problems.append(f"Prepared input checksum differs: {path}")
    if problems:
        shown = "\n".join(f"  - {p}" for p in problems[:12])
        extra = f"\n  ... and {len(problems) - 12} more" if len(problems) > 12 else ""
        raise ValueError(f"Input check failed:\n{shown}{extra}\nSee docs/INPUTS.md and docs/CONDITIONING.md.")
    spec = importlib.util.spec_from_file_location("conditioning_suite", ROOT / "experiments/conditioning/suite.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check_code()
    return len(expected)


def build_steps(paths, outdir):
    common = ["--embeddings", str(paths["embeddings"]), "--galspec", str(paths["galspec"]),
              "--specgals-home", str(paths["specgals_home"])]
    base = ["uv", "run", "--frozen"]
    suite = str(ROOT / "experiments/conditioning/suite.py")
    conditioning = ["uv", "run", "--script", "--locked", suite]
    work = outdir / "work/conditioning"
    redshift = outdir / "work/redshift"
    steps = [
        ("Evaluate redshift equations and verify reference checks", base + ["wwdc-reproduce", *common,
         "--fronts", str(ROOT / "artifacts/fronts"), "--outdir", str(redshift / "analysis")]),
        ("Build redshift figures and coefficient table", base + ["wwdc-figures", *common,
         "--selected", str(redshift / "analysis/selected.csv"), "--fronts", str(ROOT / "artifacts/fronts"),
         "--cmd-cache", str(paths["cmd_cache"]), "--outdir", str(redshift / "figures")]),
        ("Copy prepared conditioning inputs into the output directory", [sys.executable, suite, "export",
         "--source", str(paths["conditioning_archive"]), "--workdir", str(work)]),
        ("Verify the copied conditioning inputs", [sys.executable, suite, "verify", "--workdir", str(work)]),
    ]
    for title, script, arguments in (
        ("Freeze validation selections and score NORM equations", "finalize_conditioning_validation", []),
        ("Rebuild the NORM ablation table", "norm_input_ablation", ["report"]),
        ("Build the NORM frontier figure", "plot_norm_frontier", []),
        ("Compute and check the NORM latent score", "plot_cmd_norm_s", []),
        ("Build the full-sample NORM CMD figure", "plot_norm_cmd_full", []),
        ("Rebuild the fibre-mass SR comparison", "analyse_fibre_original_control", []),
    ):
        steps.append((title, conditioning + ["run", "--workdir", str(work), script, *arguments]))
    steps.append(("Compare all three regenerated tables with references", base + ["python",
                  str(ROOT / "experiments/conditioning/check_outputs.py"), "--workdir", str(work)]))
    if "norm_generation" in paths:
        steps.append(("Replot the NORM spectral sequence from saved arrays", conditioning + [
            "run", "--workdir", str(work), "preview_raw_norm_sequence", "--input",
            str(paths["norm_generation"]), "--outdir", str(work / "norm_sequence")]))
    return steps


def collect_results(outdir, paths):
    """Collect manuscript-facing outputs; distinguish archived MLP scores."""
    work = outdir / "work/conditioning"
    redshift_figures = outdir / "work/redshift/figures"
    files = [(p, outdir / "figures/redshift" / p.name) for p in sorted(redshift_figures.glob("*.pdf"))]
    if not files:
        raise ValueError("No redshift figure PDFs were produced")
    files += [
        (work / "norm_frontier/norm_frontier.pdf", outdir / "figures/normalisation/norm_frontier.pdf"),
        (work / "norm_cmd_full_2026-10-18/norm_cmd_full.pdf", outdir / "figures/normalisation/norm_cmd.pdf"),
        (work / "norm_input_ablation_2026-10-16/summary.csv", outdir / "tables/norm_ablation.csv"),
        (work / "fibre_mass_conditioning_2026-10-14/original_control_comparison/performance.csv",
         outdir / "tables/fibre_sr.csv"),
        (redshift_figures / "latent_weights.tex", outdir / "tables/latent_weights.tex"),
    ]
    if "norm_generation" in paths:
        files.append((work / "norm_sequence/raw_decoder_sequence_preview.pdf",
                      outdir / "figures/normalisation/norm_sequence.pdf"))
    for source, destination in files:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    archived = [
        ("norm_conditioning_probe_2026-10-18/mlp/summary.csv", "archived_norm_probe_mlp.csv", {"original", "norm", "zflow"}),
        ("fibre_mass_conditioning_2026-10-14/mlp_control_2026-10-15/summary.csv", "archived_fibre_mlp.csv", {"original", "norm"}),
    ]
    for relative, name, arms in archived:
        source = ROOT / "artifacts/conditioning/reference" / relative
        with source.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, [row for row in reader if row["arm"] in arms]
        with (outdir / "tables" / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    sequence = "Replotted from supplied saved arrays." if "norm_generation" in paths else "Not produced: no norm_generation arrays supplied."
    index = ["# Saved-equation reproduction results", "",
             "Redshift reference checks and three conditioning-table comparisons passed.", "",
             "- [Figures](figures/): redshift and NORM frontier/CMD PDFs.",
             "- [Tables](tables/): regenerated NORM ablation, fibre SR and latent weights.",
             "- [Run summary](summary.json): completed steps and commands.",
             "- [Logs](logs/): one log per step.",
             "- [Working files](work/): selections, hashes, arrays and detailed diagnostics.", "",
             "## Scope", "", f"NORM spectral sequence: {sequence}", "",
             "Files prefixed `archived_` are bundled reference MLP scores copied for convenience; "
             "their models were not retrained. Fibre structural analysis reuses archived conditioned summaries.", "",
             "This run does not retrain flows, regenerate embeddings, rerun symbolic searches, "
             "retrain probes/MLPs or independently verify every printed manuscript coefficient."]
    (outdir / "RESULTS.md").write_text("\n".join(index) + "\n")


def run_steps(steps, outdir, paths, config):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "logs").mkdir()
    summary = {
        "status": "running", "scope": "saved_equation_analysis",
        "config_sha256": digest(config), "runner_sha256": digest(Path(__file__)),
        "inputs": {key: str(value) for key, value in paths.items()},
        "not_run": ["flow training", "embedding generation", "symbolic searches", "MLP/probe training"],
        "norm_sequence": "replot_saved_arrays" if "norm_generation" in paths else "not_supplied",
        "steps": [],
    }

    def save():
        (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    save()
    try:
        for number, (title, command) in enumerate(steps, 1):
            logfile = Path("logs") / f"{number:02d}.log"
            record = {"name": title, "command": command, "log": str(logfile), "status": "running"}
            summary["steps"].append(record)
            save()
            print(f"[{number}/{len(steps)}] {title}", flush=True)
            with (outdir / logfile).open("w") as stream:
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
            record.update(returncode=result.returncode, status="passed" if result.returncode == 0 else "failed")
            save()
            if result.returncode:
                raise RuntimeError(f"{title} failed. See {outdir / logfile}")
        collect_results(outdir, paths)
        summary["status"] = "passed"
        save()
    except BaseException as exc:
        if summary["steps"] and summary["steps"][-1]["status"] == "running":
            summary["steps"][-1]["status"] = "failed"
        summary.update(status="failed", error=str(exc))
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("reproduction.json"), help="Input paths; relative paths resolve beside this file")
    parser.add_argument("--outdir", type=Path, default=Path("outputs/paper"), help="New or empty output directory")
    parser.add_argument("--check", action="store_true", help="Verify input paths/hashes and show the plan without writing outputs or running analysis")
    args = parser.parse_args()
    try:
        config = args.config.resolve()
        paths = load_inputs(config)
        outdir = args.outdir.resolve()
        print("Checking inputs and prepared-archive checksums…", flush=True)
        count = check_inputs(paths, outdir)
        steps = build_steps(paths, outdir)
        print(f"Inputs verified ({count} pinned prepared files).", flush=True)
        if "norm_generation" not in paths:
            print("NORM spectral sequence omitted: optional saved generation arrays were not supplied.", flush=True)
        if args.check:
            for number, (title, _) in enumerate(steps, 1):
                print(f"  {number}. {title}")
            return
        if shutil.which("uv") is None:
            raise ValueError("uv is required; install it before running the workflow")
        run_steps(steps, outdir, paths, config)
        print(f"Saved-equation reproduction passed. Open {outdir / 'RESULTS.md'}", flush=True)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
