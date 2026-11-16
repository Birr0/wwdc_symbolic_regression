#!/usr/bin/env python3
"""Reproduce the paper's analysis from existing symbolic searches.
"""

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
# Support direct execution without installing the package.
sys.path.insert(0, str(ROOT / "src"))
from wwdc_repro.shared.utils import sha256 as digest
from wwdc_repro.shared.paths import ARCHIVE_PATHS
REQUIRED = ("embeddings", "galspec", "specgals_home", "conditioning_archive") # required entries in the config
OPTIONAL = ("cmd_cache", "norm_generation") # optional entries for the config.

def load_inputs(config):
    '''
    Read and validate input paths from a JSON configuration file.
 
    Require the keys listed in REQUIRED and allow those listed in OPTIONAL.
    
    Parameters
    ----------
    config : Path
        Path to the JSON configuration file.
    
    Returns
    -------
    dict[str, Path]
        Input names mapped to resolved absolute paths. Optional entries
        with null values are omitted, except cmd_cache, which receives
        its default path.
    
    Raises
    ------
    ValueError
        If the configuration is not a JSON object, required entries are
        missing or empty, unknown keys are present, or a supplied path
        is not a nonempty string.
    json.JSONDecodeError
        If the configuration is not valid JSON.
    OSError
        If the configuration file cannot be read.
    
    Notes
    -----
    This function resolves paths but does not check whether they exist.
    '''
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
    paths.setdefault("cmd_cache", paths["conditioning_archive"] / "plot_cmd_latent_results/vagc_join.parquet") # default path to vagc
    return paths


def check_inputs(paths, outdir):
    """Validate input files, output location and pinned conditioning sources.
 
    Check that each embedding split contains Parquet shards, required
    catalogue files exist, and optional generation arrays are available.
    Verify prepared conditioning inputs against their recorded sizes and
    SHA-256 hashes. Source hashes are recorded for provenance, not enforced.
    
    Parameters
    ----------
    paths : dict[str, Path]
        Resolved input paths returned by load_inputs.
    outdir : Path
        Resolved output directory. Must be new or empty and must not
        overlap the embeddings, conditioning archive or generation
        directories.
    
    Returns
    -------
    int
        Number of prepared conditioning input files verified.
    
    Raises
    ------
    ValueError
        If an input check fails or the output location is unsuitable.
    
    Notes
    -----
    Does not create outputs or run the analyses.
    
    """
    problems = []
    for split in ("train", "val", "test"):
        # check the embeddings exist
        folder = paths["embeddings"] / split
        if not folder.is_dir() or not any(folder.glob("*.parquet")):
            problems.append(f"embeddings: no parquet shards in {folder}")
    for path in (paths["galspec"], paths["cmd_cache"], *[
        paths["specgals_home"] / f"SDSSspecgalsDR8_{i}.fit.gz" for i in (1, 2)
    ]):
        # check the catalogue files
        if not path.is_file():
            problems.append(f"Missing file: {path}")
    if "norm_generation" in paths:
        # optional: check the norm spectral sequence figure.
        for name in ("generated_spectra.npz", "observed_spectra.npz"):
            if not (paths["norm_generation"] / name).is_file():
                problems.append(f"Missing generation arrays: {paths['norm_generation'] / name}")
    for name in ("embeddings", "conditioning_archive", "norm_generation"):
        # ensure the input and output are in separate directories for cleanliness
        if name in paths:
            source = paths[name]
            if outdir == source or source in outdir.parents or outdir in source.parents:
                problems.append(f"Output directory must be separate from {name}: {source}")
    if outdir.exists() and (not outdir.is_dir() or any(outdir.iterdir())):
        # check output dir is empty
        problems.append(f"Output directory must be new or empty: {outdir}")
    archive = paths["conditioning_archive"]
    expected = json.loads((ROOT / "artifacts/conditioning/prepared_inputs.json").read_text())["files"]
    # check that the inputs are indentical to what has been used before using hashes and relative paths.
    for row in expected:
        path = archive / row["path"]
        if not path.is_file():
            problems.append(f"Missing prepared input: {path}")
        elif path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            problems.append(f"Prepared input checksum differs: {path}")
    if problems:
        # report any discovered problems.
        shown = "\n".join(f"  - {p}" for p in problems[:12])
        extra = f"\n  ... and {len(problems) - 12} more" if len(problems) > 12 else ""
        raise ValueError(f"Input check failed:\n{shown}{extra}\nSee README.md.")
    return len(expected)

def build_steps(paths, outdir, selection="train"):
    '''
    Construct the ordered commands for the reproduction workflow.

    Parameters
    ----------
    paths : dict[str, Path]
        Resolved input paths returned by load_inputs.
    outdir : Path
        Root output directory for the reproduction run.
    
    Returns
    -------
    list[tuple[str, list[str]]]
        Ordered steps, each containing a descriptive label and a command
        represented as a list of arguments.
    
    Notes
    -----
    Constructs commands without executing them or creating directories.
    Includes the spectral-sequence plotting step only when norm_generation
    is supplied.
    '''

    common = ["--embeddings", str(paths["embeddings"]), "--galspec", str(paths["galspec"]),
              "--specgals-home", str(paths["specgals_home"])]
    base = ["uv", "run", "--frozen"] # base command
    suite = str(ROOT / "experiments/conditioning/suite.py") # file to run
    conditioning = ["uv", "run", "--script", "--locked", suite]
    work = outdir / "work/conditioning"
    redshift = outdir / "work/redshift"
    # Label all steps in reproduction workflow.
    # Redshift-conditioning analysis
    steps = [
        ("Evaluate redshift equations and verify reference checks", base + ["wwdc-reproduce", *common,
         "--fronts", str(ROOT / "artifacts/fronts"), "--outdir", str(redshift / "analysis"), "--selection", selection]),
        ("Build redshift figures and equation/weight tables", base + ["wwdc-figures", *common,
         "--selected", str(redshift / "analysis/selected.csv"), "--fronts", str(ROOT / "artifacts/fronts"), 
         "--cmd-cache", str(paths["cmd_cache"]), "--outdir", str(redshift / "figures")]), # run figures.main()
        ("Copy prepared conditioning inputs into the output directory", [sys.executable, suite, "export",
         "--source", str(paths["conditioning_archive"]), "--workdir", str(work)]),
        ("Verify the copied conditioning inputs", [sys.executable, suite, "verify", "--workdir", str(work)]),
    ]
    # Normalisation-conditioning analysis
    for title, script, arguments in (
        ("Freeze selections and score NORM equations", "validate_conditioning", ["--selection", selection]),
        ("Rebuild the NORM ablation table", "norm_input_ablation", ["compare_results", "--selection", selection]),
        ("Build the NORM frontier figure", "plot_norm_frontier", []),
        ("Prepare scores and build the full-sample NORM CMD figure", "plot_norm_cmd_full", []),
        ("Rebuild the fibre-mass SR comparison", "analyse_fibre", ["--selection", selection]),
    ):
        steps.append((title, conditioning + ["run", "--workdir", str(work), script, *arguments]))
    if selection == "validation":
        steps.append(("Compare validation-selected tables with archived references", base + ["python",
                      str(ROOT / "experiments/conditioning/check_outputs.py"), "--workdir", str(work)]))
    if "norm_generation" in paths:
        steps.append(("Replot the NORM spectral sequence from saved arrays", conditioning + [
            "run", "--workdir", str(work), "generate_norm_sequence", "--input",
            str(paths["norm_generation"]), "--outdir", str(work / "norm_sequence")]))
    return steps


def collect_results(outdir, paths):
    """
    Collect generated figures and tables into the final output directories.
 
    Copy selected products from the working directories into figures/ and
    tables/. Include archived MLP summaries with explicit archived filenames. 
    Write RESULTS.md describing the outputs.
    
    Parameters
    ----------
    outdir : Path
        Root reproduction output directory containing the completed
        redshift and conditioning working directories.
    paths : dict[str, Path]
        Resolved input configuration. The presence of norm_generation
        determines whether the spectral-sequence figure is collected.
    
    Returns
    -------
    None
 
    Raises
    ------
    ValueError
        If no redshift figure PDFs are found.
    OSError
        If a required source file cannot be read or an output cannot
        be written.
    
    Notes
    -----
    Creates destination directories as needed and overwrites existing
    destination files. Does not regenerate results or retrain archived
    MLP models.
    """
    work = outdir / "work/conditioning"
    redshift_figures = outdir / "work/redshift/figures"
    files = [(p, outdir / "figures/redshift" / p.name) for p in sorted(redshift_figures.glob("*.pdf"))] # collect pdf files
    if not files:
        raise ValueError("No redshift figure PDFs were produced")
    files += [
        (work / "norm_frontier/norm_frontier.pdf", outdir / "figures/normalisation/norm_frontier.pdf"),
        (work / "norm_cmd/norm_cmd_full.pdf", outdir / "figures/normalisation/norm_cmd.pdf"),
        (work / "norm_ablation/summary.csv", outdir / "tables/norm_ablation.csv"),
        (work / "fibre_mass/original_control_comparison/performance.csv",
         outdir / "tables/fibre_sr.csv"),
        (redshift_figures / "latent_weights.tex", outdir / "tables/latent_weights.tex"),
        (redshift_figures / "front_condz.tex", outdir / "tables/front_condz.tex"),
        (redshift_figures / "front_condz.json", outdir / "tables/front_condz.json"),
    ] # pair each source file with a destination

    files.extend((redshift_figures / name, outdir / "tables" / name) for name in
                 ("truncation.csv", "truncation_table.tex", "coverage.csv", "coverage_table.tex", "truncation.json", "coverage.json"))
    # also add the csv, text and json outputs to the list of files. Mostly for table manuscripts. 

    if "norm_generation" in paths:
        files.append((work / "norm_sequence/norm_conditional_sequence.pdf",
                      outdir / "figures/normalisation/norm_sequence.pdf")) # append norm generation plot if applicable.
    for source, destination in files:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination) # copy files over.

    archived = [
        (ARCHIVE_PATHS["mlp_references"]["norm_probe"], "archived_norm_probe_mlp.csv", {"original", "norm", "zflow"}),
        (ARCHIVE_PATHS["mlp_references"]["fibre_mass"], "archived_fibre_mlp.csv", {"original", "norm"}),
    ] # mlp results to include.

    for relative, name, arms in archived:
        # read csvs and retain the requested arms
        source = ROOT / "artifacts/conditioning/reference" / relative
        with source.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, [row for row in reader if row["arm"] in arms]
        with (outdir / "tables" / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    # create a short index of the run for inspection.
    sequence = "Replotted from supplied saved arrays." if "norm_generation" in paths else "Not produced: no norm_generation arrays supplied."
    index = ["# Saved-equation reproduction results", "",
             "Completed workflow steps and checks are recorded in summary.json.", "",
             "- [Figures](figures/): redshift and NORM frontier/CMD PDFs.",
             "- [Tables](tables/): regenerated NORM ablation, fibre SR, latent weights, archived conditional+z equations, Taylor truncation and prediction coverage.",
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
    '''
    Execute the reproduction steps, record progress and collect outputs.
 
    Run commands sequentially from the reproduction repository root.
    Write each command's standard output and errors to a numbered log,
    and update summary.json before and after each step. Stop on failure.
    After all commands succeed, collect the final figures and tables.
    
    Parameters
    ----------
    steps : list[tuple[str, list[str]]]
        Ordered step descriptions and command arguments from build_steps.
    outdir : Path
        Validated output directory. Its logs subdirectory must not exist.
    paths : dict[str, Path]
        Resolved input paths used for the run summary and output collection.
    config : Path
        Configuration file whose SHA-256 hash is recorded in the summary.
    
    Returns
    -------
    None
    
    Raises
    ------
    RuntimeError
        If a command returns a nonzero exit code.
    OSError
        If a directory, log or summary cannot be written, a command cannot
        be started, or a required output cannot be collected.
    
    Notes
    -----
    Failures and interruptions inside the execution block are recorded in
    summary.json and re-raised. Completed outputs and logs are retained.
    '''
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
        for number, (title, command) in enumerate(steps, 1): # loop through the steps in order and run.
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
    if len(sys.argv) > 1 and sys.argv[1] in {"equations", "plots", "tables", "mlp"}:
        from wwdc_repro.workflows.paper_stages import main as stage_main
        stage_main(sys.argv[1:])
        return
    # coordinate functions above to build and then run the reproduction workflow.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("reproduction.json"), help="Input paths; relative paths resolve beside this file")
    parser.add_argument("--outdir", type=Path, default=Path("outputs/paper"), help="New or empty output directory")
    parser.add_argument("--check", action="store_true", help="Verify input paths/hashes and show the plan without writing outputs or running analysis")
    parser.add_argument("--selection", choices=("train", "validation"), default="train")
    args = parser.parse_args()
    try:
        config = args.config.resolve()
        paths = load_inputs(config)
        outdir = args.outdir.resolve()
        print("Checking inputs and prepared-archive checksums…", flush=True)
        count = check_inputs(paths, outdir)
        steps = build_steps(paths, outdir, selection=args.selection)
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
