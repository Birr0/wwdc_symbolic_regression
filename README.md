# WWDC symbolic regression

This code is designed to produce equations from galaxy embeddings. We condition the embeddings in order to control them and
fit symbolic regression models to discover underlying forms.

The code follows these steps:
1. Get data.
2. Configure paths to data
3. Run equations.
4. Generate plots and tables.

The flow training and embedding generation are documented separately in
[wwdc_spectra](https://github.com/Birr0/wwdc_spectra).

## 1. Install
Firstly, clone the repo:
```bash
git clone https://github.com/Birr0/wwdc_symbolic_regression.git
cd wwdc_symbolic_regression
```

Use Python 3.12 and [uv](https://docs.astral.sh/uv/) to install the package and
its dependencies. If `uv` is not already installed, install it with:

```bash
python -m pip install uv
```

From the repository root (the directory containing `pyproject.toml` and
`uv.lock`), run:

```bash
uv sync --frozen --python 3.12 --extra catalogues
source .venv/bin/activate
cp reproduction.example.json reproduction.json
```

## 2. Get the data

The data is available on Hugging Face:
[Birr001/wwdc-symbolic-regression](https://huggingface.co/datasets/Birr001/wwdc-symbolic-regression).
Download the pinned release and verify its checksums:

```bash
uv run --frozen --extra catalogues python download_data.py --profile full
```

The full bundle is approximately 36 GB and includes the NORM/joint embeddings
and scaling metadata needed for fresh searches. Keep the supplied spectrum
splits and ordering.

## 3. Configure paths

The full downloader generates a configuration with the correct local paths.
Copy it to the repository root:

```bash
cp data/paper/reproduction.json reproduction.json
```

For existing local data, edit the example configuration instead. Paths may be
absolute or relative to the configuration file.

| Setting | Input |
|---|---|
| `data_root` | Contains `galSpecExtra-dr8.fits`, `sdss/` and `sdss_II/`. |
| `embeddings` | Redshift-conditioned embeddings, with `train/`, `val/`, `test/`. |
| `norm_embeddings` | NORM-conditioned embeddings with the same splits. |
| `joint_embeddings` | Joint redshift/NORM-conditioned embeddings. |
| `cmd_cache` | Saved `vagc_join.parquet` catalogue for CMD plots. |
| `norm_generation` | Contains `generated_spectra.npz` and `observed_spectra.npz`. |

For the downloaded bundle, use `data/paper/data` as `data_root`,
`data/paper/conditioning/plot_cmd_latent_results/vagc_join.parquet` as `cmd_cache`,
and `data/paper/generation` as `norm_generation`. Keep the NORM checkpoint YAML
in its bundled `ckpts/` directory for the scaling statistics.

The full-profile configuration supports the three commands below. The `saved`
profile generates a configuration for the historical archive workflow.

## 4. Run

```bash
# Check inputs and print the plan without launching searches.
python reproduce.py equations --selection validation --check

# Fit the symbolic equations and MLP comparisons required by the tables.
python reproduce.py equations --selection validation --threads 8

# Generate figures and tables from the completed run; no model fitting.
python reproduce.py plots --selection validation
python reproduce.py tables --selection validation
```

The equation stage prepares inputs, runs 51 symbolic searches, selects equations
using validation scores with these commands, evaluates them on test, and fits the MLP
comparisons. Local searches run sequentially, with eight Operon threads per
search in this example. Plotting requires both the CMD cache and saved spectral
arrays. Plots and tables use this run's results, not substituted archived scores.

Fresh searches save full-precision equations directly, so no precision-recovery
rerun is needed. Exported equations are still checked against native training
scores; rounded table equations are for presentation. Recovery tools are retained
only for the historical archive workflow.

All commands default to `reproduction.json` and `outputs/paper`. To change these,
pass `--config FILE` and `--outdir DIR` consistently to all three commands.
The manuscript selects equations by validation performance and reports their test
scores. Use `--selection validation` consistently for all stages. Training-based
selection remains available with `--selection train` (the CLI default).

## 5. Check the results

| Output | Location |
|---|---|
| Figures | `outputs/paper/figures/` |
| CSV and LaTeX tables | `outputs/paper/tables/` |
| Equations, scores and intermediate data | `outputs/paper/work/` |
| Logs | `outputs/paper/logs/` |
| Stage completion records | `outputs/paper/equations.json`, `plots.json`, `tables.json` |

Inspect the figures, nonfinite scores and equation interpretations before updating
the manuscript. The NORM physical-unit coefficient table is generated from the
full-precision equations and saved training scaling.

## Update the manuscript assets

After completing the validation-selected run above, export figures and formatted tables
into an unpacked Overleaf project containing `draft.tex`:

```bash
PYTHONPATH=src .venv/bin/python -m tools.export_manuscript \
  --run outputs/paper \
  --project ../pre-submission/WWDC_Symbolic_Regression
```

The manuscript uses `figures/` and `\input{tables/...}`. Re-exporting updates the
generated assets without editing numerical table rows in `draft.tex`. The NORM
physical-unit coefficient table is derived from the precise equations and saved
training scaling; exports stop if the equations no longer match its stated family.
Scientific prose still requires author review. The exporter requires validation
selection. For the existing two-directory workflow, use `--fresh DIR` and
`--validation DIR` instead of `--run`; this also checks that reused CMD/anatomy
figures use unchanged equations.

## Restarting

Repeated commands skip verified completed work. After fixing an interrupted
step, preserve its previous attempt and restart it with:

```bash
python reproduce.py equations --selection validation --threads 8 --restart-incomplete
```

Use `plots` or `tables` for those stages. Restarting begins the incomplete step
again; it does not resume an Operon population. Changed inputs/settings or
damaged completed outputs require a new output directory.

<details>
<summary>Optional: SLURM</summary>

Submit from the repository root. Use a Python environment with the pinned search
dependencies listed in `experiments/conditioning/suite.py`, including pyoperon
0.4.0. Inputs and the working directory must be accessible on compute nodes.

```bash
mkdir -p logs
export SEARCH_PYTHON=/absolute/path/to/search-environment/bin/python
export SEARCH_WORKDIR=/absolute/path/to/outputs/searches
export DATA_ROOT=/absolute/path/to/data
sbatch experiments/slurm/search.sbatch \
  wwdc_repro.workflows.run_total_mass --embed-type orig+z
```

The wrapper runs seeds 42–44 as separate array tasks. It requests 8 CPUs, 16 GB
and 24 hours per task; override these and supply your cluster's partition/account
as needed. `--array=0-2%1` limits concurrency to one task. The wrapper supplies
`--seed` and `--threads`, so do not add them yourself. Preparation must finish
before submitting searches that require prepared matrices. HPC execution is
still awaiting a cluster smoke test.

</details>

<details>
<summary>Optional: historical saved results, without fresh searches</summary>

```bash
cp reproduction.archive.example.json reproduction.archive.json
# Edit the paths, including conditioning_archive.
python reproduce.py --config reproduction.archive.json --outdir outputs/archive-review
```

This separate workflow uses saved equations and labelled archived MLP results.
The downloader's smaller `saved` profile supports this route.

</details>

Code is licensed under [MIT](LICENSE).
