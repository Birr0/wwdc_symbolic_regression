# WWDC symbolic regression

Reproduce the paper from saved galaxy embeddings:
**get data → configure paths → run equations → generate plots and tables**.
Flow training and embedding generation are documented separately in
[wwdc_spectra](https://github.com/Birr0/wwdc_spectra).

## 1. Install

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). From this repository:

```bash
uv sync --frozen
cp reproduction.example.json reproduction.json
```

## 2. Get the data

The data bundle is prepared but **not yet published/pinned**. The intended
Hugging Face dataset is `Birr001/wwdc-symbolic-regression`. Existing local inputs
can be used now. Once published:

```bash
uv sync --frozen --extra catalogues
uv run --frozen --extra catalogues python download_data.py --profile full
```

The full bundle is approximately 36 GB and includes the NORM/joint embeddings
and scaling metadata needed for fresh searches. Keep the supplied spectrum
splits and ordering.

## 3. Configure paths

Edit the root-level `reproduction.json`. Paths may be absolute or relative to
that file. The example supplies the expected embedding subdirectories.

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

The downloader currently writes a configuration for the historical archive
workflow. For the commands below, use `reproduction.example.json` instead.

## 4. Run

```bash
# Check inputs and print the plan without launching searches.
python reproduce.py equations --check

# Fit the symbolic equations and MLP comparisons required by the tables.
python reproduce.py equations --threads 8

# Generate figures and tables from the completed run; no model fitting.
python reproduce.py plots
python reproduce.py tables
```

The equation stage prepares inputs, runs 51 symbolic searches, selects equations
using training scores by default, evaluates them on test, and fits the MLP
comparisons. Local searches run sequentially, with eight Operon threads per
search in this example. Plotting requires both the CMD cache and saved spectral
arrays. Plots and tables use this run's results, not substituted archived scores.

All commands default to `reproduction.json` and `outputs/paper`. To change these,
pass `--config FILE` and `--outdir DIR` consistently to all three commands.
Optional `--selection validation` retains validation-based selection; use the
same selection policy throughout.

## 5. Check the results

| Output | Location |
|---|---|
| Figures | `outputs/paper/figures/` |
| CSV and LaTeX tables | `outputs/paper/tables/` |
| Equations, scores and intermediate data | `outputs/paper/work/` |
| Logs | `outputs/paper/logs/` |
| Stage completion records | `outputs/paper/equations.json`, `plots.json`, `tables.json` |

Inspect the figures, failed/nonfinite scores and equation interpretations before
updating the manuscript. The full fresh rerun is pending; the NORM analytic-family
parameter table still needs scientific reconciliation with the generated equations.

## Restarting

Repeated commands skip verified completed work. After fixing an interrupted
step, preserve its previous attempt and restart it with:

```bash
python reproduce.py equations --threads 8 --restart-incomplete
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
