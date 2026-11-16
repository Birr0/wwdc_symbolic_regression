# WWDC symbolic regression

This code is designed to produce equations from galaxy spectra embeddings. We condition the embeddings in order to control them and
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
Download the release using:

```bash
uv run --frozen --extra catalogues python download_data.py --profile full
```

The full data is approximately 36 GB.

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
| `stellar_mlp_spectra` | Optional matched spectral cache for the total-mass MLP; see below. |
| `stellar_mlp_data_root` | Optional root containing `sdss/sdss_II_catalog`; enables automatic spectral-cache preparation. |

For the downloaded data, use `data/paper/data` as `data_root`,
`data/paper/conditioning/plot_cmd_latent_results/vagc_join.parquet` as `cmd_cache`,
and `data/paper/generation` as `norm_generation`. Keep the NORM checkpoint YAML
in its bundled `ckpts/` directory.

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

The equation stage prepares inputs, runs the symbolic searches, selects equations
using validation scores with these commands, evaluates them on the test data, and fits the MLP
comparisons.

The manuscript selects equations by validation performance and reports their test
scores. Use `--selection validation` consistently for all stages. Training-based
selection remains available with `--selection train` (the CLI default).

## 5. Check the results

The results are saved in the following places for inspection: 

| Output | Location |
|---|---|
| Figures | `outputs/paper/figures/` |
| CSV and LaTeX tables | `outputs/paper/tables/` |
| Equations, scores and intermediate data | `outputs/paper/work/` |
| Logs | `outputs/paper/logs/` |
| Stage completion records | `outputs/paper/equations.json`, `plots.json`, `tables.json` |


Code is licensed under [MIT](LICENSE).
