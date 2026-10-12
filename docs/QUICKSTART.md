# Reproduce the saved-equation analysis

Run the commands below from the repository root, which contains `pyproject.toml`.
This route evaluates bundled candidate equations; it does not train a flow or
launch a symbolic-regression search. Python 3.12+ and `uv` are required.

## 1. Install and check the package

```bash
uv sync --frozen
uv run --frozen python -m unittest discover -s tests -v
uv run --frozen wwdc-reproduce --help
```

These checks do not require the paper's external datasets. They are not a
substitute for the full numerical reproduction below.

## 2. Locate the exact inputs

Follow [INPUTS.md](INPUTS.md) for expected versions, checksums, and acquisition
limits. Public locations for the exact embeddings and model products have not
yet been assigned. Do not substitute newly generated splits for the archives.

Set paths appropriate to your machine:

```bash
WWDC_EMBEDDINGS=/absolute/path/to/embeddings/7655991_0
WWDC_GALSPEC=/absolute/path/to/galSpecExtra-dr8.fits
WWDC_SPECGALS=/absolute/path/to/sdss
```

The embeddings directory contains `train/`, `val/`, and `test/` parquet shards.
The specgals directory contains both `SDSSspecgalsDR8_*.fit.gz` catalogue files.
The eighteen candidate CSVs are already included in `artifacts/fronts/`.

## 3. Evaluate and select equations

```bash
uv run --frozen wwdc-reproduce \
  --embeddings "$WWDC_EMBEDDINGS" \
  --galspec "$WWDC_GALSPEC" \
  --specgals-home "$WWDC_SPECGALS" \
  --fronts artifacts/fronts \
  --outdir outputs/reproduction
```

Use a new or empty output directory. Inspect `checks.json` and `samples.csv`:
the expected cleaned train/validation/test counts are 352,896 / 44,246 / 44,141.

| Output | Purpose |
|---|---|
| `manifest.json` | Input/code hashes, runtime, and analysis settings |
| `candidate_scores.csv` | Training agreement, validation scores, and eligibility |
| `selected_before_test.csv` | Frozen validation choices before test evaluation |
| `selected.csv` | The same choices with test scores and failure statuses |
| `curves.csv`, `frontier.pdf` | Strict aggregate; gaps retain failed evaluations |
| `preprocessing.json`, `sample_ids.csv.gz` | Training-derived preprocessing and sample membership |
| `samples.csv` | Counts after each join/filter, sorted membership hashes, and ordered ID hashes |

See the [code guide](CODE_GUIDE.md) for how these stages fit together.

## 4. Produce the paper frontier and anatomy

```bash
uv run --frozen wwdc-figures \
  --embeddings "$WWDC_EMBEDDINGS" \
  --galspec "$WWDC_GALSPEC" \
  --specgals-home "$WWDC_SPECGALS" \
  --selected outputs/reproduction/selected.csv \
  --fronts artifacts/fronts \
  --only frontier seed-frontiers anatomy \
  --outdir outputs/paper-panels
```

This subset does not require the CMD cache. For the remaining panels and cache
recipe, see [FIGURES.md](FIGURES.md) and [INPUTS.md](INPUTS.md).

Paper frontiers mark display fallbacks with hollow triangles. These use the
highest finite test score among selections at strictly lower budgets for the
same arm and seed. They do not replace failed selections in the analysis.
Use `frontier_display_values.csv` to inspect the displayed values and
`frontier_selected_scores.csv` for the strict aggregate.

## Other reproduction routes

- **Fresh SR search:** [experiment instructions](../experiments/README.md).
  `uv run --script --locked experiments/rerun_sr.py` provisions its separate
  locked pyoperon 0.4.0 environment. Supply `--data-root`, `--arm`, `--seed`,
  and a new `--outdir` as shown in the experiment instructions.
  Historical comparison outputs are diagnostic, while
  `wwdc-reproduce` implements the current selection policy.
- **Checkpoint to embeddings:** [README instructions](../README.md#generate-embeddings-from-the-checkpoint)
  and [pinned input layout](INPUTS.md). Only a 32-spectrum inference comparison
  has been verified during migration, not full-dataset regeneration.
- **Embedding MLP comparison:** [README instructions](../README.md#reproduce-the-embedding-mlp-comparison).

Full public-input reproduction remains a release task. See
[VALIDATION.md](VALIDATION.md) for the scope of completed checks.
