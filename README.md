# WWDC symbolic regression

Reproduce the paper's analysis from saved embeddings and candidate equations,
including Section 6 (spectral normalisation) and the fibre-mass SR comparison.

**Data release status:** the exact inputs are packaged, but the Hugging Face
upload is still pending. The [data release guide](docs/DATA_RELEASE.md) gives the
download command, package contents and remaining publication step.

## Reproduce the saved results

Use Python 3.12 and `uv`. From the repository root:

```bash
uv sync --frozen
cp reproduction.example.json reproduction.json
```

Edit the four input paths in `reproduction.json`, then run:

```bash
python3 reproduce.py
```

The command checks inputs, evaluates equations, creates redshift and NORM
figures, and checks the input data. It stops
at the first failed step and points to its log. It runs the separate locked
analysis environments automatically.

Equation selection defaults to training R²: within each length budget, choose
from the training Pareto front, freeze the choice, then report its test score.
Ties favour shorter equations and then source order. Failed test evaluations
remain missing values; they do not trigger replacement equations.

Use `python3 reproduce.py --selection validation` for the previous validation
selection policy and its archived numerical reference checks. The redshift CLI
and NORM/fibre selection commands also accept `--selection train|validation`.
Some historical output filenames still contain `validation`; the `selection`
column records the policy actually used. No symbolic search is rerun by this switch.

Open **`outputs/paper/RESULTS.md`** when it finishes. Figures are in `figures/`,
tables in `tables/`, and detailed selections and diagnostics in `work/`.

The optional `norm_generation` path adds the NORM spectral-sequence figure from
saved arrays. MLP reference scores are labelled `archived_`; this workflow does
not retrain models or rerun symbolic searches. The results index records this scope.

## Where to look next

| Need | Read |
|---|---|
| Input paths, running the command and expected results | [Reproduction walkthrough](docs/QUICKSTART.md) |
| Download data or obtain the full training inputs | [Data release](docs/DATA_RELEASE.md) |
| Review coverage of every paper figure and table | [Paper review inventory](docs/PAPER_REVIEW.md) |
| What each script does | [Script guide](docs/SCRIPTS.md) |
| Fresh searches, checkpoint inference or model/probe training | [Advanced reproduction](docs/ADVANCED.md) |
| Scientific checks, provenance and release status | [Technical documentation](docs/README.md) |

Code is licensed under [MIT](LICENSE). Model training and original embedding
production use [wwdc_spectra](https://github.com/Birr0/wwdc_spectra).

Active redshift, NORM-conditioning, fibre-mass, and fitting code lives together in
[`src/wwdc_repro`](src/wwdc_repro). See [the code map](docs/SCRIPTS.md) for entry
points and review status. The existing conditioning suite command remains a
compatibility launcher; it runs package modules directly without staging code in
output folders.
