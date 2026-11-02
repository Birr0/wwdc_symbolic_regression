# Reproduce the saved paper results

The normal workflow uses one input-path file and one command. It evaluates saved
equations and recreates figures/tables; it does not launch training or fresh searches.
Use Python 3.12 and run the following commands from the repository root.

## 1. Install

```bash
uv sync --frozen
cp reproduction.example.json reproduction.json
```

The runner uses the base package environment and a separately locked conditioning
environment automatically. The first run needs access to the dependency packages.

## 2. Set your input paths

Once the [HF release](DATA_RELEASE.md) is published, the downloader writes this
file automatically:

```bash
uv run --frozen --extra catalogues python download_data.py
python3 reproduce.py --config data/paper/reproduction.json
```

Install the download dependency first with `uv sync --frozen --extra catalogues`.
Use `download_data.py --profile full` for the additional neural training inputs.
Until the upload is complete, use existing local inputs as described below.

Edit `reproduction.json`:

| Setting | What to supply |
|---|---|
| `embeddings` | The archived redshift-flow embedding directory with `train/`, `val/`, `test/` parquet shards for run `7655991_0`. |
| `galspec` | `galSpecExtra-dr8.fits`. |
| `specgals_home` | Directory containing both `SDSSspecgalsDR8_1.fit.gz` and `SDSSspecgalsDR8_2.fit.gz`. |
| `conditioning_archive` | Unpacked prepared NORM/fibre input bundle. It includes `conditioning_validation_2026-10-13/`, `fibre_mass_conditioning_2026-10-14/`, `norm_input_ablation_2026-10-16/`, `shared_structure_audit_2026-10-14/` and `plot_cmd_latent_results/vagc_join.parquet`. |
| `norm_generation` | Optional directory containing `generated_spectra.npz` and `observed_spectra.npz`. Leave `null` to omit the spectral-sequence figure. |

Absolute paths work. Relative input paths resolve beside the JSON file.
The CMD cache defaults to the file inside the conditioning bundle; an optional
`cmd_cache` setting overrides it. The bundle's 293 prepared files are checked
against committed SHA-256 hashes before any outputs are written.

**Availability:** exact embeddings, prepared inputs and model products still lack
public download locations. See [INPUTS.md](INPUTS.md) for input identities and
[CONDITIONING.md](CONDITIONING.md) for the author archive/export layout.
Do not substitute different spectrum splits. The command never downloads data.

## 3. Run

```bash
python3 reproduce.py
```

Progress prints the current step. The runner automatically:

1. Evaluates redshift candidate equations, freezes validation choices and checks numerical references.
2. Builds redshift figures and the latent-weight table.
3. Copies and verifies the prepared conditioning bundle in a separate working directory.
4. Freezes and evaluates NORM selections, rebuilds the ablation and fibre SR tables, and builds the NORM frontier/CMD.
5. Checks three regenerated tables against bundled references and collects the results.
6. Replots the NORM spectral sequence if saved arrays were supplied.

To check input paths/hashes and see the plan without running analysis:

```bash
python3 reproduce.py --check
```

To use another path file or run again into a fresh output directory:

```bash
python3 reproduce.py --config reproduction.json --outdir outputs/paper-run-02
```

The output directory must be new or empty and separate from the input archives.
Source inputs are preserved. The working copy needs about 1.16 GB for prepared
inputs, plus generated outputs. Runtime depends on the host; the workflow includes
full-data equation evaluation and CMD density calculations.

## Read the result

Open **`outputs/paper/RESULTS.md`**. The output layout is:

```text
outputs/paper/
  RESULTS.md       results index and scope
  summary.json     per-step status, commands and logs
  figures/         redshift and normalisation PDFs
  tables/          regenerated tables and labelled archived MLP scores
  logs/            one log per step
  work/            frozen selections, sample IDs, arrays and provenance
```

A successful run passes the redshift numerical reference checks and three
conditioning-table comparisons. It does not independently validate every figure's
appearance or every coefficient rounded for the manuscript.

Files prefixed `archived_` contain saved MLP reference scores, not newly trained
results. The fibre SR report reuses archived conditioned structural summaries.
The spectral sequence is replotted from saved arrays; fresh checkpoint inference
is an advanced workflow. Flows, embeddings, symbolic searches and probes/MLPs
are not retrained by this command.

If a step fails, the command stops, returns a nonzero exit status and prints the
log path. `summary.json` records the failure. Correct the issue and choose a new
output directory; partial runs are not silently resumed or overwritten.

## Advanced work

Use [ADVANCED.md](ADVANCED.md) for checkpoint inference and the earlier embedding
MLP, [CONDITIONING.md](CONDITIONING.md) for Section 6 training/inference and
individual commands, or [experiments](../experiments/README.md) for fresh symbolic
searches. The [script guide](SCRIPTS.md) explains every experiment file; it is not
required reading for the normal workflow.
