# Reproducing the NORM sections and fibre appendix

Section 6, "Conditioning on spectral normalisation", is included by the
pre-submission manuscript from `sections/normalisation.tex`. This walkthrough
covers its reproduction code, together with the joint-conditioning analysis
and fibre-mass controls.

The entry point is `experiments/conditioning/suite.py`. It runs preserved analysis
scripts in a separate working copy, with no dependency on the flower checkout or
its directory location. This is checkpoint-first reproduction: flow training and
embedding production belong to [wwdc_spectra](https://github.com/Birr0/wwdc_spectra).

## Inputs and environment

Use Python 3.12 and the dedicated script lock. This creates an isolated experiment
environment, including pyoperon, PyTorch and generation dependencies:

```bash
uv run --script --locked experiments/conditioning/suite.py list
```

Run the commands below with that prefix. The lock lives beside the launcher as
`suite.py.lock`; direct experiment dependencies are pinned in the launcher's inline
metadata. To intentionally update it, edit the pins, run
`uv lock --script experiments/conditioning/suite.py`, and repeat the checks.
`environment.json` records the earlier analysis environment's direct versions;
the new lock resolves transitive dependencies and was tested separately.

The main package's `uv.lock` and historical SR runner's lock remain separate.
For base-package installation and regression tests use `uv sync --frozen` and
`uv run --frozen python -m unittest discover -s tests -v`. Native search results
can still depend on platform and thread configuration; a lock is not a guarantee
of identical stochastic searches across platforms.

The prepared input archive is about 1.16 GB. Its 293 files and hashes are listed in
[`prepared_inputs.json`](../artifacts/conditioning/prepared_inputs.json). It contains
matrices, row IDs, exact 10,000-row indices, saved candidate fronts, frozen
selections, reference summaries and the VAGC join. It does not contain the large
raw spectra, embeddings or model weights. Public download locations remain to be
assigned; Git alone cannot currently reproduce all results.

Stage an archive into a **new** working directory. For the local author archive,
`/path/to/archive` is `examples/spectra/symbolic_regression` in the flower checkout.
An already exported bundle has the same layout and can also be used as the source.

```bash
uv run --script --locked experiments/conditioning/suite.py export \
  --source /path/to/archive --workdir /path/to/conditioning-work
uv run --script --locked experiments/conditioning/suite.py verify \
  --workdir /path/to/conditioning-work
```

Export copies files; it never links writable results back to the originals. Verify
before running analyses: regenerated summaries/manifests may intentionally change,
so verification against the initial archive can fail afterwards. Keep the source
archive unchanged. The launcher checks packaged source hashes and refuses to
replace a staged script that has been edited. Some plotting/report scripts replace
outputs in the working copy; fits and probe training refuse existing run directories.
Only load trusted pickle frames and model checkpoints.

Small reference tables are committed under `artifacts/conditioning/reference/`.
Large prepared files are deliberately outside Git. Raw embedding and checkpoint
identities are recorded in `artifacts/conditioning/raw_inputs.json`; catalogue and
saved spectrum-split requirements are in [INPUTS.md](INPUTS.md).

## Representation and variable map

| Arm | Flow conditioning | SR predictors in matched comparisons |
|---|---|---|
| `original` | No flow | Original Spender latents, NORM, z |
| `zflow` | z | Conditional latents, NORM, z |
| `norm` | NORM | Conditional latents, NORM, z |
| `znorm` | z and NORM | Conditional latents, NORM, z |

`X1`–`X10` are training-standardised latents; `X11` is standardised log10 NORM;
`X12` is raw redshift. In the without-NORM ablation, `X11` becomes raw redshift.
Do not confuse flow conditioning with explicitly supplying a variable to SR.
The historical z-flow comparison has no explicit NORM and a slightly different
sample; the matched `zflow` arm fixes those differences.

NORM is the median usable observed flux density in the rest-wavelength window
5300–5850 Angstrom, not total broad-band flux. The SR standardisation is fitted
on the full filtered training split. Displayed equations undo that scaling to
use `ell = log10(N/N0)`; this changes coefficients, not predictions. Flow context
scaling and SR scaling are distinct. Targets are `lgm_tot_p50` for total mass and
`LGM_FIB_P50` for fibre mass, both logarithmic solar masses.

## Regenerate the section figures and tables

Run these commands from the reproduction repository. Paths after `--workdir`
can be anywhere; relative artifact names are resolved inside that directory.

```bash
uv run --script --locked experiments/conditioning/suite.py run --workdir /path/to/conditioning-work \
  finalize_conditioning_validation
uv run --script --locked experiments/conditioning/suite.py run --workdir /path/to/conditioning-work \
  norm_input_ablation report
uv run --script --locked experiments/conditioning/suite.py run --workdir /path/to/conditioning-work \
  plot_cmd_norm_s
uv run --script --locked experiments/conditioning/suite.py run --workdir /path/to/conditioning-work \
  plot_norm_cmd_full
uv run --script --locked experiments/conditioning/suite.py run --workdir /path/to/conditioning-work \
  compare_norm_joint_s
uv run --script --locked experiments/conditioning/suite.py run --workdir /path/to/conditioning-work \
  analyse_fibre_original_control
```

| Result | Location inside the work directory |
|---|---|
| NORM input-ablation frontier PDF and summary | `norm_input_ablation_2026-10-16/` |
| Full-sample CMD PDF, bin counts, KDE and metadata | `norm_cmd_full_2026-10-18/` |
| Per-spectrum latent correction S and photometry | `norm_cmd_s_2026-10-16/plot_data.parquet` |
| Joint versus NORM correction comparison | `norm_joint_s_comparison_2026-10-16/` |
| Fibre performance and structure tables | `fibre_mass_conditioning_2026-10-14/original_control_comparison/` |
| Frozen validation choices and held-out scores | `conditioning_validation_2026-10-13/` |

`plot_cmd_norm_s` checks that the selected seed-42, budget-15 equation has constant
latent derivatives and that `S = sum(w_i x_i)` equals `f(x,n,z)-f(0,n,z)` on all
splits. The CMD uses catalogue `ABSMAG_g-ABSMAG_r` and the paper's magnitude
conversion and filters. Full-sample counts use every matched row (440,087 here).
The colour panel requires 25 spectra per bin. Contours estimate 10/30/50/70/90%
KDE probability mass using a reproducible 60,000-row subsample; they are not exact
fractions of the empirical histogram. No random subsampling is used for counts or
mean S. SciencePlots falls back to Matplotlib text when LaTeX is unavailable.

For the full matched total-mass structural analysis, run
`analyse_matched_conditioning`. For the fibre diagnostic reconstruction, supply the data root through the launcher, as:

```bash
uv run --script --locked experiments/conditioning/suite.py run --workdir /path/to/conditioning-work \
  --data-root /path/to/data analyse_fibre_mass
uv run --script --locked experiments/conditioning/suite.py run --workdir /path/to/conditioning-work \
  audit_shared_structure
uv run --script --locked experiments/conditioning/suite.py run --workdir /path/to/conditioning-work \
  analyse_fibre_original_control
```

The full fibre diagnostics need raw catalogues, embeddings and the saved NORM YAML;
the first command list instead reuses their archived conditional summaries. The
structural audit covers the total and fibre comparisons here; an unrelated
exploratory `new_total` sweep was removed from this packaged audit. Fixed latent
direction, additive latent dependence and complete n/z/latent separability are
separate tests. Similarity to a physical scaling is not an exact identification
of S with a mass-to-light ratio.

## Latent-only probes and the fibre MLP control

```bash
uv run --script --locked experiments/conditioning/suite.py run --workdir /path/to/conditioning-work \
  norm_conditioning_probe
uv run --script --locked experiments/conditioning/suite.py run --workdir /path/to/conditioning-work \
  redshift_conditioning_probe
```

These use **only ten latent coordinates** to predict log10 NORM or raw z.
Neither conditioning variable is included as a predictor. They report coordinate
Pearson/Spearman correlations, OLS and a 64/32 ReLU MLP across seeds 42–44. Each
MLP uses the saved SR 10,000-row training subset, fits its input/target scaling on
that subset, and uses Adam at 0.001 with batches of 1024, at most 500 epochs and
patience 40. Minimum validation MSE chooses the checkpoint before test scoring.
Histories, preprocessing, checkpoints and predictions are saved. Weak probe
performance is evidence of reduced accessible dependence, not proof of independence.

`fibre_mass_mlp_control` uses the same architecture and stopping procedure with
12 inputs: latents **plus NORM and z**. It is not a latent-only probe. The exported
bundle includes its reference result directory; move
`fibre_mass_conditioning_2026-10-14/mlp_control_2026-10-15` outside the working tree
before rerunning that script. Other training scripts also refuse existing outputs.
Some historical fibre runs reached the epoch cap; retain that limitation when
comparing their scores. The reference checkpoints are not in the bundle; rerunning
training produces new ones.

## Symbolic regression and validation

Selection requires finite full-training/validation predictions and agreement with
the native fit-subset training R² within 1e-6. At each length budget choose maximum
validation R², with shorter length then stable row order resolving ties. Persist
choices before held-out evaluation. A nonfinite test prediction is an evaluation
failure, never a reason to select a replacement. Budget 15 means length **at most**
15, not necessarily an expression of length 15.

The search uses the original operators/settings: 10,000 matched training rows,
seeds 42–44, eight threads, max search length 25, up to 1,000 generations,
10^7 evaluations and 1,000 local coefficient-optimisation iterations. Exported
length can include scale/intercept nodes up to 29. New matched-control reruns
also record native MDL/AIC/BIC; these do not replace validation selection.

For a fresh search, export a second work directory, then **move the corresponding
archived result directory outside it** before fitting. Keep the prepared matrices,
frames and saved training indices. For example, after moving
`conditioning_validation_2026-10-13/matched_controls/original_seed42` aside:

```bash
uv run --script --locked experiments/conditioning/suite.py run --workdir /path/to/fresh-work \
  matched_conditioning_controls fit --arm original --seed 42
```

Other rerun entry points:

- `matched_conditioning_controls fit --arm zflow --seed 42`: matched z-flow control.
- `norm_input_ablation fit --arm norm --seed 42`: without explicit NORM; prepared
  matrices must remain, and `norm_seed42` in its output root must first be moved.
- `fibre_mass_conditioning fit --arm norm --seed 42`: conditional fibre fits.
- `fibre_mass_original_control fit --seed 42`: original-latent fibre fit.
- `rerun_conditioning_exports`: reruns both NORM seed-42 fronts and checks exact
  archived equation matches before recovering 60-decimal exports. Move both
  `rerun_norm_cond_seed42` and `rerun_norm_cond+z_seed42` aside first.
- `run_normflow --append-norm --embed-type cond+z --seed 43` and
  `run_znormflow --append-norm --embed-type cond+z --seed 43`: regenerate archived
  candidate pools using raw embeddings/catalogues and the historical runner.
  Move the corresponding `job_results_*` directory first. That runner retains
  historical test-based diagnostic model summaries; **do not use those summaries
  for paper selection**. Follow it with `validate_conditioning` and
  `finalize_conditioning_validation`.

Raw-data stages take the launcher's `--data-root /path/to/data` before the script
name. `validate_conditioning` rebuilds matrices and initial candidate audits;
`finalize_conditioning_validation` verifies recovered exports and writes the final
frozen selection before test evaluation. `check_conditioning_physics` reconstructs
the physical scaling metadata. Preparation helpers are also included in
`matched_conditioning_controls`, `fibre_mass_conditioning`,
`fibre_mass_original_control` and `norm_input_ablation`; the archive already
contains their outputs. Do not run preparation over existing prepared directories.
The packaged `val_rescore` is helper-only: its obsolete test-gated CLI was removed.

## Conditional generation and upstream provenance

The current local upstream source differs from its Git revision. The relevant
changes, including NORM/joint training configurations and the normalisation loader,
are preserved in `artifacts/conditioning/upstream_changes.patch`; file hashes and
base revision are in `upstream.json`. Apply them to a **separate clean checkout**:

```bash
git clone https://github.com/Birr0/wwdc_spectra /path/to/wwdc-spectra-generation
git -C /path/to/wwdc-spectra-generation checkout 55bee36565e1944c4feeafec7c19525bee3ba200
git -C /path/to/wwdc-spectra-generation apply /absolute/path/to/reproduction/artifacts/conditioning/upstream_changes.patch
```

This records the observed source used for this reproduction check. It does not
prove every file was identical when the original checkpoints were trained. The
newer NORM configurations are separate from the older exploratory
`experiments/normalisation_flow.py`. Keep training in upstream's framework, with
its Hydra configurations and W&B logger; refer to its training/inference entry
points and the saved checkpoint YAML for resolved settings.

After `plot_cmd_norm_s`, run:

```bash
uv run --script --locked experiments/conditioning/suite.py run --workdir /path/to/conditioning-work \
  --data-root /path/to/data --upstream /path/to/wwdc-spectra-generation \
  generate_norm_sequence --n-spectra 5 --seed 42 \
  --factors 0.5 0.75 1 1.25 1.5 1.75 2 --outdir norm_generation_random_seed42_five
```

Required external inputs are the NORM checkpoint and YAML, archived train/test
embeddings, `sdss/sdss_II_catalog` saved spectrum splits, and the pretrained Spender
weights. Preserve saved splits; a newly downloaded random split is not equivalent.
The selected objects are random rows of the matched CMD test sample, not selected
by colour. The inverse-flow seed is held fixed **for each object**, then the
conditioning value varies. Left panels show the raw decoder outputs, with no additional median division;
right panels multiply those outputs by the requested NORM. Therefore restored
amplitude is imposed explicitly and is not independent evidence that conditioning
learned a luminosity law. The observed spectrum is included for reference.

The script saves numerical arrays, source spectrum IDs, context values and
round-trip/step-size checks alongside PDF/PNG figures. `plot_raw_norm_generation
--outdir norm_generation_random_seed42_five` shows raw decoder outputs without
that final amplitude restoration.

## Verification and remaining release work

Run `python -m unittest discover -s tests -v` in the installed repository.
After the figure/table commands, compare the regenerated tables with:

```bash
python experiments/conditioning/check_outputs.py --workdir /path/to/conditioning-work
```

The recorded local integration checks are in
`artifacts/conditioning/package_verification.json`. They distinguish regenerated
results from expensive training/search stages that were only packaged. A fresh base-package environment and a separately locked experiment environment
were also tested on the same host; this is not a cross-platform test.
Public hosting of exact inputs/checkpoints remains a release task. The paper's text is not changed by this addition.

## Figure 1 equivalent for NORM conditioning

After `norm_input_ablation report`, run:

```bash
uv run --script --locked experiments/conditioning/suite.py run --workdir /path/to/conditioning-work \
  plot_norm_frontier
```

`norm_frontier/norm_frontier.pdf` puts all four matched ablation curves on one
panel with the three-seed minimum/maximum band. It also saves a 400-dpi PNG,
numerical summary and input hash. Every curve includes z. Solid lines supply
NORM explicitly; dashed lines omit it. This uses frozen validation selections;
any nonfinite test score leaves a gap rather than selecting a replacement.
The manuscript asset is `figures/norm_frontier.pdf`, labelled `fig:norm_frontier`.

## Full-sample Figure 10

```bash
uv run --script --locked experiments/conditioning/suite.py run --workdir /path/to/conditioning-work \
  plot_joint_cmd_full
```

Outputs are in `joint_cmd_full/`: PDF/400-dpi PNG, plotting rows, bin arrays,
KDE contours and an input-hash manifest. All three splits contribute to the
descriptive CMD; equations remain the frozen budget-15 validation selections.
Each seed row shows the same full-sample count/KDE panel, then NORM and joint
NORM+z mean corrections. The minimum count of 25, bin edges, KDE seed/sample,
contour probabilities and S colour range match the full-sample NORM CMD.
The existing test-only comparison and its reported correlations are preserved.

The default generation figure uses raw decoder outputs as of 2026-10-20.
Saved arrays use schema 2: `amplitude_restored = decoded_raw * reference_norm * factor`.
The `shape` array remains a median-normalised diagnostic and is not plotted.
Older saved arrays retain their original semantics; the raw-output replotter always
recomputes the right panels from `decoded_raw` rather than reading `amplitude_restored`.

To replot existing generation arrays without running the flow again:

```bash
python experiments/conditioning/scripts/preview_raw_norm_sequence.py \
  --input /path/to/saved-generation --outdir /path/to/raw-output-plot
```
