# GitHub repository preparation

Initial inventory: 2026-10-12. This records the existing symbolic-regression reproduction project's state, not a completed release validation.

Update: the documentation index/walkthrough, standalone SR environment, data
validation, evaluator checks, and lightweight CI are now implemented. Fourteen
tests, 15 full-data reference checks, and a clean-copy install/build check pass.
Below is the original work inventory; consult [RELEASE_STATUS.md](RELEASE_STATUS.md)
for current publication boundaries and [VALIDATION.md](VALIDATION.md) for evidence.

## Scope

Confirmed scope: the symbolic-regression reproduction repository. It reproduces the analysis from saved embeddings and candidate equations, with an optional checkpoint-inference route. Embedding training and generation remain upstream in [wwdc_spectra](https://github.com/Birr0/wwdc_spectra).

## What already exists

- A local Git repository, with no remote configured at inspection time.
- A Python package under `src/wwdc_repro/`, six installed command entry points, `pyproject.toml`, and `uv.lock`.
- Eighteen archived candidate fronts, input hashes, reference checks, and training-score recovery provenance.
- Documentation of inputs, figures, provenance, validation, and selection corrections.
- Tests covering validation-only selection, training-reference recovery, and retention of failed test evaluations.
- Historical SR rerun scripts and exploratory spectrum-MLP experiments.

There are substantial existing modified and untracked files. Preserve them and inspect their diffs before preparing release commits.

## Reader journeys to document

1. **Understand the analysis:** a short scientific overview, definitions of each representation, and a diagram from embeddings/catalogues through filtering, training-only preprocessing, candidate evaluation, validation selection, and test evaluation to figures.
2. **Reproduce saved results:** installation, exact input acquisition and verification, one copyable command sequence, expected output files and checks, and known limitations.
3. **Inspect a figure:** a direct mapping from manuscript figure to command, calculation function, numerical output, and interpretation caveats.
4. **Rerun an SR search:** a standalone environment recipe and repository-relative command with explicit inputs, seed, arm, and output directory.
5. **Regenerate embeddings:** pinned upstream checkout and checkpoint instructions, clearly distinguishing the verified 32-spectrum comparison from a full generation run.

## Work before publication

### Documentation and portability

- Reconcile README descriptions with current code and audit evidence. In particular, the statement that only two arm/seed searches were checked predates the additional training-reference recovery reruns.
- Explain the two frontier outputs together: strict evaluation retains failures/gaps; manuscript figures use explicitly marked, test-informed display fallbacks. This is not replacement of validation-selected models.
- Reconcile figure styling/count claims with the current corner plot, fixed-score anatomy, and frontier outputs.
- Replace parent-workspace assumptions in experiment instructions and inspect scripts for equivalent path dependencies. Historical absolute paths inside provenance records are evidence, not automatically executable dependencies; preserve their meaning.
- Completed: standalone SR environment recipe with a separate script lockfile (`experiments/rerun_sr.py.lock`), explicit data-root argument, and repository-relative instructions. An isolated-copy `cond+z` seed-42 run matched all 23 archived equations at both export precisions; see `artifacts/standalone_search_verification.json`. The main analysis lockfile is unchanged by this work.
- Add a concise code/data-flow guide and an indexed documentation landing page. Keep detailed historical audit material available without requiring newcomers to read it first.

### Inputs and release metadata

- Assign public locations for the exact checkpoint, companion YAML, archived spectrum splits, and cached embeddings, with checksums and applicable redistribution terms. These are currently external prerequisites without assigned public download locations.
- Document the remaining CMD catalogue provenance limits already recorded in `INPUTS.md`.
- Choose a code licence and check the provenance/terms of vendored historical code and upstream dependencies.
- Add citation metadata once repository identity and paper metadata are settled.
- Select the GitHub owner/name and configure the remote when scope is confirmed.

### Validation

- Run the existing tests in the intended standalone environment.
- Verify installation and command help from a fresh checkout without parent-workspace imports.
- Run the documented saved-result workflow with the audited inputs and compare numerical outputs and sample IDs.
- Verify current paper figure recipes against their expected numerical outputs; do not infer scientific equivalence from appearance alone.
- Add CI for lightweight installation and scientific-integrity checks that do not require private data or large checkpoints.
- Perform a public-input end-to-end check once the exact inputs are hosted. Earlier local clean-clone validation is not evidence that an external reader can currently obtain all inputs.

## Suggested implementation order

Start with the documentation map and a standalone saved-result workflow, then address the SR search environment and path portability, then input publication and final clean-checkout validation. Preserve numerical behaviour while making the code easier to navigate; report any scientific changes separately.
