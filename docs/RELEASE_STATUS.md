# Review release status

Prepared 2026-10-12. This is a code review release of the symbolic-regression
reproduction project, starting from saved neural representations or the
specified released checkpoint. It does not claim a complete public-data
end-to-end reproduction while the exact inputs lack public download locations.

## Included

- Package commands for saved-equation analysis, figures, checkpoint inference,
  catalogue cache reconstruction, and the matched embedding MLP comparison.
- Eighteen archived candidate fronts and numerical/provenance audit artifacts.
- Locked analysis dependencies and a separate locked historical SR search.
- A walkthrough, code guide, figure recipes, and data/selection reviews.
- Data, equation, and selection tests, with a GitHub Actions workflow for
  installation, command checks, tests, and package building.
- Section 6 spectral-normalisation analysis, joint-conditioning and fibre-mass
  controls: preserved scripts, a locked experiment launcher, reference tables,
  input manifests and an upstream source patch. See [CONDITIONING.md](CONDITIONING.md).
  The local package passes 17 unit tests; source hashes also pass the launcher's
  check. Historical integration evidence is recorded separately in
  `artifacts/conditioning/package_verification.json`.

## Known boundaries

Local release checks passed: 14 tests, 15 full-data reference checks, exact
unchanged archived scores/selections, clean-copy installation/command checks,
and wheel/source builds. Evidence is in [VALIDATION.md](VALIDATION.md).

- Public hosting of exact embeddings, checkpoint/YAML, and archived spectrum
  splits remains outstanding; see [INPUTS.md](INPUTS.md).
- Neural training was not rerun. Full-dataset checkpoint inference was not
  verified during migration; the recorded comparison covers 32 spectra.
- Selected historical SR reruns match on the audited host; this is not a
  cross-platform determinism guarantee.
- The CMD reproduces pinned mirror inputs; their original photometry/crossmatch
  provenance remains a documented limitation.
- Spectrum splits are not guaranteed to separate physical galaxies.
- Paper frontiers use marked, test-informed display fallbacks. Strict analysis
  retains failed test evaluations without replacement.
- Expression evaluation preserves historical SymPy simplification and exported
  coefficient precision. It does not replay native Operon tree arithmetic.
- Experimental spectrum-MLP/outlier scripts are exploratory. In particular,
  `experiments/investigate_mlp_outliers.py` retains local dataset/upstream defaults
  and is not part of the standalone paper reproduction route.

## Repository and licence

The review repository is `Birr0/wwdc_symbolic_regression`, private for now at the
author's request. Repository code is covered by the [MIT licence](../LICENSE).
The vendored
historical modules originate in `Birr0/flower`; their MIT notice is preserved in
[`experiments/historical/LICENSE`](../experiments/historical/LICENSE). Upstream
`wwdc_spectra` is referenced, not bundled; its GitHub metadata did not declare
a licence when inspected. Dataset/checkpoint redistribution terms must be
handled separately from the repository's code licence.

GitHub Actions results should be checked for the current commit. Local verification
and hosted CI are recorded separately.
