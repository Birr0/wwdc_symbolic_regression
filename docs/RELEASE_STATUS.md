# Review release status

Prepared 2026-10-12. This is a code review release of the symbolic-regression
reproduction project, starting from saved neural representations or the
specified released checkpoint. It does not claim a complete public-data
end-to-end reproduction while the exact inputs lack public download locations.

The normal entry point is now `python3 reproduce.py`; see
[the walkthrough](QUICKSTART.md). Its 12-step saved-result workflow passed on the
author's inputs, including 15 redshift checks and three conditioning-table
comparisons. All 23 unit tests pass. This run reused existing locked Python
environments because the sandbox blocked uv cache writes; fresh environment
bootstrap and public-input reproduction remain unverified. See the
[workflow verification record](../artifacts/reproduction_workflow_verification.json).

The next release adds a verified 1,423-file HF bundle, download profiles and the
training source pinned in `artifacts/conditioning/release_upstream.json`.
All 29 reproduction tests and seven upstream tests pass; a local restore of the
saved data profile passes the complete workflow. The HF upload and remote
download remain pending. See [DATA_RELEASE.md](DATA_RELEASE.md) and
[`data_release_verification.json`](../artifacts/data_release_verification.json).

## Included

- Package commands for saved-equation analysis, figures, checkpoint inference,
  catalogue cache reconstruction, and the matched embedding MLP comparison.
- Eighteen archived candidate fronts and numerical/provenance audit artifacts.
- Locked analysis dependencies and a separate locked historical SR search.
- A walkthrough, code guide, figure recipes, and data/selection reviews.
- Data, equation, and selection tests, with a GitHub Actions workflow for
  installation, command checks, tests, and package building.
- Section 6 spectral-normalisation analysis and fibre-mass
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
