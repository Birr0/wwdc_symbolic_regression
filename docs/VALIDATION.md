# Migration verification, 2026-10-05

The standalone implementation was compared with the saved pre-submission
audit and, for score/anatomy, the original inspected plotting functions.
All runs used the same local data. These are numerical reproduction checks,
not independent validation of the underlying scientific interpretations.

| Check | Result |
| --- | --- |
| Selection | All 447 selected equation strings and saved test scores match the audited validation-selected results |
| Samples | Exact train/validation/test spectrum-ID hashes and counts; 12 reference checks pass |
| Zero-latent figure | All 400 exported grid rows match the audited numerical CSV exactly |
| Raw latent histograms | Entire moments JSON matches the audited recreation exactly |
| Score projection | Maximum per-row difference across three seeds 3.56e-15 |
| Anatomy | Maximum baseline difference 1.78e-15; prediction-SD difference below 1e-16 |
| CMD | All 440,140 rows match by spectrum ID; magnitudes and colours identical; score difference below 9e-16 |
| CMD cache rebuild | All 525,505 rows, all columns and all duplicate-match choices match the old cache |
| Checkpoint inference | Clean pinned upstream source, locked inference extra, first 32 test spectra; original max error 9.88e-7, conditional 1.884e-5, unconditional 1.419e-5; redshift/mask ratio exact |

The tested checkpoint tolerances are 2e-6 for original and 2e-5 for conditional
and unconditional embeddings. They are numerical tolerances, not a guarantee
for every spectrum or hardware platform. [The compact inference report](../artifacts/embedding_check_32.json)
records the environment, input fingerprints and observed errors.

Figure images were inspected for axis labels, colour scales and layout.
Styling differs from the historical manuscript figures; its numerical recipes
and the known latent-distribution recreation are retained. No new figure was
automatically copied into the paper.

The clean-clone run is recorded here after completion. It uses a new virtual
environment on this host and external audited data; it is not a second-machine
or public-download verification. Full inference across all spectra remains open.
