# Data preparation review

Reviewed 2026-10-12 against the local archived inputs. The initial review left
[`data.py`](../src/wwdc_repro/shared/data.py) unchanged; the follow-up below added explicit
validation and audit metadata. [Initial results](../artifacts/data_review.json)
and [before/after verification](../artifacts/data_validation_verification.json)
record the source hashes for each stage.

## Walkthrough

1. `load_sample` reads the astroML SDSS specgals catalogue and the separate DR8
   galSpecExtra FITS file. The astroML catalogue already incorporates its documented
   photometric/spectroscopic quality cuts and catalogue-redshift cut `z > 0.02`.
2. It loads each archived embedding split independently. Parquet files are sorted
   lexicographically, preserving the historical order (`1`, `10`, …, `2`, …).
   Changing to numeric sorting would change the seeded training draws.
3. It extracts integer spectrum IDs and inner-joins each split to astroML.
   `validate="one_to_one"` rejects duplicate join keys instead of multiplying rows.
4. It keeps `mask_ratio <= 0.5` and embedding redshift `z_x <= 0.3`, then
   inner-joins the separate galSpecExtra table by `specObjID`.
5. `clean_target` removes missing target values and the `-9999` sentinel.
6. `matrices` fits each latent scaler on the **full cleaned training split**,
   applies it to validation/test, and appends raw redshift for the `+z` arms.
   There are 10 latent features, 11 for `+z`, or one for redshift alone.
7. The 10,000-row draw happens later in `frontier.score_fronts` or the search
   wrapper, with `default_rng(seed).choice(..., replace=False)`. Scaling therefore
   uses the full training population before the smaller search sample is drawn.
8. `sample_summary` records membership hashes and counts;
   `preprocessing_state` serialises the training-derived scaler parameters.

## Exact catalogue provenance

The target constant is lowercase `lgm_tot_p50`, supplied by astroML. The separate
FITS file supplies uppercase `LGM_TOT_P50`. These names do not collide during
the join. It is therefore inaccurate to describe the current code as replacing
astroML's target with the FITS mass column.

The separate join is part of the historical sample construction and makes extra
catalogue columns available. It is not needed to obtain a mass column that
astroML already contains. Removing the join would still require checking sample
membership and downstream column dependencies; it was retained for reproduction.

On all 441,283 retained rows, the two mass columns agree exactly (maximum absolute
difference zero). This documentation correction does not change any target values.
The embedding redshift `z_x` and astroML redshift `z_y` also agree exactly on these
rows. This agreement is not an independent check of the association between an
actual spectrum's flux and its metadata, so it does not resolve the previously
identified upstream mislabelling issue.

## Results on the archived inputs

| Check | Train | Validation | Test |
|---|---:|---:|---:|
| Rows after joins and mask/redshift cuts | 352,918 | 44,250 | 44,144 |
| Rows after target cleaning | 352,896 | 44,246 | 44,141 |
| Nonfinite retained targets | 0 | 0 | 0 |
| Nonfinite redshifts | 0 | 0 | 0 |
| Nonfinite latent values, all three representations | 0 | 0 | 0 |

- All split membership hashes match `reference.json`.
- Spectrum IDs are unique within each split, with zero overlap between splits.
  This is not a claim of physical-galaxy disjointness.
- The seed-42, seed-43, and seed-44 training draws match the archived rerun IDs
  in exact order, each containing 10,000 rows.
- All six arms produce finite matrices, and appended redshifts are unchanged.
- Inspection confirms scalers are fitted only on training rows. Their use of the
  full training split rather than just the 10,000 search rows is intentional.

No numerical discrepancy was found in these checks. They verify this stage's
behaviour on the archived inputs, not the scientific correctness of the catalogues.

## Follow-up implementation

The validation and audit changes below are implemented. An exact comparison
confirmed unchanged DataFrame contents/order for all three splits, all six arms'
feature/target matrices, scaler parameters, and all three seeded draws.

1. `load_sample` requires local catalogue files and passes
   `download_if_missing=False` to astroML.
2. Historical NaN/sentinel target exclusions are unchanged; retained infinite
   targets, nonfinite redshifts/latents, and nonfinite mask ratios at the
   catalogue-matched filtering stage raise explicit errors.
3. `sample_summary` records an ordered ID hash alongside the original sorted
   membership hash. The main analysis checks row order before candidate scoring;
   the search wrapper also checks it before drawing its training sample.
4. Column origins are documented above. Existing column names remain compatible.
5. IDs must be decimal integers or the archived byte-string form. Arbitrary
   numeric substrings, decimal floats, and int64 overflow are rejected. Latent
   arrays must contain ten coordinates (plain or singleton-wrapped vectors),
   and unknown representation arms are rejected.
6. Intermediate counts travel with the frames and are written to `samples.csv`.
   Spectrum-ID uniqueness and split separation are enforced by reusable checks.

| Stage | Train | Validation | Test |
|---|---:|---:|---:|
| Archived embeddings | 422,481 | 52,810 | 52,811 |
| astroML join | 367,609 | 46,028 | 45,990 |
| Mask cut | 353,883 | 44,368 | 44,267 |
| Redshift cut | 352,918 | 44,250 | 44,144 |
| galSpecExtra join | 352,918 | 44,250 | 44,144 |
| Target cleaning | 352,896 | 44,246 | 44,141 |

Six new data tests cover ID precision, missing inputs, training-only scaling,
nonfinite values, dimensions, row order and split overlap; the three selection
tests continue to pass. A split-membership check still concerns spectrum IDs,
not physical-galaxy identity.


## Why NORM and historical redshift sample memberships differ

Traced the 41 training, six validation and six test spectra present in the
cleaned redshift sample but absent from NORM/joint conditioning. All 53 have
negative spectral normalisation in the original checkpoint-referenced caches.
They are already absent from the raw NORM/joint embedding exports, before the
reproduction catalogue joins and sample cuts.

The upstream `wwdc_spectra.data.normalisation.configure_normalisation` retains
only finite positive NORM values before taking log10. For every split, the raw
NORM and joint embedding ID sets exactly equal the parent dataset IDs passing
this cached positivity/finite filter. The historical redshift embedding ID set
exactly equals the unfiltered parent set. Thus this filter fully accounts for
the membership differences; no additional removal of mislabelled IDs is needed
to explain them. This is not an independent validation of spectrum/metadata
pairing and does not establish why individual normalisations are negative.

The six missing test IDs are 337843946807912448, 338906349901473792,
461737465674229760, 461759730784692224, 463894988757952512 and
463943917025388544. Their cached normalisations range from -0.249389 to
-0.051431. The larger raw exclusions are 10,739 train, 1,317 validation and
1,330 test rows; most also fail the subsequent redshift-analysis sample cuts.

Evidence: `outputs/conditioning-sample-difference/differing_spectra.csv`,
`verification.json` and `verify_membership.py`. No sample membership or
selection policy was changed during this investigation.
