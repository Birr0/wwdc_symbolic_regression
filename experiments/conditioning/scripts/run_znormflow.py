"""Run the symbolic-regression sweep on the joint (Z, NORM)-conditioned flow.

``run_normflow.py`` runs the search on a flow conditioned only on NORM; the published
fronts come from ``7655991_0``, conditioned only on redshift. This runs the identical
search on ``znorm_20261013104946_0`` (``spender_I_z_normalisation_flow`` in the
wwdc_spectra repo), the same flow over the same frozen spender_I latent conditioned
jointly on raw redshift and standardised log10 NORM. ``cond`` here is therefore the
latent with both transported out. The embedding's ``y`` column is ``[z, NORM]``.

Everything except the embeddings is ``run_total_mass.py`` unchanged: ``build_merged``
for the catalogue joins, ``run_cell`` (hence ``utils.fit_sym_fn``/``fit_model_fn``) for
the search, the 10k subsample, the scaling and the MLP/LR references. ``cond+z`` still
appends raw redshift, so X1..X10 are the (z, NORM)-transported latents and X11 is z.

Output goes to ``job_results_znormflow_{FEATURE}_{EMBED}_seed{SEED}/``. The
``znormflow_`` prefix keeps it clear of the redshift-flow fronts
(``job_results_{FEATURE}_...``), of ``run_normflow.py``'s ``normflow_`` arms and of
``run_norm.py``'s ``norm_``/``nonorm_`` arms. The script refuses to overwrite an
existing result directory.

``--append-norm`` widens ``cond`` from 10 to 11 entries with the flow's NORM
conditioning value (the second entry of the embedding's ``y``), the trick
``run_norm.py`` uses. The ``cond`` arm then fits X1..X10 latents + X11 NORM, and
``cond+z`` fits X1..X10 + X11 NORM + X12 raw redshift. It tests whether the mass
information the flow removed is just the redshift and flux it conditioned on. Output
is prefixed ``znormflow_condnorm_`` so it cannot collide with the plain ``znormflow_``
arms.

Usage:
    python run_znormflow.py --embed-type cond+z --seed 42   # one cell, for timing
    python run_znormflow.py                                  # all 5 arms x 3 seeds
    python run_znormflow.py --append-norm                    # cond+NORM, cond+NORM+z
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from run_total_mass import EMBED_TYPES, SEEDS, run_cell
from val_rescore import build_merged

TAG = "znormflow"
RUN_ID = "znorm_20261013104946_0"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    data_root = os.environ.get("DATA_ROOT", "data")
    parser.add_argument("--feature", default="lgm_tot_p50")
    parser.add_argument("--embed-type", default=None, choices=EMBED_TYPES)
    parser.add_argument("--seed", type=int, default=None, choices=SEEDS)
    parser.add_argument(
        "--embeddings",
        default=f"{data_root}/sdss_II/spender_I_z_normalisation_flow/embeddings/{RUN_ID}",
    )
    parser.add_argument("--galspec", default=f"{data_root}/galSpecExtra-dr8.fits")
    parser.add_argument("--specgals-home", default=f"{data_root}/sdss")
    parser.add_argument(
        "--append-norm",
        action="store_true",
        help="append the conditioning NORM to cond; runs only the cond and cond+z arms",
    )
    args = parser.parse_args()
    if args.append_norm and args.embed_type not in (None, "cond", "cond+z"):
        parser.error("--append-norm only applies to the cond and cond+z arms")

    # cond+z first: it is the arm the published claims rest on
    if args.embed_type:
        arms = [args.embed_type]
    elif args.append_norm:
        arms = ["cond", "cond+z"]
    else:
        arms = ["cond+z"] + [e for e in EMBED_TYPES if e != "cond+z"]
    seeds = [args.seed] if args.seed is not None else SEEDS

    # fail before the slow merge rather than after hours of search
    tag = f"{TAG}_condnorm" if args.append_norm else TAG
    feature_tag = f"{tag}_{args.feature}"
    cells = [
        (e, s, f"job_results_{feature_tag}_{e}_seed{s}") for e in arms for s in seeds
    ]
    clashes = [d for _, _, d in cells if os.path.exists(d)]
    if clashes:
        message = f"refusing to overwrite existing results: {clashes}"
        raise SystemExit(message)

    print(f"target: {args.feature}  embeddings: {args.embeddings}")
    merged = build_merged(args.embeddings, args.galspec, args.specgals_home)
    # the tag names both the target column fit_sym_fn looks up and the output
    # directory, so alias the real target under it (run_norm.py does the same)
    for split, frame in merged.items():
        frame[feature_tag] = frame[args.feature]
        if args.append_norm:
            cond = np.stack(frame["cond"].values)
            # y is [z, NORM]; z is already appended by the cond+z arm
            norm = np.stack(frame["y"].values)[:, 1:2]
            frame["cond"] = list(np.hstack([cond, norm]))
            print(f"  {split}: cond widened to {cond.shape[1] + 1} dims with NORM")

    for embed_type, seed, job_dir in cells:
        print(f"\n=== {feature_tag} | {embed_type} | seed {seed} ===", flush=True)
        front, summary = run_cell(feature_tag, embed_type, seed, merged)
        os.makedirs(job_dir)
        front.to_csv(f"{job_dir}/pareto_fronts.csv", index=False)
        pd.DataFrame([summary]).to_csv(f"{job_dir}/summary_metrics.csv", index=False)
        print(f"  wrote {job_dir}/", flush=True)


if __name__ == "__main__":
    main()
