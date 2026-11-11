"""Run the matched fibre-mass MLP comparison using the shared sklearn trainer."""

from pathlib import Path
import argparse

from wwdc_repro.fibre_mass.fibre_mass_conditioning import OUT as BASE
from wwdc_repro.shared.mlp import run_mlp_experiment

OUT = BASE / "mlp_sklearn"
ARMS = ("original", "norm")


def run(outdir=OUT):
    """Train and evaluate original/NORM fibre models on the saved matched splits."""
    return run_mlp_experiment(
        BASE,
        outdir,
        arms=ARMS,
        n_features=12,
        target="LGM_FIB_P50; standardized on fitted training subset",
        inputs="ten latent coordinates + NORM + z; standardized on the 10000 training rows per seed",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.outdir)


if __name__ == "__main__":
    main()
