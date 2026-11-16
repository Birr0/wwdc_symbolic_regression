"""Generate and plot spectra at different NORM values using the trained flow.

Randomly select test spectra from the saved test embeddings, invert their Spender
latents at the observed NORM, and hold each resulting conditional seed fixed
while varying the value of NORM used in the forward flow.
Decode the generated latents and plot both the normalised
spectra and their NORM-restored amplitudes alongside the observed spectra.

Save PDF/PNG figures and spectral arrays. Use --input to replot saved arrays
without loading the model.
The flow conditions on NORM only; redshift is not held fixed.
"""

from pathlib import Path
import argparse
import os
import sys

os.environ.setdefault("DATA_ROOT", str(Path("data").resolve()))
if os.environ.get("WWDC_UPSTREAM"):
    sys.path.insert(0, str(Path(os.environ["WWDC_UPSTREAM"]) / "src"))
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from wwdc_repro.shared.figstyle import STYLE

ROOT = Path(os.environ["DATA_ROOT"]) / "sdss_II/spender_I_normalisation_flow"
TAG = "norm_20261012183217_0"
OUT = Path("norm_sequence")


def plot_sequence(source: Path, out: Path) -> None:
    """Plot saved generated and observed spectra without running flow inference.

    Parameters
    ----------
    source : Path
        Directory containing generated_spectra.npz and observed_spectra.npz.
    out : Path
        Destination for norm_conditional_sequence.pdf and .png.
    """
    with np.load(source / "generated_spectra.npz") as generated:
        # load the saved generated spectra
        wave = generated["wave_rest"]
        spectra = generated["decoded_raw"]
        factors = generated["factors"]
        nref = generated["reference_norm"]
    with np.load(source / "observed_spectra.npz") as data:
        # load observed spectra
        observed_wave = data["wave_rest"]
        observed = data["normalised_flux"]
        observed_flux = data["flux"]
    n_examples = len(nref)
    # restore to flux scale
    restored = spectra * (nref[:, None] * factors[None, :])[:, :, None]
    assert spectra.shape[:2] == (n_examples, len(factors))
    assert np.isfinite(spectra).all() and np.isfinite(restored).all()
    mask = (wave >= 3700) & (wave <= 7500) # wavelength for plotting
    out.mkdir(parents=True, exist_ok=True)
    plt.style.use(STYLE)
    plt.rcParams.update({"font.size": 18, "axes.labelsize": 20,
                         "xtick.labelsize": 18, "ytick.labelsize": 18})
    fig, axes = plt.subplots(
        n_examples,
        2,
        figsize=(14, 3.4 * n_examples),
        squeeze=False,
        sharex=True,
        layout="constrained",
    )
    colour_norm = Normalize(
        vmin=float(factors.min()),
        vmax=float(factors.max())
        if factors.max() > factors.min()
        else float(factors.min() + 1),
    )
    colours = plt.cm.viridis(colour_norm(factors))
    for i in range(n_examples):
        axes[i, 0].plot(
            observed_wave[i],
            observed[i],
            color=".5",
            lw=0.65,
            alpha=0.8,
            zorder=2,
            label="Observed spectrum",
        )
        axes[i, 1].plot(
            observed_wave[i],
            observed_flux[i],
            color=".5",
            lw=0.65,
            alpha=0.8,
            zorder=4,
            label="Observed spectrum",
        )
        for j, f in enumerate(factors):
            label = rf"$N/N_{{\rm ref}}={f:g}$"
            axes[i, 0].plot(
                wave[mask], spectra[i, j, mask], color=colours[j], lw=0.9, label=label
            )
            axes[i, 1].plot(wave[mask], restored[i, j, mask], color=colours[j], lw=0.9)
        axes[i, 1].legend(fontsize=16, loc="lower right", frameon=True)
        axes[i, 0].set_ylabel("Normalised flux")
        axes[i, 1].set_ylabel(
            "Flux density\n" + r"[$10^{-17}\,\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}$]"
        )
        for ax in axes[i]:
            ax.set_xlim(3700, 7500)
    for ax in axes[-1]:
        ax.set_xlabel(r"Rest wavelength [$\mathrm{\AA}$]")
    fig.colorbar(
        ScalarMappable(norm=colour_norm, cmap="viridis"),
        ax=axes,
        location="right",
        pad=0.02,
        fraction=0.025,
        shrink=0.6,
        ticks=np.unique(factors),
        label=r"$N/N_{\rm ref}$",
    )
    fig.savefig(out / "norm_conditional_sequence.pdf", dpi=350)
    fig.savefig(out / "norm_conditional_sequence.png", dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--factors", type=float, nargs="+", default=[0.5, 0.75, 1.0, 1.5, 2.0]
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-spectra", type=int, default=3)
    ap.add_argument("--outdir", type=Path, default=OUT)
    ap.add_argument(
        "--input",
        type=Path,
        help="Replot saved spectral arrays from this directory instead of generating spectra.",
    )
    args = ap.parse_args()
    if args.input is not None:
        # if an presaved input file of generated spectra is presented
        # then plot it.
        plot_sequence(args.input.resolve(), args.outdir.resolve())
        return

    import pandas as pd
    import torch
    import yaml
    from flow_matching.solver import ODESolver
    from wwdc_spectra.models.model import LightningFlowMatching, WrappedModel
    from wwdc_spectra.models.load_spender import PretrainedSpender
    from wwdc_spectra.data.sdss import SDSS

    factors = np.asarray(args.factors, dtype=float) # convert norm factors to numpy array
    if not np.isfinite(factors).all() or (factors <= 0).any():
        ap.error("factors must be finite and positive")
    if args.n_spectra < 1:
        ap.error("n-spectra must be positive")

    out = args.outdir.resolve()
    out.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    cfg = yaml.safe_load((ROOT / f"ckpts/{TAG}.yaml").read_text())
    mu = cfg["normalisation"]["train"]["mean"]
    sd = cfg["normalisation"]["train"]["std"]
    raw = pd.read_parquet(ROOT / f"embeddings/{TAG}/test/embeddings.parquet")
    raw["specObjID"] = raw.id.astype(str).str.extract(r"(\d+)")[0].astype("int64")
    raw["context"] = np.stack(raw.y).reshape(-1)
    pool = (
        raw[np.isfinite(raw.context)]
        .sort_values("specObjID", kind="stable")
        .reset_index(drop=True)
    )
    if args.n_spectra > len(pool):
        ap.error("n-spectra exceeds available test rows")
    positions = np.random.default_rng(args.seed).choice(
        len(pool), args.n_spectra, replace=False
    )
    chosen = [pool.iloc[i] for i in positions]
    n_examples, n_factors = len(chosen), len(factors)
    base = PretrainedSpender("sdss_I", 10)
    model = LightningFlowMatching(
        base_model=base,
        lr=0.0001,
        batch_size=512,
        code_dim=10,
        hidden_dim=1024,
        n_layers=3,
        catalog=cfg["data"]["y_catalog"],
        n_steps=20,
    )
    checkpoint = ROOT / f"ckpts/{TAG}.ckpt"
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"],
        strict=True,
    )
    model.to(device).eval()
    solver = ODESolver(velocity_model=WrappedModel(model.vf))

    def tensor(a):
        return torch.as_tensor(a, dtype=torch.float32, device=device)

    def flow(x, y, reverse=False, steps=100):
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Latent/context batch mismatch: {x.shape} vs {y.shape}")
        return solver.sample(
            x_init=x,
            y=y,
            cfg_scale=1.0,
            time_grid=tensor([1.0, 0.0] if reverse else [0.0, 1.0]),
            step_size=1 / steps,
            method="midpoint",
        )

    orig = tensor(np.stack([r.orig for r in chosen]))
    ctx = tensor([[r.context] for r in chosen])
    nref = 10 ** (ctx.cpu().numpy().ravel() * sd + mu) # get the NORM factor 

    contexts = ctx.cpu().numpy() + np.log10(factors)[None, :] / sd
    with torch.inference_mode():
        seed = flow(orig, ctx, reverse=True)
        stored = tensor(np.stack([r.cond for r in chosen]))
        re20 = flow(orig, ctx, reverse=True, steps=20)
        stored_error = float((stored - re20).abs().max())
        assert stored_error < 1e-3, (
            f"Saved conditional embedding mismatch: {stored_error}"
        ) # check the flow inversion is done ok
        gen = flow(
            seed.repeat_interleave(n_factors, dim=0), tensor(contexts.reshape(-1, 1))
        ) # generate latents at each norm
        spectra = (
            model.base_model.decoder(gen)
            .cpu()
            .numpy()
            .reshape(n_examples, n_factors, -1)
        ) # decode generated latents
    wave = model.base_model.model.wave_rest.detach().cpu().numpy()
    assert np.isfinite(spectra).all()
    # Restore amplitude using the requested NORM.
    restored = spectra * (nref[:, None] * factors[None, :])[:, :, None]
    # Retrieve the actual input spectra, preserving the upstream masks and scale.
    dataset = SDSS(
        split="test", y_catalog={"variables": {"Z": {}}, "drop_variables": []}
    )
    source_ids = (
        pd.Series(dataset.dataset["object_id"])
        .astype(str)
        .str.extract(r"(\d+)")[0]
        .astype("int64")
    )
    observed = []
    observed_wave = []
    observed_norms = []
    source_rows = []
    for i, r in enumerate(chosen):
        matches = np.flatnonzero(source_ids.to_numpy() == int(r.specObjID))
        if len(matches) != 1:
            raise ValueError(
                f"Expected one input spectrum for {r.specObjID}, found {len(matches)}"
            )
        source_row = int(matches[0])
        item = dataset.dataset[source_row]
        s = item["spectrum"]
        np.testing.assert_allclose(float(item["Z"]), float(r.z), atol=1e-7, rtol=0)
        spectrum, weights, norm, restwave, _ = dataset.prepare_spectrum(
            s["flux"], s["ivar"], s["lambda"], s["mask"], item["Z"]
        )
        np.testing.assert_allclose(float(norm), nref[i], rtol=2e-6) # check the norm matches the spectrum used before
        with torch.inference_mode():
            encoded = model.base_model.encode(spectrum[None].to(device))
        error = float((encoded - orig[i : i + 1]).abs().max())
        assert error < 1e-4, (
            f"Input spectrum does not reproduce saved original latent: {error}"
        ) # check the input matches the original latent
        values = spectrum.numpy().copy()
        values[(weights.numpy() <= 0) | ~np.isfinite(values)] = np.nan
        observed.append(values)
        observed_wave.append(restwave.numpy())
        observed_norms.append(float(norm))
        source_rows.append(source_row)
    observed = np.stack(observed)
    observed_wave = np.stack(observed_wave)
    observed_norms = np.array(observed_norms)
    np.savez(
        out / "observed_spectra.npz",
        wave_rest=observed_wave,
        normalised_flux=observed,
        flux=observed * observed_norms[:, None],
        normalisation=observed_norms,
        source_test_rows=source_rows,
        specObjID=np.array([int(r.specObjID) for r in chosen], dtype=np.int64),
    ) # save observed spectra
    np.savez(
        out / "generated_spectra.npz",
        wave_rest=wave,
        decoded_raw=spectra,
        amplitude_restored=restored,
        conditional_seed=seed.cpu().numpy(),
        generated_latents=gen.cpu().numpy().reshape(n_examples, n_factors, 10),
        factors=factors,
        reference_norm=nref,
    )
    plot_sequence(out, out)


if __name__ == "__main__":
    main()
