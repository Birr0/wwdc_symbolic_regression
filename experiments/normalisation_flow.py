"""Train the pinned WWDC flow with log10 spectral normalisation as its only context.

Uses archived parent splits, exact upstream preprocessing and cached original
Spender embeddings. No stellar-mass labels or redshift conditioning are used.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

REVISION = "0895b4ed9dbe6815cbe641fd1f936c6d53197aa5"


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class NormDataset(Dataset):
    def __init__(self, source):
        self.source = source

    def __len__(self):
        return len(self.source.dataset)

    def __getitem__(self, index):
        row = self.source.dataset[index]
        s = row["spectrum"]
        _, _, norm, _, _ = self.source.prepare_spectrum(
            s["flux"], s["ivar"], s["lambda"], s["mask"], row["Z"]
        )
        return index, float(norm)


class IdentityEncoder(torch.nn.Module):
    def encode(self, x):
        return x


def prepare(args):
    from wwdc_spectra.data.sdss import SDSS
    from wwdc_repro.data import spectrum_ids

    if args.prepared.exists() and any(args.prepared.iterdir()):
        raise FileExistsError("Prepared directory must be new or empty")
    args.prepared.mkdir(parents=True, exist_ok=True)
    root = args.data_root / "sdss_II/spender_I_flow_v2/spender_I_flow_v2/embeddings/7655991_0"
    metadata = {"upstream_revision": REVISION, "splits": {}, "inputs": {},
                "normalisation": "Exact upstream SDSS.prepare_spectrum, including masks, skyline rejection and torch.median",
                "selection": "Archived parent flow splits; finite orig and finite strictly positive norm; no mass/catalogue/redshift cuts",
                "sdss_source_sha256": digest(args.upstream / "src/wwdc_spectra/data/sdss.py")}
    all_ids = set()
    for split in ("train", "val", "test"):
        source = SDSS(split=split, y_catalog={"variables": {"Z": {}}, "drop_variables": []})
        ids = spectrum_ids(pd.Series(source.dataset["object_id"]), f"raw {split}").to_numpy()
        if len(np.unique(ids)) != len(ids) or all_ids.intersection(ids.tolist()):
            raise ValueError("Duplicate spectrum IDs within or across splits")
        all_ids.update(ids.tolist())
        norms = np.full(len(ids), np.nan, dtype=np.float64)
        start = time.monotonic()
        loader = DataLoader(NormDataset(source), batch_size=512, num_workers=args.workers, shuffle=False)
        for i, (positions, values) in enumerate(loader):
            norms[positions.numpy()] = values.numpy()
            if i % 40 == 0:
                print(f"prepare {split}: {min((i+1)*512, len(ids))}/{len(ids)} ({time.monotonic()-start:.1f}s)", flush=True)
        files = sorted((root / split).glob("*.parquet"))
        if not files:
            raise FileNotFoundError(root / split)
        frame = pd.concat([pd.read_parquet(p, columns=["id", "orig", "z"]) for p in files], ignore_index=True)
        frame["specObjID"] = spectrum_ids(frame.pop("id"), f"embeddings {split}")
        if set(frame.specObjID) != set(ids):
            raise ValueError(f"{split}: embedding/raw ID mismatch")
        frame = frame.merge(pd.DataFrame({"specObjID": ids, "norm": norms}), on="specObjID", validate="one_to_one", how="left", sort=False)
        original = np.stack(frame.orig).astype(np.float32)
        valid_norm = np.isfinite(frame.norm) & (frame.norm > 0)
        valid_orig = np.isfinite(original).all(axis=1)
        keep = valid_norm & valid_orig
        frame["retained"] = keep
        frame.to_parquet(args.prepared / f"{split}_rows.parquet", index=False)
        np.savez(args.prepared / f"{split}.npz", orig=original[keep], norm=frame.norm.to_numpy()[keep],
                 ids=frame.specObjID.to_numpy()[keep], redshift=frame.z.to_numpy()[keep])
        metadata["splits"][split] = {"parent": len(frame), "retained": int(keep.sum()),
                                      "invalid_norm": int((~valid_norm).sum()), "invalid_orig": int((~valid_orig).sum()),
                                      "seconds": time.monotonic()-start,
                                      "ordered_ids_sha256": hashlib.sha256(frame.specObjID.to_numpy(dtype="<i8")[keep].tobytes()).hexdigest()}
        metadata["inputs"][split] = {str(p): digest(p) for p in files}
        print(metadata["splits"][split], flush=True)
    training = np.load(args.prepared / "train.npz")
    lognorm = np.log10(training["norm"])
    metadata["condition_mean"] = float(lognorm.mean())
    metadata["condition_std"] = float(lognorm.std())
    if not metadata["condition_std"] > 0:
        raise ValueError("Constant training condition")
    metadata["prepared_sha256"] = {s: digest(args.prepared / f"{s}.npz") for s in ("train", "val", "test")}
    (args.prepared / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


def train(args):
    import lightning as L
    from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from wwdc_spectra.models.model import LightningFlowMatching

    args.outdir.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.prepared / "manifest.json").read_text())
    L.seed_everything(args.seed, workers=True)
    loaders = {}
    for split in ("train", "val"):
        if digest(args.prepared / f"{split}.npz") != manifest["prepared_sha256"][split]:
            raise ValueError("Prepared data hash mismatch")
        a = np.load(args.prepared / f"{split}.npz")
        y = ((np.log10(a["norm"]) - manifest["condition_mean"]) / manifest["condition_std"]).astype(np.float32)[:, None]
        loaders[split] = DataLoader(TensorDataset(torch.from_numpy(a["orig"]), torch.from_numpy(y)),
                                    batch_size=512, shuffle=split == "train", num_workers=0, pin_memory=True)
    model = LightningFlowMatching(base_model=IdentityEncoder(), lr=1e-4, batch_size=512,
        code_dim=10, hidden_dim=1024, catalog={"variables": {"lognorm": {"size": 1}}, "drop_variables": []})

    class Timing(Callback):
        def on_train_epoch_start(self, trainer, module):
            self.start = time.monotonic()

        def on_train_epoch_end(self, trainer, module):
            print(f"epoch={trainer.current_epoch} seconds={time.monotonic()-self.start:.2f} metrics="
                  f"{ {k: float(v) for k,v in trainer.callback_metrics.items()} }", flush=True)

    checkpoint = ModelCheckpoint(dirpath=args.outdir / "checkpoints", monitor="val_loss", mode="min", save_top_k=1, save_last=True)
    config = {"command": sys.argv, "seed": args.seed, "upstream_revision": REVISION,
              "condition": "(log10(norm) - train_mean) / train_std; one scalar; no redshift input",
              "hidden_dim": 1024, "layers": 3, "batch_size": 512, "lr": 1e-4,
              "optimizer": "Upstream AdamW defaults", "objective": "Upstream doubled conditional/unconditional flow-matching loss",
              "max_epochs": args.epochs, "patience": 10, "min_delta": 0.0001,
              "precision": "16-mixed", "gradient_clip_val": 1.0,
              "torch": torch.__version__, "lightning": L.__version__,
              "gpu": torch.cuda.get_device_name(), "prepared_manifest": manifest,
              "script_sha256": digest(Path(__file__)),
              "upstream_model_sha256": digest(args.upstream / "src/wwdc_spectra/models/model.py"),
              "pilot_batches": args.pilot_batches}
    config["runtime_packages"] = {
        name: importlib.metadata.version(name)
        for name in ("torch", "lightning", "numpy", "pandas", "pyarrow", "datasets", "timm", "flow-matching", "spender")
    }
    (args.outdir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    trainer = L.Trainer(accelerator="gpu", devices=1, precision="16-mixed", max_epochs=args.epochs,
        gradient_clip_val=1.0, deterministic="warn", enable_progress_bar=False,
        logger=CSVLogger(str(args.outdir), name="metrics"), log_every_n_steps=10,
        callbacks=[checkpoint, EarlyStopping(monitor="val_loss", patience=10, min_delta=0.0001), Timing()],
        limit_train_batches=args.pilot_batches or 1.0,
        limit_val_batches=min(args.pilot_batches, 8) if args.pilot_batches else 1.0)
    start = time.monotonic()
    trainer.fit(model, loaders["train"], loaders["val"])
    result = {"seconds": time.monotonic()-start, "best_checkpoint": checkpoint.best_model_path,
              "best_val_loss": float(checkpoint.best_model_score), "epochs_completed": trainer.current_epoch,
              "status": "pilot_complete" if args.pilot_batches else "training_complete"}
    (args.outdir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "train"])
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--outdir", type=Path)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--pilot-batches", type=int, default=0)
    args = p.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    changes = subprocess.check_output(["git", "-C", str(args.upstream), "status", "--porcelain", "--", "src"], text=True).strip()
    if revision != REVISION or changes:
        p.error(f"A clean upstream checkout at {REVISION} is required")
    if args.stage == "train" and args.outdir is None:
        p.error("--outdir is required for training")
    torch.set_num_threads(4)
    os.environ["DATA_ROOT"] = str(args.data_root.resolve())
    sys.path.insert(0, str(args.upstream / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    (prepare if args.stage == "prepare" else train)(args)


if __name__ == "__main__":
    main()
