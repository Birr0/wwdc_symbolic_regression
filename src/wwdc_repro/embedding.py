"""Generate embeddings with the pinned upstream implementation and paper checkpoint.

Requires the inference extra, an upstream checkout, and the archived spectrum
splits. A limit supports a quick comparison without running the full dataset.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from .cli import sha256

UPSTREAM_REVISION = "0895b4ed9dbe6815cbe641fd1f936c6d53197aa5"
CHECKPOINT_SHA256 = "92d9aaf21e8523c1c45ee7b61c39ec53ea6235c308907e5a84475f1b7ff19604"
CONFIG_SHA256 = "e1e7c16ac99eb87fe9914554233fe9f7ebab04e9e93247af6bd916da8353bb3a"
PARENT_ROWS = {"train": 422481, "val": 52810, "test": 52811}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream", type=Path, required=True, help="Clean wwdc_spectra checkout"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--splits", nargs="+", choices=list(PARENT_ROWS), default=list(PARENT_ROWS)
    )
    parser.add_argument(
        "--limit", type=int, help="First N rows per split; omit for full generation"
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--compare-stored",
        type=Path,
        help="Existing embedding root to compare by spectrum ID",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Batch size and limit must be positive")
    if args.outdir.exists() and any(args.outdir.iterdir()):
        parser.error("Output directory must be new or empty")
    revision = subprocess.check_output(
        ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    changed = subprocess.check_output(
        ["git", "-C", str(args.upstream), "status", "--porcelain", "--", "src"],
        text=True,
    )
    if revision != UPSTREAM_REVISION or changed.strip():
        parser.error(f"Use a clean upstream source checkout at {UPSTREAM_REVISION}")
    run_root = args.data_root / "sdss_II/spender_I_flow_v2/spender_I_flow_v2"
    checkpoint, config = (
        run_root / "ckpts/7655991_0.ckpt",
        run_root / "ckpts/7655991_0.yaml",
    )
    for path, expected in ((checkpoint, CHECKPOINT_SHA256), (config, CONFIG_SHA256)):
        if not path.is_file() or sha256(path) != expected:
            parser.error(f"Missing or different paper checkpoint/config: {path}")
    if not (args.data_root / "sdss/sdss_II_catalog").is_dir():
        parser.error(
            "The archived sdss/sdss_II_catalog split dataset is required; do not recreate splits"
        )

    os.environ["DATA_ROOT"] = str(args.data_root.resolve())
    sys.path.insert(0, str((args.upstream / "src").resolve()))
    import numpy as np
    import pandas as pd
    import torch
    from flow_matching.solver import ODESolver
    from wwdc_spectra.data.sdss import SDSS
    from wwdc_spectra.models.load_spender import PretrainedSpender
    from wwdc_spectra.models.model import LightningFlowMatching, WrappedModel

    torch.set_num_threads(min(torch.get_num_threads(), 4))
    base = PretrainedSpender("sdss_I", 10)
    catalog = {
        "variables": {"Z": {"name": "redshift", "size": 1, "processing_fn": None}},
        "drop_variables": [],
    }
    model = LightningFlowMatching(
        base_model=base,
        lr=0.0001,
        batch_size=512,
        code_dim=10,
        hidden_dim=1024,
        catalog=catalog,
        n_steps=20,
    )
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"],
        strict=True,
    )
    model.wrapped_vf = WrappedModel(model.vf)
    model.solver = ODESolver(velocity_model=model.wrapped_vf)
    model.eval()
    args.outdir.mkdir(parents=True, exist_ok=True)
    report = {
        "upstream_revision": revision,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "config_sha256": CONFIG_SHA256,
        "device": "cpu",
        "command": sys.argv,
        "python": sys.version,
        "batch_size": args.batch_size,
        "partial": args.limit is not None,
        "solver": {
            "method": model.method,
            "n_steps": 20,
            "step_size": model.step_size,
            "conditional_cfg_scale": 1.0,
            "unconditional_cfg_scale": 0.0,
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "torchvision",
                "lightning",
                "spender",
                "flow-matching",
                "datasets",
                "numpy",
                "pandas",
                "pyarrow",
                "python-dotenv",
                "timm",
                "huggingface-hub",
            )
        },
        "splits": {},
        "passed": True,
    }
    for split in args.splits:
        dataset = SDSS(
            split=split,
            y_catalog={"variables": {"Z": {}}, "drop_variables": []},
            return_id=True,
            return_pos=True,
            return_mask_ratio=True,
        )
        if len(dataset) != PARENT_ROWS[split]:
            raise ValueError(f"Unexpected archived {split} row count: {len(dataset)}")
        n = min(args.limit, len(dataset)) if args.limit else len(dataset)
        destination = args.outdir / split
        destination.mkdir()
        stored = None
        if args.compare_stored:
            paths = sorted((args.compare_stored / split).glob("*.parquet"))
            if not paths:
                raise FileNotFoundError(f"No stored comparison shards for {split}")
            stored = pd.concat(
                [pd.read_parquet(p) for p in paths], ignore_index=True
            ).set_index("id", verify_integrity=True)
        delta_max = {key: 0.0 for key in ("orig", "cond", "uncond", "z", "mask_ratio")}
        input_digest = hashlib.sha256()
        id_digest = hashlib.sha256()
        for shard, start in enumerate(range(0, n, args.batch_size)):
            items = [dataset[i] for i in range(start, min(start + args.batch_size, n))]
            x = torch.stack([item[0] for item in items])
            y = torch.stack([item[1] for item in items]).to(torch.float32)
            input_digest.update(x.numpy().tobytes())
            ids = [item[2] for item in items]
            for identity in ids:
                id_digest.update((str(identity) + "\n").encode())
            with torch.inference_mode():
                predicted = model.predict_step(
                    x, y, embed_opt=["orig", "cond", "uncond"]
                )
            values = {
                name: array.detach().cpu().numpy() for name, array in predicted.items()
            }
            if not all(np.isfinite(array).all() for array in values.values()):
                raise ValueError(f"Nonfinite embedding in {split} at row {start}")
            frame = pd.DataFrame(
                {
                    **{key: value.tolist() for key, value in values.items()},
                    "id": ids,
                    "BESTOBJID": [item[3] for item in items],
                    "ra": [item[4] for item in items],
                    "dec": [item[5] for item in items],
                    "z": [item[6] for item in items],
                    "mask_ratio": [float(item[7]) for item in items],
                }
            )
            frame.to_parquet(destination / f"{shard}.parquet", index=False)
            if stored is not None:
                expected = stored.loc[ids]
                for key, actual in values.items():
                    delta_max[key] = max(
                        delta_max[key],
                        float(np.max(np.abs(actual - np.stack(expected[key])))),
                    )
                for key in ("z", "mask_ratio"):
                    delta_max[key] = max(
                        delta_max[key],
                        float(
                            np.max(
                                np.abs(frame[key].to_numpy() - expected[key].to_numpy())
                            )
                        ),
                    )
            print(f"{split}: {start + len(items)}/{n}", flush=True)
        tolerance = {
            "orig": 2e-6,
            "cond": 2e-5,
            "uncond": 2e-5,
            "z": 0,
            "mask_ratio": 0,
        }
        result = {
            "rows": n,
            "prepared_spectra_sha256": input_digest.hexdigest(),
            "ordered_source_id_sha256": id_digest.hexdigest(),
        }
        if stored is not None:
            result.update(
                {
                    "max_abs_error": delta_max,
                    "tolerance": tolerance,
                    "passed": all(
                        delta_max[key] <= tolerance[key] for key in tolerance
                    ),
                }
            )
            report["passed"] &= result["passed"]
        report["splits"][split] = result
    (args.outdir / "embedding_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    if not report["passed"]:
        raise SystemExit("Embedding comparison failed; inspect embedding_manifest.json")
    print(f"Completed {args.outdir}")
