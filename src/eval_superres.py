"""Zero-shot super-resolution evaluation for the trained FNO1d Burgers model.

Tests the central claim of Fourier Neural Operators (Li et al. 2020): the
learned operator is resolution-invariant. Every op in FNO1d is either a
pointwise MLP / 1x1 conv (shared across all grid points, independent of how
many there are) or a spectral conv that truncates to a *fixed* number of
Fourier modes, independent of input length N. So a model trained only at
256 points can be run -- with zero retraining and zero change to its
weights -- directly at 512 or 1024 points, just by feeding it a longer
input.

Methodology: PDEBench's native grid is 1024 points, evenly divisible by
512 and 256, so all three evaluation resolutions are built by *exact*
strided subsampling of the native field (never lossy interpolation) --
this is also exactly how the original 256-resolution training data was
built (src/data.py), so the "256" row below is a built-in sanity check.
Predictions at resolution R are compared via relative L2 against the
native ground truth subsampled to that SAME resolution R, so every row is
an apples-to-apples same-grid comparison.

Usage:
    python -m src.eval_superres
"""
from __future__ import annotations

import h5py
import numpy as np
import torch

from src.data import RAW_PATH, make_splits
from src.model import FNO1d
from src.train import relative_l2_loss

CHECKPOINT_PATH = "app/model/fno_burgers_final.pt"
FULL_RES = 1024
RESOLUTIONS = [256, 512, 1024]
BATCH_SIZE = 100


def load_native_test_set():
    """Read only the t=0 / t=final slices for the test split, at the raw
    file's native 1024 resolution (never the full (10000, 201, 1024) tensor).
    """
    with h5py.File(RAW_PATH, "r") as f:
        n = f["tensor"].shape[0]
        _, _, test_idx = make_splits(n)
        test_idx_sorted = np.sort(test_idx)  # h5py fancy indexing requires increasing order
        u0 = f["tensor"][test_idx_sorted, 0, :].astype(np.float32)
        uT = f["tensor"][test_idx_sorted, -1, :].astype(np.float32)
        x = f["x-coordinate"][:].astype(np.float32)
    return x, u0, uT


@torch.no_grad()
def eval_at_resolution(model, mean, std, x_full, u0_full, uT_full, res, device):
    stride = FULL_RES // res
    x = x_full[::stride]              # exact subsample, same convention as training
    u0 = u0_full[:, ::stride]
    uT = uT_full[:, ::stride]

    grid_t = torch.from_numpy(x).to(device)
    losses = []
    for start in range(0, u0.shape[0], BATCH_SIZE):
        batch_u0 = u0[start:start + BATCH_SIZE]
        batch_uT = uT[start:start + BATCH_SIZE]

        u0_norm = (batch_u0 - mean) / std
        uT_norm = (batch_uT - mean) / std

        inp = torch.stack([
            torch.from_numpy(u0_norm).to(device),
            grid_t.unsqueeze(0).expand(batch_u0.shape[0], -1),
        ], dim=-1)  # (batch, res, 2)
        target = torch.from_numpy(uT_norm).to(device).unsqueeze(-1)  # (batch, res, 1)

        pred = model(inp)  # (batch, res, 1) -- model runs natively at `res`, no resampling
        loss = relative_l2_loss(pred, target, norm_floor=0.1)
        losses.append(loss.item() * batch_u0.shape[0])

    return sum(losses) / u0.shape[0]


def main():
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FNO1d(modes=ckpt["modes"], width=ckpt["width"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    mean, std, train_res = ckpt["mean"], ckpt["std"], ckpt["resolution"]

    print(f"checkpoint: trained at resolution {train_res}, modes={ckpt['modes']}, "
          f"width={ckpt['width']}, reported val rel-L2={ckpt['val_loss']:.4f}")
    print(f"device: {device}")
    print("loading native-resolution (1024) test set from raw HDF5 ...")
    x_full, u0_full, uT_full = load_native_test_set()
    print(f"test set: {u0_full.shape[0]} held-out trajectories at native resolution {FULL_RES}\n")

    results = {}
    for res in RESOLUTIONS:
        results[res] = eval_at_resolution(model, mean, std, x_full, u0_full, uT_full, res, device)

    print(f"{'resolution':>10} | {'vs. training res':>20} | {'relative L2':>14}")
    print("-" * 50)
    for res in RESOLUTIONS:
        if res == train_res:
            tag = "training resolution"
        else:
            factor = res // train_res
            tag = f"{factor}x zero-shot super-res"
        print(f"{res:>10} | {tag:>20} | {results[res]:>14.6f}")


if __name__ == "__main__":
    main()
