"""Zero-shot resolution-invariance evaluation for the trained FNO2d Darcy model.

Mirrors src/eval_superres.py's methodology, but in the OPPOSITE direction.
Burgers' raw PDEBench data is native-1024 (4x the 256 training resolution),
so that script could validate true zero-shot *super*-resolution against
independent higher-resolution ground truth. Darcy's raw PDEBench data is
native-128 -- exactly the training resolution -- so there is no higher-res
ground truth available in this dataset to upsample against.

What IS verifiable here: the same architectural property (SpectralConv2d
truncates to a *fixed* number of Fourier modes, 12x12, independent of H/W;
every other op is pointwise) also means the model can run at resolutions
LOWER than training with zero retraining. 128 is evenly divisible by 64
and 32, so those coarser grids are built by *exact* strided subsampling
(never lossy interpolation) of the real native field -- exactly how you'd
build them if that were the native resolution to begin with -- giving
real, independent ground truth at every tested resolution.

Usage:
    python -m src.eval_superres_darcy
"""
from __future__ import annotations

import h5py
import numpy as np
import torch

from src.data_darcy import RAW_PATH, make_splits
from src.model_fno2d import FNO2d
from src.train import relative_l2_loss

CHECKPOINT_PATH = "app/model/fno2d_darcy_final.pt"
FULL_RES = 128
RESOLUTIONS = [128, 64, 32]
BATCH_SIZE = 50


def load_native_test_set():
    """Read nu and pressure for the test split at the raw file's native
    128x128 resolution (never the full (10000,128,128) tensor for every split)."""
    with h5py.File(RAW_PATH, "r") as f:
        n = f["nu"].shape[0]
        _, _, test_idx = make_splits(n)
        test_idx_sorted = np.sort(test_idx)  # h5py fancy indexing requires increasing order
        nu = f["nu"][test_idx_sorted].astype(np.float32)
        pressure = f["tensor"][test_idx_sorted, 0].astype(np.float32)
        x = f["x-coordinate"][:].astype(np.float32)
        y = f["y-coordinate"][:].astype(np.float32)
    return x, y, nu, pressure


@torch.no_grad()
def eval_at_resolution(model, nu_mean, nu_std, p_mean, p_std, x_full, y_full, nu_full, p_full, res, device):
    stride = FULL_RES // res
    x = x_full[::stride]
    y = y_full[::stride]
    nu = nu_full[:, ::stride, ::stride]
    pressure = p_full[:, ::stride, ::stride]

    xg = torch.from_numpy(x).view(res, 1).expand(res, res)
    yg = torch.from_numpy(y).view(1, res).expand(res, res)
    grid = torch.stack([xg, yg], dim=-1).to(device)  # (res, res, 2)

    losses = []
    for start in range(0, nu.shape[0], BATCH_SIZE):
        batch_nu = nu[start:start + BATCH_SIZE]
        batch_p = pressure[start:start + BATCH_SIZE]

        nu_norm = (batch_nu - nu_mean) / nu_std
        p_norm = (batch_p - p_mean) / p_std

        inp = torch.cat([
            torch.from_numpy(nu_norm).unsqueeze(-1).to(device),
            grid.unsqueeze(0).expand(batch_nu.shape[0], -1, -1, -1),
        ], dim=-1)  # (batch, res, res, 3)
        target = torch.from_numpy(p_norm).to(device).unsqueeze(-1)  # (batch, res, res, 1)

        pred = model(inp)  # model runs natively at `res`, no resampling
        loss = relative_l2_loss(pred, target, norm_floor=0.1)
        losses.append(loss.item() * batch_nu.shape[0])

    return sum(losses) / nu.shape[0]


def main():
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FNO2d(modes1=ckpt["modes1"], modes2=ckpt["modes2"], width=ckpt["width"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    nu_mean, nu_std = ckpt["nu_mean"], ckpt["nu_std"]
    p_mean, p_std = ckpt["p_mean"], ckpt["p_std"]
    train_res = ckpt["resolution"]

    print(f"checkpoint: trained at resolution {train_res}x{train_res}, "
          f"modes=({ckpt['modes1']},{ckpt['modes2']}), width={ckpt['width']}, "
          f"reported val rel-L2={ckpt['val_loss']:.4f}")
    print(f"device: {device}")
    print("loading native-resolution (128x128) test set from raw HDF5 ...")
    x_full, y_full, nu_full, p_full = load_native_test_set()
    print(f"test set: {nu_full.shape[0]} held-out samples at native resolution {FULL_RES}x{FULL_RES}\n")

    results = {}
    for res in RESOLUTIONS:
        results[res] = eval_at_resolution(
            model, nu_mean, nu_std, p_mean, p_std, x_full, y_full, nu_full, p_full, res, device
        )

    print(f"{'resolution':>12} | {'vs. training res':>20} | {'relative L2':>14}")
    print("-" * 54)
    for res in RESOLUTIONS:
        if res == train_res:
            tag = "training resolution"
        else:
            factor = train_res // res
            tag = f"{factor}x zero-shot coarsening"
        print(f"{res:>9}x{res:<2} | {tag:>20} | {results[res]:>14.6f}")


if __name__ == "__main__":
    main()
