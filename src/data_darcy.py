"""Data loading/preprocessing for PDEBench 2D Darcy Flow (beta=1.0).

Task framing: steady-state operator map
    nu(x, y)  -->  pressure(x, y)
i.e. given a (binary, piecewise-constant) permeability field, predict the
resulting pressure field directly. Unlike Burgers this has no time axis
at all -- every (nu, pressure) pair in the file is already a complete,
independent input/output example, so there's no slicing to do.

Native resolution (128x128) is used as-is: at that size a batch of 20
peaks at under 1GB of GPU memory (verified on a GTX 1650), so unlike
Burgers' 1024->256 downsample there's no need to shrink the grid.
"""
from __future__ import annotations

import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_THIS_DIR)

RAW_PATH = os.path.join(REPO_ROOT, "data", "raw", "darcy", "2D_DarcyFlow_beta1.0_Train.hdf5")
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
CACHE_FILE = os.path.join(PROCESSED_DIR, "darcy_fno_data.pt")

RESOLUTION = 128

SEED = 42
N_TRAIN, N_VAL, N_TEST = 8000, 1000, 1000


def load_pairs(raw_path: str = RAW_PATH):
    with h5py.File(raw_path, "r") as f:
        x = f["x-coordinate"][:].astype(np.float32)
        y = f["y-coordinate"][:].astype(np.float32)
        nu = f["nu"][:].astype(np.float32)          # (N, 128, 128)
        pressure = f["tensor"][:, 0].astype(np.float32)  # (N, 128, 128)
    return x, y, nu, pressure


def make_splits(n_samples: int, n_train: int = N_TRAIN, n_val: int = N_VAL,
                 n_test: int = N_TEST, seed: int = SEED):
    assert n_train + n_val + n_test <= n_samples
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val: n_train + n_val + n_test]
    return train_idx, val_idx, test_idx


def preprocess_and_cache(raw_path: str = RAW_PATH, out_dir: str = PROCESSED_DIR,
                          cache_file: str = CACHE_FILE, force: bool = False) -> dict:
    if os.path.exists(cache_file) and not force:
        return torch.load(cache_file, weights_only=False)

    os.makedirs(out_dir, exist_ok=True)
    x, y, nu, pressure = load_pairs(raw_path)
    n = nu.shape[0]
    train_idx, val_idx, test_idx = make_splits(n)

    # Normalization stats from the TRAIN split only.
    nu_mean, nu_std = float(nu[train_idx].mean()), float(nu[train_idx].std())
    p_mean, p_std = float(pressure[train_idx].mean()), float(pressure[train_idx].std())

    payload = {
        "x": torch.from_numpy(x),
        "y": torch.from_numpy(y),
        "nu": torch.from_numpy(nu),
        "pressure": torch.from_numpy(pressure),
        "train_idx": torch.from_numpy(train_idx),
        "val_idx": torch.from_numpy(val_idx),
        "test_idx": torch.from_numpy(test_idx),
        "nu_mean": nu_mean, "nu_std": nu_std,
        "p_mean": p_mean, "p_std": p_std,
        "resolution": RESOLUTION,
    }
    torch.save(payload, cache_file)
    return payload


class DarcyFNODataset(Dataset):
    """Each example: input = [nu(x,y), x, y] (shape (128, 128, 3)),
    target = pressure(x,y) (shape (128, 128, 1)), both normalized with
    train-set statistics.
    """

    def __init__(self, payload: dict, split: str = "train", normalize: bool = True):
        idx = {
            "train": payload["train_idx"],
            "val": payload["val_idx"],
            "test": payload["test_idx"],
        }[split]
        res = payload["resolution"]
        xg = payload["x"].view(res, 1).expand(res, res)
        yg = payload["y"].view(1, res).expand(res, res)
        self.grid = torch.stack([xg, yg], dim=-1)  # (128, 128, 2)

        self.nu = payload["nu"][idx]
        self.pressure = payload["pressure"][idx]
        self.nu_mean, self.nu_std = payload["nu_mean"], payload["nu_std"]
        self.p_mean, self.p_std = payload["p_mean"], payload["p_std"]
        self.normalize = normalize

    def __len__(self) -> int:
        return self.nu.shape[0]

    def __getitem__(self, i: int):
        nu = self.nu[i]
        pressure = self.pressure[i]
        if self.normalize:
            nu = (nu - self.nu_mean) / self.nu_std
            pressure = (pressure - self.p_mean) / self.p_std
        inp = torch.cat([nu.unsqueeze(-1), self.grid], dim=-1)  # (128, 128, 3)
        target = pressure.unsqueeze(-1)  # (128, 128, 1)
        return inp, target


if __name__ == "__main__":
    payload = preprocess_and_cache(force=True)
    print("split sizes -> train:", len(payload["train_idx"]),
          "val:", len(payload["val_idx"]),
          "test:", len(payload["test_idx"]))
    print("resolution:", payload["resolution"])
    print("nu mean/std:", payload["nu_mean"], payload["nu_std"])
    print("pressure mean/std:", payload["p_mean"], payload["p_std"])

    ds = DarcyFNODataset(payload, "train")
    xb, yb = ds[0]
    print("one sample -> input shape:", xb.shape, "target shape:", yb.shape)
