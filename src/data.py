"""Data loading/preprocessing for PDEBench 1D Burgers (nu=0.01).

Task framing: the model learns a single-shot operator map
    u(x, t=0)  -->  u(x, t=T_final)
i.e. given an initial condition it predicts the fully-evolved solution
directly, with no autoregressive time-stepping. This also matches the
FastAPI backend contract: "take an initial condition, return a
predicted solution".

Only the first and last time slices are ever read from the HDF5 file,
so we never load the full (10000, 201, 1024) tensor (~8GB) into memory.
"""
from __future__ import annotations

import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_THIS_DIR)

RAW_PATH = os.path.join(REPO_ROOT, "data", "1D_Burgers_Sols_Nu0.01.hdf5")
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
CACHE_FILE = os.path.join(PROCESSED_DIR, "burgers_fno_data.pt")

FULL_RES = 1024
TARGET_RES = 256
DOWNSAMPLE_STRIDE = FULL_RES // TARGET_RES  # 4

SEED = 42
N_TRAIN, N_VAL, N_TEST = 8000, 1000, 1000


def load_initial_final_pairs(raw_path: str = RAW_PATH, final_t_index: int = -1):
    """Read only the t=0 and t=final slices for every trajectory, downsampled in space."""
    with h5py.File(raw_path, "r") as f:
        x = f["x-coordinate"][::DOWNSAMPLE_STRIDE].astype(np.float32)
        u0 = f["tensor"][:, 0, :][:, ::DOWNSAMPLE_STRIDE].astype(np.float32)
        uT = f["tensor"][:, final_t_index, :][:, ::DOWNSAMPLE_STRIDE].astype(np.float32)
    return x, u0, uT


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
    """Build (and cache to disk) the downsampled initial/final pairs + split indices
    + normalization stats. Re-running this is cheap once cached."""
    if os.path.exists(cache_file) and not force:
        return torch.load(cache_file, weights_only=False)

    os.makedirs(out_dir, exist_ok=True)
    x, u0, uT = load_initial_final_pairs(raw_path)
    n = u0.shape[0]
    train_idx, val_idx, test_idx = make_splits(n)

    # Normalization stats computed on the TRAIN split only, to avoid leaking
    # val/test statistics into training.
    mean = float(u0[train_idx].mean())
    std = float(u0[train_idx].std())

    payload = {
        "x": torch.from_numpy(x),
        "u0": torch.from_numpy(u0),
        "uT": torch.from_numpy(uT),
        "train_idx": torch.from_numpy(train_idx),
        "val_idx": torch.from_numpy(val_idx),
        "test_idx": torch.from_numpy(test_idx),
        "mean": mean,
        "std": std,
        "resolution": TARGET_RES,
    }
    torch.save(payload, cache_file)
    return payload


class BurgersFNODataset(Dataset):
    """Each example: input = [u(x,0), x] (shape (TARGET_RES, 2)),
    target = u(x, T_final) (shape (TARGET_RES, 1)), both normalized
    with train-set statistics.
    """

    def __init__(self, payload: dict, split: str = "train", normalize: bool = True):
        idx = {
            "train": payload["train_idx"],
            "val": payload["val_idx"],
            "test": payload["test_idx"],
        }[split]
        self.grid = payload["x"]
        self.u0 = payload["u0"][idx]
        self.uT = payload["uT"][idx]
        self.mean = payload["mean"]
        self.std = payload["std"]
        self.normalize = normalize

    def __len__(self) -> int:
        return self.u0.shape[0]

    def __getitem__(self, i: int):
        u0 = self.u0[i]
        uT = self.uT[i]
        if self.normalize:
            u0 = (u0 - self.mean) / self.std
            uT = (uT - self.mean) / self.std
        inp = torch.stack([u0, self.grid], dim=-1)  # (TARGET_RES, 2): [u0(x), x]
        target = uT.unsqueeze(-1)  # (TARGET_RES, 1)
        return inp, target


if __name__ == "__main__":
    payload = preprocess_and_cache(force=True)
    print("split sizes -> train:", len(payload["train_idx"]),
          "val:", len(payload["val_idx"]),
          "test:", len(payload["test_idx"]))
    print("resolution:", payload["resolution"], "mean/std:", payload["mean"], payload["std"])

    ds = BurgersFNODataset(payload, "train")
    xb, yb = ds[0]
    print("one sample -> input shape:", xb.shape, "target shape:", yb.shape)
