"""Lightweight inference wrapper for the FastAPI backend.

Loads only the small, self-contained deployment checkpoint (model weights +
hyperparams + normalization stats) -- never the raw HDF5 dataset or the
h5py dependency.
"""
from __future__ import annotations

import numpy as np
import torch

from src.model import FNO1d


class FNOPredictor:
    def __init__(self, checkpoint_path: str, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.resolution: int = ckpt["resolution"]
        self.mean: float = ckpt["mean"]
        self.std: float = ckpt["std"]
        self.grid = np.linspace(0.0, 1.0, self.resolution, endpoint=False, dtype=np.float32)

        self.model = FNO1d(modes=ckpt["modes"], width=ckpt["width"]).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    def _resample(self, u0: np.ndarray) -> np.ndarray:
        """Linearly resample an arbitrary-length initial condition onto the
        model's training grid (uniform, `resolution` points on [0, 1))."""
        if u0.shape[0] == self.resolution:
            return u0.astype(np.float32)
        src_x = np.linspace(0.0, 1.0, u0.shape[0], endpoint=False, dtype=np.float32)
        return np.interp(self.grid, src_x, u0).astype(np.float32)

    @torch.no_grad()
    def predict(self, u0: list[float]) -> dict:
        u0_arr = self._resample(np.asarray(u0, dtype=np.float32))
        u0_norm = (u0_arr - self.mean) / self.std

        inp = torch.stack([
            torch.from_numpy(u0_norm),
            torch.from_numpy(self.grid),
        ], dim=-1).unsqueeze(0).to(self.device)  # (1, resolution, 2)

        pred_norm = self.model(inp).squeeze(0).squeeze(-1).cpu().numpy()  # (resolution,)
        pred = pred_norm * self.std + self.mean

        return {
            "x": self.grid.tolist(),
            "u0_resampled": u0_arr.tolist(),
            "prediction": pred.tolist(),
        }
