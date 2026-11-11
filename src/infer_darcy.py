"""Lightweight inference wrapper for the Darcy Flow FNO2d backend.

Loads only the small, self-contained deployment checkpoint -- never the
raw HDF5 dataset or h5py.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.model_fno2d import FNO2d


class DarcyPredictor:
    def __init__(self, checkpoint_path: str, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.resolution: int = ckpt["resolution"]
        self.nu_mean: float = ckpt["nu_mean"]
        self.nu_std: float = ckpt["nu_std"]
        self.p_mean: float = ckpt["p_mean"]
        self.p_std: float = ckpt["p_std"]

        res = self.resolution
        xg = torch.linspace(0.0, 1.0, res).view(res, 1).expand(res, res)
        yg = torch.linspace(0.0, 1.0, res).view(1, res).expand(res, res)
        self.grid = torch.stack([xg, yg], dim=-1).to(self.device)  # (res, res, 2)

        self.model = FNO2d(modes1=ckpt["modes1"], modes2=ckpt["modes2"], width=ckpt["width"]).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    def _resample(self, field: np.ndarray) -> np.ndarray:
        """Bilinearly resample an arbitrary H x W permeability field onto
        the model's training grid (resolution x resolution)."""
        if field.shape == (self.resolution, self.resolution):
            return field.astype(np.float32)
        t = torch.from_numpy(field.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        t = F.interpolate(t, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False)
        return t.squeeze(0).squeeze(0).numpy()

    @torch.no_grad()
    def predict(self, nu: list[list[float]]) -> dict:
        nu_arr = self._resample(np.asarray(nu, dtype=np.float32))
        nu_norm = (nu_arr - self.nu_mean) / self.nu_std

        inp = torch.cat([
            torch.from_numpy(nu_norm).unsqueeze(-1).to(self.device),
            self.grid,
        ], dim=-1).unsqueeze(0)  # (1, res, res, 3)

        pred_norm = self.model(inp).squeeze(0).squeeze(-1).cpu().numpy()  # (res, res)
        pred = pred_norm * self.p_std + self.p_mean

        return {
            "resolution": self.resolution,
            "nu_resampled": nu_arr.tolist(),
            "pressure": pred.tolist(),
        }
