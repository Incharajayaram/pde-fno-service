"""FastAPI backend for the FNO Burgers-equation operator.

Serves:
  GET  /                  the demo frontend (single static HTML page)
  GET  /api/health        liveness check
  GET  /api/presets       synthetic example initial conditions + real
                          held-out PDEBench test samples (with ground truth)
  POST /api/predict       {"u0": [floats]} -> predicted final solution

Only the small deployment checkpoint (and a small bundle of real test
samples) is loaded at startup -- not the full dataset, not h5py.
"""
from __future__ import annotations

import json
import os

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.infer import FNOPredictor

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(APP_DIR, "model", "fno_burgers_final.pt")
CHECKPOINT_PATH = os.environ.get("MODEL_CHECKPOINT_PATH", DEFAULT_CKPT)
REAL_SAMPLES_PATH = os.path.join(APP_DIR, "model", "real_samples.json")

app = FastAPI(title="PDE Solutions as a Service (FNO / Burgers)")
predictor: FNOPredictor | None = None
real_samples: dict | None = None


@app.on_event("startup")
def load_model() -> None:
    global predictor, real_samples
    predictor = FNOPredictor(CHECKPOINT_PATH)
    with open(REAL_SAMPLES_PATH) as f:
        real_samples = json.load(f)


class PredictRequest(BaseModel):
    u0: list[float] = Field(..., min_length=8, description="Initial condition values on a uniform grid over [0, 1)")


@app.get("/")
def index():
    return FileResponse(os.path.join(APP_DIR, "static", "index.html"))


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None}


def _synthetic_presets(x: np.ndarray) -> dict:
    rng = np.random.default_rng(42)  # fixed seed: same curve every time this preset is picked
    harmonics = rng.integers(1, 6, size=5)
    phases = rng.uniform(0, 2 * np.pi, size=5)
    amplitudes = rng.uniform(0.3, 1.0, size=5) / harmonics  # energy decays with frequency, like PDEBench ICs
    random_smooth = sum(a * np.sin(2 * np.pi * k * x + p) for a, k, p in zip(amplitudes, harmonics, phases))

    return {
        "sine_low (k=1)": (0.8 * np.sin(2 * np.pi * x)).tolist(),
        "sine_high (k=4)": (0.5 * np.sin(2 * np.pi * 4 * x)).tolist(),
        "step": np.where((x > 0.25) & (x < 0.75), 0.8, -0.2).tolist(),
        "double_step": np.select(
            [x < 0.33, x < 0.66], [-0.6, 0.6], default=0.0
        ).tolist(),
        "square_wave": (0.7 * np.sign(np.sin(2 * np.pi * 3 * x))).tolist(),
        "triangle": (0.7 * (1 - 4 * np.abs(x - 0.5))).tolist(),
        "gaussian_bump": (0.9 * np.exp(-((x - 0.5) ** 2) / (2 * 0.05 ** 2))).tolist(),
        "gaussian_wide": (0.9 * np.exp(-((x - 0.5) ** 2) / (2 * 0.15 ** 2))).tolist(),
        "two_bumps": (
            0.7 * np.exp(-((x - 0.3) ** 2) / (2 * 0.04 ** 2))
            - 0.7 * np.exp(-((x - 0.7) ** 2) / (2 * 0.04 ** 2))
        ).tolist(),
        "random_smooth": random_smooth.tolist(),
    }


@app.get("/api/presets")
def presets():
    assert predictor is not None and real_samples is not None
    x = predictor.grid
    return {
        "x": x.tolist(),
        "synthetic": _synthetic_presets(x),
        "real_samples": [
            {"label": f"Real PDEBench sample #{i + 1}", "u0": s["u0"], "true_uT": s["true_uT"]}
            for i, s in enumerate(real_samples["samples"])
        ],
    }


@app.post("/api/predict")
def predict(req: PredictRequest):
    assert predictor is not None
    if not np.all(np.isfinite(req.u0)):
        raise HTTPException(status_code=400, detail="u0 must contain only finite numbers")
    return predictor.predict(req.u0)
