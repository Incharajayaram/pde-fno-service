"""FastAPI backend for the FNO PDE operators (Burgers 1D, Darcy Flow 2D).

Serves:
  GET  /                  Burgers demo frontend
  GET  /api/health        Burgers liveness check
  GET  /api/presets       Burgers synthetic + real-sample presets
  POST /api/predict       Burgers: {"u0": [floats]} -> predicted final solution

  GET  /darcy             Darcy demo frontend
  GET  /api/darcy/health  Darcy liveness check
  GET  /api/darcy/presets Darcy synthetic + real-sample presets
  POST /api/darcy/predict Darcy: {"nu": [[floats]]} -> predicted pressure field

Only the small deployment checkpoints (and small bundles of real test
samples) are loaded at startup -- not the full datasets, not h5py.
"""
from __future__ import annotations

import json
import logging
import os

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.infer import FNOPredictor
from src.infer_darcy import DarcyPredictor

logger = logging.getLogger("uvicorn.error")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(APP_DIR, "model", "fno_burgers_final.pt")
CHECKPOINT_PATH = os.environ.get("MODEL_CHECKPOINT_PATH", DEFAULT_CKPT)
REAL_SAMPLES_PATH = os.path.join(APP_DIR, "model", "real_samples.json")

DARCY_DEFAULT_CKPT = os.path.join(APP_DIR, "model", "fno2d_darcy_final.pt")
DARCY_CHECKPOINT_PATH = os.environ.get("DARCY_MODEL_CHECKPOINT_PATH", DARCY_DEFAULT_CKPT)
DARCY_REAL_SAMPLES_PATH = os.path.join(APP_DIR, "model", "darcy_real_samples.json")

app = FastAPI(title="PDE Solutions as a Service (FNO: Burgers + Darcy)")
predictor: FNOPredictor | None = None
real_samples: dict | None = None
darcy_predictor: DarcyPredictor | None = None
darcy_real_samples: dict | None = None


@app.on_event("startup")
def load_model() -> None:
    global predictor, real_samples, darcy_predictor, darcy_real_samples
    predictor = FNOPredictor(CHECKPOINT_PATH)
    with open(REAL_SAMPLES_PATH) as f:
        real_samples = json.load(f)

    # Darcy loads gracefully: while its training run is still in progress
    # (or its checkpoint hasn't been deployed yet), the Burgers demo should
    # keep working rather than failing the whole app's startup.
    try:
        darcy_predictor = DarcyPredictor(DARCY_CHECKPOINT_PATH)
        with open(DARCY_REAL_SAMPLES_PATH) as f:
            darcy_real_samples = json.load(f)
    except FileNotFoundError:
        logger.warning("Darcy checkpoint not found at %s -- /api/darcy/* will return 503", DARCY_CHECKPOINT_PATH)


class PredictRequest(BaseModel):
    u0: list[float] = Field(..., min_length=8, description="Initial condition values on a uniform grid over [0, 1)")


class DarcyPredictRequest(BaseModel):
    nu: list[list[float]] = Field(..., min_length=4, description="Permeability field, a square 2D grid of at least 4x4")


@app.get("/")
def index():
    return FileResponse(os.path.join(APP_DIR, "static", "index.html"))


@app.get("/darcy")
def darcy_index():
    return FileResponse(os.path.join(APP_DIR, "static", "darcy.html"))


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


@app.get("/api/darcy/health")
def darcy_health():
    return {"status": "ok", "model_loaded": darcy_predictor is not None}


def _require_darcy() -> DarcyPredictor:
    if darcy_predictor is None:
        raise HTTPException(status_code=503, detail="Darcy model not loaded yet")
    return darcy_predictor


def _darcy_synthetic_presets(resolution: int) -> dict:
    res = resolution
    xx, yy = np.meshgrid(np.arange(res), np.arange(res), indexing="ij")
    lo, hi = 0.1, 1.0  # same two permeability values the training data uses

    checkerboard = np.where(((xx // (res // 4)) + (yy // (res // 4))) % 2 == 0, hi, lo)

    cx, cy, r = res / 2, res / 2, res / 4
    circular_inclusion = np.where((xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2, lo, hi)

    stripes = np.where((xx // (res // 6)) % 2 == 0, hi, lo)

    rng = np.random.default_rng(7)
    block_size = res // 8
    blocks = rng.choice([lo, hi], size=(8, 8))
    random_blocks = np.kron(blocks, np.ones((block_size, block_size)))
    random_blocks = random_blocks[:res, :res]
    if random_blocks.shape != (res, res):  # pad if res isn't a clean multiple of 8
        random_blocks = np.pad(
            random_blocks,
            ((0, res - random_blocks.shape[0]), (0, res - random_blocks.shape[1])),
            mode="edge",
        )

    return {
        "checkerboard": checkerboard.astype(np.float32).tolist(),
        "circular_inclusion": circular_inclusion.astype(np.float32).tolist(),
        "stripes": stripes.astype(np.float32).tolist(),
        "random_blocks": random_blocks.astype(np.float32).tolist(),
    }


@app.get("/api/darcy/presets")
def darcy_presets():
    pred = _require_darcy()
    assert darcy_real_samples is not None
    return {
        "resolution": pred.resolution,
        "synthetic": _darcy_synthetic_presets(pred.resolution),
        "real_samples": [
            {"label": f"Real PDEBench sample #{i + 1}", "nu": s["nu"], "true_pressure": s["true_pressure"]}
            for i, s in enumerate(darcy_real_samples["samples"])
        ],
    }


@app.post("/api/darcy/predict")
def darcy_predict(req: DarcyPredictRequest):
    pred = _require_darcy()
    arr = np.asarray(req.nu, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise HTTPException(status_code=400, detail="nu must be a square 2D grid")
    if not np.all(np.isfinite(arr)):
        raise HTTPException(status_code=400, detail="nu must contain only finite numbers")
    return pred.predict(req.nu)
