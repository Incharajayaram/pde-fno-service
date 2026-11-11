"""FastAPI backend for the FNO Burgers-equation operator.

Serves:
  GET  /                  the demo frontend (single static HTML page)
  GET  /api/health        liveness check
  GET  /api/presets       a few synthetic example initial conditions
  POST /api/predict       {"u0": [floats]} -> predicted final solution

Only the small deployment checkpoint is loaded at startup -- not the
dataset, not h5py.
"""
from __future__ import annotations

import os

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.infer import FNOPredictor

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(APP_DIR, "model", "fno_burgers_final.pt")
CHECKPOINT_PATH = os.environ.get("MODEL_CHECKPOINT_PATH", DEFAULT_CKPT)

app = FastAPI(title="PDE Solutions as a Service (FNO / Burgers)")
predictor: FNOPredictor | None = None


@app.on_event("startup")
def load_model() -> None:
    global predictor
    predictor = FNOPredictor(CHECKPOINT_PATH)


class PredictRequest(BaseModel):
    u0: list[float] = Field(..., min_length=8, description="Initial condition values on a uniform grid over [0, 1)")


@app.get("/")
def index():
    return FileResponse(os.path.join(APP_DIR, "static", "index.html"))


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.get("/api/presets")
def presets():
    assert predictor is not None
    x = predictor.grid
    return {
        "x": x.tolist(),
        "presets": {
            "sine": (0.8 * np.sin(2 * np.pi * x)).tolist(),
            "step": np.where((x > 0.25) & (x < 0.75), 0.8, -0.2).tolist(),
            "gaussian_bump": (0.9 * np.exp(-((x - 0.5) ** 2) / (2 * 0.05 ** 2))).tolist(),
        },
    }


@app.post("/api/predict")
def predict(req: PredictRequest):
    assert predictor is not None
    if not np.all(np.isfinite(req.u0)):
        raise HTTPException(status_code=400, detail="u0 must contain only finite numbers")
    return predictor.predict(req.u0)
