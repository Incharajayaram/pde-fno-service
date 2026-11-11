# PDE Solutions as a Service

Fourier Neural Operators (FNO), built from scratch in PyTorch, trained on [PDEBench](https://github.com/pdebench/PDEBench) and served as a full-stack web application. Each PDE gets its own model, its own API routes, and its own frontend page — all behind one FastAPI app.

Built for the QUEST Lab (IISc) Full Stack AI Pre-doctoral application.

**Live demo:** https://pde-fno-service.onrender.com

## Demos

### 1. 1D Burgers' Equation — `/`
Predicts the full spatial solution `u(x, T)` from an initial condition `u(x, 0)`.

- **Dataset:** PDEBench 1D Burgers, ν = 0.01
- **Model:** `FNO1d` — 4 Fourier layers, GELU, 1×1 conv skip connections, 16 Fourier modes, width 64 (~550K params)
- **Preprocessing:** spatial resolution downsampled 1024 → 256; single-timestep operator map (no autoregressive rollout)
- **Result:** held-out test relative L2 = **0.0141**
- **Frontend:** 10 synthetic initial-condition presets (sine, step, Gaussian bump, etc.) plus 8 real PDEBench samples with ground-truth overlay; type a custom initial condition directly

### 2. 2D Darcy Flow — `/darcy`
Predicts the steady-state pressure field `p(x, y)` from a permeability field `ν(x, y)` — a genuinely steady-state (no time axis) elliptic PDE, unlike Burgers.

- **Dataset:** PDEBench 2D Darcy Flow, β = 1.0
- **Model:** `FNO2d` — 4 Fourier layers, GELU, 1×1 conv skip connections, 12×12 Fourier modes, width 32 (~2.37M params), built with its own `SpectralConv2d` (two weight tensors per layer, since `rfft2` only enforces conjugate symmetry along the last axis)
- **Resolution:** 128×128 grid
- **Result:** held-out test relative L2 = **0.0435**
- **Frontend:** canvas-rendered heatmaps (permeability input, predicted pressure, and — for real samples — ground-truth pressure on a shared color scale) with 4 synthetic presets (checkerboard, circular inclusion, stripes, random blocks) and 4 real PDEBench samples

Both models are trained with Adam (lr 1e-3), a relative-L2 loss, and a step LR scheduler with early stopping (patience-based, with a grace period after each LR decay).

## Architecture

```
src/
  data.py, data_darcy.py       # HDF5 loading, downsampling, train/val/test caching
  model.py, model_fno2d.py     # FNO1d / FNO2d built from scratch
  train.py, train_darcy.py     # training loops, relative-L2 loss, early stopping
  infer.py, infer_darcy.py     # lightweight inference wrappers (checkpoint-only, no dataset)
app/
  main.py                      # FastAPI app: serves both models' routes and pages
  model/                       # committed inference checkpoints (small, self-contained)
  static/index.html, darcy.html
Dockerfile                     # CPU-only torch, single image serving both demos
```

At runtime, the backend loads only the small checkpoint files in `app/model/` (a few MB each) — never the raw PDEBench HDF5 datasets, which stay local/gitignored and are only needed for training.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/` | Burgers frontend |
| GET | `/darcy` | Darcy Flow frontend |
| GET | `/api/health` | Burgers model status |
| GET | `/api/darcy/health` | Darcy model status |
| GET | `/api/presets` | Burgers synthetic + real sample presets |
| GET | `/api/darcy/presets` | Darcy synthetic + real sample presets |
| POST | `/api/predict` | `{"u0": [...]}` → predicted `u(x, T)` |
| POST | `/api/darcy/predict` | `{"nu": [[...]]}` → predicted pressure field |

If a checkpoint is missing at startup, that PDE's routes return `503` without taking down the rest of the app (verified independently for both models).

## Running locally

```bash
pip install -r requirements-dev.txt   # includes torch + h5py, for training
# or
pip install -r requirements.txt       # inference-only, matches the Docker image

uvicorn app.main:app --reload
```

Training (requires the raw PDEBench HDF5 files under `data/`, downloaded separately — see PDEBench's own tooling):

```bash
python -m src.train --epochs 500
python -m src.train_darcy --epochs 150
```

## Deployment

Single `Dockerfile` at the repo root (CPU-only PyTorch, ~184MB base), deployed on Render from this GitHub repo's `main` branch. Both models ship in the same image.

## Other PDEBench datasets considered

Explored as candidates for extending this pipeline further: Diffusion-Sorption (1D, 4GB, zero architecture change from Burgers), Shallow-Water (2D, 6.2GB), 2D Reaction-Diffusion (13GB). Advection/CFD variants and NS_Incom (2.3TB) were ruled out as infeasible on local storage.
