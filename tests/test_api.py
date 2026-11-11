"""FastAPI endpoint + inference tests, covering every route in app/main.py.

Uses the real, small, committed checkpoints (app/model/*.pt) -- not the raw
PDEBench datasets -- so this suite runs anywhere torch is installed, with
no dataset download required. See tests/conftest.py for the shared client
fixture (loads the app once per test session).
"""
from __future__ import annotations

import json

BURGERS_SYNTHETIC_PRESET_NAMES = {
    "sine_low (k=1)", "sine_high (k=4)", "step", "double_step", "square_wave",
    "triangle", "gaussian_bump", "gaussian_wide", "two_bumps", "random_smooth",
}
DARCY_SYNTHETIC_PRESET_NAMES = {
    "checkerboard", "circular_inclusion", "stripes", "random_blocks",
    "offcenter_inclusion", "l_shape", "diagonal_bands", "sparse_holes",
}


# ---- pages ----

def test_index_page_serves_html(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "PDE Solutions as a Service" in res.text


def test_darcy_page_serves_html(client):
    res = client.get("/darcy")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "Darcy Flow" in res.text


# ---- health ----

def test_burgers_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "model_loaded": True}


def test_darcy_health(client):
    res = client.get("/api/darcy/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "model_loaded": True}


# ---- presets ----

def test_burgers_presets(client):
    res = client.get("/api/presets")
    assert res.status_code == 200
    data = res.json()
    assert set(data["synthetic"].keys()) == BURGERS_SYNTHETIC_PRESET_NAMES
    assert len(data["real_samples"]) == 8
    for sample in data["real_samples"]:
        assert len(sample["u0"]) == len(sample["true_uT"]) == len(data["x"])


def test_darcy_presets(client):
    res = client.get("/api/darcy/presets")
    assert res.status_code == 200
    data = res.json()
    assert data["resolution"] == 128
    assert set(data["synthetic"].keys()) == DARCY_SYNTHETIC_PRESET_NAMES
    assert len(data["real_samples"]) == 8
    for name, field in data["synthetic"].items():
        assert len(field) == 128 and len(field[0]) == 128, name


# ---- Burgers inference ----

def test_burgers_predict_valid_input(client):
    u0 = [0.5 * (i % 7) for i in range(32)]  # any length >= 8 is valid, resampled internally
    res = client.post("/api/predict", json={"u0": u0})
    assert res.status_code == 200
    data = res.json()
    assert set(data.keys()) == {"x", "u0_resampled", "prediction"}
    assert len(data["x"]) == len(data["u0_resampled"]) == len(data["prediction"]) == 256  # training resolution


def test_burgers_predict_rejects_too_short_input(client):
    res = client.post("/api/predict", json={"u0": [0.1, 0.2, 0.3]})  # below min_length=8
    assert res.status_code == 422


def test_burgers_predict_rejects_non_finite_input(client):
    # A syntactically valid JSON number of extreme magnitude parses to +inf;
    # this exercises the server's explicit np.isfinite guard (not pydantic's
    # own validation), since JSON itself has no "NaN"/"Infinity" literals.
    body = '{"u0": [1e400, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]}'
    res = client.post("/api/predict", content=body, headers={"content-type": "application/json"})
    assert res.status_code == 400


# ---- Darcy inference ----

def test_darcy_predict_valid_input(client):
    presets = client.get("/api/darcy/presets").json()
    nu = presets["synthetic"]["checkerboard"]
    res = client.post("/api/darcy/predict", json={"nu": nu})
    assert res.status_code == 200
    data = res.json()
    assert set(data.keys()) == {"resolution", "nu_resampled", "pressure"}
    assert data["resolution"] == 128
    assert len(data["pressure"]) == 128 and len(data["pressure"][0]) == 128


def test_darcy_predict_rejects_non_square_input(client):
    nu = [[0.1, 0.2, 0.3, 0.4, 0.5]] * 4  # 4 rows x 5 cols -- not square
    res = client.post("/api/darcy/predict", json={"nu": nu})
    assert res.status_code == 400


def test_darcy_predict_rejects_too_small_input(client):
    nu = [[0.1, 0.2, 0.3]] * 3  # below min_length=4 rows
    res = client.post("/api/darcy/predict", json={"nu": nu})
    assert res.status_code == 422
