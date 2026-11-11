"""Forward-pass mechanism and output-shape tests for FNO2d (Darcy Flow).

These never touch the raw PDEBench HDF5 data or a checkpoint -- they
exercise the architecture itself (src/model_fno2d.py) with fresh random
weights, so they run anywhere torch is installed, no dataset download
required.
"""
from __future__ import annotations

import torch

from src.model_fno2d import FNO2d


def make_model() -> FNO2d:
    return FNO2d(modes1=12, modes2=12, width=32)  # matches the deployed checkpoint's hyperparams


def test_param_count():
    model = make_model()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 2_368_001


def test_forward_pass_output_shape_matches_input_grid():
    model = make_model()
    model.eval()
    # 32/64/128 mirror the resolutions exercised in src/eval_superres_darcy.py
    for res in (32, 64, 128):
        for batch in (1, 3):
            x = torch.randn(batch, res, res, 3)
            out = model(x)
            assert out.shape == (batch, res, res, 1)


def test_forward_pass_is_finite():
    model = make_model()
    model.eval()
    x = torch.randn(2, 64, 64, 3)
    out = model(x)
    assert torch.isfinite(out).all()


def test_eval_mode_is_deterministic():
    model = make_model()
    model.eval()
    x = torch.randn(2, 64, 64, 3)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.equal(out1, out2)


def test_gradients_flow_to_all_parameters():
    model = make_model()
    model.train()
    x = torch.randn(2, 32, 32, 3)
    target = torch.randn(2, 32, 32, 1)
    out = model(x)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient at {name}"


def test_spectral_conv2d_clamps_modes_at_low_resolution():
    # At very low resolution the number of available frequency bins (along
    # either axis) can drop below modes1/modes2; SpectralConv2d must clamp
    # gracefully rather than error (see the `m1`/`m2` clamps in
    # src/model_fno2d.py).
    model = FNO2d(modes1=12, modes2=12, width=8, n_layers=1)
    model.eval()
    x = torch.randn(1, 16, 16, 3)  # h//2=8 < modes1=12, w//2+1=9 < modes2=12
    out = model(x)
    assert out.shape == (1, 16, 16, 1)
    assert torch.isfinite(out).all()
