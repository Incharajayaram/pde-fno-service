"""Forward-pass mechanism and output-shape tests for FNO1d (Burgers).

These never touch the raw PDEBench HDF5 data or a checkpoint -- they
exercise the architecture itself (src/model.py) with fresh random weights,
so they run anywhere torch is installed, no dataset download required.
"""
from __future__ import annotations

import torch

from src.model import FNO1d


def make_model() -> FNO1d:
    return FNO1d(modes=16, width=64)  # matches the deployed checkpoint's hyperparams


def test_param_count():
    model = make_model()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 549_569


def test_forward_pass_output_shape_matches_input_length():
    model = make_model()
    model.eval()
    for length in (64, 128, 256, 512):
        for batch in (1, 4):
            x = torch.randn(batch, length, 2)
            out = model(x)
            assert out.shape == (batch, length, 1)


def test_forward_pass_is_finite():
    model = make_model()
    model.eval()
    x = torch.randn(4, 256, 2)
    out = model(x)
    assert torch.isfinite(out).all()


def test_eval_mode_is_deterministic():
    model = make_model()
    model.eval()
    x = torch.randn(2, 256, 2)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.equal(out1, out2)


def test_gradients_flow_to_all_parameters():
    model = make_model()
    model.train()
    x = torch.randn(4, 256, 2)
    target = torch.randn(4, 256, 1)
    out = model(x)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient at {name}"


def test_spectral_conv_clamps_modes_at_low_resolution():
    # At very low resolution the number of rfft frequency bins can drop
    # below `modes`; SpectralConv1d must clamp gracefully rather than error
    # (see `modes = min(self.modes, x_ft.shape[-1])` in src/model.py).
    model = FNO1d(modes=16, width=8, n_layers=1)
    model.eval()
    x = torch.randn(1, 8, 2)  # rfft bins = 5 < modes=16
    out = model(x)
    assert out.shape == (1, 8, 1)
    assert torch.isfinite(out).all()
