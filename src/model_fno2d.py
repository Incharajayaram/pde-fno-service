"""2D Fourier Neural Operator, for steady-state 2D operators (e.g. Darcy
flow: permeability field -> pressure field). Mirrors model.py's FNO1d,
extended to 2D spectral convolutions.

Kept as a separate module from model.py (FNO1d) rather than a shared
dimensionality-generic base class -- with only two PDEs so far, the
extra abstraction isn't earning its keep yet (see CLAUDE conventions:
no premature generalization). Revisit if/when a third spatial
dimensionality is added.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """2D analogue of SpectralConv1d. rfft2 only halves the last axis, so
    the low frequencies along the first axis appear in two separate
    corners of the spectrum (the first `modes1` rows and the last
    `modes1` rows) -- hence two independent weight tensors, one per
    corner, instead of one.
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # modes kept along the height axis (both signs)
        self.modes2 = modes2  # modes kept along the width axis (one-sided, rfft)

        scale = 1.0 / (in_channels * out_channels)
        shape = (in_channels, out_channels, modes1, modes2)
        self.weight1_real = nn.Parameter(scale * torch.randn(*shape))
        self.weight1_imag = nn.Parameter(scale * torch.randn(*shape))
        self.weight2_real = nn.Parameter(scale * torch.randn(*shape))
        self.weight2_imag = nn.Parameter(scale * torch.randn(*shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, H, W)
        batch, _, h, w = x.shape
        x_ft = torch.fft.rfft2(x, s=(h, w))  # (batch, in_channels, H, W//2 + 1)

        m1 = min(self.modes1, h // 2)
        m2 = min(self.modes2, x_ft.shape[-1])

        weight1 = torch.complex(self.weight1_real[..., :m1, :m2], self.weight1_imag[..., :m1, :m2])
        weight2 = torch.complex(self.weight2_real[..., :m1, :m2], self.weight2_imag[..., :m1, :m2])

        out_ft = torch.zeros(batch, self.out_channels, h, x_ft.shape[-1], dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :m1, :m2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], weight1)
        out_ft[:, :, -m1:, :m2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -m1:, :m2], weight2)

        return torch.fft.irfft2(out_ft, s=(h, w))


class FourierLayer2d(nn.Module):
    def __init__(self, width: int, modes1: int, modes2: int, activation: bool = True):
        super().__init__()
        self.spectral_conv = SpectralConv2d(width, width, modes1, modes2)
        self.pointwise_conv = nn.Conv2d(width, width, kernel_size=1)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.spectral_conv(x) + self.pointwise_conv(x)
        if self.activation:
            out = F.gelu(out)
        return out


class FNO2d(nn.Module):
    """Input: (batch, H, W, 3) = [a(x,y), x, y]. Output: (batch, H, W, 1).

    Pipeline: lift (3 -> width) -> 4x FourierLayer2d -> project (width -> 1).
    """

    def __init__(self, modes1: int = 12, modes2: int = 12, width: int = 32,
                 in_channels: int = 3, out_channels: int = 1, n_layers: int = 4):
        super().__init__()
        self.lift = nn.Linear(in_channels, width)
        self.fourier_layers = nn.ModuleList([
            FourierLayer2d(width, modes1, modes2, activation=(i < n_layers - 1))
            for i in range(n_layers)
        ])
        self.project1 = nn.Linear(width, 128)
        self.project2 = nn.Linear(128, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, H, W, in_channels)
        x = self.lift(x)                 # (batch, H, W, width)
        x = x.permute(0, 3, 1, 2)        # (batch, width, H, W) for conv2d/fft2 along H,W
        for layer in self.fourier_layers:
            x = layer(x)
        x = x.permute(0, 2, 3, 1)        # (batch, H, W, width)
        x = F.gelu(self.project1(x))
        x = self.project2(x)             # (batch, H, W, out_channels)
        return x


if __name__ == "__main__":
    model = FNO2d(modes1=12, modes2=12, width=32)
    n_params = sum(p.numel() for p in model.parameters())
    dummy = torch.randn(4, 85, 85, 3)
    out = model(dummy)
    print("output shape:", out.shape)
    print("param count:", n_params)
