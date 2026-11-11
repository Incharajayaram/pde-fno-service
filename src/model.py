"""Fourier Neural Operator (1D) built from scratch, following Li et al. 2020
("Fourier Neural Operator for Parametric PDEs"): 4 spectral-conv layers,
each paired with a pointwise (1x1 conv) skip path, GELU activations,
Adam/lr=1e-3/relative-L2 loss are used in train.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """Learns a convolution by truncating to the lowest `modes` Fourier
    modes, multiplying by a learned complex weight per mode, and
    transforming back. This is the "global" operator-learning piece of
    the FNO: a conv1d in Fourier space is a pointwise complex multiply.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = 1.0 / (in_channels * out_channels)
        # Real/imaginary parts stored separately (real-valued params only).
        self.weight_real = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes))
        self.weight_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, length)
        batch, _, length = x.shape
        x_ft = torch.fft.rfft(x, n=length)  # (batch, in_channels, length//2 + 1), complex

        modes = min(self.modes, x_ft.shape[-1])
        weight = torch.complex(self.weight_real[..., :modes], self.weight_imag[..., :modes])

        out_ft = torch.zeros(batch, self.out_channels, x_ft.shape[-1],
                              dtype=torch.cfloat, device=x.device)
        # "bix,iox->box": contract the truncated input modes with the weight.
        out_ft[..., :modes] = torch.einsum("bix,iox->box", x_ft[..., :modes], weight)

        return torch.fft.irfft(out_ft, n=length)  # (batch, out_channels, length)


class FourierLayer(nn.Module):
    """One Fourier layer: spectral conv (global) + 1x1 conv (local/pointwise)
    skip connection, summed, then an activation.
    """

    def __init__(self, width: int, modes: int, activation: bool = True):
        super().__init__()
        self.spectral_conv = SpectralConv1d(width, width, modes)
        self.pointwise_conv = nn.Conv1d(width, width, kernel_size=1)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.spectral_conv(x) + self.pointwise_conv(x)
        if self.activation:
            out = F.gelu(out)
        return out


class FNO1d(nn.Module):
    """Input: (batch, length, 2) = [u0(x), x]. Output: (batch, length, 1) = u(x, T).

    Pipeline: lift (2 -> width) -> 4x FourierLayer -> project (width -> 1).
    """

    def __init__(self, modes: int = 16, width: int = 64, in_channels: int = 2,
                 out_channels: int = 1, n_layers: int = 4):
        super().__init__()
        self.lift = nn.Linear(in_channels, width)
        self.fourier_layers = nn.ModuleList([
            FourierLayer(width, modes, activation=(i < n_layers - 1))
            for i in range(n_layers)
        ])
        self.project1 = nn.Linear(width, 128)
        self.project2 = nn.Linear(128, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, length, in_channels)
        x = self.lift(x)             # (batch, length, width)
        x = x.permute(0, 2, 1)       # (batch, width, length) for conv1d/fft along length
        for layer in self.fourier_layers:
            x = layer(x)
        x = x.permute(0, 2, 1)       # (batch, length, width)
        x = F.gelu(self.project1(x))
        x = self.project2(x)         # (batch, length, out_channels)
        return x


if __name__ == "__main__":
    model = FNO1d(modes=16, width=64)
    n_params = sum(p.numel() for p in model.parameters())
    dummy = torch.randn(8, 256, 2)
    out = model(dummy)
    print("output shape:", out.shape)
    print("param count:", n_params)
