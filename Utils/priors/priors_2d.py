"""2D latent noise priors.

Mirrors ``priors_1d.py`` for rank-2 grids.  All decoders are linear and
differentiable; analytical scales are derived from the stationary output
variance of the decoder applied to ``z ~ N(0, I)``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import LatentNoisePrior
from .priors_1d import _bspline_basis_1d


class White2DPrior(LatentNoisePrior):
    """``eta = std * z`` on a 2D grid."""

    def __init__(self, shape: tuple[int, int], std: float,
                 device=None, dtype: torch.dtype = torch.float32):
        super().__init__(device=device, dtype=dtype)
        self.latent_shape = tuple(shape)
        self.output_shape = tuple(shape)
        self.std = float(std)
        self.register_buffer("_anchor", torch.zeros(1, device=device, dtype=dtype))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.std * z

    def marginal_std(self) -> torch.Tensor:
        return torch.full(self.output_shape, self.std,
                          device=self.device, dtype=self.dtype)


class Spatial2DPrior(LatentNoisePrior):
    """Separable Gaussian-smoothed 2D white noise.

    Uses an outer-product 2D Gaussian kernel.  Stationary output variance is
    ``sum(kernel2d**2)`` for ``z ~ N(0, I)``.
    """

    def __init__(self, shape: tuple[int, int], correlation_length: float,
                 std: float, device=None, dtype: torch.dtype = torch.float32):
        super().__init__(device=device, dtype=dtype)
        self.latent_shape = tuple(shape)
        self.output_shape = tuple(shape)
        self.correlation_length = float(correlation_length)

        kernel_size = max(3, int(4 * correlation_length) // 2 * 2 + 1)
        sigma = correlation_length / 2.0
        x = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
        g = torch.exp(-x ** 2 / (2 * sigma ** 2))
        g = g / g.sum()
        kernel2d = torch.outer(g, g)
        self.register_buffer(
            "_kernel", kernel2d.view(1, 1, kernel_size, kernel_size)
        )
        self._padding = kernel_size // 2

        norm = torch.sqrt(torch.sum(kernel2d ** 2)).item()
        self.scale = float(std) / norm if norm > 1e-12 else float(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = z.unsqueeze(1)
        y = F.conv2d(x, self._kernel, padding=self._padding)
        return y.squeeze(1) * self.scale

    def marginal_std(self) -> torch.Tensor:
        sig = self.scale * torch.linalg.vector_norm(self._kernel)
        return torch.full(self.output_shape, sig.item(),
                          device=self.device, dtype=self.dtype)


class BSpline2DPrior(LatentNoisePrior):
    """Tensor-product cubic B-spline prior on a 2D grid.

    ``latent_shape == (Kt, Kx)`` control points;
    ``eta = Phi_t @ c @ Phi_x^T`` evaluated at the ``(H, W)`` grid.
    """

    def __init__(self, shape: tuple[int, int],
                 n_knots: tuple[int, int] | int,
                 std: float, device=None, dtype: torch.dtype = torch.float32):
        super().__init__(device=device, dtype=dtype)
        H, W = shape
        if isinstance(n_knots, int):
            Kt, Kx = n_knots, n_knots
        else:
            Kt, Kx = n_knots
        self.latent_shape = (Kt, Kx)
        self.output_shape = (H, W)

        Phi_t = _bspline_basis_1d(H, Kt, device, dtype)              # [H, Kt]
        Phi_x = _bspline_basis_1d(W, Kx, device, dtype)              # [W, Kx]
        self.register_buffer("_Phi_t", Phi_t)
        self.register_buffer("_Phi_x", Phi_x)

        row_t = (Phi_t ** 2).sum(dim=1)
        row_x = (Phi_x ** 2).sum(dim=1)
        mean_var = (row_t.mean() * row_x.mean()).item()
        mean_sigma = mean_var ** 0.5
        self.scale = float(std) / mean_sigma if mean_sigma > 1e-12 else float(std)

    def decode(self, c: torch.Tensor) -> torch.Tensor:
        out = torch.einsum("ik,bkl,jl->bij", self._Phi_t, c, self._Phi_x)
        return out * self.scale

    def marginal_std(self) -> torch.Tensor:
        row_t = torch.sum(self._Phi_t ** 2, dim=1)                   # [H]
        row_x = torch.sum(self._Phi_x ** 2, dim=1)                   # [W]
        var = torch.outer(row_t, row_x)                              # [H, W]
        return self.scale * torch.sqrt(var)


class Spectral2DPrior(LatentNoisePrior):
    """2D colored noise via radial spectral shaping ``|k|^(-alpha/2)``."""

    def __init__(self, shape: tuple[int, int], alpha: float, std: float,
                 device=None, dtype: torch.dtype = torch.float32):
        super().__init__(device=device, dtype=dtype)
        H, W = shape
        self.latent_shape = (H, W)
        self.output_shape = (H, W)
        self.alpha = float(alpha)

        fh = torch.fft.fftfreq(H, device=device, dtype=dtype)
        fw = torch.fft.fftfreq(W, device=device, dtype=dtype)
        kh, kw = torch.meshgrid(fh, fw, indexing="ij")
        mag = torch.sqrt(kh ** 2 + kw ** 2)
        mag[0, 0] = 1.0
        mask = mag ** (-self.alpha / 2.0)
        mask[0, 0] = 0.0
        self.register_buffer("_mask", mask)

        var = torch.mean(mask ** 2).item()
        self.scale = float(std) / (var ** 0.5) if var > 1e-24 else float(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        Z = torch.fft.fft2(z, dim=(-2, -1))
        Y = self._mask * Z
        eta = torch.fft.ifft2(Y, dim=(-2, -1)).real
        return eta * self.scale

    def marginal_std(self) -> torch.Tensor:
        sig = self.scale * torch.sqrt(torch.mean(self._mask ** 2))
        return torch.full(self.output_shape, sig.item(),
                          device=self.device, dtype=self.dtype)
