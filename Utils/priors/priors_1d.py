"""1D latent noise priors.

Each prior is a differentiable ``nn.Module`` with kernels / bases registered
as buffers.  The decoder is a linear map ``eta = M z`` whose analytical scale
is computed once at construction so that ``Var(eta_i)`` equals the requested
``std^2`` under ``z ~ N(0, I)``.  No per-batch empirical renormalisation is
performed anywhere — gradients flow cleanly through ``decode``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .base import LatentNoisePrior


# ----------------------------------------------------------------------
# White
# ----------------------------------------------------------------------
class WhitePrior(LatentNoisePrior):
    """``eta = std * z`` — grid space is the latent space."""

    def __init__(self, n_points: int, std: float, device=None,
                 dtype: torch.dtype = torch.float32):
        super().__init__(device=device, dtype=dtype)
        self.latent_shape = (n_points,)
        self.output_shape = (n_points,)
        self.std = float(std)
        # Register a zero buffer purely so .to(device) tracks the module.
        self.register_buffer(
            "_anchor", torch.zeros(1, device=device, dtype=dtype)
        )

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.std * z

    def marginal_std(self) -> torch.Tensor:
        return torch.full((self.output_shape[0],), self.std,
                          device=self.device, dtype=self.dtype)


# ----------------------------------------------------------------------
# Convolution-based (Spatial Gaussian kernel; PRE-correlated)
# ----------------------------------------------------------------------
def _gaussian_kernel_1d(correlation_length: float, device, dtype) -> torch.Tensor:
    kernel_size = max(3, int(4 * correlation_length) // 2 * 2 + 1)
    sigma = correlation_length / 2.0
    x = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    k = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return k / k.sum()


class _Conv1DPrior(LatentNoisePrior):
    """Internal helper: ``eta = scale * conv1d(z, kernel)``.

    Analytical scale: for ``z ~ N(0, I)`` and stationary output far from the
    boundary, ``Var(eta) = sum(kernel**2)``.  ``scale = std / sqrt(sum k**2)``
    makes the marginal std equal ``std`` in the interior.
    """

    def __init__(self, n_points: int, kernel: torch.Tensor, std: float,
                 device=None, dtype: torch.dtype = torch.float32):
        super().__init__(device=device, dtype=dtype)
        self.latent_shape = (n_points,)
        self.output_shape = (n_points,)

        k = kernel.to(device=device, dtype=dtype).view(1, 1, -1)
        self.register_buffer("_kernel", k)
        self._padding = k.shape[-1] // 2

        norm = torch.sqrt(torch.sum(k ** 2)).item()
        self.scale = float(std) / norm if norm > 1e-12 else float(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = z.unsqueeze(1)                              # [B, 1, N]
        y = F.conv1d(x, self._kernel, padding=self._padding)
        return y.squeeze(1) * self.scale

    def marginal_std(self) -> torch.Tensor:
        # Exact: Var(eta_i) = scale^2 * sum_j k[j]^2 on the interior. At the
        # boundary, convolution with zero-padding reduces variance. We return
        # the interior value, which is what the `std` knob targets.
        std = self.scale * torch.linalg.vector_norm(self._kernel)
        return torch.full((self.output_shape[0],), std.item(),
                          device=self.device, dtype=self.dtype)


class SpatialPrior(_Conv1DPrior):
    """Gaussian-smoothed white noise: ``eta = scale * (G_L * z)``."""

    def __init__(self, n_points: int, correlation_length: float, std: float,
                 device=None, dtype: torch.dtype = torch.float32):
        kernel = _gaussian_kernel_1d(correlation_length, device=device, dtype=dtype)
        super().__init__(n_points, kernel, std, device=device, dtype=dtype)
        self.correlation_length = float(correlation_length)


class PreCorrelatedPrior(_Conv1DPrior):
    """Convolution prior using a user-supplied (e.g. additive PRE) kernel."""

    def __init__(self, n_points: int, kernel: torch.Tensor, std: float,
                 device=None, dtype: torch.dtype = torch.float32):
        if kernel.ndim != 1:
            kernel = kernel.reshape(-1)
        super().__init__(n_points, kernel, std, device=device, dtype=dtype)


# ----------------------------------------------------------------------
# GP (exact Cholesky)
# ----------------------------------------------------------------------
class GPPrior(LatentNoisePrior):
    """``eta = L z`` with ``L = chol(K + eps I)``, K chosen from gpytorch."""

    def __init__(self, n_points: int, correlation_length: float, std: float,
                 kernel_type: str = "rbf", nu: float = 1.5,
                 device=None, dtype: torch.dtype = torch.float32):
        super().__init__(device=device, dtype=dtype)
        self.latent_shape = (n_points,)
        self.output_shape = (n_points,)
        self.correlation_length = float(correlation_length)
        self.kernel_type = kernel_type.lower()
        self.nu = float(nu)

        try:
            import gpytorch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "GPPrior requires gpytorch. Install it or use SpatialPrior."
            ) from exc

        x_loc = torch.linspace(0, 1, n_points, device=device, dtype=dtype)
        if self.kernel_type == "rbf":
            covar = gpytorch.kernels.RBFKernel()
        elif self.kernel_type == "matern":
            covar = gpytorch.kernels.MaternKernel(nu=nu)
        elif self.kernel_type == "periodic":
            covar = gpytorch.kernels.PeriodicKernel()
        else:
            raise ValueError(f"Unknown GP kernel: {kernel_type!r}")

        covar.lengthscale = correlation_length / n_points
        scaled = gpytorch.kernels.ScaleKernel(covar)
        scaled.outputscale = float(std) ** 2

        with torch.no_grad():
            K = scaled(x_loc, x_loc).evaluate()
        K = K + 1e-6 * torch.eye(n_points, device=device, dtype=dtype)
        try:
            L = torch.linalg.cholesky(K)
        except RuntimeError:
            K = K + 1e-4 * torch.eye(n_points, device=device, dtype=dtype)
            L = torch.linalg.cholesky(K)
        self.register_buffer("_L", L)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self._L.T

    def marginal_std(self) -> torch.Tensor:
        # Diagonal of K = diag(L L^T) = sum(L_i^2).
        return torch.sqrt(torch.sum(self._L ** 2, dim=1))


# ----------------------------------------------------------------------
# Cubic B-spline (low-rank, C2-continuous)
# ----------------------------------------------------------------------
def _cubic_bspline_kernel(t: torch.Tensor) -> torch.Tensor:
    abs_t = torch.abs(t)
    result = torch.zeros_like(t)
    m1 = abs_t < 1
    a1 = abs_t[m1]
    result[m1] = (2.0 / 3.0) - a1 ** 2 + 0.5 * a1 ** 3
    m2 = (abs_t >= 1) & (abs_t < 2)
    a2 = abs_t[m2]
    result[m2] = (1.0 / 6.0) * (2.0 - a2) ** 3
    return result


def _bspline_basis_1d(n_points: int, n_knots: int, device, dtype) -> torch.Tensor:
    eval_pts = torch.linspace(0.0, 1.0, n_points, device=device, dtype=dtype)
    knots = torch.linspace(0.0, 1.0, n_knots, device=device, dtype=dtype)
    dk = knots[1] - knots[0]
    t = (eval_pts.unsqueeze(1) - knots.unsqueeze(0)) / dk
    return _cubic_bspline_kernel(t)                                  # [N, K]


class BSplinePrior(LatentNoisePrior):
    """``eta = scale * Phi c``; ``c`` are K << N B-spline control points."""

    def __init__(self, n_points: int, n_knots: int, std: float,
                 device=None, dtype: torch.dtype = torch.float32):
        super().__init__(device=device, dtype=dtype)
        self.latent_shape = (n_knots,)
        self.output_shape = (n_points,)
        self.n_knots = int(n_knots)

        Phi = _bspline_basis_1d(n_points, n_knots, device, dtype)    # [N, K]
        self.register_buffer("_Phi", Phi)

        row_var = (Phi ** 2).sum(dim=1)                              # [N]
        mean_sigma = torch.sqrt(row_var.mean()).item()
        self.scale = float(std) / mean_sigma if mean_sigma > 1e-12 else float(std)

    def decode(self, c: torch.Tensor) -> torch.Tensor:
        return (c @ self._Phi.T) * self.scale

    def marginal_std(self) -> torch.Tensor:
        # Var(eta_i) = scale^2 * sum_k Phi[i,k]^2.
        return self.scale * torch.sqrt(torch.sum(self._Phi ** 2, dim=1))


# ----------------------------------------------------------------------
# Spectral (1/f^alpha)
# ----------------------------------------------------------------------
class SpectralPrior(LatentNoisePrior):
    """Colored noise via spectral shaping: ``eta = ifft(mask * fft(z)) * scale``.

    Mask is ``|k|^(-alpha/2)`` so the power spectrum is ``S(k) ~ |k|^(-alpha)``.
    Sampling is differentiable in ``z`` because the FFTs and the fixed mask
    are linear; no per-batch std normalisation is used.
    """

    def __init__(self, n_points: int, alpha: float, std: float,
                 device=None, dtype: torch.dtype = torch.float32):
        super().__init__(device=device, dtype=dtype)
        self.latent_shape = (n_points,)
        self.output_shape = (n_points,)
        self.alpha = float(alpha)

        freqs = torch.fft.fftfreq(n_points, device=device, dtype=dtype)
        mag = torch.abs(freqs)
        mag[0] = 1.0  # protect DC from /0; we zero it below
        mask = mag ** (-self.alpha / 2.0)
        mask[0] = 0.0  # remove mean
        self.register_buffer("_mask", mask)

        # Analytical output variance: E[|eta|^2] = (1/N) sum |mask|^2 for z ~ N(0,I).
        var = torch.mean(mask ** 2).item()
        self.scale = float(std) / (var ** 0.5) if var > 1e-24 else float(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        Z = torch.fft.fft(z, dim=-1)
        Y = self._mask * Z
        eta = torch.fft.ifft(Y, dim=-1).real
        return eta * self.scale

    def marginal_std(self) -> torch.Tensor:
        sig = self.scale * torch.sqrt(torch.mean(self._mask ** 2))
        return torch.full((self.output_shape[0],), sig.item(),
                          device=self.device, dtype=self.dtype)


# ----------------------------------------------------------------------
# Ornstein-Uhlenbeck temporal prior (exact AR(1))
# ----------------------------------------------------------------------
class OUPrior(LatentNoisePrior):
    """Exact OU process ``dX = -X/tau dt + sigma dW`` as a linear decoder.

    Latent is a sequence of iid ``N(0, I)`` innovations.  Stationary marginal
    std equals ``std``.  Decoder is differentiable (no Python RNG loop): the
    AR(1) recursion is expanded into a lower-triangular matrix times ``z``.
    """

    def __init__(self, n_steps: int, dt: float, tau: float, std: float,
                 device=None, dtype: torch.dtype = torch.float32):
        super().__init__(device=device, dtype=dtype)
        self.latent_shape = (n_steps,)
        self.output_shape = (n_steps,)
        self.dt = float(dt)
        self.tau = float(tau)

        theta = 1.0 / tau
        rho = float(torch.exp(torch.tensor(-theta * dt)).item())
        # AR(1): x_t = rho x_{t-1} + sigma_inc * z_t with
        # sigma_inc chosen so stationary Var(x) = std^2.
        sigma_inc = float(std) * (1.0 - rho ** 2) ** 0.5

        # Build lower-triangular mixing matrix M where x = M z, x_0 = std*z_0
        # so the chain is stationary from t=0.
        idx = torch.arange(n_steps, device=device, dtype=dtype)
        M = torch.zeros(n_steps, n_steps, device=device, dtype=dtype)
        M[0, 0] = float(std)
        for t in range(1, n_steps):
            # x_t = rho^t * x_0 + sum_{s=1..t} rho^{t-s} * sigma_inc * z_s
            M[t, 0] = (rho ** t) * float(std)
            M[t, 1:t + 1] = sigma_inc * (rho ** (t - torch.arange(1, t + 1, device=device, dtype=dtype)))
        self.register_buffer("_M", M)
        _ = idx  # silence unused warning for static analysers

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self._M.T

    def marginal_std(self) -> torch.Tensor:
        return torch.sqrt(torch.sum(self._M ** 2, dim=1))
