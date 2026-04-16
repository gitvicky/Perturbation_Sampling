"""Latent-space noise priors for perturbation sampling.

Each prior exposes three pieces:

- ``sample_latent(batch, seed)`` draws a latent tensor ``z`` from the base
  (usually standard-normal) distribution.
- ``decode(z)`` maps a latent to a physical perturbation ``eta`` via a
  differentiable, side-effect-free transformation.  The composition
  ``decode(sample_latent(...))`` has the same marginal statistics as the
  corresponding legacy generator in ``Utils.noise_gen`` but is pure torch, so
  gradients flow through ``z``.
- ``log_prior(z)`` is the negative log-density of ``z`` (up to constants) used
  as a regulariser during Langevin / Adam updates.  For Gaussian latents this
  is ``mean(z**2)`` — matching the existing prior-loss shape.

The priors are used by the Langevin and Optimisation rescue loops in
``Inversion_Strategies/inversion/residual_inversion.py`` to optimise in the
*latent* space rather than on grid-point perturbations, which preserves the
function-class structure (smoothness, covariance, spline sub-space) throughout
optimisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


class LatentNoisePrior:
    """Abstract base class for latent-space noise priors."""

    latent_shape: tuple[int, ...]

    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype

    def sample_latent(self, batch: int, seed: Optional[int] = None) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(seed)
        return torch.randn(batch, *self.latent_shape, device=self.device, dtype=self.dtype)

    def decode(self, z: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def log_prior(self, z: torch.Tensor) -> torch.Tensor:
        """Default Gaussian prior: mean(z**2).  Matches existing loss scale."""
        return torch.mean(z ** 2)


class WhitePrior(LatentNoisePrior):
    """eta = std * z, z ~ N(0, I).  Grid-space is the latent space."""

    def __init__(self, n_points: int, std: float, device, dtype):
        super().__init__(device, dtype)
        self.latent_shape = (n_points,)
        self.std = float(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.std * z


class SpatialPrior(LatentNoisePrior):
    """eta = conv1d(z, G_L) * scale, with scale chosen analytically.

    The decoder is linear in ``z`` and differentiable.  We precompute a scalar
    ``scale`` so that, for ``z ~ N(0, I)`` and far from the boundary, the
    marginal standard deviation of ``eta`` equals the user-specified ``std``.
    This replaces the non-differentiable batch-level renormalisation in
    ``PDENoiseGenerator1D.spatially_correlated_noise``.
    """

    def __init__(self, n_points: int, correlation_length: float, std: float,
                 device, dtype):
        super().__init__(device, dtype)
        self.latent_shape = (n_points,)

        kernel_size = max(3, int(4 * correlation_length) // 2 * 2 + 1)
        sigma = correlation_length / 2.0
        x = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
        kernel = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel = kernel / torch.sum(kernel)
        self._kernel = kernel.view(1, 1, -1)            # [1, 1, K]
        self._padding = kernel_size // 2

        # Analytical scale: for white z of unit variance, the conv output has
        # stationary variance = sum(kernel**2).  Choose scale so the output
        # has std == user std.
        norm = torch.sqrt(torch.sum(kernel ** 2)).item()
        self.scale = float(std) / norm if norm > 1e-12 else float(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = z.unsqueeze(1)                               # [B, 1, N]
        y = F.conv1d(x, self._kernel, padding=self._padding)
        return y.squeeze(1) * self.scale


class GPPrior(LatentNoisePrior):
    """eta = L z, with L = chol(K) cached once at construction.

    Gives samples with exactly the GP covariance ``K``.  ``log_prior(z) =
    mean(z**2)`` is the canonical GP log-prior in whitened coordinates.
    """

    def __init__(self, n_points: int, correlation_length: float, std: float,
                 kernel_type: str, nu: float, device, dtype):
        super().__init__(device, dtype)
        self.latent_shape = (n_points,)

        try:
            import gpytorch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "GPPrior requires gpytorch. Install it or use SpatialPrior."
            ) from exc

        x_loc = torch.linspace(0, 1, n_points, device=device, dtype=dtype)
        if kernel_type.lower() == "rbf":
            covar = gpytorch.kernels.RBFKernel()
        elif kernel_type.lower() == "matern":
            covar = gpytorch.kernels.MaternKernel(nu=nu)
        elif kernel_type.lower() == "periodic":
            covar = gpytorch.kernels.PeriodicKernel()
        else:
            raise ValueError(f"Unknown GP kernel: {kernel_type}")

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
        self._L = L                                      # [N, N]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, N]  ->  eta: [B, N] via eta = z @ L.T  (equivalent to L @ z^T)
        return z @ self._L.T


class BSplinePrior(LatentNoisePrior):
    """eta = Phi @ c, with c the K << N control-point latents.

    ``latent_shape == (n_knots,)`` — optimisation dimensionality drops from
    ``N`` to ``K`` (typically 100 -> 16) and every iterate is a valid cubic
    spline (C^2-continuous).
    """

    def __init__(self, n_points: int, n_knots: int, std: float, device, dtype):
        super().__init__(device, dtype)
        self.latent_shape = (n_knots,)

        Phi = self._build_basis(n_points, n_knots, device, dtype)     # [N, K]
        self._Phi = Phi                                                # [N, K]

        # Analytical scale: for c ~ N(0, I), Var(eta_i) = sum_k Phi[i,k]**2.
        # Use the mean row-norm so the average marginal std matches user std.
        row_var = (Phi ** 2).sum(dim=1)
        mean_sigma = torch.sqrt(row_var.mean()).item()
        self.scale = float(std) / mean_sigma if mean_sigma > 1e-12 else float(std)

    @staticmethod
    def _cubic_bspline_1d(t: torch.Tensor) -> torch.Tensor:
        abs_t = torch.abs(t)
        result = torch.zeros_like(t)
        mask1 = abs_t < 1
        a1 = abs_t[mask1]
        result[mask1] = (2.0 / 3.0) - a1 ** 2 + 0.5 * a1 ** 3
        mask2 = (abs_t >= 1) & (abs_t < 2)
        a2 = abs_t[mask2]
        result[mask2] = (1.0 / 6.0) * (2.0 - a2) ** 3
        return result

    @classmethod
    def _build_basis(cls, n_points: int, n_knots: int, device, dtype) -> torch.Tensor:
        eval_pts = torch.linspace(0.0, 1.0, n_points, device=device, dtype=dtype)
        knots = torch.linspace(0.0, 1.0, n_knots, device=device, dtype=dtype)
        dk = knots[1] - knots[0]
        t = (eval_pts.unsqueeze(1) - knots.unsqueeze(0)) / dk
        return cls._cubic_bspline_1d(t)                                # [N, K]

    def decode(self, c: torch.Tensor) -> torch.Tensor:
        # c: [B, K] ; Phi: [N, K]  ->  [B, N]
        return (c @ self._Phi.T) * self.scale


class PreCorrelatedPrior(LatentNoisePrior):
    """eta = conv1d(z, pre_kernel) * scale, analogous to SpatialPrior but with
    a user-supplied kernel (e.g. the additive PRE kernel).
    """

    def __init__(self, n_points: int, kernel: torch.Tensor, std: float,
                 device, dtype):
        super().__init__(device, dtype)
        self.latent_shape = (n_points,)

        k = kernel.to(device=device, dtype=dtype).view(1, 1, -1)
        self._kernel = k
        self._padding = k.shape[-1] // 2

        norm = torch.sqrt(torch.sum(k ** 2)).item()
        self.scale = float(std) / norm if norm > 1e-12 else float(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = z.unsqueeze(1)
        y = F.conv1d(x, self._kernel, padding=self._padding)
        return y.squeeze(1) * self.scale


class White2DPrior(LatentNoisePrior):
    """eta = std * z, z ~ N(0, I) on a 2D grid."""

    def __init__(self, shape: tuple[int, int], std: float, device, dtype):
        super().__init__(device, dtype)
        self.latent_shape = tuple(shape)
        self.std = float(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.std * z


class Spatial2DPrior(LatentNoisePrior):
    """2D separable Gaussian-smoothed white noise.

    Decoder: eta = conv2d(z, G) * scale, with G a separable Gaussian kernel.
    Analytical scale uses the stationary output variance = sum(G**2).
    """

    def __init__(self, shape: tuple[int, int], correlation_length: float,
                 std: float, device, dtype):
        super().__init__(device, dtype)
        self.latent_shape = tuple(shape)

        kernel_size = max(3, int(4 * correlation_length) // 2 * 2 + 1)
        sigma = correlation_length / 2.0
        x = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
        g = torch.exp(-x ** 2 / (2 * sigma ** 2))
        g = g / torch.sum(g)
        # Separable 2D kernel = outer product.
        kernel2d = torch.outer(g, g)                          # [K, K]
        self._kernel = kernel2d.view(1, 1, kernel_size, kernel_size)
        self._padding = kernel_size // 2

        norm = torch.sqrt(torch.sum(kernel2d ** 2)).item()
        self.scale = float(std) / norm if norm > 1e-12 else float(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, H, W]  ->  [B, 1, H, W]  ->  conv2d  ->  [B, H, W]
        x = z.unsqueeze(1)
        y = F.conv2d(x, self._kernel, padding=self._padding)
        return y.squeeze(1) * self.scale


class BSpline2DPrior(LatentNoisePrior):
    """Tensor-product cubic B-spline prior on a 2D grid.

    ``latent_shape == (Kt, Kx)`` control points; ``eta = Phi_t @ c @ Phi_xᵀ``
    evaluated at the ``(H, W)`` grid.  Latent dimensionality is ``Kt * Kx``.
    """

    def __init__(self, shape: tuple[int, int], n_knots: tuple[int, int] | int,
                 std: float, device, dtype):
        super().__init__(device, dtype)
        H, W = shape
        if isinstance(n_knots, int):
            Kt, Kx = n_knots, n_knots
        else:
            Kt, Kx = n_knots
        self.latent_shape = (Kt, Kx)

        Phi_t = BSplinePrior._build_basis(H, Kt, device, dtype)   # [H, Kt]
        Phi_x = BSplinePrior._build_basis(W, Kx, device, dtype)   # [W, Kx]
        self._Phi_t = Phi_t
        self._Phi_x = Phi_x

        # For c ~ N(0, I_{Kt Kx}), Var(eta[i,j]) = sum_{k,l} Phi_t[i,k]^2 Phi_x[j,l]^2
        # = (Phi_t**2).sum(1)[i] * (Phi_x**2).sum(1)[j]; use mean over grid.
        row_t = (Phi_t ** 2).sum(dim=1)                           # [H]
        row_x = (Phi_x ** 2).sum(dim=1)                           # [W]
        mean_var = (row_t.mean() * row_x.mean()).item()
        mean_sigma = mean_var ** 0.5
        self.scale = float(std) / mean_sigma if mean_sigma > 1e-12 else float(std)

    def decode(self, c: torch.Tensor) -> torch.Tensor:
        # c: [B, Kt, Kx] ; output [B, H, W] = Phi_t @ c @ Phi_xᵀ.
        out = torch.einsum("ik,bkl,jl->bij", self._Phi_t, c, self._Phi_x)
        return out * self.scale


@dataclass(frozen=True)
class PriorSpec:
    """Compact description of a prior; resolved into a concrete object by
    ``build_prior`` below.
    """
    noise_type: str
    noise_std: float
    correlation_length: float = 5.0
    gp_kernel: str = "rbf"
    gp_nu: float = 1.5
    bspline_n_knots: int = 16
    pre_kernel: Optional[torch.Tensor] = None


def build_prior_1d(
    spec: PriorSpec,
    n_points: int,
    device,
    dtype=torch.float32,
) -> LatentNoisePrior:
    """Factory: build a 1D latent prior from a :class:`PriorSpec`."""
    nt = spec.noise_type
    if nt == "white":
        return WhitePrior(n_points, spec.noise_std, device, dtype)
    if nt == "spatial":
        return SpatialPrior(n_points, spec.correlation_length, spec.noise_std,
                            device, dtype)
    if nt == "gp":
        return GPPrior(n_points, spec.correlation_length, spec.noise_std,
                       spec.gp_kernel, spec.gp_nu, device, dtype)
    if nt == "bspline":
        return BSplinePrior(n_points, spec.bspline_n_knots, spec.noise_std,
                            device, dtype)
    if nt == "pre_correlated":
        if spec.pre_kernel is None:
            raise ValueError("pre_kernel must be provided for 'pre_correlated'")
        return PreCorrelatedPrior(n_points, spec.pre_kernel, spec.noise_std,
                                  device, dtype)
    raise ValueError(f"Unknown noise type: {nt}")


def build_prior_2d(
    spec: PriorSpec,
    shape: tuple[int, int],
    device,
    dtype=torch.float32,
) -> LatentNoisePrior:
    """Factory: build a 2D latent prior from a :class:`PriorSpec`.

    Currently supports ``white``, ``spatial``, and ``bspline``.  GP and
    pre-correlated 2D priors are not implemented — callers should fall back
    to the legacy grid-space path for those cases.
    """
    nt = spec.noise_type
    if nt == "white":
        return White2DPrior(shape, spec.noise_std, device, dtype)
    if nt == "spatial":
        return Spatial2DPrior(shape, spec.correlation_length, spec.noise_std,
                              device, dtype)
    if nt == "bspline":
        return BSpline2DPrior(shape, spec.bspline_n_knots, spec.noise_std,
                              device, dtype)
    raise ValueError(f"2D prior not implemented for noise_type={nt!r}")


def build_prior(
    spec: PriorSpec,
    input_shape: tuple[int, ...],
    device,
    dtype=torch.float32,
) -> LatentNoisePrior:
    """Dispatch to :func:`build_prior_1d` or :func:`build_prior_2d` based on
    ``len(input_shape)``.  Raises ``ValueError`` for higher-dimensional inputs.
    """
    if len(input_shape) == 1:
        return build_prior_1d(spec, input_shape[0], device, dtype)
    if len(input_shape) == 2:
        return build_prior_2d(spec, tuple(input_shape), device, dtype)
    raise ValueError(f"No latent prior for input of rank {len(input_shape)}")
