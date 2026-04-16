"""Legacy imperative noise generators.

This module is preserved for backward compatibility.  Every method now
delegates to the unified, differentiable priors in :mod:`Utils.priors` —
new code should prefer ``Utils.priors.build_prior(PriorSpec(...), shape)``
and sample via ``prior.sample(batch)``.

The generator methods retain their original positional signatures so
downstream callers (notebooks, experiment scripts) continue to work.
"""

from __future__ import annotations

from typing import Optional

import torch

from Utils.priors import (
    BSpline2DPrior,
    BSplinePrior,
    GPPrior,
    OUPrior,
    PreCorrelatedPrior,
    Spatial2DPrior,
    SpatialPrior,
    Spectral2DPrior,
    SpectralPrior,
    White2DPrior,
    WhitePrior,
)


def _set_seed(seed: Optional[int]) -> None:
    if seed is not None:
        torch.manual_seed(seed)


# ----------------------------------------------------------------------
# 2D generator
# ----------------------------------------------------------------------
class PDENoiseGenerator:
    """Legacy 2D noise generator. Delegates to :mod:`Utils.priors`."""

    def __init__(self, device="cpu", dtype: torch.dtype = torch.float32):
        self.device = device
        self.dtype = dtype

    # Accepts either ``(H, W)`` (single sample) or ``(B, H, W)`` (batched).
    def _unpack_shape(self, shape):
        if len(shape) == 2:
            return None, tuple(shape)
        if len(shape) == 3:
            return int(shape[0]), tuple(shape[1:])
        raise ValueError(f"Unsupported 2D shape: {shape}")

    def white_noise(self, shape, std: float = 1.0,
                    seed: Optional[int] = None) -> torch.Tensor:
        _set_seed(seed)
        batch, spatial = self._unpack_shape(shape)
        prior = White2DPrior(spatial, std, self.device, self.dtype)
        B = 1 if batch is None else batch
        z = prior.sample_latent(B)
        out = prior.decode(z)
        return out.squeeze(0) if batch is None else out

    def colored_noise_spectral(self, shape, alpha: float = 0.0,
                               std: float = 1.0) -> torch.Tensor:
        prior = Spectral2DPrior(tuple(shape), alpha, std, self.device, self.dtype)
        return prior.decode(prior.sample_latent(1)).squeeze(0)

    def spatially_correlated_noise(self, shape, correlation_length: float = 5.0,
                                   std: float = 1.0) -> torch.Tensor:
        prior = Spatial2DPrior(tuple(shape), correlation_length, std,
                               self.device, self.dtype)
        return prior.decode(prior.sample_latent(1)).squeeze(0)

    def temporal_noise_sequence(self, spatial_shape, n_timesteps: int,
                                dt: float = 0.01, tau: float = 1.0,
                                std: float = 1.0,
                                noise_type: str = "white") -> torch.Tensor:
        H, W = spatial_shape
        # One OU process per spatial point; shared parameters.
        ou = OUPrior(n_timesteps, dt, tau, std, self.device, self.dtype)
        z = ou.sample_latent(H * W)
        traj = ou.decode(z)                              # [H*W, n_timesteps]
        return traj.reshape(H, W, n_timesteps).permute(2, 0, 1).contiguous()

    def mesh_scaled_noise(self, shape, dx: float = 1.0, dy: float = 1.0,
                          std: float = 1.0) -> torch.Tensor:
        mesh_factor = (dx * dy) ** -0.5
        return self.white_noise(shape, std=std * mesh_factor)

    def boundary_compatible_noise(self, shape,
                                  boundary_type: str = "periodic") -> torch.Tensor:
        from .priors import BoundaryProject
        base = White2DPrior(tuple(shape), 1.0, self.device, self.dtype)
        if boundary_type == "periodic":
            return base.decode(base.sample_latent(1)).squeeze(0)
        wrapped = BoundaryProject(base, boundary_type)
        return wrapped.decode(wrapped.sample_latent(1)).squeeze(0)


# ----------------------------------------------------------------------
# 1D generator
# ----------------------------------------------------------------------
class PDENoiseGenerator1D:
    """Legacy 1D noise generator. Delegates to :mod:`Utils.priors`."""

    def __init__(self, device="cpu", dtype: torch.dtype = torch.float32):
        self.device = device
        self.dtype = dtype

    def white_noise(self, batch_size: int, n_points: int, std: float = 1.0,
                    seed: Optional[int] = None) -> torch.Tensor:
        prior = WhitePrior(n_points, std, self.device, self.dtype)
        return prior.decode(prior.sample_latent(batch_size, seed=seed))

    def spatially_correlated_noise(self, batch_size: int, n_points: int,
                                   correlation_length: float = 5.0,
                                   std: float = 1.0,
                                   seed: Optional[int] = None) -> torch.Tensor:
        prior = SpatialPrior(n_points, correlation_length, std,
                             self.device, self.dtype)
        return prior.decode(prior.sample_latent(batch_size, seed=seed))

    def pre_correlated_noise(self, batch_size: int, n_points: int,
                             kernel: torch.Tensor, std: float = 1.0,
                             seed: Optional[int] = None) -> torch.Tensor:
        prior = PreCorrelatedPrior(n_points, kernel, std,
                                   self.device, self.dtype)
        return prior.decode(prior.sample_latent(batch_size, seed=seed))

    def bspline_noise(self, batch_size: int, n_points: int,
                      n_knots: int = 16, std: float = 1.0,
                      seed: Optional[int] = None) -> torch.Tensor:
        prior = BSplinePrior(n_points, n_knots, std, self.device, self.dtype)
        return prior.decode(prior.sample_latent(batch_size, seed=seed))

    def gp_noise(self, batch_size: int, n_points: int,
                 correlation_length: float = 5.0, std: float = 1.0,
                 kernel_type: str = "rbf", nu: float = 1.5,
                 seed: Optional[int] = None) -> torch.Tensor:
        try:
            prior = GPPrior(n_points, correlation_length, std,
                            kernel_type=kernel_type, nu=nu,
                            device=self.device, dtype=self.dtype)
        except RuntimeError:
            # Graceful fallback when gpytorch is missing, matching legacy behaviour.
            return self.spatially_correlated_noise(
                batch_size, n_points, correlation_length=correlation_length,
                std=std, seed=seed,
            )
        return prior.decode(prior.sample_latent(batch_size, seed=seed))

    def spectral_noise(self, batch_size: int, n_points: int,
                       alpha: float = 1.0, std: float = 1.0,
                       seed: Optional[int] = None) -> torch.Tensor:
        prior = SpectralPrior(n_points, alpha, std, self.device, self.dtype)
        return prior.decode(prior.sample_latent(batch_size, seed=seed))

    def ou_noise(self, batch_size: int, n_steps: int,
                 dt: float = 0.01, tau: float = 1.0, std: float = 1.0,
                 seed: Optional[int] = None) -> torch.Tensor:
        prior = OUPrior(n_steps, dt, tau, std, self.device, self.dtype)
        return prior.decode(prior.sample_latent(batch_size, seed=seed))

    def mesh_scaled_noise(self, batch_size: int, n_points: int,
                          dx: float = 1.0, std: float = 1.0,
                          correlation_length: Optional[float] = None,
                          seed: Optional[int] = None) -> torch.Tensor:
        effective_std = std * (dx ** -0.5)
        if correlation_length is None:
            return self.white_noise(batch_size, n_points,
                                    std=effective_std, seed=seed)
        return self.spatially_correlated_noise(
            batch_size, n_points, correlation_length=correlation_length,
            std=effective_std, seed=seed,
        )


__all__ = ["PDENoiseGenerator", "PDENoiseGenerator1D"]
