"""``PriorSpec`` and the ``build_prior`` factory.

A single ``PriorSpec`` can describe any of the noise families plus optional
mesh-scaling / boundary-projection wrappers; ``build_prior`` resolves it into
a concrete :class:`LatentNoisePrior` whose rank matches ``input_shape``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .base import LatentNoisePrior
from .priors_1d import (
    BSplinePrior,
    GPPrior,
    OUPrior,
    PreCorrelatedPrior,
    SpatialPrior,
    SpectralPrior,
    WhitePrior,
)
from .priors_2d import (
    BSpline2DPrior,
    Spatial2DPrior,
    Spectral2DPrior,
    White2DPrior,
)
from .wrappers import BoundaryProject, MeshScale


@dataclass(frozen=True)
class PriorSpec:
    """Compact description of a prior; resolved by :func:`build_prior`."""

    noise_type: str
    noise_std: float
    correlation_length: float = 5.0
    gp_kernel: str = "rbf"
    gp_nu: float = 1.5
    bspline_n_knots: int = 16
    pre_kernel: Optional[torch.Tensor] = None
    spectral_alpha: float = 1.0
    ou_tau: float = 1.0
    ou_dt: float = 0.01
    boundary: Optional[str] = None                    # None|periodic|dirichlet|neumann
    mesh_spacing: Optional[tuple[float, ...]] = None  # e.g. (dx,) or (dx, dy)


def _apply_wrappers(spec: PriorSpec, prior: LatentNoisePrior) -> LatentNoisePrior:
    if spec.boundary is not None:
        prior = BoundaryProject(prior, spec.boundary)
    if spec.mesh_spacing is not None:
        prior = MeshScale(prior, spec.mesh_spacing)
    return prior


def build_prior_1d(
    spec: PriorSpec,
    n_points: int,
    device,
    dtype=torch.float32,
) -> LatentNoisePrior:
    """Build a 1D latent prior from a :class:`PriorSpec`."""
    nt = spec.noise_type
    if nt == "white":
        p = WhitePrior(n_points, spec.noise_std, device, dtype)
    elif nt == "spatial":
        p = SpatialPrior(n_points, spec.correlation_length, spec.noise_std,
                         device, dtype)
    elif nt == "gp":
        p = GPPrior(n_points, spec.correlation_length, spec.noise_std,
                    spec.gp_kernel, spec.gp_nu, device, dtype)
    elif nt == "bspline":
        p = BSplinePrior(n_points, spec.bspline_n_knots, spec.noise_std,
                         device, dtype)
    elif nt == "pre_correlated":
        if spec.pre_kernel is None:
            raise ValueError("pre_kernel must be provided for 'pre_correlated'")
        p = PreCorrelatedPrior(n_points, spec.pre_kernel, spec.noise_std,
                               device, dtype)
    elif nt == "spectral":
        p = SpectralPrior(n_points, spec.spectral_alpha, spec.noise_std,
                          device, dtype)
    elif nt == "ou":
        p = OUPrior(n_points, spec.ou_dt, spec.ou_tau, spec.noise_std,
                    device, dtype)
    else:
        raise ValueError(f"Unknown noise type: {nt!r}")
    return _apply_wrappers(spec, p)


def build_prior_2d(
    spec: PriorSpec,
    shape: tuple[int, int],
    device,
    dtype=torch.float32,
) -> LatentNoisePrior:
    """Build a 2D latent prior from a :class:`PriorSpec`."""
    nt = spec.noise_type
    if nt == "white":
        p = White2DPrior(shape, spec.noise_std, device, dtype)
    elif nt == "spatial":
        p = Spatial2DPrior(shape, spec.correlation_length, spec.noise_std,
                           device, dtype)
    elif nt == "bspline":
        p = BSpline2DPrior(shape, spec.bspline_n_knots, spec.noise_std,
                           device, dtype)
    elif nt == "spectral":
        p = Spectral2DPrior(shape, spec.spectral_alpha, spec.noise_std,
                            device, dtype)
    else:
        raise ValueError(f"2D prior not implemented for noise_type={nt!r}")
    return _apply_wrappers(spec, p)


def build_prior(
    spec: PriorSpec,
    input_shape: tuple[int, ...],
    device,
    dtype=torch.float32,
) -> LatentNoisePrior:
    """Dispatch to the 1D or 2D builder based on ``len(input_shape)``."""
    if len(input_shape) == 1:
        return build_prior_1d(spec, input_shape[0], device, dtype)
    if len(input_shape) == 2:
        return build_prior_2d(spec, tuple(input_shape), device, dtype)
    raise ValueError(f"No latent prior for input of rank {len(input_shape)}")
