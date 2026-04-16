"""Unified differentiable latent noise priors.

Public API:

- :class:`LatentNoisePrior` — abstract base (``nn.Module``).
- 1D priors: :class:`WhitePrior`, :class:`SpatialPrior`, :class:`GPPrior`,
  :class:`BSplinePrior`, :class:`PreCorrelatedPrior`, :class:`SpectralPrior`,
  :class:`OUPrior`.
- 2D priors: :class:`White2DPrior`, :class:`Spatial2DPrior`,
  :class:`BSpline2DPrior`, :class:`Spectral2DPrior`.
- Wrappers: :class:`MeshScale`, :class:`BoundaryProject`.
- Factory: :class:`PriorSpec`, :func:`build_prior`, :func:`build_prior_1d`,
  :func:`build_prior_2d`.
"""

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
from .spec import PriorSpec, build_prior, build_prior_1d, build_prior_2d
from .wrappers import BoundaryProject, MeshScale

__all__ = [
    # base
    "LatentNoisePrior",
    # 1D
    "WhitePrior",
    "SpatialPrior",
    "GPPrior",
    "BSplinePrior",
    "PreCorrelatedPrior",
    "SpectralPrior",
    "OUPrior",
    # 2D
    "White2DPrior",
    "Spatial2DPrior",
    "BSpline2DPrior",
    "Spectral2DPrior",
    # wrappers
    "MeshScale",
    "BoundaryProject",
    # factory
    "PriorSpec",
    "build_prior",
    "build_prior_1d",
    "build_prior_2d",
]
