"""Backwards-compatible re-exports from :mod:`Utils.priors`.

The latent-prior implementations have moved to the :mod:`Utils.priors`
package, which unifies them with the noise generators previously in
:mod:`Utils.noise_gen` and adds boundary / mesh-scaling wrappers.  This
module is preserved so existing imports continue to work.
"""

from Utils.priors import (
    BoundaryProject,
    BSpline2DPrior,
    BSplinePrior,
    GPPrior,
    LatentNoisePrior,
    MeshScale,
    OUPrior,
    PreCorrelatedPrior,
    PriorSpec,
    Spatial2DPrior,
    SpatialPrior,
    Spectral2DPrior,
    SpectralPrior,
    White2DPrior,
    WhitePrior,
    build_prior,
    build_prior_1d,
    build_prior_2d,
)

__all__ = [
    "LatentNoisePrior",
    "WhitePrior",
    "SpatialPrior",
    "GPPrior",
    "BSplinePrior",
    "PreCorrelatedPrior",
    "SpectralPrior",
    "OUPrior",
    "White2DPrior",
    "Spatial2DPrior",
    "BSpline2DPrior",
    "Spectral2DPrior",
    "MeshScale",
    "BoundaryProject",
    "PriorSpec",
    "build_prior",
    "build_prior_1d",
    "build_prior_2d",
]
