"""Composable wrappers over any :class:`LatentNoisePrior`.

Each wrapper is itself a prior: it forwards ``sample_latent`` / ``log_prior``
to the inner prior and composes a linear projection on top of ``decode``.
This preserves differentiability and keeps the whitened-coordinate log-prior
semantics.
"""

from __future__ import annotations

from typing import Optional

import torch

from .base import LatentNoisePrior


class _PriorWrapper(LatentNoisePrior):
    """Common scaffolding: delegate latent draws and log-prior to an inner prior."""

    def __init__(self, inner: LatentNoisePrior):
        super().__init__(device=inner.device, dtype=inner.dtype)
        self.inner = inner
        self.latent_shape = inner.latent_shape
        self.output_shape = inner.output_shape

    def sample_latent(self, batch: int, seed: Optional[int] = None) -> torch.Tensor:
        return self.inner.sample_latent(batch, seed=seed)

    def log_prior(self, z: torch.Tensor) -> torch.Tensor:
        return self.inner.log_prior(z)

    # ``decode`` overridden by concrete wrappers.


class MeshScale(_PriorWrapper):
    """Multiply the inner decoder by ``1 / sqrt(prod(spacing))``.

    For a discrete white-noise representation of a continuous field, this
    enforces ``integral xi^2`` invariance under mesh refinement.
    """

    def __init__(self, inner: LatentNoisePrior, spacing: tuple[float, ...] | float):
        super().__init__(inner)
        if isinstance(spacing, (int, float)):
            self.spacing = (float(spacing),)
        else:
            self.spacing = tuple(float(s) for s in spacing)
        vol = 1.0
        for s in self.spacing:
            vol *= s
        self.factor = vol ** -0.5

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.inner.decode(z) * self.factor

    def marginal_std(self) -> torch.Tensor:
        return self.inner.marginal_std() * self.factor


class BoundaryProject(_PriorWrapper):
    """Linear projection onto a boundary-compatible subspace.

    - ``periodic``: identity (already periodic for FFT-based priors; for
      convolution priors this is a no-op and the caller should use a
      circular-padded decoder instead if strict periodicity is required).
    - ``dirichlet``: multiply by a 0-on-boundary mask.
    - ``neumann``: clamp boundary rows/cols to their interior neighbour
      (linear reflection).
    """

    def __init__(self, inner: LatentNoisePrior, boundary: str):
        super().__init__(inner)
        boundary = boundary.lower()
        if boundary not in ("periodic", "dirichlet", "neumann"):
            raise ValueError(
                f"Unknown boundary {boundary!r}; expected "
                "'periodic', 'dirichlet', or 'neumann'."
            )
        self.boundary = boundary

        if boundary == "dirichlet":
            mask = torch.ones(inner.output_shape,
                              device=inner.device, dtype=inner.dtype)
            # Zero every outer slab of each axis.
            slicer = [slice(None)] * len(inner.output_shape)
            for axis in range(len(inner.output_shape)):
                s0 = slicer.copy(); s0[axis] = 0
                s1 = slicer.copy(); s1[axis] = -1
                mask[tuple(s0)] = 0
                mask[tuple(s1)] = 0
            self.register_buffer("_mask", mask)
        else:
            self.register_buffer(
                "_mask", torch.ones(1, device=inner.device, dtype=inner.dtype)
            )

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        eta = self.inner.decode(z)
        if self.boundary == "dirichlet":
            return eta * self._mask
        if self.boundary == "neumann":
            # Reflect the interior-adjacent slice onto the boundary.
            for axis in range(1, eta.ndim):
                # axis=0 is the batch dim.
                eta = eta.clone()
                s0 = [slice(None)] * eta.ndim
                s1 = [slice(None)] * eta.ndim
                s0[axis], s1[axis] = 0, 1
                eta[tuple(s0)] = eta[tuple(s1)]
                s0[axis], s1[axis] = -1, -2
                eta[tuple(s0)] = eta[tuple(s1)]
            return eta
        return eta  # periodic: no-op

    def marginal_std(self) -> torch.Tensor:
        s = self.inner.marginal_std()
        if self.boundary == "dirichlet":
            return s * self._mask
        return s
