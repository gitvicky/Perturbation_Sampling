"""Base class for differentiable latent noise priors.

Every concrete prior exposes the triple ``(sample_latent, decode, log_prior)``:

- ``sample_latent(batch, seed)`` draws ``z`` from the base distribution
  (``N(0, I)`` by default).
- ``decode(z)`` is a *pure*, differentiable linear map ``eta = g_theta(z)``
  realised with buffers registered on the module (so ``.to(device)`` moves
  them for free).
- ``log_prior(z)`` is the canonical whitened log-prior ``mean(z**2)`` — the
  pull-back of the physical-space GP / spline / convolution prior through the
  linear decoder.

``marginal_std()`` returns the theoretical marginal standard deviation at
each output index, derived analytically from the decoder — no empirical
renormalisation, so gradients flow cleanly to ``z`` throughout.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class LatentNoisePrior(nn.Module):
    """Abstract base class for latent-space noise priors."""

    latent_shape: tuple[int, ...]
    output_shape: tuple[int, ...]

    def __init__(self, device=None, dtype: torch.dtype = torch.float32):
        super().__init__()
        self._init_device = device
        self._init_dtype = dtype

    # ------------------------------------------------------------------
    # Device / dtype are inferred from registered buffers so .to() Just Works.
    # Fallback to the construction-time values for parameter-free priors.
    # ------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        buf = next(iter(self.buffers()), None)
        if buf is not None:
            return buf.device
        return torch.device(self._init_device) if self._init_device is not None else torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        buf = next(iter(self.buffers()), None)
        if buf is not None:
            return buf.dtype
        return self._init_dtype

    # ------------------------------------------------------------------
    # Sampling / decoding / prior
    # ------------------------------------------------------------------
    def sample_latent(self, batch: int, seed: Optional[int] = None) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(seed)
        return torch.randn(batch, *self.latent_shape,
                           device=self.device, dtype=self.dtype)

    def decode(self, z: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def log_prior(self, z: torch.Tensor) -> torch.Tensor:
        """Default Gaussian prior ``mean(z**2)`` — matches existing loss scale."""
        return torch.mean(z ** 2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # nn.Module hook
        return self.decode(z)

    def sample(self, batch: int, seed: Optional[int] = None) -> torch.Tensor:
        """Convenience: draw ``z`` and decode in one call."""
        return self.decode(self.sample_latent(batch, seed=seed))

    def marginal_std(self) -> torch.Tensor:
        """Theoretical marginal std at every output index.

        Default implementation estimates via the decoder applied to the
        identity; concrete priors override with a closed-form.
        """
        raise NotImplementedError
