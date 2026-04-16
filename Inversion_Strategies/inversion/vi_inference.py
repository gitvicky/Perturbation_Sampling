"""Per-trajectory Variational Inference for residual-bound inversion.

Given a calibrated bound ``qhat`` and a latent noise prior ``p(z) = N(0, I)``
in whitened coordinates (see :mod:`Utils.latent_priors`), we fit a Gaussian
variational posterior ``q_phi(z) = N(mu, Sigma)`` whose samples produce
perturbations ``eta = prior.decode(z)`` that keep the residual
``|D(u_pred + eta)| <= qhat`` — i.e. samples that live on the admissible
physical manifold.  The fitted posterior is then sampled and decoded to
populate the min/max envelope exactly like the existing rejection /
optimisation / Langevin / generator pipelines.

Three covariance parameterisations are exposed:

- ``mean_field``: ``Sigma = diag(sigma^2)``, L parameters per axis.
- ``low_rank``:   ``Sigma = U U^T + diag(d^2)``, U in R^{L x r} — scales to
  large latent dimensions (2D PDE grids) while modelling correlations.
- ``full``:       ``Sigma = L_chol L_chol^T`` — most expressive but
  ``O(L^2)`` memory; the :func:`fit_vi_posterior` helper guards against
  accidental use on oversized problems.

The KL term is computed in closed form against ``N(0, I)``.  The
reparameterisation ``z = mu + L_chol @ eps`` (or analogous for the other
modes) makes the ELBO gradient path-differentiable through both the prior
decoder and the residual operator.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from Utils.latent_priors import LatentNoisePrior


def _qhat_tensor(qhat, device) -> torch.Tensor:
    t = torch.tensor(np.asarray(qhat, dtype=np.float64),
                     dtype=torch.float32, device=device).abs()
    if t.ndim > 0:
        t = t.unsqueeze(0)
    return t


class VariationalPosterior(nn.Module):
    """Gaussian variational posterior over a flat latent of length ``L``.

    The posterior always treats ``z`` as a flat vector of size
    ``L = prod(latent_shape)``; callers reshape to ``latent_shape`` before
    handing it to the prior decoder.
    """

    def __init__(
        self,
        latent_shape: tuple[int, ...],
        covariance: str = "mean_field",
        rank: int = 8,
        init_log_sigma: float = -1.0,
    ):
        super().__init__()
        if covariance not in ("mean_field", "low_rank", "full"):
            raise ValueError(
                f"Unknown covariance={covariance!r}; choose from "
                "'mean_field', 'low_rank', 'full'."
            )
        self.latent_shape = tuple(latent_shape)
        self.L = int(np.prod(latent_shape))
        self.covariance = covariance
        self.rank = int(rank)

        self.mu = nn.Parameter(torch.zeros(self.L))

        if covariance == "mean_field":
            self.log_sigma = nn.Parameter(
                torch.full((self.L,), float(init_log_sigma))
            )
        elif covariance == "low_rank":
            self.U = nn.Parameter(
                0.01 * torch.randn(self.L, self.rank)
            )
            self.log_d = nn.Parameter(
                torch.full((self.L,), float(init_log_sigma))
            )
        else:  # full
            # Parameterise the Cholesky factor directly.  Diagonal is passed
            # through softplus to guarantee positivity; strictly-lower entries
            # are free parameters.
            init = math.exp(float(init_log_sigma))
            # softplus^{-1}(init) = log(exp(init) - 1)
            inv_sp = math.log(math.expm1(init)) if init > 0 else float(init_log_sigma)
            diag_raw = torch.full((self.L,), inv_sp)
            lower = torch.zeros(self.L, self.L)
            self.L_diag_raw = nn.Parameter(diag_raw)
            self.L_lower = nn.Parameter(lower)

    # ------------------------------------------------------------------
    # Cholesky helpers
    # ------------------------------------------------------------------
    def _full_chol(self) -> torch.Tensor:
        diag = torch.nn.functional.softplus(self.L_diag_raw) + 1e-8
        tril = torch.tril(self.L_lower, diagonal=-1)
        return tril + torch.diag(diag)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def rsample(self, n: int) -> torch.Tensor:
        """Draw ``n`` reparameterised samples; returns ``[n, L]``."""
        eps = torch.randn(n, self.L, device=self.mu.device, dtype=self.mu.dtype)
        if self.covariance == "mean_field":
            sigma = torch.exp(self.log_sigma)
            return self.mu + eps * sigma
        if self.covariance == "low_rank":
            d = torch.exp(self.log_d)
            eps_r = torch.randn(n, self.rank, device=self.mu.device,
                                dtype=self.mu.dtype)
            return self.mu + eps * d + eps_r @ self.U.T
        # full
        L_chol = self._full_chol()
        return self.mu + eps @ L_chol.T

    def rsample_shaped(self, n: int) -> torch.Tensor:
        """Reparameterised sample reshaped to ``(n, *latent_shape)``."""
        return self.rsample(n).view(n, *self.latent_shape)

    # ------------------------------------------------------------------
    # KL(q || N(0, I))   — closed form in every mode
    # ------------------------------------------------------------------
    def kl_to_standard_normal(self) -> torch.Tensor:
        mu2 = torch.sum(self.mu ** 2)
        if self.covariance == "mean_field":
            sigma2 = torch.exp(2 * self.log_sigma)
            tr = torch.sum(sigma2)
            logdet = 2 * torch.sum(self.log_sigma)
            return 0.5 * (tr + mu2 - self.L - logdet)
        if self.covariance == "low_rank":
            d2 = torch.exp(2 * self.log_d)
            # tr(Sigma) = sum(d^2) + sum(U**2)
            tr = torch.sum(d2) + torch.sum(self.U ** 2)
            # logdet(D^2 + U U^T) via matrix determinant lemma:
            #   = logdet(D^2) + logdet(I_r + U^T D^{-2} U)
            logdet_D = 2 * torch.sum(self.log_d)
            D_inv2 = torch.exp(-2 * self.log_d)             # [L]
            M = torch.eye(self.rank, device=self.mu.device, dtype=self.mu.dtype) \
                + self.U.T @ (D_inv2.unsqueeze(1) * self.U)
            # Cholesky is the numerically stable route.
            L_M = torch.linalg.cholesky(M)
            logdet_M = 2 * torch.sum(torch.log(torch.diagonal(L_M)))
            logdet = logdet_D + logdet_M
            return 0.5 * (tr + mu2 - self.L - logdet)
        # full
        L_chol = self._full_chol()
        tr = torch.sum(L_chol ** 2)
        logdet = 2 * torch.sum(torch.log(torch.diagonal(L_chol)))
        return 0.5 * (tr + mu2 - self.L - logdet)


def fit_vi_posterior(
    pred_tensor: torch.Tensor,
    operator,
    qhat,
    prior: LatentNoisePrior,
    config,
    *,
    joint: bool = False,
) -> VariationalPosterior:
    """Fit a per-trajectory Gaussian variational posterior via ELBO ascent.

    Returns the trained posterior in ``eval()`` mode.  Raises ``ValueError``
    if ``covariance='full'`` is requested on a latent larger than
    ``config.vi_full_cov_max_dim``.
    """
    L = int(np.prod(prior.latent_shape))
    if config.vi_covariance == "full" and L > int(config.vi_full_cov_max_dim):
        raise ValueError(
            f"Full-covariance VI requested on latent of dim {L} > "
            f"vi_full_cov_max_dim={config.vi_full_cov_max_dim}. "
            "Use vi_covariance='low_rank' instead."
        )

    device = pred_tensor.device
    q = VariationalPosterior(
        prior.latent_shape,
        covariance=config.vi_covariance,
        rank=config.vi_rank,
        init_log_sigma=config.vi_init_log_sigma,
    ).to(device=device, dtype=pred_tensor.dtype)

    opt = optim.Adam(q.parameters(), lr=config.vi_lr)
    qhat_t = _qhat_tensor(qhat, device)
    input_ndim = pred_tensor.ndim - 1  # batch axis is 0

    q.train()
    for step in range(int(config.vi_steps)):
        opt.zero_grad()
        z = q.rsample_shaped(int(config.vi_n_mc))                  # [M, *latent]
        eta = prior.decode(z)                                      # [M, *input]
        r = operator(pred_tensor + eta)

        diff = torch.abs(r) - qhat_t
        if joint:
            # Penalise the worst-violation-per-trajectory; keeps gradient
            # focused on making each sample a valid joint containment.
            flat_violation = torch.clamp(diff, min=0).reshape(diff.shape[0], -1)
            l_bound = torch.mean(flat_violation.amax(dim=1) ** 2)
        else:
            l_bound = torch.mean(torch.clamp(diff, min=0) ** 2)

        kl = q.kl_to_standard_normal() / max(q.L, 1)
        loss = config.lambda_boundary * l_bound + config.vi_kl_weight * kl
        loss.backward()
        opt.step()

        if l_bound.detach().item() < 1e-10:
            break

    q.eval()
    return q
