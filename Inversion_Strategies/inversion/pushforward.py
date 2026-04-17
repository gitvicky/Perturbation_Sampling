"""
Quantile pushforward inversion.

Maps residual-space conformal bounds into physical-space bounds via a
deterministic pushforward through the (possibly linearised) PDE operator.
No ground truth is used at calibration — only the residual field R = D(u_pred)
on the calibration set plus the known operator D.

Pipeline
--------
1. Assemble operator as a dense matrix M (Nint x T).
   - Linear ODEs (SHO/DHO): M is built from the composite stencil kernel.
   - Nonlinear (Duffing):   M = J_D(u_pred) evaluated at the test prediction.
2. Estimate residual covariance  Sigma_R  from calibration residuals
   (with Ledoit-Wolf style shrinkage for stability).
3. Pushforward covariance        Sigma_E = M^+ Sigma_R M^{+T}
   using Tikhonov-regularised pseudo-inverse to tame null-space blow-up.
4. Joint Mahalanobis CP on the residual:
       s_i = sqrt( R_i^T Sigma_R^{-1} R_i ),    qhat = Quantile_{1-alpha}(s).
   The implied residual-space set is an ellipsoid { r : r^T Sigma_R^{-1} r <= qhat^2 }.
5. Pushforward: the corresponding error ellipsoid has covariance
       C_E = qhat^2 * Sigma_E
   and pointwise bounds  |E_k| <= qhat * sqrt(Sigma_E[k,k]).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Operator assembly
# ---------------------------------------------------------------------------
def stencil_to_matrix(kernel: np.ndarray, T: int, interior: bool = True) -> np.ndarray:
    """Build the (N_out, T) convolution matrix for a 1D stencil with zero padding.

    ``interior=True`` drops the boundary rows (rows that would touch zero-padding),
    matching ``slice(1, -1)`` used elsewhere in the codebase. For a 3-point
    stencil this returns a (T-2, T) matrix.
    """
    k = np.asarray(kernel, dtype=np.float64).ravel()
    ks = k.size
    pad = ks // 2
    M = np.zeros((T, T), dtype=np.float64)
    for i in range(T):
        for j in range(ks):
            col = i + j - pad
            if 0 <= col < T:
                M[i, col] = k[j]
    if interior:
        M = M[pad:T - pad, :]
    return M


def duffing_jacobian_matrix(
    u_pred: np.ndarray, alpha: float, beta: float, delta: float, dt: float,
) -> np.ndarray:
    """Linearised Duffing residual operator  J = d R(u) / d u  at u = u_pred.

    R(u) = 2*D_tt(u) + dt*delta*D_t(u) + 2 dt^2 (alpha*u + beta*u^3)
    so the Jacobian is the same linear stencil plus a diagonal
        2 dt^2 (alpha + 3 beta u_pred^2)
    added to the identity-row contribution.
    """
    from Utils.PRE.ConvOps_0d import ConvOperator

    T = u_pred.size
    D_t = ConvOperator(order=1)
    D_tt = ConvOperator(order=2)

    # Linear part matches the DHO composite (alpha handled via diagonal below
    # so we don't double-count it): 2*D_tt + dt*delta*D_t.
    lin_kernel = 2.0 * D_tt.kernel.numpy() + dt * delta * D_t.kernel.numpy()
    M_lin = stencil_to_matrix(lin_kernel, T, interior=True)          # (T-2, T)

    diag_vals = 2.0 * dt**2 * (alpha + 3.0 * beta * u_pred**2)       # (T,)
    # Interior rows i=1..T-2 pick diagonal entry at column i.
    M_diag = np.zeros_like(M_lin)
    for i in range(M_lin.shape[0]):
        M_diag[i, i + 1] = diag_vals[i + 1]

    return M_lin + M_diag


# ---------------------------------------------------------------------------
# Covariance + pushforward
# ---------------------------------------------------------------------------
def shrunk_covariance(X: np.ndarray, shrink: float = 0.05) -> np.ndarray:
    """Empirical covariance with diagonal shrinkage toward the mean variance."""
    X = np.asarray(X, dtype=np.float64)
    X0 = X - X.mean(axis=0, keepdims=True)
    n = X.shape[0]
    S = (X0.T @ X0) / max(n - 1, 1)
    trace_mean = np.trace(S) / S.shape[0]
    return (1.0 - shrink) * S + shrink * trace_mean * np.eye(S.shape[0])


def tikhonov_pinv(M: np.ndarray, lam: float = 1e-6) -> np.ndarray:
    """Tikhonov-regularised pseudo-inverse:  (M^T M + lam I)^{-1} M^T."""
    MtM = M.T @ M
    reg = lam * np.trace(MtM) / MtM.shape[0] * np.eye(MtM.shape[0])
    return np.linalg.solve(MtM + reg, M.T)


@dataclass
class PushforwardResult:
    qhat: float
    sigma_R: np.ndarray          # (Nint, Nint)
    sigma_E: np.ndarray          # (T, T)
    pointwise_std: np.ndarray    # (T,) — sqrt(diag(sigma_E))
    lower: np.ndarray            # (T,)  u_pred - qhat * pointwise_std
    upper: np.ndarray            # (T,)  u_pred + qhat * pointwise_std


def mahalanobis_qhat(residual_cal: np.ndarray, sigma_R: np.ndarray,
                     alpha: float, jitter: float = 1e-8) -> float:
    """Split-conformal qhat using Mahalanobis nonconformity scores on residuals."""
    R = np.asarray(residual_cal, dtype=np.float64)
    n = R.shape[0]
    # Whitening via Cholesky for numerical stability.
    L = np.linalg.cholesky(sigma_R + jitter * np.eye(sigma_R.shape[0]))
    W = np.linalg.solve(L, R.T).T                       # (n, Nint)
    scores = np.sqrt(np.sum(W * W, axis=1))             # (n,)
    # Conformal quantile: ceil((n+1)(1-alpha))/n.
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def pushforward_bounds(
    u_pred: np.ndarray,
    residual_cal: np.ndarray,
    operator_matrix: np.ndarray,
    *,
    alpha: float,
    shrink: float = 0.05,
    tikhonov: float = 1e-6,
) -> PushforwardResult:
    """Compute pushforward bounds for a single prediction.

    Parameters
    ----------
    u_pred : (T,) predicted trajectory (numpy).
    residual_cal : (N_cal, Nint) calibration residuals on the interior.
    operator_matrix : (Nint, T) linear(ised) operator mapping u -> residual.
    alpha : miscoverage target.
    """
    sigma_R = shrunk_covariance(residual_cal, shrink=shrink)       # (Nint, Nint)
    M_pinv = tikhonov_pinv(operator_matrix, lam=tikhonov)          # (T, Nint)
    sigma_E = M_pinv @ sigma_R @ M_pinv.T                          # (T, T)
    pointwise_std = np.sqrt(np.clip(np.diag(sigma_E), 0.0, None))  # (T,)

    qhat = mahalanobis_qhat(residual_cal, sigma_R, alpha=alpha)
    halfwidth = qhat * pointwise_std
    return PushforwardResult(
        qhat=qhat,
        sigma_R=sigma_R,
        sigma_E=sigma_E,
        pointwise_std=pointwise_std,
        lower=u_pred - halfwidth,
        upper=u_pred + halfwidth,
    )


# ---------------------------------------------------------------------------
# Empirical coverage
# ---------------------------------------------------------------------------
def coverage_curve(
    preds: np.ndarray,
    truths: np.ndarray,
    residual_cal: np.ndarray,
    build_operator: Callable[[np.ndarray], np.ndarray],
    *,
    alphas: np.ndarray,
    interior_slice: slice = slice(1, -1),
    shrink: float = 0.05,
    tikhonov: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate realised coverage at each alpha on held-out (preds, truths).

    ``build_operator(u_pred)`` returns the operator matrix for one trajectory.
    For linear ODEs it can ignore u_pred and return a constant matrix.
    """
    preds = np.asarray(preds, dtype=np.float64)
    truths = np.asarray(truths, dtype=np.float64)
    residual_cal = np.asarray(residual_cal, dtype=np.float64)

    nominal = 1.0 - np.asarray(alphas, dtype=np.float64)
    cov_pointwise = np.zeros_like(nominal)
    cov_joint = np.zeros_like(nominal)

    for ai, alpha in enumerate(alphas):
        hits_point = []
        hits_joint = []
        for i in range(preds.shape[0]):
            u_pred = preds[i]
            M = build_operator(u_pred)
            res = pushforward_bounds(
                u_pred, residual_cal, M,
                alpha=float(alpha), shrink=shrink, tikhonov=tikhonov,
            )
            t_int = truths[i][interior_slice]
            l_int = res.lower[interior_slice]
            u_int = res.upper[interior_slice]
            inside = (t_int >= l_int) & (t_int <= u_int)
            hits_point.append(float(inside.mean()))
            hits_joint.append(float(inside.all()))
        cov_pointwise[ai] = float(np.mean(hits_point))
        cov_joint[ai] = float(np.mean(hits_joint))

    return nominal, cov_pointwise, cov_joint
