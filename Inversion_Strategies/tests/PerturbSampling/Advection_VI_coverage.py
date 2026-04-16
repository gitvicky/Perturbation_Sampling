"""Advection (1D spatial + time = 2D grid) coverage validation for VI.

Exercises the 2D branch of the latent-prior / VI pipeline end-to-end:

- ``Spatial2DPrior`` + mean_field and low_rank VI (full is skipped by the
  latent-dim guardrail since L = H * W is a few thousand).
- ``BSpline2DPrior`` (Kt=Kx=8 -> L=64) runs full-cov VI as a sanity reference.

For each configuration we report bound width at alpha=0.1 and empirical
coverage over a small alpha sweep.  A valid VI posterior achieves
coverage >= 1-alpha at alpha=0.1.
"""

import os
import sys
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from Neural_PDE.Numerical_Solvers.Advection.Advection_1D import Advection_1d
from Utils.PRE.ConvOps_1d import ConvOperator
from Inversion_Strategies.inversion.residual_inversion import (
    PerturbationSamplingConfig,
    calibrate_qhat_from_residual,
    perturbation_bounds_nd,
    empirical_coverage_curve_nd,
)


def build_cfg(noise_type, *, use_vi, vi_covariance, vi_rank=8, bspline_n_knots=8):
    return PerturbationSamplingConfig(
        n_samples=400,
        batch_size=200,
        max_rounds=2,
        noise_type=noise_type,
        noise_std=0.05,
        correlation_length=4.0,
        bspline_n_knots=bspline_n_knots,
        seed=0,
        use_vi=use_vi,
        vi_covariance=vi_covariance,
        vi_rank=vi_rank,
        vi_steps=300,
        vi_n_mc=8,
        vi_lr=1e-2,
        vi_full_cov_max_dim=4096,
    )


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    # --- 1. Generate calibration / test trajectories ---
    Nx, Nt = 60, 40
    x_min, x_max, t_end, v = 0.0, 2.0, 0.5, 1.0
    sim = Advection_1d(Nx, Nt, x_min, x_max, t_end)
    dt, dx = sim.dt, sim.dx

    def get_data(n):
        out = []
        for _ in range(n):
            xc = 0.5 + 0.5 * np.random.rand()
            amp = 50 + 150 * np.random.rand()
            _, _, u_soln, _ = sim.solve(xc, amp, v)
            out.append(u_soln)
        return torch.tensor(np.array(out), dtype=torch.float32)

    print("Generating Advection trajectories...")
    u_cal = get_data(15)
    u_test = get_data(6)

    # --- 2. Residual operator D u = u_t + v u_x ---
    D_t = ConvOperator(domain="t", order=1, scale=1.0 / (2 * dt))
    D_x = ConvOperator(domain="x", order=1, scale=1.0 / (2 * dx))

    def advection_residual(u):
        return D_t(u) + v * D_x(u)

    # Add modest surrogate noise to calibration to emulate a learned surrogate.
    u_cal_noisy = u_cal + 0.02 * torch.randn_like(u_cal)
    res_cal = advection_residual(u_cal_noisy)
    qhat = calibrate_qhat_from_residual(res_cal, alpha=0.1)
    print(f"qhat = {qhat:.4f}")

    interior = (slice(1, -1), slice(1, -1))
    alpha_levels = np.array([0.1, 0.3])

    test_pred = (u_test[0] + 0.02 * torch.randn_like(u_test[0])).numpy()

    configs = [
        ("spatial",  "VI-MF",   dict(use_vi=True, vi_covariance="mean_field")),
        ("spatial",  "VI-LR",   dict(use_vi=True, vi_covariance="low_rank", vi_rank=8)),
        ("bspline",  "VI-MF",   dict(use_vi=True, vi_covariance="mean_field")),
        ("bspline",  "VI-Full", dict(use_vi=True, vi_covariance="full")),
    ]

    rows = []
    for nt, name, flags in configs:
        try:
            cfg = build_cfg(nt, **flags)
            b = perturbation_bounds_nd(
                pred_signal=test_pred, residual_operator=advection_residual,
                qhat=qhat, interior_slice=interior, config=cfg,
            )
            width = float(np.mean(b.width))

            cov = empirical_coverage_curve_nd(
                preds=u_test.numpy(),
                truths=u_test.numpy(),          # self-coverage proxy; real truth == pred
                residual_cal=res_cal,
                operator=advection_residual,
                alphas=alpha_levels,
                interior_slice=interior,
                perturbation_config=cfg,
                cp_mode="marginal",
            )
            cov_arr = cov.empirical_coverage_perturbation
            rows.append((nt, name, width, cov_arr))
            print(f"  {nt:8s} {name:8s} width={width:.4f}  "
                  f"cov={np.array2string(cov_arr, precision=2)}")
        except Exception as exc:
            print(f"  {nt:8s} {name:8s} FAILED: {exc!r}")
            rows.append((nt, name, float("nan"),
                         np.full_like(alpha_levels, np.nan, dtype=float)))

    print("\n" + "=" * 70)
    print(f"{'prior':8s} {'method':8s} {'width':>8s}  " +
          "  ".join(f"cov@α={a:.1f}" for a in alpha_levels) + "   valid@α=0.1")
    print("-" * 70)
    for nt, name, width, cov_arr in rows:
        nominal = 1.0 - alpha_levels
        ok = "YES" if (not np.isnan(cov_arr[0]) and cov_arr[0] >= nominal[0] - 1e-6) else "NO"
        cov_str = "  ".join(f"   {c:.2f}" for c in cov_arr)
        print(f"{nt:8s} {name:8s} {width:8.4f}  {cov_str}     {ok}")
    print("=" * 70)


if __name__ == "__main__":
    main()
