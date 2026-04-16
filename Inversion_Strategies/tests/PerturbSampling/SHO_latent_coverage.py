"""SHO coverage validation for latent-space optimisation.

Trains a Neural ODE on the Simple Harmonic Oscillator, computes PRE residuals
and a calibrated qhat, then inverts bounds under several combinations of:

- noise prior: white / spatial / gp / bspline
- sampling method: MC rejection / Optim / Langevin / Generator
- optimisation space: grid (legacy) vs. latent (new)

For each combination we measure empirical coverage across alpha levels and
mean bound width at alpha=0.1. A valid method must achieve coverage >= 1-alpha.
The comparison latent-vs-grid checks the latent-space implementation doesn't
regress coverage and records whether it gives tighter / looser bounds.
"""

import os
import sys
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from Utils.PRE.ConvOps_0d import ConvOperator
from Inversion_Strategies.inversion.residual_inversion import (
    PerturbationSamplingConfig,
    calibrate_qhat_from_residual,
    empirical_coverage_curve_1d,
    perturbation_bounds_1d,
)
from Expts.SHO.SHO_NODE import (
    HarmonicOscillator, ODEFunc,
    generate_training_data, train_neural_ode, evaluate,
)


def build_cfg(noise_type, *, optimise_in_latent, use_optimisation=False,
              use_langevin=False, use_generator=False):
    common = dict(
        n_samples=4000,
        batch_size=1000,
        max_rounds=3,
        noise_std=0.5,
        seed=123,
        use_optimisation=use_optimisation,
        use_langevin=use_langevin,
        use_generator=use_generator,
        opt_steps=30,
        langevin_steps=15,
        optimise_in_latent=optimise_in_latent,
    )
    if noise_type == "spatial":
        return PerturbationSamplingConfig(noise_type="spatial", correlation_length=24.0, **common)
    if noise_type == "white":
        return PerturbationSamplingConfig(noise_type="white", **common)
    if noise_type == "gp":
        return PerturbationSamplingConfig(noise_type="gp", correlation_length=24.0, gp_kernel="rbf", **common)
    if noise_type == "bspline":
        return PerturbationSamplingConfig(noise_type="bspline", bspline_n_knots=16, **common)
    raise ValueError(noise_type)


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    m, k = 1.0, 1.0
    oscillator = HarmonicOscillator(k, m)
    t_span = (0, 10)
    n_points = 100
    n_trajectories = 50

    print("Training Neural ODE on SHO...")
    t_train, states, derivs = generate_training_data(
        oscillator, t_span, n_points, n_trajectories)
    func = ODEFunc(hidden_dim=64)
    train_neural_ode(func, t_train, states, derivs, n_epochs=500, batch_size=16)

    print("Evaluating on test trajectories...")
    t, numerical_sol, neural_sol = evaluate(
        oscillator, func, t_span, n_points,
        x_range=(-2, 2), v_range=(-2, 2), n_solves=30)

    pos = torch.tensor(neural_sol[..., 0], dtype=torch.float32)
    dt = t[1] - t[0]

    D_tt = ConvOperator(order=2)
    D_identity = ConvOperator(order=0)
    D_identity.kernel = torch.tensor([0.0, 1.0, 0.0])
    D_pos = ConvOperator(conv="spectral")
    D_pos.kernel = m * D_tt.kernel + dt ** 2 * k * D_identity.kernel

    res = D_pos(pos)
    n_cal = int(0.8 * len(res))
    residual_cal = res[:n_cal]

    alpha_levels = np.array([0.1, 0.3, 0.5])
    interior = slice(1, -1)

    rows = []
    noise_types = ["white", "spatial", "gp", "bspline"]
    methods = [
        ("MC",      dict()),
        ("Optim",   dict(use_optimisation=True)),
        ("Langevin", dict(use_langevin=True)),
        ("Gen",     dict(use_generator=True)),
    ]

    qhat_for_width = calibrate_qhat_from_residual(residual_cal, alpha=0.1)

    for nt in noise_types:
        for method_name, flags in methods:
            for latent_flag in (False, True):
                # MC on grid vs latent makes no difference (no optimisation step),
                # so skip latent=True for MC to save time.
                if method_name == "MC" and latent_flag:
                    continue
                try:
                    cfg = build_cfg(nt, optimise_in_latent=latent_flag, **flags)
                    # Bound width on a single representative trajectory.
                    bounds = perturbation_bounds_1d(
                        pred_signal=pos[n_cal].numpy(),
                        residual_operator=D_pos,
                        qhat=qhat_for_width,
                        interior_slice=interior,
                        config=cfg,
                    )
                    width = float(np.mean(bounds.width))

                    cov = empirical_coverage_curve_1d(
                        preds=neural_sol[..., 0],
                        truths=numerical_sol[..., 0],
                        residual_cal=residual_cal,
                        operator=D_pos,
                        alphas=alpha_levels,
                        interior_slice=interior,
                        perturbation_config=cfg,
                        cp_mode="marginal",
                    )
                    cov_arr = cov.empirical_coverage_perturbation
                    rows.append((nt, method_name, latent_flag, width, cov_arr))
                    print(f"  {nt:8s} {method_name:5s} latent={str(latent_flag):5s} "
                          f"width={width:.3f}  cov={np.array2string(cov_arr, precision=2)}")
                except Exception as exc:
                    print(f"  {nt:8s} {method_name:5s} latent={latent_flag} FAILED: {exc!r}")
                    rows.append((nt, method_name, latent_flag, float("nan"),
                                 np.full_like(alpha_levels, np.nan)))

    # Summary
    print("\n" + "=" * 78)
    print(f"{'noise':8s} {'method':6s} {'space':6s} {'width':>8s}  " +
          "  ".join(f"cov@α={a:.1f}" for a in alpha_levels) +
          "   valid@α=0.1")
    print("-" * 78)
    for nt, m_name, latent_flag, width, cov_arr in rows:
        space = "latent" if latent_flag else "grid"
        nominal = 1.0 - alpha_levels
        valid_first = "YES" if (not np.isnan(cov_arr[0]) and cov_arr[0] >= nominal[0] - 1e-6) else "NO"
        cov_str = "  ".join(f"   {c:.2f}" for c in cov_arr)
        print(f"{nt:8s} {m_name:6s} {space:6s} {width:8.3f}  {cov_str}     {valid_first}")
    print("=" * 78)


if __name__ == "__main__":
    main()
