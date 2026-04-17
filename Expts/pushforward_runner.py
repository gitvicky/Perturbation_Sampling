"""
Quantile-pushforward inversion: SHO, DHO, Duffing.

For each case:
  1. Train / evaluate the Neural ODE (reusing existing SHO/DHO/Duffing modules).
  2. Compute residuals on an 80/20 calibration/validation split (no GT used in
     calibration — residuals only).
  3. Assemble operator matrix M  (linear stencil for SHO/DHO, Jacobian for
     Duffing, evaluated per trajectory at u_pred).
  4. Mahalanobis CP in residual space -> qhat.
  5. Pushforward Sigma_E = M^+ Sigma_R M^{+T}  and emit pointwise bounds.
  6. Plot bounds for one held-out trajectory and the empirical coverage curve
     against nominal 1 - alpha.

Usage
-----
  source .venv/bin/activate
  python Expts/pushforward_runner.py                # run all three
  python Expts/pushforward_runner.py sho            # only SHO
  python Expts/pushforward_runner.py duffing --shrink 0.1 --tikhonov 1e-4
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Utils.PRE.ConvOps_0d import ConvOperator
from Inversion_Strategies.inversion.pushforward import (
    coverage_curve,
    duffing_jacobian_matrix,
    pushforward_bounds,
    stencil_to_matrix,
)


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
PALETTE = {
    "pushforward": "#C07A52",
    "prediction": "#5B7EC0",
    "truth": "#2C2C2C",
    "target": "#9B9B9B",
}
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.15})
FIG_DIR = os.path.join(os.path.dirname(__file__), "Figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, linewidth=0.5, color="#CCCCCC")


# ---------------------------------------------------------------------------
# Per-case setup
# ---------------------------------------------------------------------------
def _train_and_evaluate_sho():
    from Expts.SHO.SHO_NODE import (
        HarmonicOscillator, ODEFunc, generate_training_data,
        train_neural_ode, evaluate,
    )
    m, k = 1.0, 1.0
    osc = HarmonicOscillator(k, m)
    t_span, n_points, n_traj = (0, 10), 100, 50
    t_train, states, derivs = generate_training_data(osc, t_span, n_points, n_traj)
    func = ODEFunc(hidden_dim=64)
    train_neural_ode(func, t_train, states, derivs, n_epochs=500, batch_size=16)
    t, num_sol, nn_sol = evaluate(osc, func, t_span, n_points,
                                  x_range=(-2, 2), v_range=(-2, 2), n_solves=100)
    dt = float(t[1] - t[0])

    D_tt = ConvOperator(order=2)
    D_id = ConvOperator(order=0); D_id.kernel = torch.tensor([0., 1., 0.])
    D = ConvOperator(conv="spectral")
    D.kernel = m * D_tt.kernel + dt**2 * k * D_id.kernel

    return dict(
        name="SHO", t=t, pred_pos=nn_sol[..., 0], truth_pos=num_sol[..., 0],
        operator=D, linear=True,
    )


def _train_and_evaluate_dho():
    from Expts.DHO.DHO_NODE import (
        DampedHarmonicOscillator, ODEFunc, generate_training_data,
        train_neural_ode, evaluate,
    )
    m, k, c = 1.0, 1.0, 0.2
    osc = DampedHarmonicOscillator(k, m, c)
    t_span, n_points, n_traj = (0, 15), 100, 50
    t_train, states, derivs = generate_training_data(osc, t_span, n_points, n_traj)
    func = ODEFunc(hidden_dim=64)
    train_neural_ode(func, t_train, states, derivs, n_epochs=500, batch_size=16)
    t, num_sol, nn_sol = evaluate(osc, func, t_span, n_points,
                                  x_range=(-2, 2), v_range=(-2, 2), n_solves=100)
    dt = float(t[1] - t[0])

    D_t = ConvOperator(order=1)
    D_tt = ConvOperator(order=2)
    D_id = ConvOperator(order=0); D_id.kernel = torch.tensor([0., 1., 0.])
    D = ConvOperator(conv="spectral")
    D.kernel = 2*m*D_tt.kernel + dt*c*D_t.kernel + 2*dt**2*k*D_id.kernel

    return dict(
        name="DHO", t=t, pred_pos=nn_sol[..., 0], truth_pos=num_sol[..., 0],
        operator=D, linear=True,
    )


def _train_and_evaluate_duffing():
    from Expts.Duffing.Duffing_NODE import (
        DuffingOscillator, DuffingResidualOperator, ODEFunc,
        generate_training_data, train_neural_ode, evaluate,
    )
    a, b, d = 1.0, 0.5, 0.2
    osc = DuffingOscillator(a, b, d)
    t_span, n_points, n_traj = (0, 15), 100, 50
    t_train, states, derivs = generate_training_data(osc, t_span, n_points, n_traj)
    func = ODEFunc(hidden_dim=64)
    train_neural_ode(func, t_train, states, derivs, n_epochs=500, batch_size=16)
    t, num_sol, nn_sol = evaluate(osc, func, t_span, n_points,
                                  x_range=(-2, 2), v_range=(-2, 2), n_solves=100)
    dt = float(t[1] - t[0])

    residual_op = DuffingResidualOperator(a, b, d, dt)
    return dict(
        name="Duffing", t=t, pred_pos=nn_sol[..., 0], truth_pos=num_sol[..., 0],
        operator=residual_op, linear=False,
        duffing_params=(a, b, d, dt),
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_case(case: dict, *, shrink: float, tikhonov: float,
             alphas: np.ndarray) -> None:
    name = case["name"]
    t = case["t"]
    preds = np.asarray(case["pred_pos"], dtype=np.float64)     # (N, T)
    truths = np.asarray(case["truth_pos"], dtype=np.float64)   # (N, T)
    op = case["operator"]

    # Residuals on the full set.
    pos_t = torch.tensor(preds, dtype=torch.float32)
    res_full = op(pos_t).detach().cpu().numpy()
    # Operator output includes boundary rows; keep interior to match inversion.
    res_full = res_full[:, 1:-1]                              # (N, T-2)

    # 80/20 calibration / validation split.
    n_cal = int(0.8 * preds.shape[0])
    residual_cal = res_full[:n_cal]
    preds_val = preds[n_cal:]
    truths_val = truths[n_cal:]
    test_idx = 0                                              # first validation traj

    T = preds.shape[1]
    # Build operator matrix:
    if case["linear"]:
        kernel = op.kernel.detach().cpu().numpy()
        M_const = stencil_to_matrix(kernel, T, interior=True)

        def build_op(u_pred: np.ndarray) -> np.ndarray:
            return M_const
    else:
        a, b, d, dt = case["duffing_params"]

        def build_op(u_pred: np.ndarray) -> np.ndarray:
            return duffing_jacobian_matrix(u_pred, a, b, d, dt)

    # --- Bounds for the showcase trajectory (median alpha=0.1) -------------
    M_show = build_op(preds_val[test_idx])
    show = pushforward_bounds(
        preds_val[test_idx], residual_cal, M_show,
        alpha=0.1, shrink=shrink, tikhonov=tikhonov,
    )
    print(f"[{name}] qhat (alpha=0.1) = {show.qhat:.4f}")
    print(f"[{name}] mean pointwise halfwidth = "
          f"{(show.qhat * show.pointwise_std).mean():.4f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    _style(ax)
    ax.fill_between(t, show.lower, show.upper,
                    color=PALETTE["pushforward"], alpha=0.25,
                    label="Pushforward bound (α=0.1)")
    ax.plot(t, preds_val[test_idx], color=PALETTE["prediction"],
            linewidth=1.8, label="Neural ODE prediction")
    ax.plot(t, truths_val[test_idx], color=PALETTE["truth"],
            linewidth=1.6, marker=".", markersize=3, markevery=5,
            label="Ground truth")
    ax.set_title(f"{name}: Quantile Pushforward Bounds")
    ax.set_xlabel("Time"); ax.set_ylabel("Position")
    ax.legend(loc="best", edgecolor="#DDDDDD")
    fig.savefig(os.path.join(FIG_DIR, f"{name.lower()}_pushforward_bounds.png"))
    plt.close(fig)

    # --- Bounds across alpha (single trajectory) ---------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    _style(ax)
    cmap = plt.get_cmap("magma", len(alphas) + 2)
    for i, alpha in enumerate(sorted(alphas)):
        res = pushforward_bounds(preds_val[test_idx], residual_cal, M_show,
                                 alpha=float(alpha), shrink=shrink,
                                 tikhonov=tikhonov)
        ax.fill_between(t, res.lower, res.upper, color=cmap(i + 1),
                        alpha=0.3, label=f"1-α={1 - alpha:.2f}")
    ax.plot(t, preds_val[test_idx], color=PALETTE["prediction"],
            linewidth=1.8, label="Prediction")
    ax.plot(t, truths_val[test_idx], color=PALETTE["truth"],
            linewidth=1.6, marker=".", markersize=3, markevery=5,
            label="Ground truth")
    ax.set_title(f"{name}: Pushforward Bounds Across α")
    ax.set_xlabel("Time"); ax.set_ylabel("Position")
    ax.legend(loc="best", edgecolor="#DDDDDD", ncol=2, fontsize=8)
    fig.savefig(os.path.join(FIG_DIR, f"{name.lower()}_pushforward_alpha_bounds.png"))
    plt.close(fig)

    # --- Empirical coverage curve -----------------------------------------
    nominal, cov_point, cov_joint = coverage_curve(
        preds_val, truths_val, residual_cal, build_op,
        alphas=alphas, interior_slice=slice(1, -1),
        shrink=shrink, tikhonov=tikhonov,
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    _style(ax)
    ax.plot(nominal, nominal, color=PALETTE["target"], linestyle=":",
            linewidth=1.8, label="Target")
    ax.plot(nominal, cov_point, color=PALETTE["pushforward"], linewidth=2.0,
            marker="o", markersize=4, label="Pointwise coverage")
    ax.plot(nominal, cov_joint, color="#4A6FA5", linewidth=2.0,
            marker="s", markersize=4, label="Joint coverage (all-t)")
    ax.fill_between(nominal, 0, nominal, color=PALETTE["target"], alpha=0.06)
    ax.set_xlabel("Nominal $1-\\alpha$")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(f"{name}: Pushforward Coverage")
    ax.legend(loc="lower right", edgecolor="#DDDDDD")
    fig.savefig(os.path.join(FIG_DIR, f"{name.lower()}_pushforward_coverage.png"))
    plt.close(fig)

    print(f"[{name}] nominal   : {np.round(nominal, 2)}")
    print(f"[{name}] pointwise : {np.round(cov_point, 3)}")
    print(f"[{name}] joint     : {np.round(cov_joint, 3)}\n")


CASES = {
    "sho": _train_and_evaluate_sho,
    "dho": _train_and_evaluate_dho,
    "duffing": _train_and_evaluate_duffing,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Quantile pushforward inversion runner.")
    ap.add_argument("cases", nargs="*", default=list(CASES.keys()),
                    choices=list(CASES.keys()))
    ap.add_argument("--shrink", type=float, default=0.05,
                    help="Covariance shrinkage (Ledoit-Wolf-style, default 0.05).")
    ap.add_argument("--tikhonov", type=float, default=1e-6,
                    help="Relative Tikhonov regularisation for M^+ (default 1e-6).")
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
    args = ap.parse_args()

    alphas = np.asarray(args.alphas, dtype=float)
    for name in args.cases:
        print(f"=== {name.upper()} ===")
        case = CASES[name]()
        run_case(case, shrink=args.shrink, tikhonov=args.tikhonov,
                 alphas=alphas)
