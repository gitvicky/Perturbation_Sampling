"""
Quantile-pushforward inversion: SHO, DHO, Duffing, Advection-1D, Burgers-1D.

ODE cases train/evaluate the neural surrogate. PDE cases use synthetic
surrogate predictions:

    u_pred = u_true + noise

For each case:
  1. Build residuals on an 80/20 calibration/validation split.
  2. Assemble operator matrix M (constant linear stencil or local Jacobian).
  3. Mahalanobis CP in residual space -> qhat.
  4. Pushforward Sigma_E = M^+ Sigma_R M^{+T} and emit pointwise bounds.
  5. Plot one held-out trajectory/field and the empirical coverage curve.

Usage
-----
  source .venv/bin/activate
  python Expts/pushforward_runner.py
  python Expts/pushforward_runner.py sho duffing
  python Expts/pushforward_runner.py advection burgers --noise-std 0.02
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Callable

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Utils.PRE.ConvOps_0d import ConvOperator
from Inversion_Strategies.inversion.pushforward import (
    coverage_curve,
    duffing_jacobian_matrix,
    fft_coverage_curve,
    fft_pushforward_bounds,
    pushforward_bounds,
    stencil_to_matrix,
)


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
PALETTE = {
    "pushforward": "#C07A52",
    "fft": "#6E8B3D",
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
# Matrix assembly helpers (2D spatiotemporal PDEs)
# ---------------------------------------------------------------------------
def _flat_idx(ti: int, xi: int, nx: int) -> int:
    return ti * nx + xi


def _advection_matrix(nt: int, nx: int, dt: float, dx: float, v: float) -> np.ndarray:
    """Dense matrix for interior advection residual: u_t + v*u_x."""
    n_out = (nt - 2) * (nx - 2)
    n_in = nt * nx
    M = np.zeros((n_out, n_in), dtype=np.float64)

    row = 0
    for ti in range(1, nt - 1):
        for xi in range(1, nx - 1):
            M[row, _flat_idx(ti + 1, xi, nx)] += 1.0 / (2.0 * dt)
            M[row, _flat_idx(ti - 1, xi, nx)] += -1.0 / (2.0 * dt)
            M[row, _flat_idx(ti, xi + 1, nx)] += v / (2.0 * dx)
            M[row, _flat_idx(ti, xi - 1, nx)] += -v / (2.0 * dx)
            row += 1
    return M


def _burgers_jacobian_matrix(u_pred: np.ndarray, dt: float, dx: float, nu: float) -> np.ndarray:
    """Dense Jacobian of interior Burgers residual at u_pred."""
    nt, nx = u_pred.shape
    n_out = (nt - 2) * (nx - 2)
    n_in = nt * nx
    M = np.zeros((n_out, n_in), dtype=np.float64)

    row = 0
    for ti in range(1, nt - 1):
        for xi in range(1, nx - 1):
            u_c = float(u_pred[ti, xi])
            u_x = (float(u_pred[ti, xi + 1]) - float(u_pred[ti, xi - 1])) / (2.0 * dx)

            M[row, _flat_idx(ti + 1, xi, nx)] += 1.0 / (2.0 * dt)
            M[row, _flat_idx(ti - 1, xi, nx)] += -1.0 / (2.0 * dt)

            M[row, _flat_idx(ti, xi + 1, nx)] += u_c / (2.0 * dx) - nu / (dx**2)
            M[row, _flat_idx(ti, xi - 1, nx)] += -u_c / (2.0 * dx) - nu / (dx**2)
            M[row, _flat_idx(ti, xi, nx)] += u_x + 2.0 * nu / (dx**2)
            row += 1
    return M


def _coverage_curve_flat(
    preds: np.ndarray,
    truths: np.ndarray,
    residual_cal: np.ndarray,
    build_operator: Callable[[np.ndarray], np.ndarray],
    *,
    alphas: np.ndarray,
    eval_idx: np.ndarray,
    shrink: float,
    tikhonov: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
                u_pred,
                residual_cal,
                M,
                alpha=float(alpha),
                shrink=shrink,
                tikhonov=tikhonov,
            )
            t_eval = truths[i][eval_idx]
            l_eval = res.lower[eval_idx]
            u_eval = res.upper[eval_idx]
            inside = (t_eval >= l_eval) & (t_eval <= u_eval)
            hits_point.append(float(inside.mean()))
            hits_joint.append(float(inside.all()))
        cov_pointwise[ai] = float(np.mean(hits_point))
        cov_joint[ai] = float(np.mean(hits_joint))

    return nominal, cov_pointwise, cov_joint


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


def _make_advection_case(*, noise_std: float = 0.02, n_traj: int = 60, seed: int = 0):
    from Neural_PDE.Numerical_Solvers.Advection.Advection_1D import Advection_1d
    from Expts.Advection.Advection_Model import sample_ic_params, solve_parameterised

    nt, nx = 40, 60
    x_min, x_max, t_end = 0.0, 2.0, 0.5
    v = 1.0
    sim = Advection_1d(nx, nt, x_min, x_max, t_end)
    dt, dx = float(sim.dt), float(sim.dx)

    rng = np.random.default_rng(seed)
    truths = []
    for _ in range(n_traj):
        xc, width, height = sample_ic_params(rng)
        truths.append(solve_parameterised(sim, xc, width, height, v))
    truths = np.asarray(truths, dtype=np.float64)  # (N, Nt, Nx+3)

    preds = truths + noise_std * rng.standard_normal(size=truths.shape)
    nt_eff, nx_eff = truths.shape[1], truths.shape[2]
    M_const = _advection_matrix(nt_eff, nx_eff, dt, dx, v)

    def residual_fn(u: np.ndarray) -> np.ndarray:
        return (
            (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2.0 * dt)
            + v * (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / (2.0 * dx)
        )

    interior_eval_idx = np.arange(nt_eff * nx_eff).reshape(nt_eff, nx_eff)[1:-1, 1:-1].ravel()
    return dict(
        name="Advection1D",
        kind="nd",
        t=np.arange(nt_eff, dtype=np.float64) * dt,
        x=np.asarray(sim.x, dtype=np.float64),
        preds=preds,
        truths=truths,
        residual_fn=residual_fn,
        build_op=lambda _u: M_const,
        interior_eval_idx=interior_eval_idx,
    )


def _make_burgers_case(
    *,
    noise_std: float = 0.02,
    n_traj: int = 60,
    seed: int = 0,
    nu: float = 0.002,
):
    from Expts.Burgers.Burgers_Model import load_or_generate_burgers_dataset

    cache_path = os.path.join(os.path.dirname(__file__), "Burgers", "data", "Burgers_1d_cached.npz")
    all_traj, grid = load_or_generate_burgers_dataset(cache_path=cache_path, n_sims=n_traj, seed=seed)

    # Extra stride keeps the Jacobian matrix size tractable in pushforward inversion.
    t_stride, x_stride = 2, 4
    truths = np.asarray(all_traj[:, ::t_stride, ::x_stride], dtype=np.float64)
    dt = float(grid.dt * t_stride)
    x = np.asarray(grid.x[::x_stride], dtype=np.float64)
    dx = float(x[1] - x[0])

    rng = np.random.default_rng(seed)
    preds = truths + noise_std * rng.standard_normal(size=truths.shape)
    nt_eff, nx_eff = truths.shape[1], truths.shape[2]

    def residual_fn(u: np.ndarray) -> np.ndarray:
        u_c = u[:, 1:-1, 1:-1]
        u_t = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2.0 * dt)
        u_x = (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / (2.0 * dx)
        u_xx = (u[:, 1:-1, 2:] - 2.0 * u_c + u[:, 1:-1, :-2]) / (dx**2)
        return u_t + u_c * u_x - nu * u_xx

    def build_op(u_pred: np.ndarray) -> np.ndarray:
        return _burgers_jacobian_matrix(u_pred, dt=dt, dx=dx, nu=nu)

    interior_eval_idx = np.arange(nt_eff * nx_eff).reshape(nt_eff, nx_eff)[1:-1, 1:-1].ravel()
    return dict(
        name="Burgers1D",
        kind="nd",
        t=np.arange(nt_eff, dtype=np.float64) * dt,
        x=x,
        preds=preds,
        truths=truths,
        residual_fn=residual_fn,
        build_op=build_op,
        interior_eval_idx=interior_eval_idx,
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

    # FFT pushforward only makes sense for stationary (linear, constant-kernel)
    # operators. Skip for Duffing where the Jacobian is u_pred-dependent.
    fft_show = None
    fft_kernel = None
    if case["linear"]:
        fft_kernel = op.kernel.detach().cpu().numpy()
        fft_show = fft_pushforward_bounds(
            preds_val[test_idx], residual_cal, fft_kernel,
            alpha=0.1, lam=tikhonov,
        )
        print(f"[{name}] FFT qhat (alpha=0.1) = {fft_show.qhat:.4f}")
        print(f"[{name}] FFT halfwidth = "
              f"{(fft_show.qhat * fft_show.pointwise_std).mean():.4f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    _style(ax)
    ax.fill_between(t, show.lower, show.upper,
                    color=PALETTE["pushforward"], alpha=0.25,
                    label="Dense pushforward (α=0.1)")
    if fft_show is not None:
        ax.plot(t, fft_show.lower, color=PALETTE["fft"], linewidth=1.4,
                linestyle="--", label="FFT pushforward (α=0.1)")
        ax.plot(t, fft_show.upper, color=PALETTE["fft"], linewidth=1.4,
                linestyle="--")
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
    if fft_kernel is not None:
        _, fft_cov_point, fft_cov_joint = fft_coverage_curve(
            preds_val, truths_val, residual_cal, fft_kernel,
            alphas=alphas, interior_slice=slice(1, -1), lam=tikhonov,
        )
    else:
        fft_cov_point = fft_cov_joint = None

    fig, ax = plt.subplots(figsize=(8, 6))
    _style(ax)
    ax.plot(nominal, nominal, color=PALETTE["target"], linestyle=":",
            linewidth=1.8, label="Target")
    ax.plot(nominal, cov_point, color=PALETTE["pushforward"], linewidth=2.0,
            marker="o", markersize=4, label="Dense pointwise")
    ax.plot(nominal, cov_joint, color="#4A6FA5", linewidth=2.0,
            marker="s", markersize=4, label="Dense joint")
    if fft_cov_point is not None:
        ax.plot(nominal, fft_cov_point, color=PALETTE["fft"], linewidth=2.0,
                linestyle="--", marker="^", markersize=4, label="FFT pointwise")
        ax.plot(nominal, fft_cov_joint, color="#3E5F25", linewidth=2.0,
                linestyle="--", marker="v", markersize=4, label="FFT joint")
    ax.fill_between(nominal, 0, nominal, color=PALETTE["target"], alpha=0.06)
    ax.set_xlabel("Nominal $1-\\alpha$")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(f"{name}: Pushforward Coverage")
    ax.legend(loc="lower right", edgecolor="#DDDDDD", fontsize=8)
    fig.savefig(os.path.join(FIG_DIR, f"{name.lower()}_pushforward_coverage.png"))
    plt.close(fig)

    print(f"[{name}] nominal    : {np.round(nominal, 2)}")
    print(f"[{name}] dense pt   : {np.round(cov_point, 3)}")
    print(f"[{name}] dense jt   : {np.round(cov_joint, 3)}")
    if fft_cov_point is not None:
        print(f"[{name}] fft pt     : {np.round(fft_cov_point, 3)}")
        print(f"[{name}] fft jt     : {np.round(fft_cov_joint, 3)}")
    print()


def run_case_nd(case: dict, *, shrink: float, tikhonov: float, alphas: np.ndarray) -> None:
    name = case["name"]
    t = np.asarray(case["t"], dtype=np.float64)
    x = np.asarray(case["x"], dtype=np.float64)
    preds = np.asarray(case["preds"], dtype=np.float64)   # (N, Nt, Nx)
    truths = np.asarray(case["truths"], dtype=np.float64) # (N, Nt, Nx)
    residual_fn = case["residual_fn"]
    build_op = case["build_op"]
    eval_idx = np.asarray(case["interior_eval_idx"], dtype=np.int64)

    res_full = residual_fn(preds)                          # (N, Nt-2, Nx-2)
    n_cal = int(0.8 * preds.shape[0])
    residual_cal = res_full[:n_cal].reshape(n_cal, -1)
    preds_val = preds[n_cal:]
    truths_val = truths[n_cal:]
    test_idx = 0

    nt, nx = preds.shape[1], preds.shape[2]
    u_show = preds_val[test_idx]
    M_show = build_op(u_show)
    show = pushforward_bounds(
        u_show.reshape(-1),
        residual_cal,
        M_show,
        alpha=0.1,
        shrink=shrink,
        tikhonov=tikhonov,
    )
    lower = show.lower.reshape(nt, nx)
    upper = show.upper.reshape(nt, nx)

    mid_t = nt // 2
    x_interior = x[1:-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    _style(ax)
    ax.fill_between(
        x_interior,
        lower[mid_t, 1:-1],
        upper[mid_t, 1:-1],
        color=PALETTE["pushforward"],
        alpha=0.25,
        label="Pushforward bound (alpha=0.1)",
    )
    ax.plot(x_interior, preds_val[test_idx][mid_t, 1:-1], color=PALETTE["prediction"], linewidth=1.8, label="Prediction")
    ax.plot(x_interior, truths_val[test_idx][mid_t, 1:-1], color=PALETTE["truth"], linewidth=1.6, label="Ground truth")
    ax.set_title(f"{name}: Mid-time Pushforward Bounds")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.legend(loc="best", edgecolor="#DDDDDD")
    fig.savefig(os.path.join(FIG_DIR, f"{name.lower()}_pushforward_bounds.png"))
    plt.close(fig)

    preds_val_flat = preds_val.reshape(preds_val.shape[0], -1)
    truths_val_flat = truths_val.reshape(truths_val.shape[0], -1)
    nominal, cov_point, cov_joint = _coverage_curve_flat(
        preds=preds_val_flat,
        truths=truths_val_flat,
        residual_cal=residual_cal,
        build_operator=lambda u_flat: build_op(u_flat.reshape(nt, nx)),
        alphas=alphas,
        eval_idx=eval_idx,
        shrink=shrink,
        tikhonov=tikhonov,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    _style(ax)
    ax.plot(nominal, nominal, color=PALETTE["target"], linestyle=":", linewidth=1.8, label="Target")
    ax.plot(nominal, cov_point, color=PALETTE["pushforward"], linewidth=2.0, marker="o", markersize=4, label="Pointwise coverage")
    ax.plot(nominal, cov_joint, color="#4A6FA5", linewidth=2.0, marker="s", markersize=4, label="Joint coverage (all cells)")
    ax.fill_between(nominal, 0, nominal, color=PALETTE["target"], alpha=0.06)
    ax.set_xlabel("Nominal $1-\\alpha$")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(f"{name}: Pushforward Coverage")
    ax.legend(loc="lower right", edgecolor="#DDDDDD")
    fig.savefig(os.path.join(FIG_DIR, f"{name.lower()}_pushforward_coverage.png"))
    plt.close(fig)

    print(f"[{name}] qhat (alpha=0.1) = {show.qhat:.4f}")
    print(f"[{name}] mean pointwise halfwidth = {(show.qhat * show.pointwise_std).mean():.4f}")
    print(f"[{name}] nominal   : {np.round(nominal, 2)}")
    print(f"[{name}] pointwise : {np.round(cov_point, 3)}")
    print(f"[{name}] joint     : {np.round(cov_joint, 3)}\n")


ODE_CASES = {
    "sho": _train_and_evaluate_sho,
    "dho": _train_and_evaluate_dho,
    "duffing": _train_and_evaluate_duffing,
}
PDE_CASES = ("advection", "burgers")
ALL_CASES = list(ODE_CASES.keys()) + list(PDE_CASES)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Quantile pushforward inversion runner.")
    ap.add_argument("cases", nargs="*", default=ALL_CASES, choices=ALL_CASES)
    ap.add_argument("--shrink", type=float, default=0.05,
                    help="Covariance shrinkage (Ledoit-Wolf-style, default 0.05).")
    ap.add_argument("--tikhonov", type=float, default=1e-6,
                    help="Relative Tikhonov regularisation for M^+ (default 1e-6).")
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
    ap.add_argument(
        "--noise-std",
        type=float,
        default=0.02,
        help="Std of additive synthetic noise for PDE pushforward cases (u_pred=u_true+noise).",
    )
    args = ap.parse_args()

    alphas = np.asarray(args.alphas, dtype=float)
    for name in args.cases:
        print(f"=== {name.upper()} ===")
        if name in ODE_CASES:
            run_case(
                ODE_CASES[name](),
                shrink=args.shrink,
                tikhonov=args.tikhonov,
                alphas=alphas,
            )
        elif name == "advection":
            run_case_nd(
                _make_advection_case(noise_std=args.noise_std),
                shrink=args.shrink,
                tikhonov=args.tikhonov,
                alphas=alphas,
            )
        elif name == "burgers":
            run_case_nd(
                _make_burgers_case(noise_std=args.noise_std),
                shrink=args.shrink,
                tikhonov=args.tikhonov,
                alphas=alphas,
            )
