"""
Experiment runner for residual bound inversion on Neural ODE test cases.

Each experiment follows the same pipeline:
  1. Train a Neural ODE on synthetic trajectories
  2. Build a composite FD-stencil kernel encoding the ODE
  3. Compute residuals and calibrate qhat via conformal prediction
  4. Invert residual bounds to physical space (point-wise, interval FFT, perturbation)
  5. Plot bound comparisons and empirical coverage curves

Linear ODEs (SHO, DHO): all three inversion methods apply.
Nonlinear ODEs (Pendulum, Duffing): only perturbation sampling is valid,
  since the convolution theorem requires linearity.

Usage:
  python Expts/experiment_runner.py                        # run all, default spatial noise
  python Expts/experiment_runner.py sho                    # run only SHO
  python Expts/experiment_runner.py --noise-type bspline   # run all with B-spline noise
  python Expts/experiment_runner.py sho --noise-type gp    # SHO with GP noise
  python Expts/experiment_runner.py pendulum duffing       # run only nonlinear cases
"""
# %%a
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Utils.PRE.ConvOps_0d import ConvOperator

from Inversion_Strategies.inversion.residual_inversion import (
    IntervalFFTSlicing,
    PerturbationSamplingConfig,
    calibrate_qhat_from_residual,
    empirical_coverage_curve_1d,
    invert_residual_bounds_1d,
    perturbation_bounds_1d,
)

# -- Shared plot style ---------------------------------------------------------
PALETTE = {
    'pointwise':    '#E07A5F',   # terra cotta
    'intervalfft':  '#3D405B',   # charcoal indigo
    'perturbation': '#81B29A',   # sage green
    'truth':        '#2C2C2C',   # near-black
    'prediction':   '#5B7EC0',   # steel blue
    'target':       '#9B9B9B',   # warm grey
}

plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})


NOISE_TYPES = ("spatial", "white", "gp", "bspline")


def _build_perturbation_config(noise_type, seed=123):
    """Build a PerturbationSamplingConfig for the given noise type."""
    common = dict(
        n_samples=20000,
        batch_size=1000,
        max_rounds=5,
        noise_std=0.5,
        seed=seed,
    )
    if noise_type == "spatial":
        return PerturbationSamplingConfig(noise_type="spatial", correlation_length=24.0, **common)
    elif noise_type == "white":
        return PerturbationSamplingConfig(noise_type="white", **common)
    elif noise_type == "gp":
        return PerturbationSamplingConfig(noise_type="gp", correlation_length=24.0, gp_kernel="rbf", **common)
    elif noise_type == "bspline":
        return PerturbationSamplingConfig(noise_type="bspline", bspline_n_knots=16, **common)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}. Choose from {NOISE_TYPES}")


def _style_ax(ax):
    """Apply clean spine and grid styling to an axis."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)
    ax.grid(True, alpha=0.25, linewidth=0.5, color='#CCCCCC')
    ax.tick_params(direction='out', length=4, width=0.6)

#Empirical Coverage Plotting
def _save_coverage_plot(case_name, coverage_result, save_name):
    """Plot empirical vs nominal coverage for all three inversion methods."""
    fig, ax = plt.subplots(figsize=(8, 6))
    _style_ax(ax)

    nom = coverage_result.nominal_coverage

    # Ideal diagonal
    ax.plot(nom, nom, color=PALETTE['target'], linewidth=1.8, linestyle=':',
            label='Target Coverage', zorder=1)

    # Point-wise
    ax.plot(nom, coverage_result.empirical_coverage_pointwise,
            color=PALETTE['pointwise'], linewidth=2.0, linestyle='--',
            marker='s', markersize=5, markeredgewidth=0,
            label='Point-wise Inversion', zorder=3)

    # Interval FFT
    ax.plot(nom, coverage_result.empirical_coverage_intervalfft,
            color=PALETTE['intervalfft'], linewidth=2.2,
            marker='o', markersize=5, markeredgewidth=0,
            label='Interval FFT Set Propagation', zorder=4)

    # Perturbation sampling
    if coverage_result.empirical_coverage_perturbation is not None:
        ax.plot(nom, coverage_result.empirical_coverage_perturbation,
                color=PALETTE['perturbation'], linewidth=2.0, linestyle='-.',
                marker='D', markersize=4.5, markeredgewidth=0,
                label='Perturbation Sampling', zorder=3)

    # Shade undercoverage region
    ax.fill_between(nom, 0, nom, color=PALETTE['target'], alpha=0.06, zorder=0)

    ax.set_xlabel('Nominal Coverage $(1 - \\alpha)$')
    ax.set_ylabel('Empirical Coverage')
    ax.set_title(f'{case_name}: Empirical Coverage Comparison')
    ax.legend(loc='lower right', edgecolor='#DDDDDD')

    save_path = os.path.join(os.path.dirname(__file__), '..', 'Paper', 'images', save_name)
    fig.savefig(save_path)
    plt.close(fig)


def _save_coverage_plot_nonlinear(case_name, nominal, empirical_perturbation, save_name):
    """Plot empirical vs nominal coverage for nonlinear cases (perturbation only)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    _style_ax(ax)

    nom = np.asarray(nominal, dtype=float)
    emp = np.asarray(empirical_perturbation, dtype=float)

    # Ideal diagonal
    ax.plot(nom, nom, color=PALETTE['target'], linewidth=1.8, linestyle=':',
            label='Target Coverage', zorder=1)

    # Perturbation sampling
    ax.plot(nom, emp,
            color=PALETTE['perturbation'], linewidth=2.0, linestyle='-.',
            marker='D', markersize=4.5, markeredgewidth=0,
            label='Perturbation Sampling', zorder=3)

    # Shade undercoverage region
    ax.fill_between(nom, 0, nom, color=PALETTE['target'], alpha=0.06, zorder=0)

    ax.set_xlabel('Nominal Coverage $(1 - \\alpha)$')
    ax.set_ylabel('Empirical Coverage')
    ax.set_title(f'{case_name}: Empirical Coverage (Nonlinear — Perturbation Only)')
    ax.legend(loc='lower right', edgecolor='#DDDDDD')

    save_path = os.path.join(os.path.dirname(__file__), '..', 'Paper', 'images', save_name)
    fig.savefig(save_path)
    plt.close(fig)


def _perturbation_coverage_curve(
    preds, truths, residual_cal, residual_operator, *,
    alphas, interior_slice, perturbation_config,
):
    """Compute empirical coverage curve using perturbation sampling only.

    Used for nonlinear ODEs where point-wise / interval FFT inversion
    is not applicable.
    """
    from Inversion_Strategies.inversion.residual_inversion import (
        _trajectory_coverage_1d,
    )
    nominal = []
    cov_perturb = []

    preds = np.asarray(preds, dtype=float)
    truths = np.asarray(truths, dtype=float)

    for alpha in tqdm(alphas, desc="Coverage alphas", unit="α"):
        qhat = calibrate_qhat_from_residual(residual_cal, alpha=float(alpha))
        cover_flags = []

        for i in tqdm(range(preds.shape[0]),
                      desc=f"  α={float(alpha):.2f} trajectories",
                      leave=False, unit="traj"):
            perturb_bounds = perturbation_bounds_1d(
                pred_signal=preds[i],
                residual_operator=residual_operator,
                qhat=qhat,
                interior_slice=interior_slice,
                config=perturbation_config,
            )
            truth_i = truths[i][interior_slice]
            cover_flags.append(
                _trajectory_coverage_1d(truth_i[None, :], perturb_bounds)
            )

        nominal.append(1.0 - float(alpha))
        cov_perturb.append(float(np.mean(cover_flags)))

    return np.asarray(nominal), np.asarray(cov_perturb)


def run_sho(transductive=False, noise_type="spatial"):
    """Simple Harmonic Oscillator: m*x'' + k*x = 0"""
    from Expts.SHO.SHO_NODE import HarmonicOscillator, ODEFunc, generate_training_data, train_neural_ode, evaluate
    print(f"Running SHO Experiment ({'transductive' if transductive else 'inductive'}, noise={noise_type})...")

    # --- 1. Train Neural ODE ---
    m, k = 1.0, 1.0
    oscillator = HarmonicOscillator(k, m)

    t_span = (0, 10)
    n_points = 100
    n_trajectories = 50
    t_train, states, derivs = generate_training_data(
        oscillator, t_span, n_points, n_trajectories)

    func = ODEFunc(hidden_dim=64)
    train_neural_ode(func, t_train, states, derivs, n_epochs=500, batch_size=16)

    t, numerical_sol, neural_sol = evaluate(
        oscillator, func, t_span, n_points, x_range=(-2,2), v_range=(-2,2), n_solves=100)

    pos = torch.tensor(neural_sol[...,0], dtype=torch.float32)
    dt = t[1]-t[0]

    # --- 2. Build composite kernel: m*D_tt + dt^2*k*I ---
    D_tt = ConvOperator(order=2)
    D_identity = ConvOperator(order=0)
    D_identity.kernel = torch.tensor([0.0, 1.0, 0.0])

    D_pos = ConvOperator(conv='spectral')
    D_pos.kernel = m*D_tt.kernel + dt**2*k*D_identity.kernel

    # --- 3. Compute residuals and calibrate qhat ---
    res = D_pos(pos)
    if transductive:
        residual_cal = res           # all data for calibration
        test_idx = 0                 # predict on first sample
    else:
        n_cal = int(0.8 * len(res))
        residual_cal = res[:n_cal]   # calibration set (80%)
        test_idx = n_cal             # first held-out sample

    qhat = calibrate_qhat_from_residual(residual_cal, alpha=0.1)
    print(f"  [1/5] Calibrated qhat = {qhat:.4f}")

    # --- 4. Invert bounds (all three methods) ---
    print("  [2/5] Point-wise + Interval FFT inversion...")
    point_bounds, interval_bounds = invert_residual_bounds_1d(
        pred_signal=pos[test_idx].numpy(),
        kernel=D_pos.kernel.numpy(),
        qhat=qhat,
        operator=D_pos,
        interior_slice=slice(1, -1),
        intervalfft_slicing=IntervalFFTSlicing(center_start=3, center_end=-1, n_right_edges=3, output_offset=2),
        integrate_slice_pad=False,
    )
    print("  [2/5] Point-wise + Interval FFT inversion... done")

    print(f"  [3/5] Perturbation sampling inversion ({noise_type})...")
    perturb_cfg = _build_perturbation_config(noise_type, seed=123)
    perturb_bounds = perturbation_bounds_1d(
        pred_signal=pos[test_idx].numpy(),
        residual_operator=D_pos,
        qhat=qhat,
        interior_slice=slice(1, -1),
        config=perturb_cfg,
    )
    print(f"  [3/5] Perturbation sampling inversion ({noise_type})... done")

    # Verify round-trip: differentiate then integrate should recover the signal
    pos_res = D_pos.differentiate(pos, correlation=False, slice_pad=False)
    pos_retrieved = D_pos.integrate(pos_res, correlation=False, slice_pad=False)
    assert torch.allclose(pos, pos_retrieved[:, 1:-1], atol=1e-3), \
        f"Round-trip failed: max error = {(pos - pos_retrieved[:, 1:-1]).abs().max().item():.2e}"

    # --- 5. Plot bounds comparison ---
    print("  [4/5] Saving bounds comparison plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    _style_ax(ax)
    tt = t[1:-1]

    # Interval FFT fill (drawn first so it sits behind)
    ax.fill_between(tt, interval_bounds.lower, interval_bounds.upper,
                    color=PALETTE['intervalfft'], alpha=0.15, zorder=1,
                    label='Interval FFT Set Bound')
    ax.plot(tt, interval_bounds.lower, color=PALETTE['intervalfft'],
            linewidth=1.0, alpha=0.5, zorder=2)
    ax.plot(tt, interval_bounds.upper, color=PALETTE['intervalfft'],
            linewidth=1.0, alpha=0.5, zorder=2)

    # Point-wise bounds
    ax.plot(tt, point_bounds.lower, color=PALETTE['pointwise'], linestyle='--',
            linewidth=1.6, zorder=3, label='Point-wise Bound')
    ax.plot(tt, point_bounds.upper, color=PALETTE['pointwise'], linestyle='--',
            linewidth=1.6, zorder=3)

    # Perturbation bounds
    ax.plot(tt, perturb_bounds.lower, color=PALETTE['perturbation'], linestyle='-.',
            linewidth=1.6, zorder=3, label=f'Perturbation Bound ({noise_type})')
    ax.plot(tt, perturb_bounds.upper, color=PALETTE['perturbation'], linestyle='-.',
            linewidth=1.6, zorder=3)

    # Signals on top
    ax.plot(tt, pos[test_idx, 1:-1].numpy(), color=PALETTE['prediction'],
            linewidth=1.8, zorder=5, label='Neural ODE Prediction')
    ax.plot(tt, numerical_sol[test_idx, 1:-1, 0], color=PALETTE['truth'],
            linewidth=1.8, marker='.', markersize=3, markevery=5,
            zorder=6, label='Ground Truth')

    ax.set_xlabel('Time')
    ax.set_ylabel('Position')
    ax.set_title('SHO: Physical Bounds Comparison')
    ax.legend(loc='best', edgecolor='#DDDDDD', ncol=2)
    save_path = os.path.join(os.path.dirname(__file__), '..', 'Paper', 'images', 'sho_bounds_comparison.png')
    fig.savefig(save_path)
    plt.close(fig)

    # --- 6. Empirical coverage curve across alpha levels ---
    print("  [5/5] Computing empirical coverage curves...")
    alpha_levels = np.arange(0.10, 0.91, 0.10)
    sho_coverage = empirical_coverage_curve_1d(
        preds=neural_sol[..., 0],
        truths=numerical_sol[..., 0],
        residual_cal=residual_cal,
        kernel=D_pos.kernel.numpy(),
        operator=D_pos,
        alphas=alpha_levels,
        interior_slice=slice(1, -1),
        intervalfft_slicing=IntervalFFTSlicing(center_start=3, center_end=-1, n_right_edges=3, output_offset=2),
        integrate_slice_pad=False,
        perturbation_config=perturb_cfg,
    )
    _save_coverage_plot("SHO", sho_coverage, "sho_coverage_comparison.png")
    print("  SHO experiment complete.\n")


def run_dho(transductive=False, noise_type="spatial"):
    """Damped Harmonic Oscillator: m*x'' + c*x' + k*x = 0"""
    from Expts.DHO.DHO_NODE import DampedHarmonicOscillator, ODEFunc, generate_training_data, train_neural_ode, evaluate
    print(f"Running DHO Experiment ({'transductive' if transductive else 'inductive'}, noise={noise_type})...")

    # --- 1. Train Neural ODE ---
    m, k, c = 1.0, 1.0, 0.2
    oscillator = DampedHarmonicOscillator(k, m, c)

    t_span = (0, 15)
    n_points = 100
    n_trajectories = 50

    t_train, states, derivs = generate_training_data(
        oscillator, t_span, n_points, n_trajectories)

    func = ODEFunc(hidden_dim=64)
    train_neural_ode(func, t_train, states, derivs, n_epochs=500, batch_size=16)

    t, numerical_sol, neural_sol = evaluate(
        oscillator, func, t_span, n_points, x_range=(-2,2), v_range=(-2,2), n_solves=100)

    pos = torch.tensor(neural_sol[...,0], dtype=torch.float32)
    dt = t[1]-t[0]

    # --- 2. Build composite kernel: 2m*D_tt + dt*c*D_t + 2*dt^2*k*I ---
    D_t = ConvOperator(order=1)
    D_tt = ConvOperator(order=2)
    D_identity = ConvOperator(order=0)
    D_identity.kernel = torch.tensor([0.0, 1.0, 0.0])

    D_damped = ConvOperator(conv='spectral')
    D_damped.kernel = 2*m*D_tt.kernel + dt*c*D_t.kernel + 2*dt**2*k*D_identity.kernel

    # --- 3. Compute residuals and calibrate qhat ---
    res = D_damped(pos)
    if transductive:
        residual_cal = res           # all data for calibration
        test_idx = 0                 # predict on first sample
    else:
        n_cal = int(0.8 * len(res))
        residual_cal = res[:n_cal]   # calibration set (80%)
        test_idx = n_cal             # first held-out sample

    qhat = calibrate_qhat_from_residual(residual_cal, alpha=0.1)
    print(f"  [1/5] Calibrated qhat = {qhat:.4f}")

    # --- 4. Invert bounds (all three methods) ---
    print("  [2/5] Point-wise + Interval FFT inversion...")
    point_bounds, interval_bounds = invert_residual_bounds_1d(
        pred_signal=pos[test_idx].numpy(),
        kernel=D_damped.kernel.numpy(),
        qhat=qhat,
        operator=D_damped,
        interior_slice=slice(1, -1),
        intervalfft_slicing=IntervalFFTSlicing(center_start=3, center_end=-1, n_right_edges=3, output_offset=2),
        integrate_slice_pad=False,
    )
    print("  [2/5] Point-wise + Interval FFT inversion... done")

    print(f"  [3/5] Perturbation sampling inversion ({noise_type})...")
    perturb_cfg = _build_perturbation_config(noise_type, seed=321)
    perturb_bounds = perturbation_bounds_1d(
        pred_signal=pos[test_idx].numpy(),
        residual_operator=D_damped,
        qhat=qhat,
        interior_slice=slice(1, -1),
        config=perturb_cfg,
    )
    print(f"  [3/5] Perturbation sampling inversion ({noise_type})... done")

    # Verify round-trip: differentiate then integrate should recover the signal
    pos_res = D_damped.differentiate(pos, correlation=False, slice_pad=False)
    pos_retrieved = D_damped.integrate(pos_res, correlation=False, slice_pad=False)
    assert torch.allclose(pos, pos_retrieved[:, 1:-1], atol=1e-3), \
        f"Round-trip failed: max error = {(pos - pos_retrieved[:, 1:-1]).abs().max().item():.2e}"


    # --- 5. Plot bounds comparison ---
    print("  [4/5] Saving bounds comparison plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    _style_ax(ax)
    tt = t[1:-1]

    # Interval FFT fill (drawn first so it sits behind)
    ax.fill_between(tt, interval_bounds.lower, interval_bounds.upper,
                    color=PALETTE['intervalfft'], alpha=0.15, zorder=1,
                    label='Interval FFT Set Bound')
    ax.plot(tt, interval_bounds.lower, color=PALETTE['intervalfft'],
            linewidth=1.0, alpha=0.5, zorder=2)
    ax.plot(tt, interval_bounds.upper, color=PALETTE['intervalfft'],
            linewidth=1.0, alpha=0.5, zorder=2)

    # Point-wise bounds
    ax.plot(tt, point_bounds.lower, color=PALETTE['pointwise'], linestyle='--',
            linewidth=1.6, zorder=3, label='Point-wise Bound')
    ax.plot(tt, point_bounds.upper, color=PALETTE['pointwise'], linestyle='--',
            linewidth=1.6, zorder=3)

    # Perturbation bounds
    ax.plot(tt, perturb_bounds.lower, color=PALETTE['perturbation'], linestyle='-.',
            linewidth=1.6, zorder=3, label=f'Perturbation Bound ({noise_type})')
    ax.plot(tt, perturb_bounds.upper, color=PALETTE['perturbation'], linestyle='-.',
            linewidth=1.6, zorder=3)

    # Signals on top
    ax.plot(tt, pos[test_idx, 1:-1].numpy(), color=PALETTE['prediction'],
            linewidth=1.8, zorder=5, label='Neural ODE Prediction')
    ax.plot(tt, numerical_sol[test_idx, 1:-1, 0], color=PALETTE['truth'],
            linewidth=1.8, marker='.', markersize=3, markevery=5,
            zorder=6, label='Ground Truth')

    ax.set_xlabel('Time')
    ax.set_ylabel('Position')
    ax.set_title('DHO: Physical Bounds Comparison')
    ax.legend(loc='best', edgecolor='#DDDDDD', ncol=2)
    save_path = os.path.join(os.path.dirname(__file__), '..', 'Paper', 'images', 'dho_bounds_comparison.png')
    fig.savefig(save_path)
    plt.close(fig)

    # --- 6. Empirical coverage curve across alpha levels ---
    print("  [5/5] Computing empirical coverage curves...")
    alpha_levels = np.arange(0.10, 0.91, 0.10)
    dho_coverage = empirical_coverage_curve_1d(
        preds=neural_sol[..., 0],
        truths=numerical_sol[..., 0],
        residual_cal=residual_cal,
        kernel=D_damped.kernel.numpy(),
        operator=D_damped,
        alphas=alpha_levels,
        interior_slice=slice(1, -1),
        intervalfft_slicing=IntervalFFTSlicing(center_start=3, center_end=-1, n_right_edges=3, output_offset=2),
        integrate_slice_pad=False,
        perturbation_config=perturb_cfg,
    )
    _save_coverage_plot("DHO", dho_coverage, "dho_coverage_comparison.png")
    print("  DHO experiment complete.\n")


def run_pendulum(transductive=False, noise_type="spatial"):
    """Nonlinear Pendulum: x'' + (g/L)*sin(x) = 0"""
    from Expts.Pendulum.Pendulum_NODE import (
        NonlinearPendulum, NonlinearPendulumResidualOperator,
        ODEFunc, generate_training_data, train_neural_ode, evaluate,
    )
    print(f"Running Pendulum Experiment ({'transductive' if transductive else 'inductive'}, noise={noise_type})...")

    # --- 1. Train Neural ODE ---
    g, L = 9.81, 9.81   # g/L = 1.0 for unit natural frequency
    pendulum = NonlinearPendulum(g, L)

    t_span = (0, 10)
    n_points = 100
    n_trajectories = 50

    t_train, states, derivs = generate_training_data(
        pendulum, t_span, n_points, n_trajectories,
        x_range=(-np.pi/2, np.pi/2), v_range=(-1, 1))

    func = ODEFunc(hidden_dim=64)
    train_neural_ode(func, t_train, states, derivs, n_epochs=500, batch_size=16)

    t, numerical_sol, neural_sol = evaluate(
        pendulum, func, t_span, n_points,
        x_range=(-np.pi/2, np.pi/2), v_range=(-1, 1), n_solves=100)

    pos = torch.tensor(neural_sol[..., 0], dtype=torch.float32)
    dt = t[1] - t[0]

    # --- 2. Build nonlinear residual operator ---
    # The pendulum residual x'' + (g/L)*sin(x) is nonlinear, so we cannot
    # express it as a single convolution kernel. Point-wise and interval FFT
    # inversion require linearity — only perturbation sampling is valid here.
    residual_op = NonlinearPendulumResidualOperator(g, L, dt)

    # --- 3. Compute residuals and calibrate qhat ---
    res = residual_op(pos)
    if transductive:
        residual_cal = res
        test_idx = 0
    else:
        n_cal = int(0.8 * len(res))
        residual_cal = res[:n_cal]
        test_idx = n_cal

    qhat = calibrate_qhat_from_residual(residual_cal, alpha=0.1)
    print(f"  [1/4] Calibrated qhat = {qhat:.4f}")

    # --- 4. Invert bounds (perturbation sampling only) ---
    print(f"  [2/4] Perturbation sampling inversion ({noise_type})...")
    perturb_cfg = _build_perturbation_config(noise_type, seed=456)
    perturb_bounds = perturbation_bounds_1d(
        pred_signal=pos[test_idx].numpy(),
        residual_operator=residual_op,
        qhat=qhat,
        interior_slice=slice(1, -1),
        config=perturb_cfg,
    )
    print(f"  [2/4] Perturbation sampling inversion ({noise_type})... done")

    # --- 5. Plot bounds comparison ---
    print("  [3/4] Saving bounds comparison plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    _style_ax(ax)
    tt = t[1:-1]

    # Perturbation bounds fill
    ax.fill_between(tt, perturb_bounds.lower, perturb_bounds.upper,
                    color=PALETTE['perturbation'], alpha=0.15, zorder=1,
                    label=f'Perturbation Bound ({noise_type})')
    ax.plot(tt, perturb_bounds.lower, color=PALETTE['perturbation'],
            linewidth=1.0, alpha=0.5, zorder=2)
    ax.plot(tt, perturb_bounds.upper, color=PALETTE['perturbation'],
            linewidth=1.0, alpha=0.5, zorder=2)

    # Signals on top
    ax.plot(tt, pos[test_idx, 1:-1].numpy(), color=PALETTE['prediction'],
            linewidth=1.8, zorder=5, label='Neural ODE Prediction')
    ax.plot(tt, numerical_sol[test_idx, 1:-1, 0], color=PALETTE['truth'],
            linewidth=1.8, marker='.', markersize=3, markevery=5,
            zorder=6, label='Ground Truth')

    ax.set_xlabel('Time')
    ax.set_ylabel('Angle (rad)')
    ax.set_title('Nonlinear Pendulum: Physical Bounds (Perturbation Sampling)')
    ax.legend(loc='best', edgecolor='#DDDDDD', ncol=2)
    save_path = os.path.join(os.path.dirname(__file__), '..', 'Paper', 'images', 'pendulum_bounds_comparison.png')
    fig.savefig(save_path)
    plt.close(fig)

    # --- 6. Empirical coverage curve across alpha levels ---
    print("  [4/4] Computing empirical coverage curves (perturbation only)...")
    alpha_levels = np.arange(0.10, 0.91, 0.10)
    nominal, emp_perturb = _perturbation_coverage_curve(
        preds=neural_sol[..., 0],
        truths=numerical_sol[..., 0],
        residual_cal=residual_cal,
        residual_operator=residual_op,
        alphas=alpha_levels,
        interior_slice=slice(1, -1),
        perturbation_config=perturb_cfg,
    )
    _save_coverage_plot_nonlinear("Pendulum", nominal, emp_perturb,
                                  "pendulum_coverage_comparison.png")
    print("  Pendulum experiment complete.\n")


def run_duffing(transductive=False, noise_type="spatial"):
    """Duffing Oscillator: x'' + delta*x' + alpha*x + beta*x^3 = 0"""
    from Expts.Duffing.Duffing_NODE import (
        DuffingOscillator, DuffingResidualOperator,
        ODEFunc, generate_training_data, train_neural_ode, evaluate,
    )
    print(f"Running Duffing Experiment ({'transductive' if transductive else 'inductive'}, noise={noise_type})...")

    # --- 1. Train Neural ODE ---
    alpha_coeff, beta_coeff, delta_coeff = 1.0, 0.5, 0.2
    oscillator = DuffingOscillator(alpha_coeff, beta_coeff, delta_coeff)

    t_span = (0, 15)
    n_points = 100
    n_trajectories = 50

    t_train, states, derivs = generate_training_data(
        oscillator, t_span, n_points, n_trajectories)

    func = ODEFunc(hidden_dim=64)
    train_neural_ode(func, t_train, states, derivs, n_epochs=500, batch_size=16)

    t, numerical_sol, neural_sol = evaluate(
        oscillator, func, t_span, n_points,
        x_range=(-2, 2), v_range=(-2, 2), n_solves=100)

    pos = torch.tensor(neural_sol[..., 0], dtype=torch.float32)
    dt = t[1] - t[0]

    # --- 2. Build nonlinear residual operator ---
    # The Duffing residual x'' + delta*x' + alpha*x + beta*x^3 = 0 has a
    # cubic nonlinearity. The linear part is identical to the DHO, but the
    # beta*x^3 term means the full operator is nonlinear — only perturbation
    # sampling is valid for inversion.
    residual_op = DuffingResidualOperator(alpha_coeff, beta_coeff, delta_coeff, dt)

    # --- 3. Compute residuals and calibrate qhat ---
    res = residual_op(pos)
    if transductive:
        residual_cal = res
        test_idx = 0
    else:
        n_cal = int(0.8 * len(res))
        residual_cal = res[:n_cal]
        test_idx = n_cal

    qhat = calibrate_qhat_from_residual(residual_cal, alpha=0.1)
    print(f"  [1/4] Calibrated qhat = {qhat:.4f}")

    # --- 4. Invert bounds (perturbation sampling only) ---
    print(f"  [2/4] Perturbation sampling inversion ({noise_type})...")
    perturb_cfg = _build_perturbation_config(noise_type, seed=789)
    perturb_bounds = perturbation_bounds_1d(
        pred_signal=pos[test_idx].numpy(),
        residual_operator=residual_op,
        qhat=qhat,
        interior_slice=slice(1, -1),
        config=perturb_cfg,
    )
    print(f"  [2/4] Perturbation sampling inversion ({noise_type})... done")

    # --- 5. Plot bounds comparison ---
    print("  [3/4] Saving bounds comparison plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    _style_ax(ax)
    tt = t[1:-1]

    # Perturbation bounds fill
    ax.fill_between(tt, perturb_bounds.lower, perturb_bounds.upper,
                    color=PALETTE['perturbation'], alpha=0.15, zorder=1,
                    label=f'Perturbation Bound ({noise_type})')
    ax.plot(tt, perturb_bounds.lower, color=PALETTE['perturbation'],
            linewidth=1.0, alpha=0.5, zorder=2)
    ax.plot(tt, perturb_bounds.upper, color=PALETTE['perturbation'],
            linewidth=1.0, alpha=0.5, zorder=2)

    # Signals on top
    ax.plot(tt, pos[test_idx, 1:-1].numpy(), color=PALETTE['prediction'],
            linewidth=1.8, zorder=5, label='Neural ODE Prediction')
    ax.plot(tt, numerical_sol[test_idx, 1:-1, 0], color=PALETTE['truth'],
            linewidth=1.8, marker='.', markersize=3, markevery=5,
            zorder=6, label='Ground Truth')

    ax.set_xlabel('Time')
    ax.set_ylabel('Displacement')
    ax.set_title('Duffing Oscillator: Physical Bounds (Perturbation Sampling)')
    ax.legend(loc='best', edgecolor='#DDDDDD', ncol=2)
    save_path = os.path.join(os.path.dirname(__file__), '..', 'Paper', 'images', 'duffing_bounds_comparison.png')
    fig.savefig(save_path)
    plt.close(fig)

    # --- 6. Empirical coverage curve across alpha levels ---
    print("  [4/4] Computing empirical coverage curves (perturbation only)...")
    alpha_levels = np.arange(0.10, 0.91, 0.10)
    nominal, emp_perturb = _perturbation_coverage_curve(
        preds=neural_sol[..., 0],
        truths=numerical_sol[..., 0],
        residual_cal=residual_cal,
        residual_operator=residual_op,
        alphas=alpha_levels,
        interior_slice=slice(1, -1),
        perturbation_config=perturb_cfg,
    )
    _save_coverage_plot_nonlinear("Duffing", nominal, emp_perturb,
                                  "duffing_coverage_comparison.png")
    print("  Duffing experiment complete.\n")


EXPERIMENTS = {
    'sho': run_sho,
    'dho': run_dho,
    'pendulum': run_pendulum,
    'duffing': run_duffing,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run residual bound inversion experiments.')
    parser.add_argument(
        'experiments',
        nargs='*',
        default=list(EXPERIMENTS.keys()),
        choices=list(EXPERIMENTS.keys()),
        help=f'Experiments to run (default: all). Choices: {", ".join(EXPERIMENTS.keys())}',
    )
    parser.add_argument(
        '--transductive', action='store_true',
        help='Use all data for calibration (transductive CP) instead of 80/20 split',
    )
    parser.add_argument(
        '--noise-type', default='spatial', choices=NOISE_TYPES,
        help=f'Noise model for perturbation sampling (default: spatial). Choices: {", ".join(NOISE_TYPES)}',
    )
    args = parser.parse_args()

    for name in args.experiments:
        EXPERIMENTS[name](transductive=args.transductive, noise_type=args.noise_type)

# %%
