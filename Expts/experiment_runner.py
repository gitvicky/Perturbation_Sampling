"""
Experiment runner for residual bound inversion on Neural ODE test cases.

Each experiment follows the same pipeline:
  1. Train a Neural ODE on synthetic trajectories
  2. Build a composite FD-stencil kernel encoding the ODE
  3. Compute residuals and calibrate qhat via conformal prediction
  4. Invert residual bounds to physical space (Perturbation Sampling)
  5. Plot bounds and empirical coverage curves

Standard, Optimization, Langevin, and Generative perturbation sampling methods are supported.
All methods are model-agnostic and applicable to both linear and nonlinear ODEs.

Usage:
  python Expts/experiment_runner.py                        # run all, default spatial noise
  python Expts/experiment_runner.py sho                    # run only SHO
  python Expts/experiment_runner.py --noise-type bspline   # run all with B-spline noise
  python Expts/experiment_runner.py sho --noise-type gp    # SHO with GP noise
  python Expts/experiment_runner.py duffing                # run only Duffing
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
    PerturbationSamplingConfig,
    calibrate_qhat_from_residual,
    calibrate_qhat_joint_from_residual,
    empirical_coverage_curve_1d,
    perturbation_bounds_1d,
)

# -- Shared plot style ---------------------------------------------------------
PALETTE = {
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

def _method_label(use_optim, use_langevin, use_generator, use_vi=False, vi_covariance="mean_field"):
    """Return a short label for the sampling method used."""
    if use_vi:
        tag = {"mean_field": "MF", "low_rank": "LR", "full": "Full"}.get(vi_covariance, vi_covariance)
        return f"VI-{tag}"
    if use_generator:
        return "Gen"
    if use_langevin:
        return "Langevin"
    if use_optim:
        return "Optim"
    return "MC"

def _build_perturbation_config(noise_type, seed=123, use_optim=False, use_langevin=False, use_generator=False,
                                use_vi=False, vi_covariance="mean_field", vi_rank=8):
    """Build a PerturbationSamplingConfig for the given noise type."""
    common = dict(
        n_samples=20000,
        batch_size=1000,
        max_rounds=5,
        noise_std=0.5,
        seed=seed,
        use_optim=use_optim,
        use_langevin=use_langevin,
        use_generator=use_generator,
        use_vi=use_vi,
        vi_covariance=vi_covariance,
        vi_rank=vi_rank,
        opt_steps=100,
        langevin_steps=20,
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
    """Plot empirical vs nominal coverage for perturbation sampling."""
    fig, ax = plt.subplots(figsize=(8, 6))
    _style_ax(ax)

    nom = coverage_result.nominal_coverage

    # Ideal diagonal
    ax.plot(nom, nom, color=PALETTE['target'], linewidth=1.8, linestyle=':',
            label='Target Coverage', zorder=1)

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
    ax.set_title(f'{case_name}: Empirical Coverage')
    ax.legend(loc='lower right', edgecolor='#DDDDDD')

    save_path = os.path.join(os.path.dirname(__file__), 'Figures', save_name)
    fig.savefig(save_path)
    plt.close(fig)

def _save_coverage_plot_nonlinear(case_name, nominal, empirical_perturbation, save_name):
    """Plot empirical vs nominal coverage for nonlinear cases."""
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
    ax.set_title(f'{case_name}: Empirical Coverage')
    ax.legend(loc='lower right', edgecolor='#DDDDDD')

    save_path = os.path.join(os.path.dirname(__file__), 'Figures', save_name)
    fig.savefig(save_path)
    plt.close(fig)

def _compute_alpha_bounds(
    pred_signal, residual_cal, residual_operator, *,
    alphas, interior_slice, perturbation_config, cp_mode="marginal",
):
    """Compute physical-space bounds for a single trajectory at each alpha level.

    Returns a list of (alpha, InversionBounds) tuples.
    """
    joint = cp_mode == "joint"
    alpha_bounds = []
    for alpha in tqdm(alphas, desc="Alpha bounds", unit="α"):
        if joint:
            qhat_scalar, modulation = calibrate_qhat_joint_from_residual(residual_cal, alpha=float(alpha))
            qhat = qhat_scalar * modulation
        else:
            qhat = calibrate_qhat_from_residual(residual_cal, alpha=float(alpha))

        bounds = perturbation_bounds_1d(
            pred_signal=pred_signal,
            residual_operator=residual_operator,
            qhat=qhat,
            interior_slice=interior_slice,
            config=perturbation_config,
            joint=joint,
        )
        alpha_bounds.append((float(alpha), bounds))
    return alpha_bounds


def _save_alpha_bounds_plot(
    case_name, t_interior, pred_interior, truth_interior,
    alpha_bounds, save_name,
):
    """Plot physical-space bounds at each alpha level as nested shaded bands."""
    import matplotlib.cm as cm

    fig, ax = plt.subplots(figsize=(10, 6))
    _style_ax(ax)

    # Sort from widest (smallest alpha → highest coverage) to narrowest
    alpha_bounds_sorted = sorted(alpha_bounds, key=lambda ab: ab[0])

    cmap = plt.get_cmap('magma', len(alpha_bounds_sorted) + 2)
    for idx, (alpha, bounds) in enumerate(alpha_bounds_sorted):
        colour = cmap(idx + 1)
        coverage_pct = f"{100 * (1 - alpha):.0f}"
        ax.fill_between(
            t_interior, bounds.lower, bounds.upper,
            color=colour, alpha=0.30, zorder=1 + idx,
            label=f'$1-\\alpha = {coverage_pct}\\%$',
        )

    # Prediction and ground truth on top
    ax.plot(t_interior, pred_interior, color=PALETTE['prediction'],
            linewidth=1.8, zorder=len(alpha_bounds_sorted) + 2,
            label='Neural ODE Prediction')
    ax.plot(t_interior, truth_interior, color=PALETTE['truth'],
            linewidth=1.8, marker='.', markersize=3, markevery=5,
            zorder=len(alpha_bounds_sorted) + 3, label='Ground Truth')

    ax.set_xlabel('Time')
    ax.set_ylabel('Position')
    ax.set_title(f'{case_name}: Physical Bounds across $\\alpha$')
    ax.legend(loc='best', edgecolor='#DDDDDD', ncol=2, fontsize=8)

    save_path = os.path.join(os.path.dirname(__file__), 'Figures', save_name)
    fig.savefig(save_path)
    plt.close(fig)


def _perturbation_coverage_curve(
    preds, truths, residual_cal, residual_operator, *,
    alphas, interior_slice, perturbation_config, cp_mode="marginal",
):
    """Compute empirical coverage curve using perturbation sampling.
    """
    from Inversion_Strategies.inversion.residual_inversion import (
        _trajectory_coverage_nd as _trajectory_coverage_1d,
    )
    nominal = []
    cov_perturb = []
    joint = cp_mode == "joint"

    preds = np.asarray(preds, dtype=float)
    truths = np.asarray(truths, dtype=float)

    for alpha in tqdm(alphas, desc="Coverage alphas", unit="α"):
        if joint:
            qhat_scalar, modulation = calibrate_qhat_joint_from_residual(residual_cal, alpha=float(alpha))
            qhat = qhat_scalar * modulation
        else:
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
                joint=joint,
            )
            truth_i = truths[i][interior_slice]
            cover_flags.append(
                _trajectory_coverage_1d(truth_i[None, :], perturb_bounds)
            )

        nominal.append(1.0 - float(alpha))
        cov_perturb.append(float(np.mean(cover_flags)))

    return np.asarray(nominal), np.asarray(cov_perturb)

def run_sho(transductive=False, noise_type="spatial", cp_mode="marginal", use_optim=False, use_langevin=False, use_generator=False,
            use_vi=False, vi_covariance="mean_field", vi_rank=8):
    """Simple Harmonic Oscillator: m*x'' + k*x = 0"""
    from Expts.SHO.SHO_NODE import HarmonicOscillator, ODEFunc, generate_training_data, train_neural_ode, evaluate
    print(f"Running SHO Experiment ({'transductive' if transductive else 'inductive'}, noise={noise_type}, cp={cp_mode}, opt={use_optim}, langevin={use_langevin}, gen={use_generator}, vi={use_vi}({vi_covariance}))...")

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

    joint = cp_mode == "joint"
    if joint:
        qhat_scalar, modulation = calibrate_qhat_joint_from_residual(residual_cal, alpha=0.1)
        qhat = qhat_scalar * modulation
        print(f"  [1/4] Joint calibrated qhat_scalar = {qhat_scalar:.4f}, modulation range = [{modulation.min():.4f}, {modulation.max():.4f}]")
    else:
        qhat = calibrate_qhat_from_residual(residual_cal, alpha=0.1)
        print(f"  [1/4] Calibrated qhat = {qhat:.4f}")

    # --- 4. Invert bounds (Perturbation Sampling) ---
    print(f"  [2/4] Perturbation sampling inversion ({noise_type})...")
    perturb_cfg = _build_perturbation_config(noise_type, seed=123, use_optim=use_optim, use_langevin=use_langevin, use_generator=use_generator,
                                             use_vi=use_vi, vi_covariance=vi_covariance, vi_rank=vi_rank)
    perturb_bounds = perturbation_bounds_1d(
        pred_signal=pos[test_idx].numpy(),
        residual_operator=D_pos,
        qhat=qhat,
        interior_slice=slice(1, -1),
        config=perturb_cfg,
        joint=joint,
    )
    print(f"  [2/4] Perturbation sampling inversion ({noise_type})... done")

    # Verify round-trip: differentiate then integrate should recover the signal
    pos_res = D_pos.differentiate(pos, correlation=False, slice_pad=False)
    pos_retrieved = D_pos.integrate(pos_res, correlation=False, slice_pad=False)
    assert torch.allclose(pos, pos_retrieved[:, 1:-1], atol=1e-3), \
        f"Round-trip failed: max error = {(pos - pos_retrieved[:, 1:-1]).abs().max().item():.2e}"

    # --- 5. Plot bounds ---
    print(f"  [3/5] Saving bounds plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    _style_ax(ax)
    tt = t[1:-1]

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
    method = _method_label(use_optim, use_langevin, use_generator, use_vi, vi_covariance)
    ax.set_title(f'SHO: Physical Bounds ({cp_mode}, {method})')
    ax.legend(loc='best', edgecolor='#DDDDDD', ncol=2)
    save_path = os.path.join(os.path.dirname(__file__), 'Figures', f'sho_bounds_{"joint" if joint else "marginal"}_{method}.png')
    fig.savefig(save_path)
    plt.close(fig)

    # --- 6. Empirical coverage curve across alpha levels ---
    print(f"  [4/5] Computing empirical coverage curves...")
    alpha_levels = np.arange(0.10, 0.91, 0.20)
    sho_coverage = empirical_coverage_curve_1d(
        preds=neural_sol[..., 0],
        truths=numerical_sol[..., 0],
        residual_cal=residual_cal,
        operator=D_pos,
        alphas=alpha_levels,
        interior_slice=slice(1, -1),
        perturbation_config=perturb_cfg,
        cp_mode=cp_mode,
    )
    suffix = f"_{'joint' if joint else 'marginal'}_{method}"
    _save_coverage_plot(f"SHO ({cp_mode}, {method})", sho_coverage, f"sho_coverage{suffix}.png")

    # --- 7. Physical-space bounds at each alpha ---
    print(f"  [5/5] Computing physical bounds across alpha levels...")
    alpha_bounds = _compute_alpha_bounds(
        pred_signal=pos[test_idx].numpy(),
        residual_cal=residual_cal,
        residual_operator=D_pos,
        alphas=alpha_levels,
        interior_slice=slice(1, -1),
        perturbation_config=perturb_cfg,
        cp_mode=cp_mode,
    )
    _save_alpha_bounds_plot(
        f"SHO ({cp_mode}, {method})", tt,
        pred_interior=pos[test_idx, 1:-1].numpy(),
        truth_interior=numerical_sol[test_idx, 1:-1, 0],
        alpha_bounds=alpha_bounds,
        save_name=f"sho_alpha_bounds{suffix}.png",
    )
    print("  SHO experiment complete.\n")


def run_dho(transductive=False, noise_type="spatial", cp_mode="marginal", use_optim=False, use_langevin=False, use_generator=False,
            use_vi=False, vi_covariance="mean_field", vi_rank=8):
    """Damped Harmonic Oscillator: m*x'' + c*x' + k*x = 0"""
    from Expts.DHO.DHO_NODE import DampedHarmonicOscillator, ODEFunc, generate_training_data, train_neural_ode, evaluate
    print(f"Running DHO Experiment ({'transductive' if transductive else 'inductive'}, noise={noise_type}, cp={cp_mode}, opt={use_optim}, langevin={use_langevin}, gen={use_generator}, vi={use_vi}({vi_covariance}))...")

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

    joint = cp_mode == "joint"
    if joint:
        qhat_scalar, modulation = calibrate_qhat_joint_from_residual(residual_cal, alpha=0.1)
        qhat = qhat_scalar * modulation
        print(f"  [1/4] Joint calibrated qhat_scalar = {qhat_scalar:.4f}, modulation range = [{modulation.min():.4f}, {modulation.max():.4f}]")
    else:
        qhat = calibrate_qhat_from_residual(residual_cal, alpha=0.1)
        print(f"  [1/4] Calibrated qhat = {qhat:.4f}")

    # --- 4. Invert bounds (Perturbation Sampling) ---
    print(f"  [2/4] Perturbation sampling inversion ({noise_type})...")
    perturb_cfg = _build_perturbation_config(noise_type, seed=321, use_optim=use_optim, use_langevin=use_langevin, use_generator=use_generator,
                                             use_vi=use_vi, vi_covariance=vi_covariance, vi_rank=vi_rank)
    perturb_bounds = perturbation_bounds_1d(
        pred_signal=pos[test_idx].numpy(),
        residual_operator=D_damped,
        qhat=qhat,
        interior_slice=slice(1, -1),
        config=perturb_cfg,
        joint=joint,
    )
    print(f"  [2/4] Perturbation sampling inversion ({noise_type})... done")

    # Verify round-trip: differentiate then integrate should recover the signal
    pos_res = D_damped.differentiate(pos, correlation=False, slice_pad=False)
    pos_retrieved = D_damped.integrate(pos_res, correlation=False, slice_pad=False)
    assert torch.allclose(pos, pos_retrieved[:, 1:-1], atol=1e-3), \
        f"Round-trip failed: max error = {(pos - pos_retrieved[:, 1:-1]).abs().max().item():.2e}"

    # --- 5. Plot bounds ---
    print(f"  [3/5] Saving bounds plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    _style_ax(ax)
    tt = t[1:-1]

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
    method = _method_label(use_optim, use_langevin, use_generator, use_vi, vi_covariance)
    ax.set_title(f'DHO: Physical Bounds ({cp_mode}, {method})')
    ax.legend(loc='best', edgecolor='#DDDDDD', ncol=2)
    save_path = os.path.join(os.path.dirname(__file__), 'Figures', f'dho_bounds_{"joint" if joint else "marginal"}_{method}.png')
    fig.savefig(save_path)
    plt.close(fig)

    # --- 6. Empirical coverage curve across alpha levels ---
    print(f"  [4/5] Computing empirical coverage curves...")
    alpha_levels = np.arange(0.10, 0.91, 0.10)
    dho_coverage = empirical_coverage_curve_1d(
        preds=neural_sol[..., 0],
        truths=numerical_sol[..., 0],
        residual_cal=residual_cal,
        operator=D_damped,
        alphas=alpha_levels,
        interior_slice=slice(1, -1),
        perturbation_config=perturb_cfg,
        cp_mode=cp_mode,
    )
    suffix = f"_{'joint' if joint else 'marginal'}_{method}"
    _save_coverage_plot(f"DHO ({cp_mode}, {method})", dho_coverage, f"dho_coverage{suffix}.png")

    # --- 7. Physical-space bounds at each alpha ---
    print(f"  [5/5] Computing physical bounds across alpha levels...")
    alpha_bounds = _compute_alpha_bounds(
        pred_signal=pos[test_idx].numpy(),
        residual_cal=residual_cal,
        residual_operator=D_damped,
        alphas=alpha_levels,
        interior_slice=slice(1, -1),
        perturbation_config=perturb_cfg,
        cp_mode=cp_mode,
    )
    _save_alpha_bounds_plot(
        f"DHO ({cp_mode}, {method})", tt,
        pred_interior=pos[test_idx, 1:-1].numpy(),
        truth_interior=numerical_sol[test_idx, 1:-1, 0],
        alpha_bounds=alpha_bounds,
        save_name=f"dho_alpha_bounds{suffix}.png",
    )
    print("  DHO experiment complete.\n")


def run_duffing(transductive=False, noise_type="spatial", cp_mode="marginal", use_optim=False, use_langevin=False, use_generator=False,
            use_vi=False, vi_covariance="mean_field", vi_rank=8):
    """Duffing Oscillator: x'' + delta*x' + alpha*x + beta*x^3 = 0"""
    from Expts.Duffing.Duffing_NODE import (
        DuffingOscillator, DuffingResidualOperator,
        ODEFunc, generate_training_data, train_neural_ode, evaluate,
    )
    print(f"Running Duffing Experiment ({'transductive' if transductive else 'inductive'}, noise={noise_type}, cp={cp_mode}, opt={use_optim}, langevin={use_langevin}, gen={use_generator}, vi={use_vi}({vi_covariance}))...")

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
    # cubic nonlinearity.
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

    joint = cp_mode == "joint"
    if joint:
        qhat_scalar, modulation = calibrate_qhat_joint_from_residual(residual_cal, alpha=0.1)
        qhat = qhat_scalar * modulation
        print(f"  [1/4] Joint calibrated qhat_scalar = {qhat_scalar:.4f}, modulation range = [{modulation.min():.4f}, {modulation.max():.4f}]")
    else:
        qhat = calibrate_qhat_from_residual(residual_cal, alpha=0.1)
        print(f"  [1/4] Calibrated qhat = {qhat:.4f}")

    # --- 4. Invert bounds (perturbation sampling only) ---
    print(f"  [2/4] Perturbation sampling inversion ({noise_type})...")
    perturb_cfg = _build_perturbation_config(noise_type, seed=789, use_optim=use_optim, use_langevin=use_langevin, use_generator=use_generator,
                                             use_vi=use_vi, vi_covariance=vi_covariance, vi_rank=vi_rank)
    perturb_bounds = perturbation_bounds_1d(
        pred_signal=pos[test_idx].numpy(),
        residual_operator=residual_op,
        qhat=qhat,
        interior_slice=slice(1, -1),
        config=perturb_cfg,
        joint=joint,
    )
    print(f"  [2/4] Perturbation sampling inversion ({noise_type})... done")

    # --- 5. Plot bounds ---
    print("  [3/5] Saving bounds plot...")
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
    method = _method_label(use_optim, use_langevin, use_generator, use_vi, vi_covariance)
    ax.set_title(f'Duffing Oscillator: Physical Bounds ({cp_mode}, {method})')
    ax.legend(loc='best', edgecolor='#DDDDDD', ncol=2)
    save_path = os.path.join(os.path.dirname(__file__), 'Figures', f'duffing_bounds_{"joint" if joint else "marginal"}_{method}.png')
    fig.savefig(save_path)
    plt.close(fig)

    # --- 6. Empirical coverage curve across alpha levels ---
    print("  [4/5] Computing empirical coverage curves (perturbation only)...")
    alpha_levels = np.arange(0.10, 0.91, 0.10)
    nominal, emp_perturb = _perturbation_coverage_curve(
        preds=neural_sol[..., 0],
        truths=numerical_sol[..., 0],
        residual_cal=residual_cal,
        residual_operator=residual_op,
        alphas=alpha_levels,
        interior_slice=slice(1, -1),
        perturbation_config=perturb_cfg,
        cp_mode=cp_mode,
    )
    suffix = f"_{'joint' if joint else 'marginal'}_{method}"
    _save_coverage_plot_nonlinear(f"Duffing ({cp_mode}, {method})", nominal, emp_perturb,
                                  f"duffing_coverage{suffix}.png")

    # --- 7. Physical-space bounds at each alpha ---
    print(f"  [5/5] Computing physical bounds across alpha levels...")
    alpha_bounds = _compute_alpha_bounds(
        pred_signal=pos[test_idx].numpy(),
        residual_cal=residual_cal,
        residual_operator=residual_op,
        alphas=alpha_levels,
        interior_slice=slice(1, -1),
        perturbation_config=perturb_cfg,
        cp_mode=cp_mode,
    )
    _save_alpha_bounds_plot(
        f"Duffing ({cp_mode}, {method})", tt,
        pred_interior=pos[test_idx, 1:-1].numpy(),
        truth_interior=numerical_sol[test_idx, 1:-1, 0],
        alpha_bounds=alpha_bounds,
        save_name=f"duffing_alpha_bounds{suffix}.png",
    )
    print("  Duffing experiment complete.\n")

EXPERIMENTS = {
    'sho': run_sho,
    'dho': run_dho,
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
    CP_MODES = ("marginal", "joint")
    parser.add_argument(
        '--cp-mode', default='marginal', choices=CP_MODES,
        help=f'Conformal prediction mode (default: marginal). Choices: {", ".join(CP_MODES)}',
    )
    parser.add_argument(
        '--use-optim', action='store_true',
        help='Use Method 2: Differentiable Rejection (Inference-Time Optimization)',
    )
    parser.add_argument(
        '--use-langevin', action='store_true',
        help='Use Method 1: Posterior Sampling (Langevin Dynamics)',
    )
    parser.add_argument(
        '--use-generator', action='store_true',
        help='Use Method 3: Generative Modeling (Boundary Generator)',
    )
    parser.add_argument(
        '--use-vi', dest='use_vi', action='store_true',
        help='Use Method 4: Variational Inference (per-trajectory Gaussian posterior)',
    )
    parser.add_argument(
        '--vi-covariance', default='mean_field',
        choices=('mean_field', 'low_rank', 'full'),
        help='Variational posterior covariance (default: mean_field)',
    )
    parser.add_argument(
        '--vi-rank', type=int, default=8,
        help='Rank r for vi-covariance=low_rank (default: 8)',
    )
    args = parser.parse_args()

    _advanced = [args.use_optim, args.use_langevin, args.use_generator, args.use_vi]
    if sum(bool(f) for f in _advanced) > 1:
        parser.error("use-optim / --use-langevin / --use-generator / --use-vi are mutually exclusive")

    for name in args.experiments:
        EXPERIMENTS[name](
            transductive=args.transductive,
            noise_type=args.noise_type,
            cp_mode=args.cp_mode,
            use_optim=args.use_optim,
            use_langevin=args.use_langevin,
            use_generator=args.use_generator,
            use_vi=args.use_vi,
            vi_covariance=args.vi_covariance,
            vi_rank=args.vi_rank,
        )

# %%
