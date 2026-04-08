"""
Noise method comparison for perturbation sampling inversion.

Runs perturbation sampling with every available noise strategy on both
SHO and DHO cases, producing:
  - A multi-panel bounds comparison plot (one subplot per noise method)
  - (optional) An empirical coverage curve comparing all noise methods

Usage:
  python Expts/noise_comparison.py                   # both cases, bounds only
  python Expts/noise_comparison.py sho               # SHO only
  python Expts/noise_comparison.py --coverage         # include coverage curves
  python Expts/noise_comparison.py --coverage --alpha-steps 5   # fewer alpha levels (faster)
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Utils.PRE.ConvOps_0d import ConvOperator
from Inversion_Strategies.inversion.residual_inversion import (
    IntervalFFTSlicing,
    PerturbationSamplingConfig,
    calibrate_qhat_from_residual,
    invert_residual_bounds_1d,
    perturbation_bounds_1d,
)

# ---------------------------------------------------------------------------
# Noise method definitions
# ---------------------------------------------------------------------------
# Each entry: (label, config-builder kwargs).
# The config-builder receives these plus common kwargs (n_samples, etc.)
# and the composite kernel (for pre_correlated).

NOISE_METHODS = {
    "white": {
        "label": "White",
        "config_kw": dict(noise_type="white"),
    },
    "spatial": {
        "label": "Spatial (Gaussian conv)",
        "config_kw": dict(noise_type="spatial", correlation_length=24.0),
    },
    "gp_rbf": {
        "label": "GP (RBF)",
        "config_kw": dict(noise_type="gp", correlation_length=24.0, gp_kernel="rbf"),
    },
    "gp_matern": {
        "label": "GP (Mat\u00e9rn \u03bd=1.5)",
        "config_kw": dict(noise_type="gp", correlation_length=24.0, gp_kernel="matern", gp_nu=1.5),
    },
    "bspline": {
        "label": "B-spline (6 knots)",
        "config_kw": dict(noise_type="bspline", bspline_n_knots=6),
    },
    "pre_correlated": {
        "label": "PRE-correlated",
        "config_kw": dict(noise_type="pre_correlated"),
    },
}

# Visual style per noise method — distinct colours and linestyles
NOISE_STYLES = {
    "white":          dict(color="#E07A5F", ls="-",  lw=1.5),   # terra cotta
    "spatial":        dict(color="#81B29A", ls="--", lw=1.5),   # sage green
    "gp_rbf":         dict(color="#6A4C93", ls="-.", lw=1.5),   # purple
    "gp_matern":      dict(color="#1982C4", ls=":",  lw=1.8),   # blue
    "bspline":        dict(color="#FF9F1C", ls="-",  lw=1.5),   # amber
    "pre_correlated": dict(color="#F72585", ls="--", lw=1.5),   # magenta
}

PALETTE = {
    'intervalfft':  '#3D405B',
    'pointwise':    '#AAAAAA',
    'truth':        '#2C2C2C',
    'prediction':   '#5B7EC0',
    'target':       '#9B9B9B',
}

plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)
    ax.grid(True, alpha=0.25, linewidth=0.5, color='#CCCCCC')
    ax.tick_params(direction='out', length=4, width=0.6)


def _build_config(method_key, composite_kernel, seed=123):
    """Build a PerturbationSamplingConfig for a given noise method key."""
    kw = dict(NOISE_METHODS[method_key]["config_kw"])
    common = dict(n_samples=4000, batch_size=1000, max_rounds=2, noise_std=0.10, seed=seed)

    if method_key == "pre_correlated":
        kw["pre_kernel"] = composite_kernel.clone()

    return PerturbationSamplingConfig(**common, **kw)


# ---------------------------------------------------------------------------
# Case setup (shared between bounds and coverage)
# ---------------------------------------------------------------------------

def _setup_sho():
    """Train SHO Neural ODE and return everything needed for inversion."""
    from Expts.SHO.SHO_NODE import HarmonicOscillator, ODEFunc, generate_training_data, train_neural_ode, evaluate

    m, k = 1.0, 1.0
    oscillator = HarmonicOscillator(k, m)
    t_train, states, derivs = generate_training_data(oscillator, (0, 10), 100, 50)
    func = ODEFunc(hidden_dim=64)
    train_neural_ode(func, t_train, states, derivs, n_epochs=500, batch_size=16)
    t, numerical_sol, neural_sol = evaluate(oscillator, func, (0, 10), 100,
                                            x_range=(-2, 2), v_range=(-2, 2), n_solves=100)

    pos = torch.tensor(neural_sol[..., 0], dtype=torch.float32)
    dt = t[1] - t[0]

    D_tt = ConvOperator(order=2)
    D_id = ConvOperator(order=0)
    D_id.kernel = torch.tensor([0.0, 1.0, 0.0])
    D_pos = ConvOperator(conv='spectral')
    D_pos.kernel = m * D_tt.kernel + dt**2 * k * D_id.kernel

    res = D_pos(pos)
    n_cal = int(0.8 * len(res))
    residual_cal = res[:n_cal]
    test_idx = n_cal

    return dict(
        case_name="SHO", t=t, pos=pos, numerical_sol=numerical_sol, neural_sol=neural_sol,
        operator=D_pos, residual_cal=residual_cal, test_idx=test_idx,
        interior_slice=slice(1, -1),
        intervalfft_slicing=IntervalFFTSlicing(center_start=3, center_end=-1, n_right_edges=3, output_offset=2),
    )


def _setup_dho():
    """Train DHO Neural ODE and return everything needed for inversion."""
    from Expts.DHO.DHO_NODE import DampedHarmonicOscillator, ODEFunc, generate_training_data, train_neural_ode, evaluate

    m, k, c = 1.0, 1.0, 0.2
    oscillator = DampedHarmonicOscillator(k, m, c)
    t_train, states, derivs = generate_training_data(oscillator, (0, 15), 100, 50)
    func = ODEFunc(hidden_dim=64)
    train_neural_ode(func, t_train, states, derivs, n_epochs=500, batch_size=16)
    t, numerical_sol, neural_sol = evaluate(oscillator, func, (0, 15), 100,
                                            x_range=(-2, 2), v_range=(-2, 2), n_solves=100)

    pos = torch.tensor(neural_sol[..., 0], dtype=torch.float32)
    dt = t[1] - t[0]

    D_t = ConvOperator(order=1)
    D_tt = ConvOperator(order=2)
    D_id = ConvOperator(order=0)
    D_id.kernel = torch.tensor([0.0, 1.0, 0.0])
    D_damped = ConvOperator(conv='spectral')
    D_damped.kernel = 2 * m * D_tt.kernel + dt * c * D_t.kernel + 2 * dt**2 * k * D_id.kernel

    res = D_damped(pos)
    n_cal = int(0.8 * len(res))
    residual_cal = res[:n_cal]
    test_idx = n_cal

    return dict(
        case_name="DHO", t=t, pos=pos, numerical_sol=numerical_sol, neural_sol=neural_sol,
        operator=D_damped, residual_cal=residual_cal, test_idx=test_idx,
        interior_slice=slice(1, -1),
        intervalfft_slicing=IntervalFFTSlicing(center_start=3, center_end=-1, n_right_edges=3, output_offset=2),
    )


CASE_BUILDERS = {"sho": _setup_sho, "dho": _setup_dho}


# ---------------------------------------------------------------------------
# Bounds comparison
# ---------------------------------------------------------------------------

def run_bounds_comparison(setup, methods):
    """Run perturbation sampling with each noise method and plot comparison."""
    case = setup["case_name"]
    t = setup["t"]
    pos = setup["pos"]
    test_idx = setup["test_idx"]
    operator = setup["operator"]
    interior = setup["interior_slice"]
    slicing = setup["intervalfft_slicing"]
    residual_cal = setup["residual_cal"]
    numerical_sol = setup["numerical_sol"]

    tt = t[interior]
    pred_np = pos[test_idx].numpy()
    truth_np = numerical_sol[test_idx, :, 0]

    qhat = calibrate_qhat_from_residual(residual_cal, alpha=0.1)
    print(f"  qhat = {qhat:.4f}")

    # Reference bounds (point-wise + interval FFT) — computed once
    point_bounds, interval_bounds = invert_residual_bounds_1d(
        pred_signal=pred_np, kernel=operator.kernel.numpy(), qhat=qhat,
        operator=operator, interior_slice=interior,
        intervalfft_slicing=slicing, integrate_slice_pad=False,
    )

    # Perturbation bounds per noise method
    perturb_results = {}
    for key in methods:
        label = NOISE_METHODS[key]["label"]
        print(f"  Perturbation sampling: {label} ...", end=" ", flush=True)
        t0 = time.time()
        cfg = _build_config(key, operator.kernel, seed=123)
        bounds = perturbation_bounds_1d(
            pred_signal=pred_np, residual_operator=operator,
            qhat=qhat, interior_slice=interior, config=cfg,
        )
        elapsed = time.time() - t0
        perturb_results[key] = bounds
        print(f"done ({elapsed:.1f}s)")

    # --- Multi-panel plot: one subplot per noise method ---
    n_methods = len(methods)
    ncols = min(3, n_methods)
    nrows = int(np.ceil(n_methods / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows),
                             sharex=True, sharey=True, squeeze=False)

    for idx, key in enumerate(methods):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        _style_ax(ax)
        bounds = perturb_results[key]
        sty = NOISE_STYLES[key]

        # Interval FFT reference (light background)
        ax.fill_between(tt, interval_bounds.lower, interval_bounds.upper,
                        color=PALETTE['intervalfft'], alpha=0.10, zorder=1)

        # Point-wise reference (thin dashed grey)
        ax.plot(tt, point_bounds.lower, color=PALETTE['pointwise'],
                ls='--', lw=0.8, alpha=0.6, zorder=2)
        ax.plot(tt, point_bounds.upper, color=PALETTE['pointwise'],
                ls='--', lw=0.8, alpha=0.6, zorder=2)

        # Perturbation bounds (this method)
        ax.fill_between(tt, bounds.lower, bounds.upper,
                        color=sty['color'], alpha=0.18, zorder=3)
        ax.plot(tt, bounds.lower, color=sty['color'], ls=sty['ls'],
                lw=sty['lw'], zorder=4)
        ax.plot(tt, bounds.upper, color=sty['color'], ls=sty['ls'],
                lw=sty['lw'], zorder=4)

        # Prediction + truth
        ax.plot(tt, pred_np[interior], color=PALETTE['prediction'],
                lw=1.4, zorder=5)
        ax.plot(tt, truth_np[interior], color=PALETTE['truth'],
                lw=1.2, marker='.', markersize=2, markevery=5, zorder=6)

        ax.set_title(NOISE_METHODS[key]["label"], fontsize=11, fontweight='bold')
        if r == nrows - 1:
            ax.set_xlabel('Time')
        if c == 0:
            ax.set_ylabel('Position')

    # Hide unused subplots
    for idx in range(n_methods, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    # Shared legend
    legend_handles = [
        Line2D([], [], color=PALETTE['intervalfft'], alpha=0.4, lw=6, label='Interval FFT'),
        Line2D([], [], color=PALETTE['pointwise'], ls='--', lw=1, label='Point-wise'),
        Line2D([], [], color='grey', lw=1.5, label='Perturbation bound'),
        Line2D([], [], color=PALETTE['prediction'], lw=1.4, label='Prediction'),
        Line2D([], [], color=PALETTE['truth'], lw=1.2, marker='.', markersize=4, label='Ground truth'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=5,
               frameon=True, edgecolor='#DDDDDD', fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f'{case}: Perturbation Bounds by Noise Method (90% target)', fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    save_dir = os.path.join(os.path.dirname(__file__), '..', 'Paper', 'images')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{case.lower()}_noise_bounds_comparison.png')
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved bounds plot: {save_path}")

    # --- Overlay plot: all methods on one axis ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    _style_ax(ax2)

    # Interval FFT reference
    ax2.fill_between(tt, interval_bounds.lower, interval_bounds.upper,
                     color=PALETTE['intervalfft'], alpha=0.10, zorder=1,
                     label='Interval FFT')

    for key in methods:
        bounds = perturb_results[key]
        sty = NOISE_STYLES[key]
        label = NOISE_METHODS[key]["label"]
        ax2.plot(tt, bounds.lower, color=sty['color'], ls=sty['ls'], lw=sty['lw'], zorder=3)
        ax2.plot(tt, bounds.upper, color=sty['color'], ls=sty['ls'], lw=sty['lw'],
                 zorder=3, label=label)

    ax2.plot(tt, pred_np[interior], color=PALETTE['prediction'], lw=1.6,
             zorder=5, label='Prediction')
    ax2.plot(tt, truth_np[interior], color=PALETTE['truth'], lw=1.4,
             marker='.', markersize=3, markevery=5, zorder=6, label='Ground truth')

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Position')
    ax2.set_title(f'{case}: Perturbation Bounds Overlay (90% target)')
    ax2.legend(loc='best', edgecolor='#DDDDDD', fontsize=8, ncol=2)

    save_path2 = os.path.join(save_dir, f'{case.lower()}_noise_bounds_overlay.png')
    fig2.savefig(save_path2)
    plt.close(fig2)
    print(f"  Saved overlay plot: {save_path2}")

    return perturb_results


# ---------------------------------------------------------------------------
# Coverage comparison
# ---------------------------------------------------------------------------

def run_coverage_comparison(setup, methods, alpha_steps=10):
    """Compute empirical coverage for each noise method and plot comparison."""
    case = setup["case_name"]
    pos = setup["pos"]
    operator = setup["operator"]
    interior = setup["interior_slice"]
    slicing = setup["intervalfft_slicing"]
    residual_cal = setup["residual_cal"]
    numerical_sol = setup["numerical_sol"]
    neural_sol = setup["neural_sol"]

    preds = neural_sol[..., 0]
    truths = numerical_sol[..., 0]
    kernel_np = operator.kernel.numpy()

    alphas = np.linspace(0.05, 0.95, alpha_steps)
    nominal = 1.0 - alphas
    n_traj = preds.shape[0]

    # Compute point-wise and interval FFT coverage once
    cov_point = np.zeros(len(alphas))
    cov_interval = np.zeros(len(alphas))
    # Storage for each noise method's perturbation coverage
    cov_perturb = {key: np.zeros(len(alphas)) for key in methods}

    for ai, alpha in enumerate(alphas):
        qhat = calibrate_qhat_from_residual(residual_cal, alpha=float(alpha))
        print(f"  alpha={alpha:.2f} (qhat={qhat:.4f}) ...", end=" ", flush=True)

        point_flags = []
        interval_flags = []
        perturb_flags = {key: [] for key in methods}

        for i in range(n_traj):
            truth_i = truths[i][interior]

            # Point-wise + interval FFT (computed once per trajectory)
            pb, ib = invert_residual_bounds_1d(
                pred_signal=preds[i], kernel=kernel_np, qhat=qhat,
                operator=operator, interior_slice=interior,
                intervalfft_slicing=slicing, integrate_slice_pad=False,
            )
            point_flags.append(float(np.all((truth_i >= pb.lower) & (truth_i <= pb.upper))))
            interval_flags.append(float(np.all((truth_i >= ib.lower) & (truth_i <= ib.upper))))

            # Perturbation for each noise method
            for key in methods:
                cfg = _build_config(key, operator.kernel, seed=123)
                try:
                    prb = perturbation_bounds_1d(
                        pred_signal=preds[i], residual_operator=operator,
                        qhat=qhat, interior_slice=interior, config=cfg,
                    )
                    contained = float(np.all((truth_i >= prb.lower) & (truth_i <= prb.upper)))
                except RuntimeError:
                    contained = 0.0
                perturb_flags[key].append(contained)

        cov_point[ai] = np.mean(point_flags)
        cov_interval[ai] = np.mean(interval_flags)
        for key in methods:
            cov_perturb[key][ai] = np.mean(perturb_flags[key])
        print("done")

    # --- Coverage plot ---
    fig, ax = plt.subplots(figsize=(8, 6))
    _style_ax(ax)

    # Diagonal target
    ax.plot(nominal, nominal, color=PALETTE['target'], lw=1.8, ls=':',
            label='Target', zorder=1)
    ax.fill_between(nominal, 0, nominal, color=PALETTE['target'], alpha=0.06, zorder=0)

    # Point-wise + Interval FFT
    ax.plot(nominal, cov_point, color=PALETTE['pointwise'], lw=2.0, ls='--',
            marker='s', markersize=4, markeredgewidth=0, label='Point-wise', zorder=2)
    ax.plot(nominal, cov_interval, color=PALETTE['intervalfft'], lw=2.2,
            marker='o', markersize=5, markeredgewidth=0, label='Interval FFT', zorder=2)

    # Each noise method
    markers = ['D', '^', 'v', '<', '>', 'p']
    for mi, key in enumerate(methods):
        sty = NOISE_STYLES[key]
        ax.plot(nominal, cov_perturb[key], color=sty['color'], lw=1.8,
                ls=sty['ls'], marker=markers[mi % len(markers)], markersize=4.5,
                markeredgewidth=0, label=f'Perturb: {NOISE_METHODS[key]["label"]}', zorder=3)

    ax.set_xlabel('Nominal Coverage $(1 - \\alpha)$')
    ax.set_ylabel('Empirical Coverage')
    ax.set_title(f'{case}: Coverage Comparison by Noise Method')
    ax.legend(loc='lower right', edgecolor='#DDDDDD', fontsize=8)
    ax.set_xlim(nominal.min() - 0.02, nominal.max() + 0.02)
    ax.set_ylim(-0.02, 1.05)

    save_dir = os.path.join(os.path.dirname(__file__), '..', 'Paper', 'images')
    save_path = os.path.join(save_dir, f'{case.lower()}_noise_coverage_comparison.png')
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved coverage plot: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Compare perturbation sampling noise methods for residual bound inversion.')
    parser.add_argument(
        'cases', nargs='*', default=['sho', 'dho'], choices=['sho', 'dho'],
        help='Which ODE cases to run (default: both)')
    parser.add_argument(
        '--coverage', action='store_true',
        help='Also compute empirical coverage curves (slow)')
    parser.add_argument(
        '--alpha-steps', type=int, default=10,
        help='Number of alpha levels for coverage sweep (default: 10)')
    parser.add_argument(
        '--methods', nargs='+', default=list(NOISE_METHODS.keys()),
        choices=list(NOISE_METHODS.keys()),
        help=f'Noise methods to compare (default: all). Choices: {", ".join(NOISE_METHODS.keys())}')
    args = parser.parse_args()

    for case_name in args.cases:
        print(f"\n{'='*60}")
        print(f"  {case_name.upper()}: Noise Method Comparison")
        print(f"{'='*60}")

        print(f"  Setting up {case_name.upper()} (training Neural ODE)...")
        setup = CASE_BUILDERS[case_name]()

        print(f"\n  --- Bounds comparison ---")
        run_bounds_comparison(setup, args.methods)

        if args.coverage:
            print(f"\n  --- Coverage comparison ({args.alpha_steps} alpha levels) ---")
            run_coverage_comparison(setup, args.methods, alpha_steps=args.alpha_steps)

    print("\nDone.")


if __name__ == '__main__':
    main()
