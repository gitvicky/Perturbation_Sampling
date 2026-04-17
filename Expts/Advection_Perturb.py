"""
1D Advection Perturbation-Sampling Experiment
=============================================

Tests PRE-based conformal bound inversion on the 1D linear advection PDE

        u_t + v * u_x = 0,    v = 1.0,  x in [0, 2],  t in [0, 0.5]

as a scaling test of the perturbation-sampling framework on a spatiotemporal
(rather than purely temporal) problem. Mirrors the knobs exposed by
`Expts/experiment_runner.py`: noise model, CP mode (marginal/joint),
advanced samplers (optim / Langevin / generator / VI), and optional
empirical-coverage curves.

Pipeline
--------
1. Train an autoregressive 1D U-Net surrogate (`Expts/Advection/Advection_UNet.py`)
   on `(u_t, u_{t+1})` pairs drawn from the numerical solver
   (`Neural_PDE.Numerical_Solvers.Advection.Advection_1d`).
2. Roll out calibration + test trajectories under the trained U-Net from
   fresh random Gaussian-pulse initial conditions; keep the numerical
   solutions as ground truth.
3. Build the PDE residual operator from central finite-difference stencils
   wrapped as `ConvOperator`s (one temporal, one spatial).
4. Calibrate `qhat` at target coverage `1 - alpha = 0.9` from residual
   scores on the U-Net calibration rollouts (marginal or joint CP).
5. Invert the residual bound `[-qhat, +qhat]` to physical-space bounds on
   a test trajectory, optionally with gradient-guided variants.
6. Save bounds + (optional) empirical-coverage curve to `Expts/Figures/`.

Running
-------
From the repo root, with the project virtualenv active::

    source .venv/bin/activate

    # Baseline rejection sampling, spatial noise, marginal CP
    python Expts/Advection_Perturb.py

    # Alternative noise models
    python Expts/Advection_Perturb.py --noise-type white
    python Expts/Advection_Perturb.py --noise-type gp
    python Expts/Advection_Perturb.py --noise-type bspline

    # Joint CP calibration
    python Expts/Advection_Perturb.py --cp-mode joint

    # Gradient-guided samplers
    python Expts/Advection_Perturb.py --use-optim
    python Expts/Advection_Perturb.py --use-langevin
    python Expts/Advection_Perturb.py --use-generator

    # Variational-inference sampler (choose covariance structure)
    python Expts/Advection_Perturb.py --use-vi --vi-covariance low_rank --vi-rank 8

    # Also compute and plot empirical coverage curve over alpha levels
    python Expts/Advection_Perturb.py --plot-coverage

Outputs (in `Expts/Figures/`)
-----------------------------
- `advection_bounds_<cp_mode>_<method>_<noise>.png` — profile slice +
  spatiotemporal lower-bound heatmap.
- `advection_coverage_<cp_mode>_<method>_<noise>.png` — nominal vs.
  empirical coverage curve (only when `--plot-coverage` is set).
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Neural_PDE.Numerical_Solvers.Advection.Advection_1D import Advection_1d
from Utils.PRE.ConvOps_1d import ConvOperator
from Inversion_Strategies.inversion.residual_inversion import (
    PerturbationSamplingConfig,
    calibrate_qhat_from_residual,
    calibrate_qhat_joint_from_residual,
    empirical_coverage_curve_nd,
    perturbation_bounds_nd,
)
from Expts.Advection.Advection_UNet import (
    build_model,
    generate_training_data,
    train_unet,
    evaluate,
)

# -- Shared plot style (matches experiment_runner) -----------------------------
PALETTE = {
    'perturbation': '#81B29A',
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

NOISE_TYPES = ("spatial", "white", "gp", "bspline")
CP_MODES = ("marginal", "joint")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'Figures')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'Advection', 'models')
MODEL_KINDS = ("unet", "fno")


def _default_model_path(kind):
    return os.path.join(MODELS_DIR, f"advection_{kind}.pt")


def _style_ax(ax):
    """Apply clean spine/grid styling (matches experiment_runner)."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)
    ax.grid(True, alpha=0.25, linewidth=0.5, color='#CCCCCC')
    ax.tick_params(direction='out', length=4, width=0.6)


def _method_label(use_optim, use_langevin, use_generator, use_vi=False, vi_covariance="mean_field"):
    """Return a short tag for the sampling method (for filenames/titles)."""
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


def _build_perturbation_config(noise_type, seed=0, use_optim=False, use_langevin=False,
                                use_generator=False, use_vi=False,
                                vi_covariance="mean_field", vi_rank=8):
    """Build a `PerturbationSamplingConfig` for the chosen noise model.

    Mirrors the helper in `experiment_runner.py` but uses advection-scale
    defaults (smaller `noise_std`, shorter correlation length).
    """
    common = dict(
        n_samples=2000,
        batch_size=200,
        max_rounds=3,
        noise_std=0.05,
        seed=seed,
        use_optim=use_optim,
        use_langevin=use_langevin,
        use_generator=use_generator,
        use_vi=use_vi,
        vi_covariance=vi_covariance,
        vi_rank=vi_rank,
        vi_steps=300,
        opt_steps=30,
        langevin_steps=20,
        gen_train_steps=200,
    )
    if noise_type == "spatial":
        return PerturbationSamplingConfig(noise_type="spatial", correlation_length=4.0, **common)
    elif noise_type == "white":
        return PerturbationSamplingConfig(noise_type="white", **common)
    elif noise_type == "gp":
        return PerturbationSamplingConfig(noise_type="gp", correlation_length=4.0, gp_kernel="rbf", **common)
    elif noise_type == "bspline":
        return PerturbationSamplingConfig(noise_type="bspline", bspline_n_knots=16, **common)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}. Choose from {NOISE_TYPES}")


def _save_coverage_plot(case_name, nominal, empirical, save_name):
    """Plot empirical vs nominal coverage and save to `Figures/`."""
    fig, ax = plt.subplots(figsize=(8, 6))
    _style_ax(ax)

    nom = np.asarray(nominal, dtype=float)
    emp = np.asarray(empirical, dtype=float)

    ax.plot(nom, nom, color=PALETTE['target'], linewidth=1.8, linestyle=':',
            label='Target Coverage', zorder=1)
    ax.plot(nom, emp, color=PALETTE['perturbation'], linewidth=2.0, linestyle='-.',
            marker='D', markersize=4.5, markeredgewidth=0,
            label='Perturbation Sampling', zorder=3)
    ax.fill_between(nom, 0, nom, color=PALETTE['target'], alpha=0.06, zorder=0)

    ax.set_xlabel('Nominal Coverage $(1 - \\alpha)$')
    ax.set_ylabel('Empirical Coverage')
    ax.set_title(f'{case_name}: Empirical Coverage')
    ax.legend(loc='lower right', edgecolor='#DDDDDD')

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, save_name))
    plt.close(fig)


def run_advection_experiment(noise_type="spatial", cp_mode="marginal",
                              use_optim=False, use_langevin=False, use_generator=False,
                              use_vi=False, vi_covariance="mean_field", vi_rank=8,
                              plot_coverage=False, n_train=200, n_cal=100, n_test=10,
                              n_epochs=200, unet_features=32,
                              model_kind="unet", fno_modes=16, fno_layers=4,
                              model_path=None, retrain=False):
    """Run the 1D advection PRE-inversion experiment.

    Generates calibration + test trajectories, calibrates `qhat` on mocked
    surrogate residuals, inverts the residual bound to physical-space
    bounds for a single test trajectory, and (optionally) sweeps alpha to
    produce an empirical-coverage curve.

    Parameters
    ----------
    noise_type : {"spatial", "white", "gp", "bspline"}
        Perturbation noise model (see `_build_perturbation_config`).
    cp_mode : {"marginal", "joint"}
        Conformal prediction calibration mode. "joint" scales `qhat` by a
        per-cell modulation learned from the calibration residuals.
    use_optim, use_langevin, use_generator, use_vi : bool
        Mutually exclusive advanced-sampler flags.
    vi_covariance : {"mean_field", "low_rank", "full"}
        Covariance parameterisation for the VI posterior.
    vi_rank : int
        Rank used when `vi_covariance == "low_rank"`.
    plot_coverage : bool
        If True, sweep alpha over `[0.1, 0.9]` and save an empirical
        coverage curve in addition to the bounds plot. Note: this is
        considerably more expensive than the single-alpha bounds run.
    n_train, n_cal, n_test : int
        Number of trajectories used for U-Net training, CP calibration,
        and held-out testing (respectively).
    n_epochs : int
        Number of training epochs for the autoregressive U-Net.
    unet_features : int
        Base channel width for the U-Net encoder / FNO lifted width.
    model_kind : {"unet", "fno"}
        Neural surrogate architecture.
    fno_modes, fno_layers : int
        FNO-only hyperparameters: retained Fourier modes and number of
        spectral blocks.
    model_path : str or None
        Path to the surrogate checkpoint. If None, defaults to
        `Expts/Advection/models/advection_<kind>.pt`. If the file exists
        and `retrain=False`, the weights are loaded instead of retraining.
    retrain : bool
        If True, force retraining and overwrite any existing checkpoint.
    """
    if model_path is None:
        model_path = _default_model_path(model_kind)
    method = _method_label(use_optim, use_langevin, use_generator, use_vi, vi_covariance)
    print(f"Running Advection Experiment (surrogate={model_kind}, noise={noise_type}, "
          f"cp={cp_mode}, method={method})...")

    # 1. Setup Simulation
    Nx = 60
    Nt = 40
    x_min, x_max = 0.0, 2.0
    t_end = 0.5
    v = 1.0
    sim = Advection_1d(Nx, Nt, x_min, x_max, t_end)
    dt, dx = sim.dt, sim.dx

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Train (or load) autoregressive surrogate on (u_t, u_{t+1}) pairs
    model = build_model(kind=model_kind, features=unet_features,
                        modes=fno_modes, n_layers=fno_layers)
    if (not retrain) and os.path.exists(model_path):
        print(f"  [1/5] Loading cached {model_kind.upper()} weights from {model_path}...")
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model = model.to(device)
    else:
        reason = "forced retrain" if retrain else "no cached checkpoint"
        print(f"  [1/5] Training 1D {model_kind.upper()} surrogate ({reason}; "
              f"{n_train} trajectories, {n_epochs} epochs, device={device})...")
        train_inputs, train_targets, _ = generate_training_data(sim, n_train, v=v, seed=0)
        train_unet(model, train_inputs, train_targets,
                   n_epochs=n_epochs, batch_size=64, lr=1e-3, device=device)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"        Saved {model_kind.upper()} weights to {model_path}")

    # 3. Roll out calibration + test trajectories under the trained U-Net
    print(f"  [2/5] Generating calibration ({n_cal}) and test ({n_test}) rollouts...")
    u_cal_truth, u_cal_pred = evaluate(sim, model, n_cal, v=v, seed=1, device=device)
    u_test_truth, u_test_pred = evaluate(sim, model, n_test, v=v, seed=2, device=device)

    # 4. Define Residual Operator: u_t + v * u_x = 0
    D_t = ConvOperator(domain='t', order=1, scale=1.0/(2*dt))
    D_x = ConvOperator(domain='x', order=1, scale=1.0/(2*dx))

    def advection_residual(u):
        """Central-difference residual of `u_t + v*u_x` on a `[BS, Nt, Nx+3]` batch."""
        return D_t(u) + v * D_x(u)

    # 5. Calibrate qhat on U-Net residuals (marginal or joint CP)
    print(f"  [3/5] Calibrating qhat ({cp_mode}) on U-Net predictions...")
    res_cal = advection_residual(u_cal_pred)

    joint = cp_mode == "joint"
    if joint:
        qhat_scalar, modulation = calibrate_qhat_joint_from_residual(res_cal, alpha=0.1)
        qhat = qhat_scalar * modulation
        print(f"        Joint qhat_scalar = {qhat_scalar:.4f}, "
              f"modulation range = [{modulation.min():.4f}, {modulation.max():.4f}]")
    else:
        qhat = calibrate_qhat_from_residual(res_cal, alpha=0.1)
        print(f"        Marginal qhat = {qhat:.4f}")

    # 6. Invert bounds using perturbation sampling
    print(f"  [4/5] Perturbation sampling inversion (noise={noise_type}, method={method})...")
    test_idx = 0
    u_pred = u_test_pred[test_idx]
    u_truth = u_test_truth[test_idx]

    perturb_cfg = _build_perturbation_config(
        noise_type, seed=0,
        use_optim=use_optim, use_langevin=use_langevin, use_generator=use_generator,
        use_vi=use_vi, vi_covariance=vi_covariance, vi_rank=vi_rank,
    )

    # Advection domain is (Nt, Nx+3). Use interior slice for both axes.
    interior_slice = (slice(1, -1), slice(1, -1))

    bounds = perturbation_bounds_nd(
        pred_signal=u_pred.numpy(),
        residual_operator=advection_residual,
        qhat=qhat,
        config=perturb_cfg,
        interior_slice=interior_slice,
        joint=joint,
    )

    # 7. Plot bounds
    print("  [5/5] Saving bounds plot...")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    suffix = f"{model_kind}_{cp_mode}_{method}_{noise_type}"

    fig = plt.figure(figsize=(15, 5))
    t_idx = Nt // 2
    x_interior = sim.x[1:-1]

    ax1 = fig.add_subplot(1, 2, 1)
    _style_ax(ax1)
    ax1.set_title(f"Advection Profile at t={t_idx*dt:.2f} ({cp_mode}, {method}, {noise_type})")
    ax1.plot(x_interior, u_truth[t_idx, 1:-1], color=PALETTE['truth'],
             linewidth=1.8, label='Truth')
    ax1.plot(x_interior, u_pred[t_idx, 1:-1], color=PALETTE['prediction'],
             linewidth=1.8, linestyle='--', label=f'{model_kind.upper()} Prediction')
    ax1.fill_between(x_interior, bounds.lower[t_idx-1], bounds.upper[t_idx-1],
                     color=PALETTE['perturbation'], alpha=0.3, label='Perturbation Bound')
    ax1.set_xlabel('x')
    ax1.set_ylabel('u')
    ax1.legend(loc='best', edgecolor='#DDDDDD')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("Spatiotemporal Error Bounds (Lower)")
    im = ax2.imshow(bounds.lower, aspect='auto',
                    extent=[x_interior[0], x_interior[-1], t_end, 0])
    fig.colorbar(im, ax=ax2, label='u_lower')
    ax2.set_xlabel('x')
    ax2.set_ylabel('t')

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"advection_bounds_{suffix}.png"))
    plt.close(fig)
    print(f"        Saved advection_bounds_{suffix}.png")

    # 6. Optional: empirical coverage curve across alpha levels
    if plot_coverage:
        print("  [+] Computing empirical coverage curve...")
        alpha_levels = np.arange(0.10, 0.91, 0.20)

        preds_np = u_test_pred.numpy()
        truths_np = u_test_truth.numpy()

        coverage = empirical_coverage_curve_nd(
            preds=preds_np,
            truths=truths_np,
            residual_cal=res_cal,
            operator=advection_residual,
            alphas=alpha_levels,
            interior_slice=interior_slice,
            perturbation_config=perturb_cfg,
            cp_mode=cp_mode,
        )
        _save_coverage_plot(
            f"Advection ({cp_mode}, {method}, {noise_type})",
            coverage.nominal_coverage,
            coverage.empirical_coverage_perturbation,
            f"advection_coverage_{suffix}.png",
        )
        print(f"        Saved advection_coverage_{suffix}.png")

    print("  Advection experiment complete.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="1D Advection PRE-inversion experiment.")
    parser.add_argument('--noise-type', default='spatial', choices=NOISE_TYPES,
                        help=f'Noise model for perturbation sampling (default: spatial).')
    parser.add_argument('--cp-mode', default='marginal', choices=CP_MODES,
                        help='Conformal prediction mode (default: marginal).')
    parser.add_argument('--use-optim', action='store_true',
                        help='Differentiable rejection (inference-time optimisation).')
    parser.add_argument('--use-langevin', action='store_true',
                        help='Posterior sampling via Langevin dynamics.')
    parser.add_argument('--use-generator', action='store_true',
                        help='Generative boundary sampler.')
    parser.add_argument('--use-vi', dest='use_vi', action='store_true',
                        help='Variational inference sampler.')
    parser.add_argument('--vi-covariance', default='mean_field',
                        choices=('mean_field', 'low_rank', 'full'),
                        help='VI posterior covariance (default: mean_field).')
    parser.add_argument('--vi-rank', type=int, default=8,
                        help='Rank for vi-covariance=low_rank (default: 8).')
    parser.add_argument('--plot-coverage', action='store_true',
                        help='Also compute and plot empirical coverage curve over alpha.')
    parser.add_argument('--n-train', type=int, default=200,
                        help='Number of trajectories for U-Net training (default: 200).')
    parser.add_argument('--n-cal', type=int, default=100,
                        help='Number of CP calibration rollouts (default: 100).')
    parser.add_argument('--n-test', type=int, default=10,
                        help='Number of held-out test rollouts (default: 10).')
    parser.add_argument('--n-epochs', type=int, default=20,
                        help='U-Net training epochs (default: 200).')
    parser.add_argument('--unet-features', type=int, default=32,
                        help='U-Net base channel width / FNO lifted width (default: 32).')
    parser.add_argument('--model', dest='model_kind', default='unet', choices=MODEL_KINDS,
                        help='Neural surrogate architecture (default: unet).')
    parser.add_argument('--fno-modes', type=int, default=16,
                        help='FNO retained Fourier modes (default: 16).')
    parser.add_argument('--fno-layers', type=int, default=4,
                        help='FNO spectral blocks (default: 4).')
    parser.add_argument('--model-path', default=None,
                        help='Surrogate checkpoint path (default: Expts/Advection/models/advection_<kind>.pt).')
    parser.add_argument('--retrain', action='store_true',
                        help='Force retraining and overwrite any cached checkpoint.')
    args = parser.parse_args()

    _advanced = [args.use_optim, args.use_langevin, args.use_generator, args.use_vi]
    if sum(bool(f) for f in _advanced) > 1:
        parser.error("--use-optim / --use-langevin / --use-generator / --use-vi are mutually exclusive")

    run_advection_experiment(
        noise_type=args.noise_type,
        cp_mode=args.cp_mode,
        use_optim=args.use_optim,
        use_langevin=args.use_langevin,
        use_generator=args.use_generator,
        use_vi=args.use_vi,
        vi_covariance=args.vi_covariance,
        vi_rank=args.vi_rank,
        plot_coverage=args.plot_coverage,
        n_train=args.n_train,
        n_cal=args.n_cal,
        n_test=args.n_test,
        n_epochs=args.n_epochs,
        unet_features=args.unet_features,
        model_kind=args.model_kind,
        fno_modes=args.fno_modes,
        fno_layers=args.fno_layers,
        model_path=args.model_path,
        retrain=args.retrain,
    )
