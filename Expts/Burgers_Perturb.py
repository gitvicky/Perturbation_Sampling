"""
1D Burgers Perturbation-Sampling Experiment
===========================================

PRE-based conformal bound inversion for the 1D viscous Burgers equation

        u_t + u*u_x = nu*u_xx,    x in [0, 2],  t in [0, 1.25]

Uses the same dataset generation strategy as
`Neural_PDE/Numerical_Solvers/Burgers/Data_Gen.py` (LHS over initial-condition
parameters with the Burgers spectral solver), with on-disk caching to avoid
regenerating trajectories on repeated runs.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Utils.PRE.ConvOps_1d import ConvOperator
from Inversion_Strategies.inversion.residual_inversion import (
    PerturbationSamplingConfig,
    calibrate_qhat_from_residual,
    calibrate_qhat_joint_from_residual,
    empirical_coverage_curve_nd,
    perturbation_bounds_nd,
)
from Expts.Burgers.Burgers_Model import (
    build_model,
    train_model,
    make_train_pairs,
    evaluate_rollouts,
    load_or_generate_burgers_dataset,
)


PALETTE = {
    "perturbation": "#81B29A",
    "truth": "#2C2C2C",
    "prediction": "#5B7EC0",
    "target": "#9B9B9B",
}

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    }
)

NOISE_TYPES = ("spatial", "white", "gp", "bspline")
CP_MODES = ("marginal", "joint")
MODEL_KINDS = ("unet", "fno")

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "Figures")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "Burgers", "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "Burgers", "data")


def _default_model_path(kind: str) -> str:
    return os.path.join(MODELS_DIR, f"burgers_{kind}.pt")


def _default_data_path() -> str:
    return os.path.join(DATA_DIR, "Burgers_1d_cached.npz")


def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.grid(True, alpha=0.25, linewidth=0.5, color="#CCCCCC")
    ax.tick_params(direction="out", length=4, width=0.6)


def _method_label(use_optim, use_langevin, use_generator, use_vi=False, vi_covariance="mean_field"):
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


def _build_perturbation_config(
    noise_type,
    *,
    seed=0,
    use_optim=False,
    use_langevin=False,
    use_generator=False,
    use_vi=False,
    vi_covariance="mean_field",
    vi_rank=8,
    n_samples=None,
    batch_size=None,
    max_rounds=None,
    opt_steps=None,
    langevin_steps=None,
):
    # Burgers has a much larger grid than advection and a nonlinear residual
    # (u * u_x) + second derivative term, so we use lighter defaults.
    default_n_samples = 6000
    default_batch_size = 600
    default_max_rounds = 2
    default_opt_steps = 12
    default_langevin_steps = 10

    if use_optim:
        default_n_samples = 3000
    if use_langevin:
        default_n_samples = 2500

    common = dict(
        n_samples=default_n_samples if n_samples is None else int(n_samples),
        batch_size=default_batch_size if batch_size is None else int(batch_size),
        max_rounds=default_max_rounds if max_rounds is None else int(max_rounds),
        noise_std=0.03,
        seed=seed,
        use_optim=use_optim,
        use_langevin=use_langevin,
        use_generator=use_generator,
        use_vi=use_vi,
        vi_covariance=vi_covariance,
        vi_rank=vi_rank,
        vi_steps=300,
        opt_steps=default_opt_steps if opt_steps is None else int(opt_steps),
        langevin_steps=default_langevin_steps if langevin_steps is None else int(langevin_steps),
        gen_train_steps=200,
    )
    if noise_type == "spatial":
        return PerturbationSamplingConfig(noise_type="spatial", correlation_length=5.0, **common)
    if noise_type == "white":
        return PerturbationSamplingConfig(noise_type="white", **common)
    if noise_type == "gp":
        return PerturbationSamplingConfig(noise_type="gp", correlation_length=5.0, gp_kernel="rbf", **common)
    if noise_type == "bspline":
        return PerturbationSamplingConfig(noise_type="bspline", bspline_n_knots=16, **common)
    raise ValueError(f"Unknown noise type: {noise_type}. Choose from {NOISE_TYPES}")


def _save_coverage_plot(case_name, nominal, empirical, save_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    _style_ax(ax)
    nom = np.asarray(nominal, dtype=float)
    emp = np.asarray(empirical, dtype=float)
    ax.plot(nom, nom, color=PALETTE["target"], linewidth=1.8, linestyle=":", label="Target Coverage", zorder=1)
    ax.plot(
        nom,
        emp,
        color=PALETTE["perturbation"],
        linewidth=2.0,
        linestyle="-.",
        marker="D",
        markersize=4.5,
        markeredgewidth=0,
        label="Perturbation Sampling",
        zorder=3,
    )
    ax.fill_between(nom, 0, nom, color=PALETTE["target"], alpha=0.06, zorder=0)
    ax.set_xlabel("Nominal Coverage $(1 - \\alpha)$")
    ax.set_ylabel("Empirical Coverage")
    ax.set_title(f"{case_name}: Empirical Coverage")
    ax.legend(loc="lower right", edgecolor="#DDDDDD")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, save_name))
    plt.close(fig)


def run_burgers_experiment(
    noise_type="spatial",
    cp_mode="marginal",
    *,
    use_optim=False,
    use_langevin=False,
    use_generator=False,
    use_vi=False,
    vi_covariance="mean_field",
    vi_rank=8,
    plot_coverage=False,
    n_train=200,
    n_cal=100,
    n_test=20,
    n_epochs=20,
    model_features=32,
    model_kind="unet",
    fno_modes=16,
    fno_layers=4,
    model_path=None,
    retrain=False,
    data_path=None,
    data_seed=0,
    force_regen_data=False,
    split_seed=0,
    nu=0.002,
    n_samples=None,
    batch_size=None,
    max_rounds=None,
    opt_steps=None,
    langevin_steps=None,
):
    if model_path is None:
        model_path = _default_model_path(model_kind)
    if data_path is None:
        data_path = _default_data_path()

    method = _method_label(use_optim, use_langevin, use_generator, use_vi, vi_covariance)
    print(f"Running Burgers Experiment (surrogate={model_kind}, noise={noise_type}, cp={cp_mode}, method={method})...")

    n_val = int(n_cal + n_test)
    n_total = int(n_train + n_val)
    print(f"  [1/6] Loading/generating Burgers trajectories (n={n_total})...")
    all_traj, grid = load_or_generate_burgers_dataset(
        cache_path=data_path,
        n_sims=n_total,
        seed=data_seed,
        force_regen=force_regen_data,
    )

    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(all_traj.shape[0])
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]

    train_traj = all_traj[train_idx]
    val_traj = all_traj[val_idx]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(kind=model_kind, features=model_features, modes=fno_modes, n_layers=fno_layers)

    if (not retrain) and os.path.exists(model_path):
        print(f"  [2/6] Loading cached {model_kind.upper()} weights from {model_path}...")
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model = model.to(device)
    else:
        reason = "forced retrain" if retrain else "no cached checkpoint"
        print(
            f"  [2/6] Training {model_kind.upper()} surrogate ({reason}; "
            f"{n_train} trajectories, {n_epochs} epochs, device={device})..."
        )
        train_x, train_y = make_train_pairs(train_traj)
        train_model(model, train_x, train_y, n_epochs=n_epochs, batch_size=64, lr=1e-3, device=device)
        model_dir = os.path.dirname(model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"        Saved {model_kind.upper()} weights to {model_path}")

    print(f"  [3/6] Rolling out validation trajectories (n={n_val})...")
    u_val_truth, u_val_pred = evaluate_rollouts(model, val_traj, device=device)

    dt, dx = grid.dt, grid.dx
    D_t = ConvOperator(domain="t", order=1, scale=1.0 / (2 * dt))
    D_x = ConvOperator(domain="x", order=1, scale=1.0 / (2 * dx))
    D_xx = ConvOperator(domain="x", order=2, scale=1.0 / (dx**2))

    def burgers_residual(u):
        return D_t(u) + u * D_x(u) - nu * D_xx(u)

    print(f"  [4/6] Calibrating qhat ({cp_mode}) on surrogate residuals...")
    res_cal = burgers_residual(u_val_pred)
    joint = cp_mode == "joint"
    if joint:
        qhat_scalar, modulation = calibrate_qhat_joint_from_residual(res_cal, alpha=0.1)
        qhat = qhat_scalar * modulation
        print(
            f"        Joint qhat_scalar = {qhat_scalar:.4f}, "
            f"modulation range = [{modulation.min():.4f}, {modulation.max():.4f}]"
        )
    else:
        qhat = calibrate_qhat_from_residual(res_cal, alpha=0.1)
        print(f"        Marginal qhat = {qhat:.4f}")

    print(f"  [5/6] Perturbation sampling inversion (noise={noise_type}, method={method})...")
    u_pred = u_val_pred[0]
    perturb_cfg = _build_perturbation_config(
        noise_type,
        seed=0,
        use_optim=use_optim,
        use_langevin=use_langevin,
        use_generator=use_generator,
        use_vi=use_vi,
        vi_covariance=vi_covariance,
        vi_rank=vi_rank,
        n_samples=n_samples,
        batch_size=batch_size,
        max_rounds=max_rounds,
        opt_steps=opt_steps,
        langevin_steps=langevin_steps,
    )
    interior_slice = (slice(1, -1), slice(1, -1))
    bounds = perturbation_bounds_nd(
        pred_signal=u_pred.numpy(),
        residual_operator=burgers_residual,
        qhat=qhat,
        config=perturb_cfg,
        interior_slice=interior_slice,
        joint=joint,
    )

    print("  [6/6] Saving bounds plot...")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    suffix = f"{model_kind}_{cp_mode}_{method}_{noise_type}"

    fig = plt.figure(figsize=(15, 5))
    t_idx = u_pred.shape[0] // 2
    x_interior = grid.x[1:-1]

    ax1 = fig.add_subplot(1, 2, 1)
    _style_ax(ax1)
    ax1.set_title(f"Burgers Profile at t={t_idx * dt:.2f} ({cp_mode}, {method}, {noise_type})")
    ax1.plot(x_interior, u_val_truth[0][t_idx, 1:-1], color=PALETTE["truth"], linewidth=1.8, label="Truth")
    ax1.plot(
        x_interior,
        u_pred[t_idx, 1:-1],
        color=PALETTE["prediction"],
        linewidth=1.8,
        linestyle="--",
        label=f"{model_kind.upper()} Prediction",
    )
    ax1.fill_between(
        x_interior,
        bounds.lower[t_idx - 1],
        bounds.upper[t_idx - 1],
        color=PALETTE["perturbation"],
        alpha=0.3,
        label="Perturbation Bound",
    )
    ax1.set_xlabel("x")
    ax1.set_ylabel("u")
    ax1.legend(loc="best", edgecolor="#DDDDDD")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("Spatiotemporal Error Bounds (Lower)")
    im = ax2.imshow(bounds.lower, aspect="auto", extent=[x_interior[0], x_interior[-1], dt * (u_pred.shape[0] - 1), 0])
    fig.colorbar(im, ax=ax2, label="u_lower")
    ax2.set_xlabel("x")
    ax2.set_ylabel("t")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"burgers_bounds_{suffix}.png"))
    plt.close(fig)
    print(f"        Saved burgers_bounds_{suffix}.png")

    if plot_coverage:
        print("  [+] Computing empirical coverage curve...")
        alpha_levels = np.arange(0.10, 0.91, 0.20)
        coverage = empirical_coverage_curve_nd(
            preds=u_val_pred.numpy(),
            truths=u_val_truth.numpy(),
            residual_cal=res_cal,
            operator=burgers_residual,
            alphas=alpha_levels,
            interior_slice=interior_slice,
            perturbation_config=perturb_cfg,
            cp_mode=cp_mode,
        )
        _save_coverage_plot(
            f"Burgers ({cp_mode}, {method}, {noise_type})",
            coverage.nominal_coverage,
            coverage.empirical_coverage_perturbation,
            f"burgers_coverage_{suffix}.png",
        )
        print(f"        Saved burgers_coverage_{suffix}.png")

    print("  Burgers experiment complete.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="1D Burgers PRE-inversion experiment.")
    parser.add_argument("--noise-type", default="spatial", choices=NOISE_TYPES)
    parser.add_argument("--cp-mode", default="marginal", choices=CP_MODES)
    parser.add_argument("--use-optim", action="store_true")
    parser.add_argument("--use-langevin", action="store_true")
    parser.add_argument("--use-generator", action="store_true")
    parser.add_argument("--use-vi", dest="use_vi", action="store_true")
    parser.add_argument("--vi-covariance", default="mean_field", choices=("mean_field", "low_rank", "full"))
    parser.add_argument("--vi-rank", type=int, default=8)
    parser.add_argument("--plot-coverage", action="store_true")
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument(
        "--n-cal",
        type=int,
        default=100,
        help="Validation trajectories contributed by former calibration count.",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=20,
        help="Validation trajectories contributed by former test count.",
    )
    parser.add_argument("--n-epochs", type=int, default=20)
    parser.add_argument("--model-features", type=int, default=32)
    parser.add_argument("--model", dest="model_kind", default="unet", choices=MODEL_KINDS)
    parser.add_argument("--fno-modes", type=int, default=16)
    parser.add_argument("--fno-layers", type=int, default=4)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--data-path", default=None, help="Cached dataset path.")
    parser.add_argument("--data-seed", type=int, default=0, help="LHS seed for generated Burgers data.")
    parser.add_argument("--split-seed", type=int, default=0, help="Train/validation split seed.")
    parser.add_argument("--force-regen-data", action="store_true", help="Ignore cache and regenerate dataset.")
    parser.add_argument("--nu", type=float, default=0.002, help="Viscosity coefficient in residual.")
    parser.add_argument("--n-samples", type=int, default=None, help="Perturbation samples (auto-tuned if unset).")
    parser.add_argument("--batch-size", type=int, default=None, help="Perturbation batch size (auto-tuned if unset).")
    parser.add_argument("--max-rounds", type=int, default=None, help="Max perturbation rounds (auto-tuned if unset).")
    parser.add_argument("--opt-steps", type=int, default=None, help="Optimisation rescue steps (auto-tuned if unset).")
    parser.add_argument("--langevin-steps", type=int, default=None, help="Langevin steps (auto-tuned if unset).")
    args = parser.parse_args()

    _advanced = [args.use_optim, args.use_langevin, args.use_generator, args.use_vi]
    if sum(bool(f) for f in _advanced) > 1:
        parser.error("--use-optim / --use-langevin / --use-generator / --use-vi are mutually exclusive")

    run_burgers_experiment(
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
        model_features=args.model_features,
        model_kind=args.model_kind,
        fno_modes=args.fno_modes,
        fno_layers=args.fno_layers,
        model_path=args.model_path,
        retrain=args.retrain,
        data_path=args.data_path,
        data_seed=args.data_seed,
        force_regen_data=args.force_regen_data,
        split_seed=args.split_seed,
        nu=args.nu,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        max_rounds=args.max_rounds,
        opt_steps=args.opt_steps,
        langevin_steps=args.langevin_steps,
    )
