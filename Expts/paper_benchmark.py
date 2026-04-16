"""
Paper benchmarking driver for ODE experiments (SHO, DHO, Duffing).

Runs all requested combinations:
  - methods: MC, Optim, Langevin, VI-Full
  - noise: spatial, white, gp, bspline
  - alpha: 0.1, 0.25, 0.5, 0.75, 0.9

Outputs:
  - Paper/images/* (alpha=0.1 bounds and empirical coverage plots, spatial-noise method comparisons)
  - Paper/images/paper_results_full.csv
  - Paper/images/paper_results_noise_avg.csv
  - Paper/images/paper_runtime_alpha_01.csv
  - Paper/results_tables.tex
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import replace
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(ROOT := os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Utils.PRE.ConvOps_0d import ConvOperator
from Inversion_Strategies.inversion.residual_inversion import (
    PerturbationSamplingConfig,
    calibrate_qhat_from_residual,
    perturbation_bounds_1d,
)


PAPER_DIR = os.path.join(ROOT, "Paper")
IMAGES_DIR = os.path.join(PAPER_DIR, "images")
RESULTS_TEX = os.path.join(PAPER_DIR, "results_tables.tex")

ALPHAS = [0.10, 0.25, 0.50, 0.75, 0.90]
NOISES = ["spatial", "white", "gp", "bspline"]
METHODS = ["MC", "Optim", "Langevin", "VI-Full"]
CASE_SEEDS = {"sho": 101, "dho": 202, "duffing": 303}
METHOD_SEEDS = {"MC": 11, "Optim": 22, "Langevin": 33, "VI-Full": 44}
NOISE_SEEDS = {"spatial": 1, "white": 2, "gp": 3, "bspline": 4}


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _cfg_for(method: str, noise: str, seed: int) -> PerturbationSamplingConfig:
    cfg = PerturbationSamplingConfig(
        n_samples=6000,
        batch_size=500,
        max_rounds=5,
        noise_type=noise,
        noise_std=0.5,
        correlation_length=24.0,
        gp_kernel="rbf",
        bspline_n_knots=16,
        seed=seed,
        opt_steps=40,
        langevin_steps=20,
        vi_steps=300,
        vi_n_mc=8,
        vi_covariance="full",
        vi_rank=8,
    )
    if method == "Optim":
        return replace(cfg, use_optimisation=True)
    if method == "Langevin":
        return replace(cfg, use_langevin=True)
    if method == "VI-Full":
        return replace(cfg, use_vi=True, vi_covariance="full")
    return cfg


def _prepare_sho(seed: int, epochs: int, n_eval: int):
    from Expts.SHO.SHO_NODE import (
        HarmonicOscillator,
        ODEFunc,
        evaluate,
        generate_training_data,
        train_neural_ode,
    )

    _set_seed(seed)
    oscillator = HarmonicOscillator(k=1.0, m=1.0)
    t_span = (0, 10)
    n_points = 100
    n_trajectories = 50
    t_train, states, derivs = generate_training_data(oscillator, t_span, n_points, n_trajectories)
    func = ODEFunc(hidden_dim=64)
    train_neural_ode(func, t_train, states, derivs, n_epochs=epochs, batch_size=16)
    t, numerical_sol, neural_sol = evaluate(
        oscillator, func, t_span, n_points, x_range=(-2, 2), v_range=(-2, 2), n_solves=n_eval
    )
    pos = torch.tensor(neural_sol[..., 0], dtype=torch.float32)
    dt = t[1] - t[0]
    D_tt = ConvOperator(order=2)
    D_identity = ConvOperator(order=0)
    D_identity.kernel = torch.tensor([0.0, 1.0, 0.0])
    op = ConvOperator(conv="spectral")
    op.kernel = D_tt.kernel + dt**2 * D_identity.kernel
    return t, numerical_sol[..., 0], neural_sol[..., 0], pos, op


def _prepare_dho(seed: int, epochs: int, n_eval: int):
    from Expts.DHO.DHO_NODE import (
        DampedHarmonicOscillator,
        ODEFunc,
        evaluate,
        generate_training_data,
        train_neural_ode,
    )

    _set_seed(seed)
    oscillator = DampedHarmonicOscillator(k=1.0, m=1.0, c=0.2)
    t_span = (0, 15)
    n_points = 100
    n_trajectories = 50
    t_train, states, derivs = generate_training_data(oscillator, t_span, n_points, n_trajectories)
    func = ODEFunc(hidden_dim=64)
    train_neural_ode(func, t_train, states, derivs, n_epochs=epochs, batch_size=16)
    t, numerical_sol, neural_sol = evaluate(
        oscillator, func, t_span, n_points, x_range=(-2, 2), v_range=(-2, 2), n_solves=n_eval
    )
    pos = torch.tensor(neural_sol[..., 0], dtype=torch.float32)
    dt = t[1] - t[0]
    D_t = ConvOperator(order=1)
    D_tt = ConvOperator(order=2)
    D_identity = ConvOperator(order=0)
    D_identity.kernel = torch.tensor([0.0, 1.0, 0.0])
    op = ConvOperator(conv="spectral")
    op.kernel = 2 * D_tt.kernel + dt * 0.2 * D_t.kernel + 2 * dt**2 * D_identity.kernel
    return t, numerical_sol[..., 0], neural_sol[..., 0], pos, op


def _prepare_duffing(seed: int, epochs: int, n_eval: int):
    from Expts.Duffing.Duffing_NODE import (
        DuffingOscillator,
        DuffingResidualOperator,
        ODEFunc,
        evaluate,
        generate_training_data,
        train_neural_ode,
    )

    _set_seed(seed)
    alpha_coeff, beta_coeff, delta_coeff = 1.0, 0.5, 0.2
    oscillator = DuffingOscillator(alpha=alpha_coeff, beta=beta_coeff, delta=delta_coeff)
    t_span = (0, 15)
    n_points = 100
    n_trajectories = 50
    t_train, states, derivs = generate_training_data(oscillator, t_span, n_points, n_trajectories)
    func = ODEFunc(hidden_dim=64)
    train_neural_ode(func, t_train, states, derivs, n_epochs=epochs, batch_size=16)
    t, numerical_sol, neural_sol = evaluate(
        oscillator, func, t_span, n_points, x_range=(-2, 2), v_range=(-2, 2), n_solves=n_eval
    )
    pos = torch.tensor(neural_sol[..., 0], dtype=torch.float32)
    dt = t[1] - t[0]
    op = DuffingResidualOperator(alpha_coeff, beta_coeff, delta_coeff, dt)
    return t, numerical_sol[..., 0], neural_sol[..., 0], pos, op


def _plot_spatial_method_bounds(case: str, t, truth, pred, test_idx: int, method_bounds: dict[str, np.ndarray]):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    tt = t[1:-1]
    for ax, method in zip(axes, METHODS):
        lower, upper = method_bounds[method]
        ax.fill_between(tt, lower, upper, alpha=0.2, label="Bounds")
        ax.plot(tt, pred[test_idx, 1:-1], linewidth=1.5, label="Prediction")
        ax.plot(tt, truth[test_idx, 1:-1], linewidth=1.2, label="Truth")
        ax.set_title(method)
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle(f"{case.upper()}: alpha=0.10 bounds by method (spatial noise)")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(os.path.join(IMAGES_DIR, f"{case}_alpha01_methods_spatial.png"), dpi=200)
    plt.close(fig)


def _plot_spatial_method_coverage(case: str, method_cov: dict[str, list[tuple[float, float]]]):
    fig, ax = plt.subplots(figsize=(7, 5))
    nominal = np.array([1.0 - a for a in ALPHAS], dtype=float)
    ax.plot(nominal, nominal, "k:", label="Target")
    for method in METHODS:
        emp = np.array([v for _, v in sorted(method_cov[method], key=lambda x: x[0])], dtype=float)
        ax.plot(nominal, emp, marker="o", label=method)
    ax.set_xlabel("Nominal coverage $(1-\\alpha)$")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(f"{case.upper()}: empirical coverage by method (spatial noise)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, f"{case}_coverage_methods_spatial.png"), dpi=200)
    plt.close(fig)


def _safe_bounds(pred_signal, operator, qhat, config, interior_slice):
    trial_cfg = config
    for _ in range(4):
        try:
            return perturbation_bounds_1d(
                pred_signal=pred_signal,
                residual_operator=operator,
                qhat=qhat,
                interior_slice=interior_slice,
                config=trial_cfg,
                joint=False,
            )
        except RuntimeError:
            trial_cfg = replace(
                trial_cfg,
                n_samples=int(trial_cfg.n_samples * 2),
                max_rounds=trial_cfg.max_rounds + 1,
            )
    pred_interior = np.asarray(pred_signal[interior_slice], dtype=float)
    return perturbation_bounds_1d(
        pred_signal=pred_signal,
        residual_operator=operator,
        qhat=qhat,
        interior_slice=interior_slice,
        config=trial_cfg,
        fallback_lower=pred_interior,
        fallback_upper=pred_interior,
        joint=False,
    )


def _write_csv(path: str, rows: list[dict], fieldnames: list[str]):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean_group(rows: list[dict], keys: list[str], mean_cols: list[str]) -> list[dict]:
    grouped: dict[tuple, dict[str, float]] = {}
    counts: dict[tuple, int] = {}
    for r in rows:
        gk = tuple(r[k] for k in keys)
        if gk not in grouped:
            grouped[gk] = {k: float(r[k]) for k in mean_cols}
            counts[gk] = 1
        else:
            for c in mean_cols:
                grouped[gk][c] += float(r[c])
            counts[gk] += 1
    out = []
    for gk, sums in grouped.items():
        row = {k: v for k, v in zip(keys, gk)}
        n = counts[gk]
        for c in mean_cols:
            row[c] = sums[c] / n
        out.append(row)
    return out


def _write_latex_tables(noise_avg_rows: list[dict], runtime_rows: list[dict]):
    lines = []
    lines.append("% Auto-generated by Expts/paper_benchmark.py")
    lines.append("\\begin{table*}[t]")
    lines.append("\\caption{Noise-averaged empirical coverage and mean interval width across target $\\alpha$ levels.}")
    lines.append("\\label{tab:coverage-width-noise-avg}")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{llccc}")
    lines.append("\\toprule")
    lines.append("Experiment & Method & $\\alpha$ & Empirical coverage & Avg. width \\\\")
    lines.append("\\midrule")
    for r in sorted(noise_avg_rows, key=lambda x: (x["experiment"], x["method"], float(x["alpha"]))):
        lines.append(
            f'{r["experiment"].upper()} & {r["method"]} & {float(r["alpha"]):.2f} & '
            f'{float(r["empirical_coverage"]):.3f} & {float(r["average_width"]):.3f} \\\\'
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    lines.append("")
    lines.append("\\begin{table}[t]")
    lines.append("\\caption{Average runtime (seconds) for a single-$\\alpha$ inversion at $\\alpha=0.10$, averaged over noise types.}")
    lines.append("\\label{tab:runtime-alpha01}")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{llc}")
    lines.append("\\toprule")
    lines.append("Experiment & Method & Runtime (s) \\\\")
    lines.append("\\midrule")
    for r in sorted(runtime_rows, key=lambda x: (x["experiment"], x["method"])):
        lines.append(
            f'{r["experiment"].upper()} & {r["method"]} & {float(r["runtime_seconds_alpha_01"]):.2f} \\\\'
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    with open(RESULTS_TEX, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark all ODE paper configurations.")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs per ODE model.")
    parser.add_argument("--n-eval", type=int, default=30, help="Number of evaluation trajectories.")
    parser.add_argument("--seed", type=int, default=123, help="Global seed.")
    args = parser.parse_args()

    os.makedirs(IMAGES_DIR, exist_ok=True)
    _set_seed(args.seed)

    case_builders: dict[str, Callable] = {
        "sho": _prepare_sho,
        "dho": _prepare_dho,
        "duffing": _prepare_duffing,
    }

    full_rows: list[dict] = []

    for case_name, builder in case_builders.items():
        print(f"\n=== Preparing {case_name.upper()} model/data ===")
        t, truth, pred, pred_torch, operator = builder(
            seed=args.seed + CASE_SEEDS[case_name],
            epochs=args.epochs,
            n_eval=args.n_eval,
        )
        residuals = operator(pred_torch)
        n_cal = int(0.8 * len(residuals))
        residual_cal = residuals[:n_cal]
        test_idx = n_cal
        eval_indices = list(range(n_cal, len(pred)))
        interior = slice(1, -1)

        method_bounds_for_plot: dict[str, np.ndarray] = {}
        method_cov_for_plot: dict[str, list[tuple[float, float]]] = {m: [] for m in METHODS}

        for method in METHODS:
            for noise in NOISES:
                combo_seed = (
                    args.seed
                    + CASE_SEEDS[case_name]
                    + METHOD_SEEDS[method]
                    + NOISE_SEEDS[noise]
                )
                config = _cfg_for(method=method, noise=noise, seed=combo_seed)
                print(f"Running {case_name.upper()} | {method} | {noise}")
                for alpha in ALPHAS:
                    qhat = calibrate_qhat_from_residual(residual_cal, alpha=alpha)

                    t0 = time.perf_counter()
                    bounds_test = _safe_bounds(pred[test_idx], operator, qhat, config, interior)
                    elapsed = time.perf_counter() - t0

                    flags = []
                    for idx in eval_indices:
                        bounds_i = _safe_bounds(pred[idx], operator, qhat, config, interior)
                        truth_i = truth[idx][1:-1]
                        inside = np.logical_and(truth_i >= bounds_i.lower, truth_i <= bounds_i.upper).all()
                        flags.append(float(inside))
                    empirical_cov = float(np.mean(flags))
                    avg_width = float(np.mean(bounds_test.width))

                    full_rows.append(
                        {
                            "experiment": case_name,
                            "method": method,
                            "noise_type": noise,
                            "alpha": alpha,
                            "nominal_coverage": 1.0 - alpha,
                            "empirical_coverage": empirical_cov,
                            "average_width": avg_width,
                            "runtime_seconds_single_alpha": elapsed if abs(alpha - 0.10) < 1e-12 else "",
                        }
                    )

                    if noise == "spatial":
                        method_cov_for_plot[method].append((alpha, empirical_cov))
                        if abs(alpha - 0.10) < 1e-12:
                            method_bounds_for_plot[method] = (bounds_test.lower, bounds_test.upper)

        _plot_spatial_method_bounds(case_name, t, truth, pred, test_idx, method_bounds_for_plot)
        _plot_spatial_method_coverage(case_name, method_cov_for_plot)

    full_csv = os.path.join(IMAGES_DIR, "paper_results_full.csv")
    _write_csv(
        full_csv,
        full_rows,
        [
            "experiment",
            "method",
            "noise_type",
            "alpha",
            "nominal_coverage",
            "empirical_coverage",
            "average_width",
            "runtime_seconds_single_alpha",
        ],
    )

    noise_avg = _mean_group(
        rows=full_rows,
        keys=["experiment", "method", "alpha"],
        mean_cols=["empirical_coverage", "average_width"],
    )
    noise_avg_csv = os.path.join(IMAGES_DIR, "paper_results_noise_avg.csv")
    _write_csv(
        noise_avg_csv,
        noise_avg,
        ["experiment", "method", "alpha", "empirical_coverage", "average_width"],
    )

    alpha01_runtime_rows = [
        r for r in full_rows
        if str(r["runtime_seconds_single_alpha"]) != ""
    ]
    runtime_avg = _mean_group(
        rows=alpha01_runtime_rows,
        keys=["experiment", "method"],
        mean_cols=["runtime_seconds_single_alpha"],
    )
    runtime_rows = [
        {
            "experiment": r["experiment"],
            "method": r["method"],
            "runtime_seconds_alpha_01": r["runtime_seconds_single_alpha"],
        }
        for r in runtime_avg
    ]
    runtime_csv = os.path.join(IMAGES_DIR, "paper_runtime_alpha_01.csv")
    _write_csv(runtime_csv, runtime_rows, ["experiment", "method", "runtime_seconds_alpha_01"])

    _write_latex_tables(noise_avg_rows=noise_avg, runtime_rows=runtime_rows)
    print(f"\nSaved:\n- {full_csv}\n- {noise_avg_csv}\n- {runtime_csv}\n- {RESULTS_TEX}")


if __name__ == "__main__":
    main()
