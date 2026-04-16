"""
Hyperparameter-impact benchmark for perturbation sampling.

Sweeps five hyperparameters of `PerturbationSamplingConfig` and measures
empirical coverage + mean bound width on the SHO, DHO and Duffing test
cases, under both marginal and joint CP, for the Langevin and
Optimisation advanced samplers.

Hyperparameter roles
--------------------
    n_samples        Total candidate perturbations drawn per trajectory.
                     Bounds are per-point min/max over accepted samples.
                     More samples -> tighter Monte-Carlo bounds, higher
                     joint-acceptance rate.

    max_rounds       Maximum retry rounds per `std_retry_factor`. Each
                     round draws fresh noise; protects against timesteps
                     with zero accepted samples.

    noise_std        Std of the base perturbation noise (or latent scale
                     when `optimise_in_latent=True`). Sets search radius.
                     Too small -> bounds collapse on prediction.
                     Too large -> rejection rate explodes.

    opt_steps        (Optim only) Adam steps to rescue rejected samples
                     via the boundary hinge loss. More steps -> more
                     rescued extremal candidates -> wider bounds +
                     higher acceptance.

    langevin_steps   (Langevin only) MCMC random-walk steps per batch.
                     More steps -> better mixing into the valid manifold.

Usage
-----
    python Inversion_Strategies/tests/PerturbSampling/benchmark_hyperparameters.py \
        --cases sho dho duffing \
        --methods optim langevin \
        --cp-modes marginal joint \
        --n-eval 20
"""
import argparse
import csv
import os
import sys
from dataclasses import asdict

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Make project root importable from inside the tests folder.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

from Utils.PRE.ConvOps_0d import ConvOperator
from Inversion_Strategies.inversion.residual_inversion import (
    PerturbationSamplingConfig,
    calibrate_qhat_from_residual,
    calibrate_qhat_joint_from_residual,
    perturbation_bounds_1d,
    _trajectory_coverage_nd,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Baseline config (everything not being swept is held here).
BASELINE = dict(
    n_samples=10000,
    batch_size=1000,
    max_rounds=5,
    noise_type="spatial",
    noise_std=0.10,
    correlation_length=24.0,
    opt_steps=200,
    langevin_steps=20,
    seed=123,
)

# Hyperparameter grids.
SWEEPS = {
    "n_samples":      [1000, 5000, 10000, 20000, 40000],
    "max_rounds":     [1, 2, 5, 10, 20],
    "noise_std":      [0.01, 0.05, 0.1, 0.2, 0.5],
    "opt_steps":      [50, 100, 200, 500, 1000],
    "langevin_steps": [5, 10, 20, 50, 100],
}

# Only sweep hyperparameters relevant to each method.
HP_FOR_METHOD = {
    "optim":    ["n_samples", "max_rounds", "noise_std", "opt_steps"],
    "langevin": ["n_samples", "max_rounds", "noise_std", "langevin_steps"],
}

ALPHA = 0.10                 # target 90% nominal coverage
CASES = ("sho", "dho", "duffing")
CP_MODES = ("marginal", "joint")
METHODS = ("optim", "langevin")

FIG_DIR = os.path.join(os.path.dirname(__file__), "benchmark_figures")
os.makedirs(FIG_DIR, exist_ok=True)
CSV_PATH = os.path.join(FIG_DIR, "benchmark_results.csv")


# ---------------------------------------------------------------------------
# Trainers — run once per case then cached in memory
# ---------------------------------------------------------------------------

def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_sho(seed: int = 0) -> dict:
    from Expts.SHO.SHO_NODE import (
        HarmonicOscillator, ODEFunc,
        generate_training_data, train_neural_ode, evaluate,
    )
    _seed_all(seed)
    m, k = 1.0, 1.0
    osc = HarmonicOscillator(k, m)
    t_train, states, derivs = generate_training_data(osc, (0, 10), 100, 50)
    func = ODEFunc(hidden_dim=64)
    train_neural_ode(func, t_train, states, derivs, n_epochs=500, batch_size=16)
    t, num_sol, neural_sol = evaluate(
        osc, func, (0, 10), 100, x_range=(-2, 2), v_range=(-2, 2), n_solves=100,
    )
    pos = torch.tensor(neural_sol[..., 0], dtype=torch.float32)
    dt = t[1] - t[0]

    D_tt = ConvOperator(order=2)
    D_identity = ConvOperator(order=0)
    D_identity.kernel = torch.tensor([0.0, 1.0, 0.0])
    op = ConvOperator(conv='spectral')
    op.kernel = m * D_tt.kernel + dt ** 2 * k * D_identity.kernel
    return dict(t=t, pos=pos, num_sol=num_sol, neural_sol=neural_sol, op=op)


def train_dho(seed: int = 0) -> dict:
    from Expts.DHO.DHO_NODE import (
        DampedHarmonicOscillator, ODEFunc,
        generate_training_data, train_neural_ode, evaluate,
    )
    _seed_all(seed)
    m, k, c = 1.0, 1.0, 0.2
    osc = DampedHarmonicOscillator(k, m, c)
    t_train, states, derivs = generate_training_data(osc, (0, 15), 100, 50)
    func = ODEFunc(hidden_dim=64)
    train_neural_ode(func, t_train, states, derivs, n_epochs=500, batch_size=16)
    t, num_sol, neural_sol = evaluate(
        osc, func, (0, 15), 100, x_range=(-2, 2), v_range=(-2, 2), n_solves=100,
    )
    pos = torch.tensor(neural_sol[..., 0], dtype=torch.float32)
    dt = t[1] - t[0]

    D_t = ConvOperator(order=1)
    D_tt = ConvOperator(order=2)
    D_identity = ConvOperator(order=0)
    D_identity.kernel = torch.tensor([0.0, 1.0, 0.0])
    op = ConvOperator(conv='spectral')
    op.kernel = 2 * m * D_tt.kernel + dt * c * D_t.kernel + 2 * dt ** 2 * k * D_identity.kernel
    return dict(t=t, pos=pos, num_sol=num_sol, neural_sol=neural_sol, op=op)


def train_duffing(seed: int = 0) -> dict:
    from Expts.Duffing.Duffing_NODE import (
        DuffingOscillator, DuffingResidualOperator, ODEFunc,
        generate_training_data, train_neural_ode, evaluate,
    )
    _seed_all(seed)
    a, b, d = 1.0, 0.5, 0.2
    osc = DuffingOscillator(a, b, d)
    t_train, states, derivs = generate_training_data(osc, (0, 15), 100, 50)
    func = ODEFunc(hidden_dim=64)
    train_neural_ode(func, t_train, states, derivs, n_epochs=500, batch_size=16)
    t, num_sol, neural_sol = evaluate(
        osc, func, (0, 15), 100, x_range=(-2, 2), v_range=(-2, 2), n_solves=100,
    )
    pos = torch.tensor(neural_sol[..., 0], dtype=torch.float32)
    dt = t[1] - t[0]
    op = DuffingResidualOperator(a, b, d, dt)
    return dict(t=t, pos=pos, num_sol=num_sol, neural_sol=neural_sol, op=op)


TRAINERS = {"sho": train_sho, "dho": train_dho, "duffing": train_duffing}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def build_config(method: str, override: dict) -> PerturbationSamplingConfig:
    """Assemble a PerturbationSamplingConfig from BASELINE + override."""
    params = dict(BASELINE)
    params.update(override)
    params["use_optimisation"] = (method == "optim")
    params["use_langevin"] = (method == "langevin")
    return PerturbationSamplingConfig(**params)


def evaluate_coverage_and_width(
    data: dict, cp_mode: str, config: PerturbationSamplingConfig,
    *, n_eval: int, alpha: float = ALPHA,
) -> tuple[float, float, float]:
    """Compute empirical coverage, mean width, mean acceptance count.

    Uses an 80/20 calibration split on the pre-computed residuals and runs
    the sampler on `n_eval` held-out trajectories.
    """
    pos = data["pos"]
    op = data["op"]
    preds = data["neural_sol"][..., 0]
    truths = data["num_sol"][..., 0]

    res = op(pos)
    n_cal = int(0.8 * len(res))
    residual_cal = res[:n_cal]

    joint = (cp_mode == "joint")
    if joint:
        qhat_scalar, modulation = calibrate_qhat_joint_from_residual(residual_cal, alpha=alpha)
        qhat = qhat_scalar * modulation
    else:
        qhat = calibrate_qhat_from_residual(residual_cal, alpha=alpha)

    n_eval = min(n_eval, preds.shape[0] - n_cal)
    cover_flags: list[float] = []
    widths: list[float] = []
    for i in range(n_cal, n_cal + n_eval):
        bounds = perturbation_bounds_1d(
            pred_signal=preds[i],
            residual_operator=op,
            qhat=qhat,
            interior_slice=slice(1, -1),
            config=config,
            joint=joint,
        )
        truth_i = truths[i][1:-1]
        cover_flags.append(_trajectory_coverage_nd(truth_i[None, :], bounds))
        widths.append(float(np.mean(bounds.width)))
    return float(np.mean(cover_flags)), float(np.mean(widths)), float(n_eval)


def run_sweeps(
    cases: list[str], methods: list[str], cp_modes: list[str],
    *, n_eval: int, hp_filter: list[str] | None = None,
) -> list[dict]:
    """Run the full cross-product sweep. Returns rows for CSV/plotting."""
    rows: list[dict] = []

    # Train each case once — expensive — then reuse.
    trained: dict[str, dict] = {}
    for case in cases:
        print(f"\n=== Training {case.upper()} ===")
        trained[case] = TRAINERS[case]()

    total = 0
    for case in cases:
        for method in methods:
            hps = HP_FOR_METHOD[method]
            if hp_filter:
                hps = [h for h in hps if h in hp_filter]
            for _ in hps:
                total += len(SWEEPS[_]) * len(cp_modes)

    with tqdm(total=total, desc="Benchmark") as pbar:
        for case in cases:
            data = trained[case]
            for method in methods:
                hps = HP_FOR_METHOD[method]
                if hp_filter:
                    hps = [h for h in hps if h in hp_filter]
                for hp in hps:
                    for value in SWEEPS[hp]:
                        for cp_mode in cp_modes:
                            cfg = build_config(method, {hp: value})
                            try:
                                cov, width, n_done = evaluate_coverage_and_width(
                                    data, cp_mode, cfg, n_eval=n_eval,
                                )
                                rows.append(dict(
                                    case=case, cp_mode=cp_mode, method=method,
                                    hp=hp, value=value,
                                    coverage=cov, width=width, n_eval=int(n_done),
                                ))
                                pbar.set_postfix_str(
                                    f"{case}|{cp_mode}|{method}|{hp}={value}: "
                                    f"cov={cov:.3f} w={width:.3f}",
                                )
                            except RuntimeError as e:
                                rows.append(dict(
                                    case=case, cp_mode=cp_mode, method=method,
                                    hp=hp, value=value,
                                    coverage=float("nan"), width=float("nan"),
                                    n_eval=0, error=str(e),
                                ))
                                pbar.write(f"FAILED {case}|{cp_mode}|{method}|{hp}={value}: {e}")
                            pbar.update(1)
    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

PALETTE = {"sho": "#5B7EC0", "dho": "#81B29A", "duffing": "#E07A5F"}
MARKERS = {"sho": "o", "dho": "s", "duffing": "D"}


def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, linewidth=0.5, color="#CCCCCC")


def plot_hp_sweep(rows: list[dict], method: str, cp_mode: str, hp: str) -> None:
    """One figure per (method, cp_mode, hp): coverage vs value, one line per case."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax in axes:
        _style_ax(ax)

    # Determine target coverage line.
    target = 1.0 - ALPHA

    x_is_log = hp in {"n_samples", "opt_steps", "langevin_steps", "max_rounds"}
    for case in CASES:
        sel = [
            r for r in rows
            if r["case"] == case and r["method"] == method
            and r["cp_mode"] == cp_mode and r["hp"] == hp
        ]
        if not sel:
            continue
        sel.sort(key=lambda r: r["value"])
        xs = [r["value"] for r in sel]
        covs = [r["coverage"] for r in sel]
        widths = [r["width"] for r in sel]
        color = PALETTE[case]
        marker = MARKERS[case]
        axes[0].plot(xs, covs, color=color, marker=marker, linewidth=1.8,
                     markersize=6, label=case.upper())
        axes[1].plot(xs, widths, color=color, marker=marker, linewidth=1.8,
                     markersize=6, label=case.upper())

    axes[0].axhline(target, color="#9B9B9B", linestyle=":", linewidth=1.4,
                    label=f"Target $1-\\alpha={target:.2f}$")
    axes[0].set_ylabel("Empirical coverage")
    axes[0].set_xlabel(hp)
    axes[0].set_title(f"Coverage  ({method}, {cp_mode})")
    axes[0].legend(loc="best", edgecolor="#DDDDDD")
    axes[0].set_ylim(-0.02, 1.05)

    axes[1].set_ylabel("Mean bound width")
    axes[1].set_xlabel(hp)
    axes[1].set_title(f"Width  ({method}, {cp_mode})")
    axes[1].legend(loc="best", edgecolor="#DDDDDD")

    if x_is_log:
        for ax in axes:
            ax.set_xscale("log")

    fig.tight_layout()
    fname = f"hp_{hp}__{method}__{cp_mode}.png"
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_all(rows: list[dict]) -> None:
    seen: set[tuple[str, str, str]] = set()
    for r in rows:
        key = (r["method"], r["cp_mode"], r["hp"])
        if key in seen:
            continue
        seen.add(key)
        plot_hp_sweep(rows, *key)


def write_csv(rows: list[dict]) -> None:
    if not rows:
        return
    cols = ["case", "cp_mode", "method", "hp", "value", "coverage", "width", "n_eval", "error"]
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {CSV_PATH}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark hyperparameter impact on perturbation-sampling coverage.",
    )
    parser.add_argument("--cases", nargs="+", default=list(CASES), choices=CASES)
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    parser.add_argument("--cp-modes", nargs="+", default=list(CP_MODES), choices=CP_MODES)
    parser.add_argument("--hp", nargs="+", default=None,
                        help="Restrict to a subset of hyperparameters (default: all relevant).")
    parser.add_argument("--n-eval", type=int, default=20,
                        help="Held-out trajectories for coverage estimation (default 20).")
    args = parser.parse_args()

    rows = run_sweeps(
        cases=args.cases,
        methods=args.methods,
        cp_modes=args.cp_modes,
        n_eval=args.n_eval,
        hp_filter=args.hp,
    )
    write_csv(rows)
    plot_all(rows)
    print(f"Figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
