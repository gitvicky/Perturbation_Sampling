"""
Bessel Equation experiments with inverse-UQ in physical space.

Equation (order n):
    x^2 y'' + x y' + (x^2 - n^2) y = 0
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from tqdm import tqdm
from scipy.integrate import solve_ivp
from scipy.special import jv, jvp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from Utils.PRE.ConvOps_0d import ConvOperator
from ConvTheorem.inversion.residual_inversion import (
    CoverageResult,
    IntervalFFTSlicing,
    PerturbationSamplingConfig,
    calibrate_qhat_from_residual,
    invert_residual_bounds_1d,
    perturbation_bounds_1d,
)


class BesselEquation:
    def __init__(self, n: int = 0):
        self.n = n

    def get_state_derivative(self, x, state):
        y, v = state
        x_safe = x if abs(x) >= 1e-8 else 1e-8
        dy_dx = v
        dv_dx = -v / x_safe - (1.0 - (self.n**2) / (x_safe**2)) * y
        return np.array([dy_dx, dv_dx], dtype=float)

    def solve_ode(self, x_span, initial_state, x_eval=None):
        solution = solve_ivp(
            fun=self.get_state_derivative,
            t_span=x_span,
            y0=initial_state,
            t_eval=x_eval,
            method="RK45",
            rtol=1e-8,
            atol=1e-8,
        )
        return solution.t, solution.y.T

    def canonical_initial_state(self, x0: float):
        if self.n == 0 and x0 <= 1e-6:
            return np.array([1.0, 0.0], dtype=float)
        return np.array([jv(self.n, x0), jvp(self.n, x0, 1)], dtype=float)


class ODEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x, y):
        x_safe = torch.clamp(x.reshape(-1, 1), min=1e-6)
        state_x = torch.cat([y, x_safe], dim=1)
        return self.net(state_x)


def generate_training_data(equation, x_span, n_points, n_trajectories):
    x_eval = np.linspace(x_span[0], x_span[1], n_points)
    states = []
    derivs = []
    for _ in range(n_trajectories):
        phase = np.random.uniform(0, 2 * np.pi)
        amp = np.random.uniform(0.5, 1.5)
        x0 = x_span[0]
        y0 = amp * np.cos(phase) * jv(equation.n, x0)
        v0 = amp * np.cos(phase) * jvp(equation.n, x0, 1)
        initial_state = np.array([y0, v0], dtype=float)
        _, sol = equation.solve_ode(x_span, initial_state, x_eval)
        states.append(sol)
        derivs.append(np.array([equation.get_state_derivative(x, s) for x, s in zip(x_eval, sol)]))
    return x_eval, np.stack(states), np.stack(derivs)


def train_neural_ode(func, train_x, train_states, train_derivs, n_epochs, batch_size):
    optimizer = torch.optim.Adam(func.parameters())
    train_x = torch.FloatTensor(train_x)
    train_states = torch.FloatTensor(train_states)
    train_derivs = torch.FloatTensor(train_derivs)
    n_samples = train_states.shape[0] * train_states.shape[1]
    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        x_flat = train_x.repeat(train_states.shape[0])
        states_flat = train_states.reshape(-1, 2)
        derivs_flat = train_derivs.reshape(-1, 2)
        indices = torch.randperm(n_samples)
        x_flat = x_flat[indices]
        states_flat = states_flat[indices]
        derivs_flat = derivs_flat[indices]
        for i in range(0, n_samples, batch_size):
            bx = x_flat[i : i + batch_size]
            bs = states_flat[i : i + batch_size]
            bd = derivs_flat[i : i + batch_size]
            pred = func(bx, bs)
            loss = torch.mean((pred - bd) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * bs.shape[0]
        losses.append(epoch_loss / n_samples)
    return losses


def evaluate(equation, neural_ode, x_span, n_points, n_solves):
    x_eval = np.linspace(x_span[0], x_span[1], n_points)
    x_tensor = torch.FloatTensor(x_eval)
    num_solns = []
    neural_solns = []
    for _ in tqdm(range(n_solves)):
        phase = np.random.uniform(0, 2 * np.pi)
        amp = np.random.uniform(0.5, 1.5)
        x0 = x_span[0]
        y0 = amp * np.cos(phase) * jv(equation.n, x0)
        v0 = amp * np.cos(phase) * jvp(equation.n, x0, 1)
        initial_state = np.array([y0, v0], dtype=float)
        _, numerical_solution = equation.solve_ode(x_span, initial_state, x_eval)
        state_0 = torch.FloatTensor(initial_state)
        neural_solution = odeint(
            lambda t, y: neural_ode(t.reshape(-1, 1), y.reshape(-1, 2)).reshape(-1),
            state_0,
            x_tensor,
        )
        num_solns.append(numerical_solution)
        neural_solns.append(neural_solution.detach().numpy())
    return x_eval, np.asarray(num_solns), np.asarray(neural_solns)


def _save_coverage_plot(case_name, coverage_result, save_name):
    plt.figure(figsize=(8, 6))
    plt.plot(coverage_result.nominal_coverage, coverage_result.nominal_coverage, color="black", linewidth=2, label="Ground Truth Target (Ideal)")
    plt.plot(coverage_result.nominal_coverage, coverage_result.empirical_coverage_pointwise, "r--", linewidth=2, label="Point-wise Inversion")
    plt.plot(coverage_result.nominal_coverage, coverage_result.empirical_coverage_intervalfft, color="gray", linewidth=2.5, label="Interval FFT Set Propagation")
    if coverage_result.empirical_coverage_perturbation is not None:
        plt.plot(coverage_result.nominal_coverage, coverage_result.empirical_coverage_perturbation, color="tab:blue", linestyle="-.", linewidth=2, label="Perturbation Sampling")
    plt.xlabel("Nominal Coverage (1 - alpha)")
    plt.ylabel("Empirical Coverage")
    plt.title(f"{case_name}: Empirical Coverage Comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_path = os.path.join(os.path.dirname(__file__), "..", "..", "Paper", "images", save_name)
    plt.savefig(save_path)
    plt.close()


def _trajectory_contained(truth, lower, upper):
    return float(np.logical_and(truth >= lower, truth <= upper).all())


def _coverage_curve_with_fallback(preds, truths, residual_cal, kernel, operator, alphas, perturb_cfg):
    nominal = []
    cov_point = []
    cov_interval = []
    cov_perturb = []

    for alpha in alphas:
        qhat = calibrate_qhat_from_residual(residual_cal, alpha=float(alpha))
        point_flags = []
        interval_flags = []
        perturb_flags = []
        for i in range(preds.shape[0]):
            point_bounds, interval_bounds = invert_residual_bounds_1d(
                pred_signal=preds[i],
                kernel=kernel,
                qhat=qhat,
                operator=operator,
                interior_slice=slice(1, -1),
                intervalfft_slicing=IntervalFFTSlicing(center_start=3, center_end=-1, n_right_edges=3, output_offset=2),
                integrate_slice_pad=False,
            )
            truth_i = truths[i, 1:-1]
            point_flags.append(_trajectory_contained(truth_i, point_bounds.lower, point_bounds.upper))
            interval_flags.append(_trajectory_contained(truth_i, interval_bounds.lower, interval_bounds.upper))
            perturb_bounds = perturbation_bounds_1d(
                pred_signal=preds[i],
                residual_operator=operator,
                qhat=qhat,
                interior_slice=slice(1, -1),
                config=perturb_cfg,
                fallback_lower=interval_bounds.lower,
                fallback_upper=interval_bounds.upper,
            )
            perturb_flags.append(_trajectory_contained(truth_i, perturb_bounds.lower, perturb_bounds.upper))
        nominal.append(1.0 - float(alpha))
        cov_point.append(float(np.mean(point_flags)))
        cov_interval.append(float(np.mean(interval_flags)))
        cov_perturb.append(float(np.mean(perturb_flags)))

    return CoverageResult(
        nominal_coverage=np.asarray(nominal, dtype=float),
        empirical_coverage_pointwise=np.asarray(cov_point, dtype=float),
        empirical_coverage_intervalfft=np.asarray(cov_interval, dtype=float),
        empirical_coverage_perturbation=np.asarray(cov_perturb, dtype=float),
    )


def _build_frozen_bessel_operator(x, n):
    dx = x[1] - x[0]
    x_interior = x[1:-1]
    x_mean = float(np.mean(x_interior))
    x2_mean = float(np.mean(x_interior**2))
    D_x = ConvOperator(order=1)
    D_xx = ConvOperator(order=2)
    D_identity = ConvOperator(order=0)
    D_identity.kernel = torch.tensor([0.0, 1.0, 0.0])
    D_bessel = ConvOperator(conv="spectral")
    D_bessel.kernel = D_xx.kernel + (dx / x_mean) * D_x.kernel + (dx**2) * (1.0 - (n**2) / x2_mean) * D_identity.kernel
    return D_bessel


def run_bessel(n=1, alpha=0.1):
    print(f"Running Bessel Experiment (n={n})...")
    equation = BesselEquation(n=n)
    x_span = (0.1, 15.0)
    n_points = 100
    n_trajectories = 50
    n_solves = 5

    x_train, states, derivs = generate_training_data(equation, x_span, n_points, n_trajectories)
    func = ODEFunc(hidden_dim=64)
    train_neural_ode(func, x_train, states, derivs, n_epochs=500, batch_size=32)
    x, numerical_sol, neural_sol = evaluate(equation, func, x_span, n_points=n_points, n_solves=n_solves)

    D_bessel = _build_frozen_bessel_operator(x, n)
    y_pred = torch.tensor(neural_sol[..., 0], dtype=torch.float32)
    res = D_bessel(y_pred)
    residual_cal = res[:4]
    qhat = calibrate_qhat_from_residual(residual_cal, alpha=alpha)

    test_idx = min(4, n_solves - 1)
    point_bounds, interval_bounds = invert_residual_bounds_1d(
        pred_signal=y_pred[test_idx].numpy(),
        kernel=D_bessel.kernel.numpy(),
        qhat=qhat,
        operator=D_bessel,
        interior_slice=slice(1, -1),
        intervalfft_slicing=IntervalFFTSlicing(center_start=3, center_end=-1, n_right_edges=3, output_offset=2),
        integrate_slice_pad=False,
    )
    perturb_cfg = PerturbationSamplingConfig(
        n_samples=4000,
        batch_size=1000,
        max_rounds=2,
        noise_type="spatial",
        noise_std=0.10,
        correlation_length=24.0,
        seed=101 + int(n),
    )
    perturb_bounds = perturbation_bounds_1d(
        pred_signal=y_pred[test_idx].numpy(),
        residual_operator=D_bessel,
        qhat=qhat,
        interior_slice=slice(1, -1),
        config=perturb_cfg,
        fallback_lower=interval_bounds.lower,
        fallback_upper=interval_bounds.upper,
    )

    plt.figure(figsize=(10, 6))
    plt.plot(x[1:-1], numerical_sol[test_idx, 1:-1, 0], color="black", linewidth=1.2, label="Ground Truth")
    plt.plot(x[1:-1], y_pred[test_idx, 1:-1].numpy(), "b-", label="Predicted State")
    plt.plot(x[1:-1], point_bounds.lower, "r--", label="Point-wise Bound")
    plt.plot(x[1:-1], point_bounds.upper, "r--")
    plt.fill_between(x[1:-1], interval_bounds.lower, interval_bounds.upper, color="gray", alpha=0.4, label="Guaranteed Set Bound (Interval FFT)")
    plt.plot(x[1:-1], perturb_bounds.lower, color="tab:blue", linestyle="-.", label="Perturbation Bound")
    plt.plot(x[1:-1], perturb_bounds.upper, color="tab:blue", linestyle="-.")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Bessel (n={n}): Physical Bounds Comparison")
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(os.path.dirname(__file__), "..", "..", "Paper", "images", f"bessel_n{n}_bounds_comparison.png")
    plt.savefig(save_path)
    plt.close()

    alpha_levels = np.arange(0.05, 0.96, 0.10)
    coverage = _coverage_curve_with_fallback(
        preds=neural_sol[..., 0],
        truths=numerical_sol[..., 0],
        residual_cal=residual_cal,
        kernel=D_bessel.kernel.numpy(),
        operator=D_bessel,
        alphas=alpha_levels,
        perturb_cfg=PerturbationSamplingConfig(
            n_samples=2000,
            batch_size=500,
            max_rounds=3,
            noise_type="spatial",
            noise_std=0.10,
            correlation_length=24.0,
            seed=777 + int(n),
        ),
    )
    _save_coverage_plot(f"Bessel n={n}", coverage, f"bessel_n{n}_coverage_comparison.png")

    np.save(os.path.join(os.path.dirname(__file__), f"Bessel_numerical_outputs_n{n}"), numerical_sol)
    np.save(os.path.join(os.path.dirname(__file__), f"Bessel_neural_outputs_n{n}"), neural_sol)


if __name__ == "__main__":
    for n_val in (0, 1, 2):
        run_bessel(n=n_val, alpha=0.1)
