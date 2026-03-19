"""
Cauchy-Euler equation experiments with inverse-UQ in physical space.

Equation:
    x^2 y'' + a x y' + b y = 0
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


class CauchyEulerEquation:
    def __init__(self, a=1.0, b=4.0):
        self.a = a
        self.b = b

    def get_state_derivative(self, x, state):
        y, z = state
        x_safe = x if abs(x) >= 1e-8 else 1e-8
        dy_dx = z
        dz_dx = -(self.a / x_safe) * z - (self.b / (x_safe**2)) * y
        return np.array([dy_dx, dz_dx], dtype=float)

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
        inputs = torch.cat([x_safe, y], dim=1)
        return self.net(inputs)


def generate_training_data(equation, x_span, n_points, n_trajectories):
    x_eval = np.linspace(x_span[0], x_span[1], n_points)
    x_values = []
    states = []
    derivatives = []
    for _ in range(n_trajectories):
        y0 = np.random.uniform(-2, 2)
        z0 = np.random.uniform(-2, 2)
        initial_state = np.array([y0, z0], dtype=float)
        x, solution = equation.solve_ode(x_span, initial_state, x_eval)
        x_values.append(x)
        states.append(solution)
        derivs = np.array([equation.get_state_derivative(x_val, state) for x_val, state in zip(x, solution)])
        derivatives.append(derivs)
    return np.stack(x_values), np.stack(states), np.stack(derivatives)


def train_neural_ode(func, train_x, train_states, train_derivs, n_epochs, batch_size):
    optimizer = torch.optim.Adam(func.parameters())
    losses = []
    train_x = torch.FloatTensor(train_x)
    train_states = torch.FloatTensor(train_states)
    train_derivs = torch.FloatTensor(train_derivs)
    n_samples = train_states.shape[0]
    for _ in range(n_epochs):
        epoch_loss = 0.0
        indices = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i : i + batch_size]
            batch_x = train_x[batch_indices].reshape(-1, 1)
            batch_states = train_states[batch_indices].reshape(-1, 2)
            batch_derivs = train_derivs[batch_indices].reshape(-1, 2)
            pred_derivs = func(batch_x, batch_states)
            loss = torch.mean((pred_derivs - batch_derivs) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        losses.append(epoch_loss)
    return losses


def evaluate(equation, neural_ode, x_span, y_range, z_range, n_solves, n_points=100):
    x = torch.linspace(x_span[0], x_span[1], n_points)
    num_solns = []
    neural_solns = []
    for _ in tqdm(range(n_solves)):
        y0 = np.random.uniform(*y_range)
        z0 = np.random.uniform(*z_range)
        initial_state = np.array([y0, z0], dtype=float)
        _, numerical_solution = equation.solve_ode(x_span, initial_state, x.numpy())
        state_0 = torch.FloatTensor(initial_state)
        neural_solution = odeint(
            lambda t, y: neural_ode(t.reshape(-1, 1), y.reshape(-1, 2)).reshape(-1),
            state_0,
            x,
        )
        num_solns.append(numerical_solution)
        neural_solns.append(neural_solution.detach().numpy())
    return x.numpy(), np.asarray(num_solns), np.asarray(neural_solns)


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


def _build_frozen_cauchy_operator(x, a, b):
    dx = x[1] - x[0]
    x_interior = x[1:-1]
    x_mean = float(np.mean(x_interior))
    x2_mean = float(np.mean(x_interior**2))
    D_x = ConvOperator(order=1)
    D_xx = ConvOperator(order=2)
    D_identity = ConvOperator(order=0)
    D_identity.kernel = torch.tensor([0.0, 1.0, 0.0])
    D_ce = ConvOperator(conv="spectral")
    D_ce.kernel = D_xx.kernel + (a * dx / x_mean) * D_x.kernel + (b * dx**2 / x2_mean) * D_identity.kernel
    return D_ce


def run_cauchy_euler(a=1.0, b=4.0, alpha=0.1):
    print(f"Running Cauchy-Euler Experiment (a={a}, b={b})...")
    equation = CauchyEulerEquation(a=a, b=b)
    x_span = (0.1, 5.0)
    n_points = 100
    n_trajectories = 50
    n_solves = 5

    x_values, states, derivs = generate_training_data(equation, x_span, n_points, n_trajectories)
    func = ODEFunc(hidden_dim=64)
    train_neural_ode(func, x_values, states, derivs, n_epochs=500, batch_size=16)
    x, numerical_sol, neural_sol = evaluate(
        equation,
        func,
        x_span,
        y_range=(-2, 2),
        z_range=(-2, 2),
        n_solves=n_solves,
        n_points=n_points,
    )

    D_ce = _build_frozen_cauchy_operator(x, a=a, b=b)
    y_pred = torch.tensor(neural_sol[..., 0], dtype=torch.float32)
    res = D_ce(y_pred)
    residual_cal = res[:4]
    qhat = calibrate_qhat_from_residual(residual_cal, alpha=alpha)

    test_idx = min(4, n_solves - 1)
    point_bounds, interval_bounds = invert_residual_bounds_1d(
        pred_signal=y_pred[test_idx].numpy(),
        kernel=D_ce.kernel.numpy(),
        qhat=qhat,
        operator=D_ce,
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
        seed=205,
    )
    perturb_bounds = perturbation_bounds_1d(
        pred_signal=y_pred[test_idx].numpy(),
        residual_operator=D_ce,
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
    plt.title(f"Cauchy-Euler (a={a}, b={b}): Physical Bounds Comparison")
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(os.path.dirname(__file__), "..", "..", "Paper", "images", f"cauchy_euler_a{a:g}_b{b:g}_bounds_comparison.png")
    plt.savefig(save_path)
    plt.close()

    alpha_levels = np.arange(0.05, 0.96, 0.10)
    coverage = _coverage_curve_with_fallback(
        preds=neural_sol[..., 0],
        truths=numerical_sol[..., 0],
        residual_cal=residual_cal,
        kernel=D_ce.kernel.numpy(),
        operator=D_ce,
        alphas=alpha_levels,
        perturb_cfg=PerturbationSamplingConfig(
            n_samples=2000,
            batch_size=500,
            max_rounds=3,
            noise_type="spatial",
            noise_std=0.10,
            correlation_length=24.0,
            seed=1205,
        ),
    )
    _save_coverage_plot(f"Cauchy-Euler a={a:g}, b={b:g}", coverage, f"cauchy_euler_a{a:g}_b{b:g}_coverage_comparison.png")

    np.save(os.path.join(os.path.dirname(__file__), f"CE_numerical_outputs_a{int(a*10)}_b{int(b*10)}"), numerical_sol)
    np.save(os.path.join(os.path.dirname(__file__), f"CE_neural_outputs_a{int(a*10)}_b{int(b*10)}"), neural_sol)


if __name__ == "__main__":
    run_cauchy_euler(a=1.0, b=4.0, alpha=0.1)
