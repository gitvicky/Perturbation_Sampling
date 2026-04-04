import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ConvTheorem.SHO.SHO_node_test import HarmonicOscillator, ODEFunc, generate_training_data, train_neural_ode, evaluate
from ConvTheorem.DHO.DHO_NODE import DampedHarmonicOscillator

from Utils.PRE.ConvOps_0d import ConvOperator

from ConvTheorem.inversion.residual_inversion import (
    IntervalFFTSlicing,
    PerturbationSamplingConfig,
    calibrate_qhat_from_residual,
    empirical_coverage_curve_1d,
    invert_residual_bounds_1d,
    perturbation_bounds_1d,
)

def _save_coverage_plot(case_name, coverage_result, save_name):
    plt.figure(figsize=(8, 6))
    plt.plot(
        coverage_result.nominal_coverage,
        coverage_result.nominal_coverage,
        color='black',
        linewidth=2.0,
        label='Ground Truth Target (Ideal)',
    )
    plt.plot(
        coverage_result.nominal_coverage,
        coverage_result.empirical_coverage_pointwise,
        'r--',
        linewidth=2.0,
        label='Point-wise Inversion',
    )
    plt.plot(
        coverage_result.nominal_coverage,
        coverage_result.empirical_coverage_intervalfft,
        color='gray',
        linewidth=2.5,
        label='Interval FFT Set Propagation',
    )
    if coverage_result.empirical_coverage_perturbation is not None:
        plt.plot(
            coverage_result.nominal_coverage,
            coverage_result.empirical_coverage_perturbation,
            color='tab:blue',
            linestyle='-.',
            linewidth=2.0,
            label='Perturbation Sampling',
        )
    plt.xlabel('Nominal Coverage (1 - alpha)')
    plt.ylabel('Empirical Coverage')
    plt.title(f'{case_name}: Empirical Coverage Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_path = os.path.join(os.path.dirname(__file__), '..', 'Paper', 'images', save_name)
    plt.savefig(save_path)
    plt.close()


def run_sho():
    print("Running SHO Experiment...")
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
        oscillator, func, t_span, n_points, x_range=(-2,2), v_range=(-2,2), n_solves=5)

    pos = torch.tensor(neural_sol[...,0], dtype=torch.float32)
    dt = t[1]-t[0]

    D_tt = ConvOperator(order=2)
    D_identity = ConvOperator(order=0)
    D_identity.kernel = torch.tensor([0.0, 1.0, 0.0])

    D_pos = ConvOperator(conv='spectral')
    D_pos.kernel = m*D_tt.kernel + dt**2*k*D_identity.kernel

    res = D_pos(pos)
    residual_cal = res[:4]
    residual_pred = res[4:]

    qhat = calibrate_qhat_from_residual(residual_cal, alpha=0.1)

    test_idx = 4
    point_bounds, interval_bounds = invert_residual_bounds_1d(
        pred_signal=pos[test_idx].numpy(),
        kernel=D_pos.kernel.numpy(),
        qhat=qhat,
        operator=D_pos,
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
        seed=123,
    )
    perturb_bounds = perturbation_bounds_1d(
        pred_signal=pos[test_idx].numpy(),
        residual_operator=D_pos,
        qhat=qhat,
        interior_slice=slice(1, -1),
        config=perturb_cfg,
    )

    pos_res = D_pos.differentiate(pos, correlation=False, slice_pad=False)
    pos_retrieved = D_pos.integrate(pos_res, correlation=False, slice_pad=False)

    plt.figure(figsize=(10, 6))
    plt.plot(t[1:-1], numerical_sol[test_idx, 1:-1, 0], color='black', linewidth=1.2, label='Ground Truth')
    plt.plot(t[1:-1], pos[test_idx, 1:-1].numpy(), 'b-', label='Predicted Position')
    
    # Plot approximate bounds
    plt.plot(t[1:-1], point_bounds.lower, 'r--', label='Point-wise Bound')
    plt.plot(t[1:-1], point_bounds.upper, 'r--')
    
    # Plot guaranteed bounds
    plt.fill_between(t[1:-1], interval_bounds.lower, interval_bounds.upper, color='gray', alpha=0.4, label='Guaranteed Set Bound (Interval FFT)')
    plt.plot(t[1:-1], perturb_bounds.lower, color='tab:blue', linestyle='-.', label='Perturbation Bound')
    plt.plot(t[1:-1], perturb_bounds.upper, color='tab:blue', linestyle='-.')
    
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('SHO: Physical Bounds Comparison (Point-wise vs. Set Propagation)')
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(os.path.dirname(__file__), '..', 'Paper', 'images', 'sho_bounds_comparison.png')
    plt.savefig(save_path)
    plt.close()

    alpha_levels = np.arange(0.05, 0.96, 0.10)
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

def run_dho():
    print("Running DHO Experiment...")
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
        oscillator, func, t_span, n_points, x_range=(-2,2), v_range=(-2,2), n_solves=5)

    pos = torch.tensor(neural_sol[...,0], dtype=torch.float32)
    dt = t[1]-t[0]

    D_t = ConvOperator(order=1)
    D_tt = ConvOperator(order=2)
    D_identity = ConvOperator(order=0)
    D_identity.kernel = torch.tensor([0.0, 1.0, 0.0])

    D_damped = ConvOperator(conv='spectral')
    D_damped.kernel = 2*m*D_tt.kernel + dt*c*D_t.kernel + 2*dt**2*k*D_identity.kernel

    res = D_damped(pos)
    residual_cal = res[:4]
    residual_pred = res[4:]

    qhat = calibrate_qhat_from_residual(residual_cal, alpha=0.1)
    test_idx = 4
    point_bounds, interval_bounds = invert_residual_bounds_1d(
        pred_signal=pos[test_idx].numpy(),
        kernel=D_damped.kernel.numpy(),
        qhat=qhat,
        operator=D_damped,
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
        seed=321,
    )
    perturb_bounds = perturbation_bounds_1d(
        pred_signal=pos[test_idx].numpy(),
        residual_operator=D_damped,
        qhat=qhat,
        interior_slice=slice(1, -1),
        config=perturb_cfg,
    )


    plt.figure(figsize=(10, 6))
    plt.plot(t[1:-1], numerical_sol[test_idx, 1:-1, 0], color='black', linewidth=1.2, label='Ground Truth')
    plt.plot(t[1:-1], pos[test_idx, 1:-1].numpy(), 'b-', label='Predicted Position')
    
    # Plot approximate bounds
    plt.plot(t[1:-1], point_bounds.lower, 'r--', label='Point-wise Bound')
    plt.plot(t[1:-1], point_bounds.upper, 'r--')
    
    # Plot guaranteed bounds
    plt.fill_between(t[1:-1], interval_bounds.lower, interval_bounds.upper, color='gray', alpha=0.4, label='Guaranteed Set Bound (Interval FFT)')
    plt.plot(t[1:-1], perturb_bounds.lower, color='tab:blue', linestyle='-.', label='Perturbation Bound')
    plt.plot(t[1:-1], perturb_bounds.upper, color='tab:blue', linestyle='-.')
    
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('DHO: Physical Bounds Comparison (Point-wise vs. Set Propagation)')
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(os.path.dirname(__file__), '..', 'Paper', 'images', 'dho_bounds_comparison.png')
    plt.savefig(save_path)
    plt.close()

    alpha_levels = np.arange(0.05, 0.96, 0.10)
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

if __name__ == '__main__':
    run_sho()
    run_dho()
