import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ConvTheorem.SHO.SHO_node_test import HarmonicOscillator, ODEFunc, generate_training_data, train_neural_ode, evaluate
from ConvTheorem.DHO.DHO_NODE import DampedHarmonicOscillator
from Utils.PRE.ConvOps_0d import ConvOperator
from Neural_PDE.UQ.inductive_cp import calibrate, emp_cov

sys.path.append(os.path.join(os.path.dirname(__file__), 'intervalFFT'))
from interval import interval
from intervalFFT import intervalFFT, inverse_intervalFFT, Real, complex_prod
from scipy.fft import fft, ifft

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

    ncf_scores = np.abs(residual_cal.numpy().flatten())
    qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=0.1)
    
    # 1) Pointwise (Approximate) Inversion 
    qhat_tensor = torch.tensor(np.full(pos.shape[1], qhat), dtype=torch.float32).unsqueeze(0)
    inv_qhat = D_pos.integrate(qhat_tensor, slice_pad=False)[0].numpy()
    
    # 2) Interval FFT Inversion (Guaranteed Set Propagation)
    # Using the exact same logic as PRE_set_prop
    test_idx = 4
    signal_padded = np.concatenate(([0], pos[test_idx].numpy(), [0]))
    kernel_pad = np.concatenate((D_pos.kernel.numpy(), np.zeros(len(signal_padded) - len(D_pos.kernel))))
    
    signal_fft = fft(signal_padded)
    kernel_fft = fft(kernel_pad)
    
    convolved = ifft(signal_fft * kernel_fft)
    inverse_kernel = 1.0 / (kernel_fft + 1e-16)
    
    # Convolved signal with strict qhat bound interval 
    # Notice we use the residual bound to create intervals around the convolved signal
    # Padding the boundaries just like original pre_set_prop
    convolved_noedges = convolved[3:-1]
    right_edges = convolved[0:3]
    left_edges = convolved[-1]
    
    # Apply uncertainty bounds (qhat) around the residual predictions
    # This creates a rigorous mathematical set
    convolved_set_center = [interval([x.real - qhat, x.real + qhat]) for x in convolved_noedges]
    convolved_set_right = [interval([x.real - qhat, x.real + qhat]) for x in right_edges]
    convolved_set_left = [interval([left_edges.real - qhat, left_edges.real + qhat])]
    
    convolved_set = convolved_set_right + convolved_set_center + convolved_set_left
    
    print("Computing Interval FFT for SHO (this may take a minute)...")
    convolved_set_fft = intervalFFT(convolved_set)
    convolved_set_fft_kernel = [complex_prod(z, c) for z, c in zip(convolved_set_fft, inverse_kernel)]
    retrieved_signal = inverse_intervalFFT(convolved_set_fft_kernel)
    signal_bounds = [Real(z) for z in retrieved_signal]
    
    # Extract boundaries
    signal_bounds_back = signal_bounds[2:2+len(t[1:-1])]
    upper_bounds_set = [float(interval_obj[0].sup) for interval_obj in signal_bounds_back]
    lower_bounds_set = [float(interval_obj[0].inf) for interval_obj in signal_bounds_back]

    pos_res = D_pos.differentiate(pos, correlation=False, slice_pad=False)
    pos_retrieved = D_pos.integrate(pos_res, correlation=False, slice_pad=False)

    plt.figure(figsize=(10, 6))
    plt.plot(t[1:-1], pos[test_idx, 1:-1].numpy(), 'b-', label='Predicted Position')
    
    # Plot approximate bounds
    plt.plot(t[1:-1], pos[test_idx, 1:-1].numpy() - np.abs(inv_qhat[1:-1]), 'r--', label='Point-wise Bound')
    plt.plot(t[1:-1], pos[test_idx, 1:-1].numpy() + np.abs(inv_qhat[1:-1]), 'r--')
    
    # Plot guaranteed bounds
    plt.fill_between(t[1:-1], lower_bounds_set, upper_bounds_set, color='gray', alpha=0.4, label='Guaranteed Set Bound (Interval FFT)')
    
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('SHO: Physical Bounds Comparison (Point-wise vs. Set Propagation)')
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(os.path.dirname(__file__), '..', 'Paper', 'images', 'sho_bounds_comparison.png')
    plt.savefig(save_path)
    plt.close()

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

    ncf_scores = np.abs(residual_cal.numpy().flatten())
    qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=0.1)

    # 1) Point-wise (Approximate) Inversion
    qhat_tensor = torch.tensor(np.full(pos.shape[1], qhat), dtype=torch.float32).unsqueeze(0)
    inv_qhat = D_damped.integrate(qhat_tensor, slice_pad=False)[0].numpy()

    # 2) Interval FFT Inversion (Guaranteed Set Propagation)
    test_idx = 4
    signal_padded = np.concatenate(([0], pos[test_idx].numpy(), [0]))
    kernel_pad = np.concatenate((D_damped.kernel.numpy(), np.zeros(len(signal_padded) - len(D_damped.kernel))))
    
    signal_fft = fft(signal_padded)
    kernel_fft = fft(kernel_pad)
    
    convolved = ifft(signal_fft * kernel_fft)
    inverse_kernel = 1.0 / (kernel_fft + 1e-16)
    
    convolved_noedges = convolved[3:-1]
    right_edges = convolved[0:3]
    left_edges = convolved[-1]
    
    # Apply uncertainty bounds (qhat) around the residual predictions
    convolved_set_center = [interval([x.real - qhat, x.real + qhat]) for x in convolved_noedges]
    convolved_set_right = [interval([x.real - qhat, x.real + qhat]) for x in right_edges]
    convolved_set_left = [interval([left_edges.real - qhat, left_edges.real + qhat])]
    
    convolved_set = convolved_set_right + convolved_set_center + convolved_set_left
    
    print("Computing Interval FFT for DHO (this may take a minute)...")
    convolved_set_fft = intervalFFT(convolved_set)
    convolved_set_fft_kernel = [complex_prod(z, c) for z, c in zip(convolved_set_fft, inverse_kernel)]
    retrieved_signal = inverse_intervalFFT(convolved_set_fft_kernel)
    signal_bounds = [Real(z) for z in retrieved_signal]
    
    # Extract boundaries
    signal_bounds_back = signal_bounds[2:2+len(t[1:-1])]
    upper_bounds_set = [float(interval_obj[0].sup) for interval_obj in signal_bounds_back]
    lower_bounds_set = [float(interval_obj[0].inf) for interval_obj in signal_bounds_back]


    plt.figure(figsize=(10, 6))
    plt.plot(t[1:-1], pos[test_idx, 1:-1].numpy(), 'b-', label='Predicted Position')
    
    # Plot approximate bounds
    plt.plot(t[1:-1], pos[test_idx, 1:-1].numpy() - np.abs(inv_qhat[1:-1]), 'r--', label='Point-wise Bound')
    plt.plot(t[1:-1], pos[test_idx, 1:-1].numpy() + np.abs(inv_qhat[1:-1]), 'r--')
    
    # Plot guaranteed bounds
    plt.fill_between(t[1:-1], lower_bounds_set, upper_bounds_set, color='gray', alpha=0.4, label='Guaranteed Set Bound (Interval FFT)')
    
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('DHO: Physical Bounds Comparison (Point-wise vs. Set Propagation)')
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(os.path.dirname(__file__), '..', 'Paper', 'images', 'dho_bounds_comparison.png')
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    run_sho()
    run_dho()
