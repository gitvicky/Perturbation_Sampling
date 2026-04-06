#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Initial tests for various sampling methods to approximate the bounds within the input space.

Case: Simple Harmonic Oscillator

Eqn:
     m(d²x/dt²) + kx = 0

First order Expressions:
    dx/dt = v
    dv/dt = -(k/m)x

Cases being explored: 
    1. White noise injection to neural-ode outputs with acceptance-rejectance criteria taken from CP-PRE bounds. 
 
"""

# %%
from Inversion_Strategies.tests.PerturbSampling.SHO_Neural_ODE import * 

#Executions 
# Initialize system
m, k = 1.0, 1.0 
oscillator = HarmonicOscillator(k, m)

# Generate training data
t_span = (0, 10)
n_points = 100
n_trajectories = 50
t, states, derivs = generate_training_data(
    oscillator, t_span, n_points, n_trajectories)

# Initialize and train neural ODE
func = ODEFunc(hidden_dim=64)
losses = train_neural_ode(
    func, t, states, derivs, n_epochs=100, batch_size=16)

# Compare solutions
initial_state = np.array([1.0, 0.0])  # x0 = 1, v0 = 0
t, numerical_sol, neural_sol = compare_solutions(
    oscillator, func, t_span, initial_state)

# Plot results
plot_comparison(t, numerical_sol, neural_sol)


# %% 

#PRE Estimations

soln = neural_sol
x = torch.tensor(soln[:, 0], dtype=torch.float32).unsqueeze(0)
v = torch.tensor(soln[:, 1], dtype=torch.float32).unsqueeze(0)

from Utils.PRE.ConvOps_0d import ConvOperator
dt = t[1]-t[0]
D_t = ConvOperator(order=1)#, scale=alpha)
D_tt = ConvOperator(order=2)#, scale=alpha)

D_identity = ConvOperator(order=0) #Identity 
D_identity.kernel = torch.tensor([0, 1, 0])

D_pos = ConvOperator(conv='direct')
D_pos.kernel = m*D_tt.kernel + dt**2*k*D_identity.kernel

D_pos_spectral = ConvOperator(conv='spectral')
D_pos_spectral.kernel = m*D_tt.kernel + dt**2*k*D_identity.kernel

# %% 
# #Plotting the residuals 

# #Position Residual 
# plt.figure()
# plt.plot(t[1:-1], D_pos(x)[0,1:-1], 'b-', label='direct_residual')
# plt.plot(t[1:-1], D_pos_spectral(x)[0,1:-1], 'r--', label='spectral_residual')
# plt.plot(t[1:-1], D_pos.differentiate(x, correlation=True)[0, 1:-1], 'k:', label='custom_spectral')

# plt.xlabel('Time')
# plt.ylabel('Residual')
# plt.title('Position Residual')
# plt.legend()
# plt.grid(True)


# # #Velocity Residual 
# # plt.figure()
# # plt.plot(t[1:-1], D_vel(v)[0,1:-1], 'b-', label='velocity_residual')
# # plt.xlabel('Time')
# # plt.ylabel('Residual')
# # plt.title('Velocity Residual')
# # plt.legend()
# # plt.grid(True)


# %%
#Performing CP-PRE
from Utils.CP.inductive_cp import * 

t, numerical_sol, neural_sol = evaluate(
    oscillator, func, t_span, n_points, x_range=(-2,2), v_range=(-2,2), n_solves=200)

pos = torch.tensor(neural_sol[...,0], dtype=torch.float32)
res = D_pos(pos)
residual_cal = res[:100]
residual_pred = res[100:]

ncf_scores = np.abs(residual_cal) 

# %%
#Plotting the bounds in the residual space
alpha = 0.1
qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=alpha)

plt.plot(t[1:-1], residual_pred[0, 1:-1], 'b-', label='PRE')
plt.plot(t[1:-1], -qhat[1:-1], 'r--', label='Lower Bound')
plt.plot(t[1:-1], +qhat[1:-1], 'g--', label='Upper Bound')
plt.xlabel('Time')
plt.ylabel('Residual')
plt.title('Marginal CP-PRE : Position')
plt.legend()
plt.grid(True)

# %% 
#Perturbed predictions. Do this for a single prediction first with 500 odd perturbed variations. 
from Utils.noise_gen import PDENoiseGenerator1D
noise_gen = PDENoiseGenerator1D()

idx = 10
pred = pos[idx:idx+1] #Doing this for a single prediction

n_samples = 10000
noise = noise_gen.spatially_correlated_noise(n_samples, n_points, correlation_length=32, std=0.1) #Gaussian Kernel
# noise = noise_gen.gp_noise(n_samples, n_points, correlation_length=32, std=0.01) #GP samples
# noise = noise_gen.pre_correlated_noise(n_samples, n_points, kernel=D_pos.kernel, std=0.01)

perturbed_pred = pred + noise 
residual_perturbed_pred = D_pos(perturbed_pred)
perturb_within_bounds = torch.abs(residual_perturbed_pred) <= torch.abs(torch.tensor(qhat))  # Shape: [n_samples, n_points]

#Computing the mean and std dev. from the predictions within the bounds. 

n_samples, n_points = perturbed_pred.shape
valid_time_indices = range(1, n_points-1)

pred_slice = perturbed_pred[:, valid_time_indices]
mask_slice = perturb_within_bounds[:, valid_time_indices]
mask_float = mask_slice.float()

# Coverage and count
coverage = mask_float.mean(dim=0) #Within the residual space
n_valid = mask_slice.sum(dim=0).float() #Points within the residual bounds.

# Means
masked_pred = pred_slice * mask_float
mean = masked_pred.sum(dim=0) / (n_valid + 1e-8)

# Vectorized std computation
# Compute squared differences from mean
diff_sq = (pred_slice - mean.unsqueeze(0)) ** 2  # Broadcasting
masked_diff_sq = diff_sq * mask_float

# Compute variance and std
variance = masked_diff_sq.sum(dim=0) / (n_valid + 1e-8)
std = torch.sqrt(variance)

# Handle cases where n_valid <= 1
std = torch.where(n_valid <= 1, torch.zeros_like(std), std)

plt.figure()
plt.plot(valid_time_indices, numerical_sol[idx, 1:-1, 0], c='black', label='soln.', lw=0.5)
# plt.plot(valid_time_indices, pred[0, 1:-1], c='red', label='pred.', lw=0.5)
plt.plot(valid_time_indices, mean, label='mean', lw=0.5)
plt.fill_between(valid_time_indices, mean - 2*std, mean + 2*std,  alpha=0.3, label='±2σ', color='orange')
plt.legend()
plt.title(r'$\alpha =  $' +  str(alpha))
print(n_valid/n_samples)
# %%
#Experimenting with different alpha values: The bounds become more erratic and discontinuous as we increase the alpha value 
# But we increase the number of samples it becomes smoother but across alpha there is a variation in smoothness 
#However it seems that the width of the bound (or its general trend) remains the same 

# %%
# # Marginal CP 

# ncf_scores = np.abs(residual_cal.numpy())
# alpha = 0.1
# qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=alpha)
# prediction_sets =  [- qhat, + qhat]


# plt.plot(t[1:-1], residual_pred[0, 1:-1], 'b-', label='PRE')
# plt.plot(t[1:-1], -qhat[1:-1], 'r--', label='Lower Bound')
# plt.plot(t[1:-1], +qhat[1:-1], 'g--', label='Upper Bound')
# plt.xlabel('Time')
# plt.ylabel('Residual')
# plt.title('Joint CP-PRE : Position')
# plt.legend()
# plt.grid(True)


# # Joint - CP 

# modulation = modulation_func(residual_cal.numpy(), np.zeros(residual_cal.shape))
# ncf_scores = ncf_metric_joint(residual_cal.numpy(), np.zeros(residual_cal.shape), modulation)
# alpha = 0.1
# qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=alpha)
# prediction_sets =  [- qhat*modulation, + qhat*modulation]


# plt.plot(t[1:-1], residual_pred[0, 1:-1], 'b-', label='PRE')
# plt.plot(t[1:-1], -qhat*modulation[1:-1], 'r--', label='Lower Bound')
# plt.plot(t[1:-1], +qhat*modulation[1:-1], 'g--', label='Upper Bound')
# plt.xlabel('Time')
# plt.ylabel('Residual')
# plt.title('Joint CP-PRE : Position')
# plt.legend()
# plt.grid(True)


# # %%
# #Inverse Bounds with Joint CP Bounds
# idx = 10
# pred = pos[idx:idx+1] #Doing this for a single prediction

# n_samples = 1000
# noise = noise_gen.spatially_correlated_noise(n_samples, n_points, correlation_length=32, std=0.5) #Gaussian Kernel
# # noise = noise_gen.gp_noise(n_samples, n_points, correlation_length=32, std=0.5) #GP samples
# perturbed_pred = pred + noise 
# residual_perturbed_pred = D_pos(perturbed_pred)
# perturb_within_bounds = torch.abs(residual_perturbed_pred) <= torch.abs(torch.tensor(qhat))  # Shape: [n_samples, n_points]
# # perturb_within_bounds = torch.abs(residual_perturbed_pred) <= torch.abs(torch.tensor(qhat*modulation))  # Shape: [n_samples, n_points]

# #Computing the mean and std dev. from the predictions within the bounds. 

# n_samples, n_points = perturbed_pred.shape
# valid_time_indices = range(1, n_points-1)

# pred_slice = perturbed_pred[:, valid_time_indices]
# mask_slice = perturb_within_bounds[:, valid_time_indices]
# mask_float = mask_slice.float()

# # Coverage and count
# coverage = mask_float.mean(dim=0) #Within the residual space
# n_valid = mask_slice.sum(dim=0).float() #Points within the residual bounds.

# # Means
# masked_pred = pred_slice * mask_float
# mean = masked_pred.sum(dim=0) / (n_valid + 1e-8)

# # Vectorized std computation
# # Compute squared differences from mean
# diff_sq = (pred_slice - mean.unsqueeze(0)) ** 2  # Broadcasting
# masked_diff_sq = diff_sq * mask_float

# # Compute variance and std
# variance = masked_diff_sq.sum(dim=0) / (n_valid + 1e-8)
# std = torch.sqrt(variance)

# # Handle cases where n_valid <= 1
# std = torch.where(n_valid <= 1, torch.zeros_like(std), std)

# plt.figure()
# plt.plot(valid_time_indices, numerical_sol[idx, 1:-1, 0], c='black', label='soln.', lw=0.5)
# plt.plot(valid_time_indices, pred[0, 1:-1], c='red', label='pred.', lw=0.5)
# plt.plot(valid_time_indices, mean, label='mean', lw=0.5)
# plt.fill_between(valid_time_indices, mean - 2*std, mean + 2*std,  alpha=0.3, label='±2σ', color='orange')
# plt.legend()
# plt.title(r'$\alpha =  $' +  str(alpha))

# print(n_valid/n_samples)
# # %%
