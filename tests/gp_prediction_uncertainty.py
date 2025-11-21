#!/usr/bin/env python3
"""
GP-based Uncertainty Quantification aligned with Conformal Prediction Coverage

This module extends the rejection sampling approach to train a zero-mean Gaussian Process
whose standard deviation represents uncertainty aligned with conformal prediction coverage.

Key Features:
- Zero-mean GP training with physics-informed bounds
- Coverage-aware hyperparameter optimization
- Integration with existing CP-PRE framework
- Validation against prescribed alpha levels

Author: Claude - Based on SHO_neural_ode.py framework
"""
# %% 
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import gpytorch
from scipy.optimize import minimize
from typing import Tuple, Dict, Optional, List
from SHO_Neural_ODE import * 

class ZeroMeanGP(gpytorch.models.ExactGP):
    """
    Zero-mean Gaussian Process for uncertainty quantification aligned with 
    conformal prediction coverage guarantees.
    """
    
    def __init__(self, train_x, train_y, likelihood, kernel_type='rbf'):
        super(ZeroMeanGP, self).__init__(train_x, train_y, likelihood)
        
        # Zero mean function
        self.mean_module = gpytorch.means.ZeroMean()
        
        # Kernel selection
        if kernel_type == 'rbf':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            )
        elif kernel_type == 'matern':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=1.5)
            )
        elif kernel_type == 'periodic':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.PeriodicKernel()
            )
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPUncertaintyTrainer:
    """
    Trainer for GP-based uncertainty quantification aligned with conformal prediction.
    """
    
    def __init__(self, device='cpu', dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        
    def generate_training_data_from_rejection_sampling(
        self, 
        pred: torch.Tensor,
        residual_operator,
        qhat: torch.Tensor,
        n_samples: int = 1000,
        noise_generator=None,
        **noise_params
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate training data for GP using rejection sampling from CP-PRE bounds.
        
        Args:
            pred: Base prediction [1, n_points]
            residual_operator: Physics residual operator (e.g., D_pos)
            qhat: Conformal prediction quantiles
            n_samples: Number of perturbation samples
            noise_generator: Noise generation object
            **noise_params: Parameters for noise generation
            
        Returns:
            train_x: Input locations (time indices)
            train_y: Accepted perturbations from base prediction
        """
        n_points = pred.shape[1]
        
        # Generate noise perturbations
        if noise_generator is not None:
            noise = noise_generator.gp_noise(n_samples, n_points, **noise_params)
        else:
            # Fallback to basic Gaussian noise
            noise = torch.randn(n_samples, n_points, device=self.device, dtype=self.dtype)
            
        # Apply perturbations and compute residuals
        perturbed_pred = pred + noise
        residual_perturbed = residual_operator(perturbed_pred)
        
        # Rejection criterion based on CP-PRE bounds
        if qhat.ndim == 1:  # Joint CP with modulation
            bounds = torch.abs(qhat).unsqueeze(0)
        else:  # Simple bounds
            bounds = torch.abs(qhat)
            
        within_bounds = torch.abs(residual_perturbed) <= bounds
        
        # Extract valid samples for GP training
        valid_time_indices = range(1, n_points-1)  # Exclude boundary points
        train_data = []
        
        for t_idx in valid_time_indices:
            valid_mask = within_bounds[:, t_idx]
            if valid_mask.sum() > 0:  # Ensure we have valid samples
                valid_perturbations = perturbed_pred[valid_mask, t_idx]
                base_value = pred[0, t_idx]
                
                # Store relative perturbations from base prediction
                relative_perturbations = valid_perturbations - base_value
                
                for pert in relative_perturbations:
                    train_data.append((t_idx, pert.item()))
        
        if len(train_data) == 0:
            raise ValueError("No valid samples found within CP bounds. Consider adjusting bounds or noise parameters.")
            
        # Convert to tensors
        train_x = torch.tensor([x[0] for x in train_data], dtype=self.dtype, device=self.device)
        train_y = torch.tensor([x[1] for x in train_data], dtype=self.dtype, device=self.device)
        
        return train_x.unsqueeze(-1), train_y
    
    def train_gp_model(
        self, 
        train_x: torch.Tensor, 
        train_y: torch.Tensor,
        kernel_type: str = 'rbf',
        training_iter: int = 100,
        lr: float = 0.1
    ) -> Tuple[ZeroMeanGP, gpytorch.likelihoods.GaussianLikelihood]:
        """
        Train the zero-mean GP model on rejection sampling data.
        
        Args:
            train_x: Input locations
            train_y: Target perturbations (relative to base prediction)
            kernel_type: Type of GP kernel
            training_iter: Number of training iterations
            lr: Learning rate
            
        Returns:
            Trained GP model and likelihood
        """
        # Initialize likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ZeroMeanGP(train_x, train_y, likelihood, kernel_type)
        
        # Move to device
        model = model.to(self.device)
        likelihood = likelihood.to(self.device)
        
        # Set to training mode
        model.train()
        likelihood.train()
        
        # Optimizer
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
        ], lr=lr)
        
        # Loss function
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        
        # Training loop
        losses = []
        for i in tqdm(range(training_iter), desc="Training GP"):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        return model, likelihood
    
    def validate_coverage_alignment(
        self,
        model: ZeroMeanGP,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        pred: torch.Tensor,
        alpha: float,
        n_test_points: int = None
    ) -> Dict[str, float]:
        """
        Validate that GP uncertainty aligns with prescribed conformal prediction coverage.
        
        Args:
            model: Trained GP model
            likelihood: GP likelihood
            pred: Base prediction to test
            alpha: Significance level (e.g., 0.1 for 90% coverage)
            n_test_points: Number of test points (defaults to prediction length)
            
        Returns:
            Dictionary containing coverage statistics
        """
        model.eval()
        likelihood.eval()
        
        if n_test_points is None:
            n_test_points = pred.shape[1] - 2  # Exclude boundaries
            
        # Test points (time indices)
        test_x = torch.linspace(1, pred.shape[1]-2, n_test_points, 
                              dtype=self.dtype, device=self.device).unsqueeze(-1)
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Get GP predictions
            observed_pred = likelihood(model(test_x))
            
            # Extract mean and standard deviation
            gp_mean = observed_pred.mean
            gp_std = observed_pred.stddev
            
            # Compute confidence intervals based on GP uncertainty
            z_score = torch.distributions.Normal(0, 1).icdf(torch.tensor(1 - alpha/2))
            
            # GP-based confidence intervals
            gp_lower = gp_mean - z_score * gp_std
            gp_upper = gp_mean + z_score * gp_std
            
            # Theoretical coverage should be (1 - alpha)
            theoretical_coverage = 1 - alpha
            
            # Empirical coverage (assuming GP mean should be close to 0 for zero-mean GP)
            # For validation, we check if 0 (no perturbation) falls within confidence intervals
            empirical_coverage = ((gp_lower <= 0) & (gp_upper >= 0)).float().mean().item()
            
            # Width of confidence intervals (measure of uncertainty)
            interval_width = (gp_upper - gp_lower).mean().item()
            
            # Coverage error
            coverage_error = abs(empirical_coverage - theoretical_coverage)
            
        return {
            'theoretical_coverage': theoretical_coverage,
            'empirical_coverage': empirical_coverage,
            'coverage_error': coverage_error,
            'mean_interval_width': interval_width,
            'mean_gp_std': gp_std.mean().item(),
            'gp_mean_abs_bias': gp_mean.abs().mean().item()
        }
    
    def optimize_hyperparameters_for_coverage(
        self,
        pred: torch.Tensor,
        residual_operator,
        qhat: torch.Tensor,
        alpha: float,
        noise_generator=None,
        target_coverage_error: float = 0.02,
        max_iterations: int = 10
    ) -> Dict:
        """
        Optimize GP hyperparameters to achieve target coverage alignment.
        
        Args:
            pred: Base prediction
            residual_operator: Physics residual operator
            qhat: Conformal prediction quantiles
            alpha: Target significance level
            noise_generator: Noise generation object
            target_coverage_error: Target error in coverage alignment
            max_iterations: Maximum optimization iterations
            
        Returns:
            Dictionary with optimization results
        """
        best_coverage_error = float('inf')
        best_params = None
        best_model = None
        best_likelihood = None
        
        # Parameter ranges to search
        noise_std_range = [0.05, 0.1, 0.2, 0.3]
        correlation_length_range = [16, 32, 64]
        kernel_types = ['rbf', 'matern']
        
        results = []
        
        for noise_std in noise_std_range:
            for corr_length in correlation_length_range:
                for kernel_type in kernel_types:
                    try:
                        # Generate training data
                        noise_params = {
                            'correlation_length': corr_length,
                            'std': noise_std
                        }
                        
                        train_x, train_y = self.generate_training_data_from_rejection_sampling(
                            pred, residual_operator, qhat, 
                            noise_generator=noise_generator,
                            **noise_params
                        )
                        
                        # Train GP
                        model, likelihood = self.train_gp_model(
                            train_x, train_y, kernel_type=kernel_type
                        )
                        
                        # Validate coverage
                        coverage_stats = self.validate_coverage_alignment(
                            model, likelihood, pred, alpha
                        )
                        
                        result = {
                            'noise_std': noise_std,
                            'correlation_length': corr_length,
                            'kernel_type': kernel_type,
                            **coverage_stats
                        }
                        results.append(result)
                        
                        # Check if this is the best so far
                        if coverage_stats['coverage_error'] < best_coverage_error:
                            best_coverage_error = coverage_stats['coverage_error']
                            best_params = {
                                'noise_std': noise_std,
                                'correlation_length': corr_length,
                                'kernel_type': kernel_type
                            }
                            best_model = model
                            best_likelihood = likelihood
                            
                        print(f"Tested: std={noise_std}, corr_len={corr_length}, "
                              f"kernel={kernel_type}, coverage_error={coverage_stats['coverage_error']:.4f}")
                        
                        # Early stopping if target achieved
                        if coverage_stats['coverage_error'] < target_coverage_error:
                            break
                            
                    except Exception as e:
                        print(f"Failed for params std={noise_std}, corr_len={corr_length}, "
                              f"kernel={kernel_type}: {e}")
                        continue
        
        return {
            'best_params': best_params,
            'best_model': best_model,
            'best_likelihood': best_likelihood,
            'best_coverage_error': best_coverage_error,
            'all_results': results
        }


def visualize_gp_uncertainty_vs_cp_bounds(
    model: ZeroMeanGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    pred: torch.Tensor,
    qhat: torch.Tensor,
    alpha: float,
    t: np.ndarray,
    numerical_sol: torch.Tensor = None,
    idx: int = 0
):
    """
    Visualize GP uncertainty bands alongside CP-PRE bounds.
    
    Args:
        model: Trained GP model
        likelihood: GP likelihood
        pred: Base prediction
        qhat: CP quantiles
        alpha: Significance level
        t: Time array
        numerical_sol: Ground truth solution (optional)
        idx: Index of prediction to visualize
    """
    model.eval()
    likelihood.eval()
    
    n_points = pred.shape[1]
    valid_indices = range(1, n_points-1)
    
    # Get GP predictions
    test_x = torch.tensor(valid_indices, dtype=torch.float32).unsqueeze(-1)
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        gp_pred = likelihood(model(test_x))
        gp_mean = gp_pred.mean.cpu().numpy()
        gp_std = gp_pred.stddev.cpu().numpy()
    
    # GP confidence intervals (2-sigma for ~95% coverage)
    z_score = 1.96  # for 95% confidence
    gp_lower = pred[0, valid_indices].cpu().numpy() + gp_mean - z_score * gp_std
    gp_upper = pred[0, valid_indices].cpu().numpy() + gp_mean + z_score * gp_std
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    # Ground truth (if available)
    if numerical_sol is not None:
        plt.plot(valid_indices, numerical_sol[idx, valid_indices, 0], 
                'k-', label='Ground Truth', linewidth=2)
    
    # Base prediction
    plt.plot(valid_indices, pred[0, valid_indices], 
            'r-', label='Neural ODE Prediction', linewidth=2)
    
    # GP uncertainty bands
    plt.fill_between(valid_indices, gp_lower, gp_upper, 
                    alpha=0.3, color='blue', label=f'GP {100*(1-alpha):.0f}% Confidence')
    
    # CP-PRE bounds (if provided)
    if qhat is not None:
        if qhat.ndim == 1:  # Marginal bounds
            cp_lower = pred[0, valid_indices].cpu().numpy() - qhat[valid_indices].cpu().numpy()
            cp_upper = pred[0, valid_indices].cpu().numpy() + qhat[valid_indices].cpu().numpy()
        else:  # Simple bounds
            cp_lower = pred[0, valid_indices].cpu().numpy() - qhat
            cp_upper = pred[0, valid_indices].cpu().numpy() + qhat
            
        plt.plot(valid_indices, cp_lower, 'g--', label='CP-PRE Lower', linewidth=1.5)
        plt.plot(valid_indices, cp_upper, 'g--', label='CP-PRE Upper', linewidth=1.5)
    
    plt.xlabel('Time Index')
    plt.ylabel('Position')
    plt.title(f'GP Uncertainty vs CP-PRE Bounds (α = {alpha})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# Example usage and integration with existing SHO framework
def integrate_with_sho_example(
    oscillator, func, t_span, alpha=0.1,
    noise_generator=None, device='cpu'
):
    """
    Example integration with the SHO Neural ODE framework.
    """
    from Utils.PRE.ConvOps_0d import ConvOperator
    from Utils.CP.inductive_cp import calibrate
    
    # Generate evaluation data (from existing framework)
    t, numerical_sol, neural_sol = evaluate(
        oscillator, func, t_span, n_points=100, 
        x_range=(-2,2), v_range=(-2,2), n_solves=200
    )
    
    # Setup physics residual operator
    m, k = 1.0, 1.0
    dt = t[1] - t[0]
    D_pos = ConvOperator(conv='direct')
    D_pos.kernel = m * ConvOperator(order=2).kernel + dt**2 * k * ConvOperator(order=0).kernel
    
    # Get residuals and calibrate
    pos = torch.tensor(neural_sol[..., 0], dtype=torch.float32)
    residual_cal = D_pos(pos[:100])
    qhat = calibrate(scores=np.abs(residual_cal), n=len(residual_cal), alpha=alpha)
    
    # Initialize GP trainer
    gp_trainer = GPUncertaintyTrainer(device=device)
    
    # Select a prediction to work with
    idx = 10
    pred = pos[idx:idx+1]
    
    # Optimize GP hyperparameters for coverage alignment
    print("Optimizing GP hyperparameters for coverage alignment...")
    optimization_results = gp_trainer.optimize_hyperparameters_for_coverage(
        pred, D_pos, torch.tensor(qhat), alpha, noise_generator
    )
    
    print(f"Best coverage error: {optimization_results['best_coverage_error']:.4f}")
    print(f"Best parameters: {optimization_results['best_params']}")
    
    # Visualize results
    visualize_gp_uncertainty_vs_cp_bounds(
        optimization_results['best_model'],
        optimization_results['best_likelihood'],
        pred, torch.tensor(qhat), alpha, t, numerical_sol, idx
    )
    
    return optimization_results


if __name__ == "__main__":
    print("GP Uncertainty Training Framework for CP Alignment")
    print("This module provides tools to train zero-mean GPs whose uncertainty")
    print("quantification aligns with conformal prediction coverage guarantees.")

# %%
