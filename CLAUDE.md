# CLAUDE.md

## Project Overview

This project inverts **Physics Residual Error (PRE) bounds from residual space to physical space** for neural PDE/ODE solvers. It extends the ICML paper "Calibrated Physics-Informed Uncertainty Quantification" (Gopakumar et al., 2025) which provides conformal prediction (CP) coverage guarantees in the residual space. The core research question: given calibrated bounds `[-qhat, +qhat]` on the PDE residual, what are the corresponding bounds on the predicted field?

**Current focus**: Advanced Perturbation Sampling for both linear and nonlinear PDEs. We use gradient-guided methods (Optimization, Langevin Dynamics, and Generative Modeling) to scale to high-dimensional, highly-constrained physical manifolds.

**Paper reference**: `2502.04406v2` — Calibrated Physics-Informed Uncertainty Quantification.

## Key Concepts

- **PRE (Physics Residual Error)**: The PDE residual `D(u_pred)` evaluated via finite-difference stencils as convolutional kernels. A data-free nonconformity score for conformal prediction.
- **Conformal Prediction**: Calibrates `qhat` from residual scores at coverage level `1 - alpha`. Provides distribution-free coverage guarantees.
- **Perturbation Sampling**: Samples perturbed predictions, checks residual containment. This is the primary method used, as it is model-agnostic and works for nonlinear PDEs.
- **Advanced Sampling Methods**:
  1. **Standard Rejection**: Monte Carlo sampling with binary accept/reject (base method).
  2. **Differentiable Rejection (Optimization)**: Backpropagates residual violations through the physics operator to "rescue" rejected samples via gradient descent.
  3. **Posterior Sampling (Langevin)**: Uses the gradient of the residual loss to guide a random walk (MCMC) into the valid physical manifold.
  4. **Generative Modeling**: Trains a lightweight neural network (MLP/CNN) to map standard Gaussian noise directly onto the valid physical manifold for zero-rejection inference.

## Repository Structure

```
Expts/                              # Experiment suite
  experiment_runner.py              # Entry point: runs SHO, DHO, Pendulum, Duffing
  Advection_Perturb.py              # 1D Advection PDE experiment (Complex PDE Scaling)
  SHO/
    SHO_NODE.py                     # Simple Harmonic Oscillator (Neural ODE)
  DHO/
    DHO_NODE.py                     # Damped Harmonic Oscillator (Neural ODE)
  Pendulum/
    Pendulum_NODE.py                # Nonlinear Pendulum
  Duffing/
    Duffing_NODE.py                 # Nonlinear Duffing Oscillator

Inversion_Strategies/               # Core inversion methods
  inversion/
    __init__.py
    residual_inversion.py           # Perturbation sampling (rejection, opt, langevin, gen)
  tests/
    PerturbSampling/                # Perturbation sampling test scripts

Utils/                              # Shared utilities
  PRE/
    ConvOps_0d.py                   # ConvOperator — wraps FD stencils (0D/temporal)
    ConvOps_1d.py                   # 1D spatial/spatiotemporal convolution operator
    ConvOps_2d.py                   # 2D spatial convolution operator
    Stencils.py                     # Finite-difference stencil definitions
  CP/
    inductive_cp.py                 # Conformal prediction calibration
  noise_gen.py                      # Noise generation for perturbation sampling

Neural_PDE/                         # Git submodule — neural surrogate framework
  Numerical_Solvers/                # Numerical PDE solvers (Advection, Burgers, etc.)
  Models/                           # Trained model weights
```

## Running Experiments

```bash
# Activate the virtual environment
source .venv/bin/activate

# Run standard experiments (SHO, DHO, Pendulum, Duffing)
python Expts/experiment_runner.py

# Test advanced sampling flags:
python Expts/experiment_runner.py sho --use-optimization
python Expts/experiment_runner.py sho --use-langevin
python Expts/experiment_runner.py sho --use-generator

# Run complex PDE scaling (Advection):
python Expts/Advection_Perturb.py
```

## Experiment Workflow

1. **Train Neural PDE/ODE** on synthetic trajectories.
2. **Evaluate** on test trajectories to get predictions.
3. **Compute residuals** via `ConvOperator` (FD stencil convolution).
4. **Calibrate** `qhat` using `calibrate_qhat_from_residual(residual_cal, alpha)`.
5. **Invert bounds** using `perturbation_bounds(pred, operator, qhat, config)`.
6. **Evaluate coverage** with `evaluate_coverage_nd()`.

## Validation

The primary metric is **empirical coverage**: for a given target `1 - alpha`, what fraction of ground truth trajectories fall within the inverted bounds? A valid method must achieve coverage >= `1 - alpha`.

- Perturbation sampling (Standard): High rejection rate in high dimensions.
- Perturbation sampling (Opt/Langevin/Gen): Scalable to high dimensions with 100% acceptance.

## Dependencies

```
torch, torchdiffeq, neuraloperator, torch-fftconv
scipy, numpy, matplotlib, tqdm
```

Python 3.11 via `.venv`.
