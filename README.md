# ResidualBound Inversion

Inverting conformal prediction bounds on the Physics Residual Error (PRE) from residual space to physical space for neural ODE/PDE solvers. Extends the work in [Calibrated Physics-Informed Uncertainty Quantification](https://arxiv.org/abs/2502.04406) (Gopakumar et al., 2025).

## Problem Statement

Given a neural surrogate that approximates solutions to a differential equation, the PRE provides a data-free nonconformity score by evaluating the PDE/ODE residual via finite-difference stencils. Conformal prediction calibrates bounds `[-qhat, +qhat]` on this residual. This project inverts those residual-space bounds back to the physical solution space, answering: *if the residual is bounded, what are the corresponding bounds on the predicted field?*

## Inversion Methods: Advanced Perturbation Sampling

We implement a suite of Advanced Perturbation Sampling methods to map residual bounds to the physical domain. These methods are model-agnostic and applicable to both linear and nonlinear differential equations.

1.  **Standard Perturbation Sampling** — Samples correlated noise (Spatial, GP, or B-Spline) and accepts perturbations whose residuals fall within the calibrated conformal bounds.
2.  **Differentiable Rejection (Optimization)** — For rejected samples, performs inference-time optimization to "push" the perturbation into the valid residual region.
3.  **Posterior Sampling (Langevin Dynamics)** — Uses MCMC/Langevin steps to refine perturbations, ensuring they satisfy the physics constraints while maintaining diversity.
4.  **Generative Modeling (Boundary Generator)** — Trains a small generative network to directly produce valid perturbations that lie on the boundary of the physical uncertainty envelope.

## Experiments

### Active Experiments

Run via the unified experiment runner:

```bash
source .venv/bin/activate
python Expts/experiment_runner.py
```

| Experiment | Equation | Description |
|---|---|---|
| **SHO** | `m x'' + k x = 0` | Simple Harmonic Oscillator. Linear test case. |
| **DHO** | `m x'' + c x' + k x = 0` | Damped Harmonic Oscillator. Linear with damping. |
| **Duffing** | `x'' + dx' + ax + bx^3 = 0` | Duffing Oscillator. Nonlinear with cubic term. |

#### Selecting Experiments

Run a specific experiment by name:

```bash
python Expts/experiment_runner.py sho        # Simple Harmonic Oscillator only
python Expts/experiment_runner.py dho        # Damped Harmonic Oscillator only
python Expts/experiment_runner.py duffing    # Duffing Oscillator only
python Expts/experiment_runner.py sho dho    # Multiple experiments
```

#### Sampling Strategies

By default, experiments use **Standard Rejection Sampling** (Monte Carlo with binary accept/reject). Use these flags to switch to an advanced sampling strategy:

| Flag | Method | Description |
|---|---|---|
| *(default)* | Standard Rejection (MC) | Monte Carlo sampling with binary accept/reject |
| `--use-optimisation` | Differentiable Rejection (Optim) | Backpropagates residual violations to rescue rejected samples via gradient descent |
| `--use-mcmc` | Posterior Sampling (Langevin) | MCMC/Langevin dynamics to walk into the valid physical manifold |
| `--use-generator` | Generative Modeling (Gen) | Trains a small neural network to directly produce valid perturbations |

```bash
# Standard rejection sampling (default)
python Expts/experiment_runner.py sho

# Differentiable rejection (inference-time optimization)
python Expts/experiment_runner.py sho --use-optimisation

# Posterior sampling (Langevin dynamics)
python Expts/experiment_runner.py sho --use-mcmc

# Generative modeling (boundary generator)
python Expts/experiment_runner.py sho --use-generator
```

#### Noise Types

Control the noise model used for perturbation sampling with `--noise-type`:

| Noise Type | Description |
|---|---|
| `spatial` (default) | Spatially correlated noise |
| `white` | White (uncorrelated) noise |
| `gp` | Gaussian process noise (RBF kernel) |
| `bspline` | B-spline basis noise |

```bash
python Expts/experiment_runner.py sho --noise-type gp
python Expts/experiment_runner.py duffing --noise-type bspline --use-mcmc
```

#### Conformal Prediction Mode

Choose between marginal and joint conformal prediction calibration:

```bash
python Expts/experiment_runner.py sho --cp-mode marginal   # default
python Expts/experiment_runner.py sho --cp-mode joint
```

#### Transductive Mode

Use all data for calibration (transductive CP) instead of the default 80/20 split:

```bash
python Expts/experiment_runner.py sho --transductive
```

#### Complex PDE Scaling (Advection)

For the 1D Advection PDE experiment, run separately:

```bash
python Expts/Advection_Perturb.py
```

Each experiment:
1. Trains a Neural ODE on synthetic trajectories.
2. Computes residuals via `ConvOperator` or nonlinear operator.
3. Calibrates `qhat` using conformal prediction.
4. Inverts bounds using Advanced Perturbation Sampling.
5. Produces bound comparison plots and empirical coverage curves.

Outputs are saved to `Paper/images/`.

## Repository Structure

```
Expts/                          # Experiment scripts
  experiment_runner.py          # Unified entry point
  SHO/                         # Simple Harmonic Oscillator
  DHO/                         # Damped Harmonic Oscillator
  Pendulum/                    # Nonlinear Pendulum
  Duffing/                     # Duffing Oscillator

Inversion_Strategies/           # Core inversion implementations
  inversion/                   # Perturbation Sampling (Standard, Opt, Langevin, Gen)
  tests/                       # Tests for inversion methods

Utils/                          # Shared utilities
  PRE/                         # ConvOperator, stencils, boundary conditions
  CP/                          # Conformal prediction calibration
  noise_gen.py                 # Correlated noise generation

Neural_PDE/                     # Git submodule — neural surrogate framework
Paper/                          # LaTeX report and generated figures
Notes/                          # Research notes and reference papers
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.11. Key dependencies: `torch`, `torchdiffeq`, `scipy`, `numpy`, `matplotlib`.

## Validation

The primary metric is **empirical coverage**: for a target level `1 - alpha`, what fraction of ground truth trajectories fall within the inverted bounds? Advanced Perturbation Sampling typically achieves well-calibrated coverage across both linear and nonlinear cases.
