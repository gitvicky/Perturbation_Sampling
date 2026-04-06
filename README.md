# ResidualBound Inversion

Inverting conformal prediction bounds on the Physics Residual Error (PRE) from residual space to physical space for neural ODE/PDE solvers. Extends the work in [Calibrated Physics-Informed Uncertainty Quantification](https://arxiv.org/abs/2502.04406) (Gopakumar et al., 2025).

## Problem Statement

Given a neural surrogate that approximates solutions to a differential equation, the PRE provides a data-free nonconformity score by evaluating the PDE/ODE residual via finite-difference stencils. Conformal prediction calibrates bounds `[-qhat, +qhat]` on this residual. This project inverts those residual-space bounds back to the physical solution space, answering: *if the residual is bounded, what are the corresponding bounds on the predicted field?*

## Inversion Methods

Three strategies are implemented and compared:

1. **Point-wise Inversion** — Divides the Fourier transform of the residual bounds by the kernel spectrum: `F^{-1}(F(bounds) / F(kernel))`. Fast, but loses coverage guarantees (~86% at 95% target) because it ignores interval dependencies in the FFT.

2. **Interval FFT (Set Propagation)** — Represents the residual bounds as zonotopes and propagates them through the FFT using interval arithmetic. Preserves all dependencies, achieving guaranteed coverage (100% empirical at 95% target).

3. **Perturbation Sampling** — Samples perturbed predictions, checks whether each perturbation's residual falls within the calibrated bounds, and takes the envelope of accepted samples. Model-agnostic; applicable to nonlinear PDEs.

## Experiments

### Active Experiments

Run via the unified experiment runner:

```bash
source .venv/bin/activate
python Expts/experiment_runner.py
```

| Experiment | ODE | Description |
|---|---|---|
| **SHO** | `m x'' + k x = 0` | Simple Harmonic Oscillator. Neural ODE trained on synthetic trajectories. Composite kernel: `m*D_tt + dt^2*k*I`. |
| **DHO** | `m x'' + c x' + k x = 0` | Damped Harmonic Oscillator. Adds first-derivative damping term to the SHO kernel: `2m*D_tt + dt*c*D_t + 2*dt^2*k*I`. |

Each experiment:
1. Trains a Neural ODE on synthetic trajectories (`torchdiffeq`)
2. Computes residuals via `ConvOperator` (FD stencil convolution)
3. Calibrates `qhat` using conformal prediction
4. Inverts bounds using all three methods
5. Produces bound comparison plots and empirical coverage curves

Outputs are saved to `Paper/images/`.

### Archived Experiments

Earlier experiments in `earlier/` (not currently run by the experiment runner):

| Experiment | Equation | Notes |
|---|---|---|
| **Bessel** | Bessel ODE | Pre-computed `.npy` outputs |
| **Cauchy-Euler** | Cauchy-Euler ODE | Pre-computed `.npy` outputs |
| **Advection** | 1D Advection PDE | FNO-based surrogate (`neuraloperator`) |

## Repository Structure

```
Expts/                          # Experiment scripts
  experiment_runner.py          # Entry point for SHO + DHO
  SHO/                         # Simple Harmonic Oscillator
  DHO/                         # Damped Harmonic Oscillator

Inversion_Strategies/           # Core inversion implementations
  inversion/                   # Point-wise, Interval FFT, Perturbation Sampling
  intervalFFT/                 # Interval arithmetic FFT + zonotope propagation
  tests/                       # Tests for inversion methods

Utils/                          # Shared utilities
  PRE/                         # ConvOperator, stencils, boundary conditions
  CP/                          # Conformal prediction calibration
  noise_gen.py                 # Correlated noise generation

Neural_PDE/                     # Git submodule — neural surrogate framework
earlier/                        # Archived experiments (Bessel, Cauchy-Euler, Advection)
Paper/                          # LaTeX report and generated figures
Notes/                          # Research notes and reference papers
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.11. Key dependencies: `torch`, `torchdiffeq`, `scipy`, `numpy`, `python-interval`, `matplotlib`.

## Validation

The primary metric is **empirical coverage**: for a target level `1 - alpha`, what fraction of ground truth trajectories fall within the inverted bounds?

| Method | Coverage at 95% target |
|---|---|
| Point-wise | ~86% (undercoverage) |
| Interval FFT | 100% (guaranteed) |
| Perturbation Sampling | Depends on sample count and noise config |
