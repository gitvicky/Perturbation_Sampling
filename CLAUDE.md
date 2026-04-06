# CLAUDE.md

## Project Overview

This project inverts **Physics Residual Error (PRE) bounds from residual space to physical space** for neural PDE/ODE solvers. It extends the ICML paper "Calibrated Physics-Informed Uncertainty Quantification" (Gopakumar et al., 2025) which provides conformal prediction (CP) coverage guarantees in the residual space. The core research question: given calibrated bounds `[-qhat, +qhat]` on the PDE residual, what are the corresponding bounds on the predicted field?

**Current focus**: ODEs (SHO, DHO). Bessel, Cauchy-Euler, and 1D Advection PDE experiments are in `earlier/` (archived). Higher-dimensional PDEs are future work.

**Paper reference**: `2502.04406v2` — Calibrated Physics-Informed Uncertainty Quantification.

## Key Concepts

- **PRE (Physics Residual Error)**: The PDE residual `D(u_pred)` evaluated via finite-difference stencils as convolutional kernels. A data-free nonconformity score for conformal prediction.
- **Conformal Prediction**: Calibrates `qhat` from residual scores at coverage level `1 - alpha`. Provides distribution-free coverage guarantees.
- **Convolution Theorem inversion**: For linear PDEs, `D(u) = kernel * u` in time domain = pointwise multiply in Fourier domain. Inversion divides by the kernel FFT.
- **Three inversion methods**:
  1. **Point-wise**: Simple `F^{-1}(F(bounds) / F(kernel))` — fast but loses coverage (~86% at 95% target) due to ignoring interval dependencies.
  2. **Interval FFT (Set Propagation)**: Represents bounds as zonotopes, propagates through FFT with interval arithmetic — achieves 100% empirical coverage.
  3. **Perturbation Sampling**: Samples perturbed predictions, checks residual containment — model-agnostic, works for nonlinear PDEs.

## Repository Structure

```
Expts/                              # Experiment suite
  experiment_runner.py              # Entry point: runs SHO and DHO experiments
  SHO/
    SHO_NODE.py                     # Simple Harmonic Oscillator (Neural ODE)
    PRE_set_prop.jl                 # Julia reference implementation of set propagation
    intervalFFT.jl                  # Julia reference implementation of interval FFT
  DHO/
    DHO_NODE.py                     # Damped Harmonic Oscillator (Neural ODE)

Inversion_Strategies/               # Core inversion methods
  inversion/
    __init__.py
    residual_inversion.py           # All 3 inversion methods + coverage curves
  intervalFFT/
    intervalFFT.py                  # FFT with interval arithmetic (Python)
    zonotope.py                     # Zonotope data structure for set representation
    pre_set_prop.py                 # PRE set propagation algorithm
    example.py                      # Standalone usage example
  tests/
    PerturbSampling/                # Perturbation sampling test scripts
    vector_residuals_test.py        # Vector residual tests
    vector_tests.py                 # Vector operation tests

Utils/                              # Shared utilities
  PRE/
    ConvOps_0d.py                   # ConvOperator — wraps FD stencils as conv kernels (0D/temporal)
    ConvOps_1d.py                   # 1D spatial convolution operator
    ConvOps_2d.py                   # 2D spatial convolution operator
    ConvOps_Spatial.py              # Spatial convolution utilities
    Stencils.py                     # Finite-difference stencil definitions
    VectorConvOps.py                # Vector-valued convolution operators
    VectorConvOps_Spatial.py        # Spatial vector convolution operators
    boundary_conditions.py          # Boundary condition handling
    fft_conv_pytorch/               # PyTorch FFT convolution implementation
  CP/
    inductive_cp.py                 # Conformal prediction calibration (calibrate, emp_cov)
  noise_gen.py                      # Correlated noise for perturbation sampling

Neural_PDE/                         # Git submodule — neural surrogate framework
  UQ/inductive_cp.py               # CP calibration (referenced by inversion code)
  Models/                           # Trained model weights

earlier/                            # Archived experiments (Bessel, Cauchy-Euler, Advection)
Notes/                              # Research notes and reference PDFs
Paper/                              # LaTeX report + generated plots in Paper/images/
tests/                              # Exploratory tests (Fourier continuation, GP uncertainty)
```

## Running Experiments

```bash
# Activate the virtual environment
source .venv/bin/activate

# Run all active experiments (SHO + DHO)
python Expts/experiment_runner.py

# Individual ODE test cases:
python Expts/SHO/SHO_NODE.py
python Expts/DHO/DHO_NODE.py
```

Outputs: coverage statistics to console, comparison plots saved to `Paper/images/`.

## Experiment Workflow

1. **Train Neural ODE** on synthetic trajectories (torchdiffeq)
2. **Evaluate** on test trajectories to get predictions
3. **Compute residuals** via `ConvOperator` (FD stencil convolution)
4. **Calibrate** `qhat` using `calibrate_qhat_from_residual(residual_cal, alpha)`
5. **Invert bounds** from residual to physical space using `invert_residual_bounds_1d()`
6. **Evaluate coverage** across alpha levels with `empirical_coverage_curve_1d()`

## Validation

The primary metric is **empirical coverage**: for a given target `1 - alpha`, what fraction of ground truth trajectories fall within the inverted bounds? A valid method must achieve coverage >= `1 - alpha`.

- Point-wise inversion: typically undercoverage (~86% at 95% target)
- Interval FFT: achieves >= target coverage (100% empirical)
- Perturbation sampling: depends on sample count and noise configuration

## Key APIs

```python
# ConvOperator — finite-difference stencil as convolution kernel
D = ConvOperator(order=2)              # 2nd derivative stencil
D = ConvOperator(order=0, conv='spectral')  # custom kernel, spectral convolution
residual = D(signal)                   # apply operator
D.differentiate(signal)                # spectral differentiation
D.integrate(signal)                    # spectral integration (divide by kernel FFT)

# Inversion
from Inversion_Strategies.inversion.residual_inversion import (
    calibrate_qhat_from_residual,
    invert_residual_bounds_1d,
    perturbation_bounds_1d,
    empirical_coverage_curve_1d,
    IntervalFFTSlicing,
    PerturbationSamplingConfig,
)
```

## Dependencies

```
torch, torchdiffeq, neuraloperator, torch-fftconv
scipy, numpy, python-interval, matplotlib, tqdm
```

Python 3.11 via `.venv`.

## Development Notes

- `sys.path` manipulation is used throughout to resolve cross-module imports — run scripts from the project root.
- The `Neural_PDE/` directory is a git submodule.
- Interval arithmetic uses `python-interval` library. Use `interval([a, b])` (list syntax) not `interval(a, b)` for proper bounds.
- For ODE cases, the composite kernel combines temporal derivative stencils scaled by `dt` and physical parameters (e.g., `m*D_tt + dt^2*k*D_identity` for SHO).
- `IntervalFFTSlicing` controls how the interval FFT output is aligned back to the original signal — edge effects from FFT require careful slicing.
- Julia reference implementations (`PRE_set_prop.jl`, `intervalFFT.jl`) exist in `Expts/SHO/` alongside the Python versions.
