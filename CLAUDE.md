# CLAUDE.md

## Project Overview

This project inverts **Physics Residual Error (PRE) bounds from residual space to physical space** for neural PDE/ODE solvers. It extends the ICML paper "Calibrated Physics-Informed Uncertainty Quantification" (Gopakumar et al., 2025) which provides conformal prediction (CP) coverage guarantees in the residual space. The core research question: given calibrated bounds `[-qhat, +qhat]` on the PDE residual, what are the corresponding bounds on the predicted field?

**Current focus**: ODEs first, then 1D PDEs (Advection). Higher-dimensional PDEs are future work.

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
ConvTheorem/                    # Main experimental suite
  experiment_runner.py          # Entry point: runs all ODE/PDE experiments
  inversion/
    residual_inversion.py       # Core inversion logic (all 3 methods + coverage curves)
  intervalFFT/
    intervalFFT.py              # FFT with interval arithmetic
    zonotope.py                 # Zonotope data structure for set representation
    pre_set_prop.py             # PRE set propagation algorithm
  SHO/SHO_node_test.py          # Simple Harmonic Oscillator (Neural ODE)
  DHO/DHO_NODE.py               # Damped Harmonic Oscillator
  Bessel/Bessel_NODE.py         # Bessel ODE
  Cauchy_Euler/Cauchy_Euler_NODE.py
  Advection/                    # 1D Advection PDE (FNO-based)
    Advection_FNO.py
    Advection_PRE.py

Neural_PDE/                     # Git submodule — neural surrogate framework
  UQ/inductive_cp.py            # Conformal prediction calibration (calibrate function)
  Utils/PRE/                    # ConvOps for 0d, 1d, 2d residual computation
  Models/                       # Trained model weights

Utils/
  PRE/ConvOps_0d.py             # ConvOperator class — wraps FD stencils as conv kernels
  noise_gen.py                  # Correlated noise for perturbation sampling

PerturbSampling/                # Standalone perturbation sampling experiments
Paper/                          # LaTeX report + generated plots in Paper/images/
```

## Running Experiments

```bash
# Activate the virtual environment
source .venv/bin/activate

# Run all experiments (SHO, DHO, Bessel, Cauchy-Euler, Advection)
python ConvTheorem/experiment_runner.py

# Individual ODE test cases can be run directly:
python ConvTheorem/SHO/SHO_node_test.py
python ConvTheorem/DHO/DHO_NODE.py
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
from ConvTheorem.inversion.residual_inversion import (
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
