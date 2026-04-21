# CLAUDE.md

## Project Overview

This project inverts **Physics Residual Error (PRE) bounds from residual space to physical space** for neural PDE/ODE solvers. It extends the ICML paper "Calibrated Physics-Informed Uncertainty Quantification" (Gopakumar et al., 2025) which provides conformal prediction (CP) coverage guarantees in the residual space. The core research question: given calibrated bounds `[-qhat, +qhat]` on the PDE residual, what are the corresponding bounds on the predicted field?

**Current focus**: Advanced Perturbation Sampling for both linear and nonlinear PDEs. We use gradient-guided methods (Optimization, Langevin Dynamics, Generative Modeling, Variational Inference) and distributional methods (Pushforward inversion) to scale to high-dimensional, highly-constrained physical manifolds.

**Paper reference**: `2502.04406v2` — Calibrated Physics-Informed Uncertainty Quantification.

## Key Concepts

- **PRE (Physics Residual Error)**: The PDE residual `D(u_pred)` evaluated via finite-difference stencils as convolutional kernels. A data-free nonconformity score for conformal prediction.
- **Conformal Prediction**: Calibrates `qhat` from residual scores at coverage level `1 - alpha`. Provides distribution-free coverage guarantees.
- **Perturbation Sampling**: Samples perturbed predictions, checks residual containment. This is the primary method used, as it is model-agnostic and works for nonlinear PDEs.
- **Advanced Sampling Methods**:
  1. **Standard Rejection**: Monte Carlo sampling with binary accept/reject (base method).
  2. **Differentiable Rejection (Optimization)**: Backpropagates residual violations through the physics operator to "rescue" rejected samples via gradient descent.
  3. **Posterior Sampling (Langevin)**: Uses the gradient of the residual loss to guide a Langevin random walk into the valid physical manifold.
  4. **Generative Modeling**: Trains a lightweight neural network (MLP/CNN) to map standard Gaussian noise directly onto the valid physical manifold for zero-rejection inference.
  5. **Variational Inference (VI)**: Learns a parametric posterior over perturbations. Supports three covariance structures: `mean_field` (diagonal), `low_rank` (rank-r + diagonal), and `full` (Cholesky). See `VariationalPosterior` in `residual_inversion.py`.
  6. **Pushforward Inversion**: Distributional alternative using Mahalanobis conformal scores in residual space with a pushforward covariance (shrunk covariance / Tikhonov pseudoinverse). See `Inversion_Strategies/inversion/pushforward.py`.

## Repository Structure

```
Expts/                              # Experiment suite
  experiment_runner.py              # ODE entry point: SHO, DHO, Duffing
  train.py                          # Config-driven (YAML) training entry point
  evaluate.py                       # Config-driven evaluation entry point
  pushforward_runner.py             # Pushforward inversion runner
  noise_comparison.py               # Ablation across noise strategies
  paper_benchmark.py                # Benchmarking across methods × noise × alpha
  pipeline/
    adapters.py                     # Task adapters (ode_sho/dho/duffing, pde_advection_1d, pde_burgers_1d)
    io_utils.py                     # YAML config loading and merging
  Advection_Perturb.py              # 1D Advection PDE experiment
  Burgers_Perturb.py                # 1D viscous Burgers experiment
  SHO/SHO_NODE.py                   # Simple Harmonic Oscillator (Neural ODE)
  DHO/DHO_NODE.py                   # Damped Harmonic Oscillator (Neural ODE)
  Duffing/Duffing_NODE.py           # Nonlinear Duffing Oscillator
  Advection/Advection_Model.py      # U-Net / FNO surrogates, autoregressive rollout
  Burgers/Burgers_Model.py          # Spectral solver + dataset caching

Inversion_Strategies/               # Core inversion methods
  inversion/
    residual_inversion.py           # Perturbation sampling (rejection, opt, langevin, gen, VI)
    pushforward.py                  # Mahalanobis conformal + pushforward covariance
  intervalFFT/                      # Interval arithmetic / zonotope methods
  tests/
    PerturbSampling/                # Coverage tests (incl. SHO_VI_coverage, Advection_VI_coverage,
                                    #   SHO_latent_coverage, benchmark_hyperparameters)
    priors/test_priors.py           # Prior unit tests

Utils/                              # Shared utilities
  PRE/                              # ConvOps_{0d,1d,2d}.py + Stencils.py
  CP/inductive_cp.py                # Conformal prediction calibration
  priors/                           # Prior package: priors_1d, priors_2d, wrappers, spec
                                    #   (White, Spatial, GP, BSpline, Spectral, OU, PreCorrelated)
  latent_priors.py                  # Backwards-compat re-exports from Utils.priors
  noise_gen.py                      # PDENoiseGenerator (0D) + PDENoiseGenerator1D

Neural_PDE/                         # Git submodule — neural surrogate framework
Paper/                              # Paper-specific scripts and results
```

## Running Experiments

```bash
# Activate the virtual environment
source .venv/bin/activate

# Run standard experiments (SHO, DHO, Duffing)
python Expts/experiment_runner.py

# Advanced sampling flags (experiment_runner.py):
python Expts/experiment_runner.py sho --use-optim
python Expts/experiment_runner.py sho --use-langevin
python Expts/experiment_runner.py sho --use-generator
python Expts/experiment_runner.py sho --use-vi --vi-covariance low_rank --vi-rank 8

# Other experiment_runner.py flags:
#   experiments         positional, choices: sho, dho, duffing (multi-select)
#   --cp-mode           marginal | joint                (default: marginal)
#   --noise-type        spatial | white | gp | bspline  (default: spatial)
#   --model-path PATH   load a specific checkpoint
#   --retrain           force retraining
#   --train-epochs N    (default: 500)
#   --n-trajectories N  (default: 50)

# Run PDE experiments (standalone scripts):
python Expts/Advection_Perturb.py
python Expts/Burgers_Perturb.py

# Config-driven training / evaluation (YAML):
python Expts/train.py --config Expts/configs/train/sho_toy.yaml
python Expts/evaluate.py --config Expts/configs/evaluate/sho_toy.yaml

# Pushforward inversion and benchmarks:
python Expts/pushforward_runner.py
python Expts/noise_comparison.py
python Expts/paper_benchmark.py
```

### Config-driven pipeline (train.py / evaluate.py)

Both entry points take a single `--config` YAML. Top-level sections expected by
`train.py`: `Experiment` (holds `task`), `Run` (seed, run_dir), `Physics`, `Model`,
`Data`, `Opt`. Task strings dispatch through `pipeline.adapters.get_adapter(task)`:

- `ode_sho`, `ode_dho`, `ode_duffing` → `ODEAdapter`
- `pde_advection_1d` → `Advection1DAdapter`
- `pde_burgers_1d` → `Burgers1DAdapter`

Each adapter implements `train(config, paths) -> dict` and `evaluate(config, paths) -> dict`.
`evaluate.py` deep-merges the eval config on top of the training config so the eval
YAML only needs overrides. `pipeline.io_utils.resolve_run_paths` returns a
`RunPaths` dataclass with `run_dir`, `model_path`, `norms_path`, `config_path`,
`train_metrics_path`, `eval_metrics_path`. Training writes `train_metrics.yaml`;
evaluation writes `eval_metrics.yaml` (task, n_eval, nt, nx, mse, mae, dt, dx).

Example configs: `Expts/configs/train/{sho_toy,advection_toy}.yaml`,
`Expts/configs/evaluate/sho_toy.yaml`.

## Experiment Workflow

1. **Train Neural PDE/ODE** on synthetic trajectories.
2. **Evaluate** on test trajectories to get predictions.
3. **Compute residuals** via `ConvOperator` (FD stencil convolution).
4. **Calibrate** `qhat` — marginal: `calibrate_qhat_from_residual(residual_cal, alpha)`;
   joint: `calibrate_qhat_joint_from_residual(...)`; Mahalanobis (pushforward): `mahalanobis_qhat(...)`.
5. **Invert bounds** — perturbation: `perturbation_bounds_1d` / `perturbation_bounds_nd`
   (aliased as `perturbation_bounds`); pushforward: `pushforward_bounds(...)`.
6. **Evaluate coverage** — `empirical_coverage_curve_1d` / `empirical_coverage_curve_nd`
   (aliased as `empirical_coverage_curve`); pushforward: `coverage_curve(...)`.

### Key public API

`residual_inversion.py`: classes `InversionBounds`, `BoundaryGenerator`,
`PerturbationSamplingConfig`, `CoverageResult`, `VariationalPosterior`; functions
`calibrate_qhat_from_residual`, `calibrate_qhat_joint_from_residual`,
`train_boundary_generator`, `perturbation_bounds_{1d,nd}`,
`empirical_coverage_curve_{1d,nd}`.

`pushforward.py`: class `PushforwardResult`; functions `stencil_to_matrix`,
`duffing_jacobian_matrix`, `shrunk_covariance`, `tikhonov_pinv`,
`mahalanobis_qhat`, `pushforward_bounds`, `coverage_curve`.

## Validation

The primary metric is **empirical coverage**: for a given target `1 - alpha`, what fraction of ground truth trajectories fall within the inverted bounds? A valid method must achieve coverage >= `1 - alpha`.

- Perturbation sampling (Standard): High rejection rate in high dimensions.
- Perturbation sampling (Opt/Langevin/Gen/VI): Scalable to high dimensions with 100% acceptance.
- Pushforward inversion: Closed-form bounds; no sampling — use when the physics operator is (approximately) linear.

## Dependencies

```
torch, torchdiffeq, neuraloperator, torch-fftconv
scipy, numpy, matplotlib, tqdm
```

Python 3.11 via `.venv`.
