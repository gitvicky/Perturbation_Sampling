from dataclasses import dataclass
from typing import Optional, Sequence, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from Neural_PDE.UQ.inductive_cp import calibrate
from Utils.noise_gen import PDENoiseGenerator, PDENoiseGenerator1D

@dataclass(frozen=True)
class InversionBounds:
    lower: np.ndarray
    upper: np.ndarray
    width: np.ndarray


class BoundaryGenerator(nn.Module):
    def __init__(self, input_shape: tuple[int, ...], hidden_dim: int = 128):
        super().__init__()
        self.input_shape = input_shape
        self.n_flat = int(np.prod(input_shape))
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n_flat, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_flat)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [batch, n_flat]
        out = self.net(z)
        return out.view(-1, *self.input_shape)


@dataclass(frozen=True)
class PerturbationSamplingConfig:
    n_samples: int = 4000
    batch_size: int = 1000
    max_rounds: int = 2
    noise_type: str = "spatial"
    noise_std: float = 0.10
    correlation_length: float = 32.0
    gp_kernel: str = "rbf"
    gp_nu: float = 1.5
    bspline_n_knots: int = 16
    pre_kernel: Optional[torch.Tensor] = None
    seed: Optional[int] = None
    std_retry_factors: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625)

    # Method 1 & 2: Advanced Sampling
    use_optimisation: bool = False
    use_mcmc: bool = False
    mcmc_steps: int = 50
    mcmc_step_size: float = 1e-3
    mcmc_noise_scale: float = 1e-4
    opt_steps: int = 50
    opt_lr: float = 1e-2
    lambda_prior: float = 1e-3
    lambda_boundary: float = 1.0

    # Method 3: Generative Modeling
    use_generator: bool = False
    gen_train_steps: int = 500
    gen_lr: float = 1e-3
    gen_hidden_dim: int = 128


@dataclass(frozen=True)
class CoverageResult:
    nominal_coverage: np.ndarray
    empirical_coverage_perturbation: Optional[np.ndarray] = None


def calibrate_qhat_from_residual(residual_cal: torch.Tensor, alpha: float) -> float:
    scores = np.abs(residual_cal.detach().cpu().numpy().flatten())
    return float(calibrate(scores=scores, n=len(scores), alpha=alpha))


def calibrate_qhat_joint_from_residual(
    residual_cal: torch.Tensor, alpha: float, eps: float = 1e-16,
) -> tuple[float, np.ndarray]:
    """Joint conformal calibration over residuals.

    Returns (qhat, modulation) where the per-point residual bound is
    ``qhat * modulation``.  The nonconformity score for each trajectory is
    ``max_t |residual(t)| / modulation(t)`` — a single scalar per trajectory —
    so the resulting ``qhat`` controls *joint* (simultaneous) coverage.
    """
    res_np = residual_cal.detach().cpu().numpy()                  # [N, T]
    modulation = np.std(res_np, axis=0)                           # [T]
    modulation = np.maximum(modulation, eps)                      # avoid /0
    scores = np.max(np.abs(res_np) / modulation[None, :], axis=tuple(range(1, res_np.ndim)))  # [N]
    qhat = float(calibrate(scores=scores, n=len(scores), alpha=alpha))
    return qhat, modulation


def train_boundary_generator(
    pred_tensor: torch.Tensor,
    operator,
    qhat,
    config: PerturbationSamplingConfig
) -> nn.Module:
    input_shape = pred_tensor.shape[1:]
    n_flat = int(np.prod(input_shape))
    generator = BoundaryGenerator(input_shape, config.gen_hidden_dim).to(pred_tensor.device)
    optimizer = optim.Adam(generator.parameters(), lr=config.gen_lr)
    
    qhat_t = torch.tensor(np.asarray(qhat, dtype=np.float64), 
                          dtype=torch.float32, device=pred_tensor.device).abs()
    if qhat_t.ndim > 0:
        qhat_t = qhat_t.unsqueeze(0)

    generator.train()
    for _ in range(config.gen_train_steps):
        optimizer.zero_grad()
        z = torch.randn(config.batch_size, n_flat, device=pred_tensor.device)
        noise = generator(z)
        
        p_gen = pred_tensor + noise
        r_gen = operator(p_gen)
        
        diff = torch.abs(r_gen) - qhat_t
        l_bound = torch.mean(torch.clamp(diff, min=0)**2)
        l_prior = torch.mean(noise**2)
        
        loss = config.lambda_boundary * l_bound + config.lambda_prior * l_prior
        loss.backward()
        optimizer.step()
        
    generator.eval()
    return generator


def _trajectory_coverage_nd(truth: np.ndarray, bounds: InversionBounds) -> float:
    in_lower = truth >= bounds.lower[None, ...]
    in_upper = truth <= bounds.upper[None, ...]
    axes = tuple(range(1, truth.ndim))
    contained = np.logical_and(in_lower, in_upper).all(axis=axes)
    return float(contained.mean())


def perturbation_bounds_nd(
    pred_signal: np.ndarray,
    residual_operator,
    qhat,
    *,
    interior_slice: Union[slice, tuple[slice, ...]] = (slice(1, -1),),
    config: PerturbationSamplingConfig = PerturbationSamplingConfig(),
    fallback_lower: Optional[np.ndarray] = None,
    fallback_upper: Optional[np.ndarray] = None,
    joint: bool = False,
) -> InversionBounds:
    pred_signal = np.asarray(pred_signal, dtype=np.float32)
    input_shape = pred_signal.shape
    n_flat = int(np.prod(input_shape))
    
    if isinstance(interior_slice, slice):
        interior_slice = (interior_slice,)
    
    device = next(residual_operator.parameters()).device if hasattr(residual_operator, 'parameters') and any(residual_operator.parameters()) else 'cpu'
    pred_tensor = torch.tensor(pred_signal, dtype=torch.float32).to(device).unsqueeze(0)

    generator = None
    if config.use_generator:
        generator = train_boundary_generator(pred_tensor, residual_operator, qhat, config)

    noise_gen_2d = PDENoiseGenerator(device=pred_tensor.device, dtype=pred_tensor.dtype)
    noise_gen_1d = PDENoiseGenerator1D(device=pred_tensor.device, dtype=pred_tensor.dtype)

    with torch.no_grad():
        dummy = torch.zeros((1, *input_shape), device=device)
        interior_dummy = dummy[(slice(None), *interior_slice)]
        interior_shape = interior_dummy.shape[1:]

    lower = None
    upper = None
    counts = None
    last_missing = int(np.prod(interior_shape))

    for retry_id, std_factor in enumerate(config.std_retry_factors):
        trial_std = float(config.noise_std) * float(std_factor)
        # Reset bounds each retry for standard noise (different scale → different bounds).
        # For generator: accumulate across retries since each draws fresh random z
        # and the generator doesn't depend on noise_std.
        if not config.use_generator or retry_id == 0:
            lower = torch.full(interior_shape, float("inf"), dtype=torch.float32, device=pred_tensor.device)
            upper = torch.full(interior_shape, float("-inf"), dtype=torch.float32, device=pred_tensor.device)
            counts = torch.zeros(interior_shape, dtype=torch.long, device=pred_tensor.device)

        remaining = int(config.n_samples)
        round_idx = 0
        while remaining > 0 and round_idx < config.max_rounds:
            draw = min(config.batch_size, remaining)
            seed = None if config.seed is None else int(config.seed + 100 * retry_id + round_idx)

            if config.use_generator:
                # In generator mode, draw generator samples on every retry.
                # std_retry_factors still gives extra independent attempts,
                # and bounds/counts are accumulated across retries.
                with torch.no_grad():
                    z = torch.randn(draw, n_flat, device=pred_tensor.device)
                    noise = generator(z)
            elif len(input_shape) == 1:
                # 1D signals: use PDENoiseGenerator1D which has proper 1D correlated noise
                n_points = input_shape[0]
                if config.noise_type == "spatial":
                    noise = noise_gen_1d.spatially_correlated_noise(draw, n_points, correlation_length=config.correlation_length, std=trial_std, seed=seed)
                elif config.noise_type == "white":
                    noise = noise_gen_1d.white_noise(draw, n_points, std=trial_std, seed=seed)
                elif config.noise_type == "gp":
                    noise = noise_gen_1d.gp_noise(draw, n_points, correlation_length=config.correlation_length, std=trial_std, kernel_type=config.gp_kernel, nu=config.gp_nu, seed=seed)
                elif config.noise_type == "bspline":
                    noise = noise_gen_1d.bspline_noise(draw, n_points, n_knots=config.bspline_n_knots, std=trial_std, seed=seed)
                elif config.noise_type == "pre_correlated":
                    if config.pre_kernel is None:
                        raise ValueError("pre_kernel must be set in config for noise_type='pre_correlated'")
                    noise = noise_gen_1d.pre_correlated_noise(draw, n_points, kernel=config.pre_kernel.clone(), std=trial_std, seed=seed)
                else:
                    raise ValueError(f"Unknown noise_type: {config.noise_type}")
            elif config.noise_type == "white":
                noise = noise_gen_2d.white_noise((draw, *input_shape), std=trial_std, seed=seed)
            elif config.noise_type == "spatial":
                noises = []
                for i in range(draw):
                    noises.append(noise_gen_2d.spatially_correlated_noise(input_shape, correlation_length=config.correlation_length, std=trial_std))
                noise = torch.stack(noises)
            else:
                noise = noise_gen_2d.white_noise((draw, *input_shape), std=trial_std, seed=seed)

            # Method 1: MCMC - Langevin
            if config.use_mcmc and not config.use_generator:
                noise = noise.clone().detach().requires_grad_(True)
                qhat_loss = torch.tensor(np.asarray(qhat, dtype=np.float64),
                                         dtype=torch.float32, device=pred_tensor.device).abs()
                if qhat_loss.ndim > 0:
                    qhat_loss = qhat_loss.unsqueeze(0)
                
                with torch.enable_grad():
                    for _ in range(config.mcmc_steps):
                        if noise.grad is not None:
                            noise.grad.zero_()
                        p_mcmc = pred_tensor + noise
                        r_mcmc = residual_operator(p_mcmc)
                        diff = torch.abs(r_mcmc) - qhat_loss
                        l_bound = torch.mean(torch.clamp(diff, min=0)**2)
                        l_prior = torch.mean(noise**2)
                        loss = config.lambda_boundary * l_bound + config.lambda_prior * l_prior
                        loss.backward()
                        with torch.no_grad():
                            grad = noise.grad
                            noise -= (config.mcmc_step_size / 2) * grad
                            noise += np.sqrt(config.mcmc_step_size) * config.mcmc_noise_scale * torch.randn_like(noise)
                noise = noise.detach()

            perturbed = pred_tensor + noise
            residuals = residual_operator(perturbed)
            
            qhat_t = torch.tensor(np.asarray(qhat, dtype=np.float64),
                                  dtype=torch.float32, device=pred_tensor.device).abs()
            if qhat_t.ndim > 0:
                qhat_t = qhat_t.unsqueeze(0)
            
            within = torch.abs(residuals) <= qhat_t

            if joint:
                # Accept/reject entire trajectories
                traj_ok = within.reshape(draw, -1).all(dim=1)
                for _ in range(len(input_shape)):
                    traj_ok = traj_ok.unsqueeze(-1)
                within = traj_ok.expand_as(within)

            # --- Accumulate pre-optimisation bounds ---
            full_slice = (slice(None), *interior_slice)
            pred_slice = perturbed[full_slice]
            mask_slice = within[full_slice]

            inf_tensor = torch.full_like(pred_slice, float("inf"))
            ninf_tensor = torch.full_like(pred_slice, float("-inf"))
            batch_min = torch.where(mask_slice, pred_slice, inf_tensor).amin(dim=0)
            batch_max = torch.where(mask_slice, pred_slice, ninf_tensor).amax(dim=0)

            lower = torch.minimum(lower, batch_min)
            upper = torch.maximum(upper, batch_max)
            counts = counts + mask_slice.sum(dim=0)

            # Method 2: Optimisation — rescue rejected samples as additional bounds
            if config.use_optimisation and not config.use_generator:
                rejected = ~within.reshape(draw, -1).all(dim=1)
                if rejected.any():
                    noise_opt = noise[rejected].clone().detach().requires_grad_(True)
                    optimizer = optim.Adam([noise_opt], lr=config.opt_lr)
                    qhat_loss = qhat_t.clone().detach()

                    with torch.enable_grad():
                        for _ in range(config.opt_steps):
                            optimizer.zero_grad()
                            p_opt = pred_tensor + noise_opt
                            r_opt = residual_operator(p_opt)
                            diff = torch.abs(r_opt) - qhat_loss
                            l_bound = torch.mean(torch.clamp(diff, min=0)**2)
                            l_prior = torch.mean(noise_opt**2)
                            loss = config.lambda_boundary * l_bound + config.lambda_prior * l_prior
                            loss.backward()
                            optimizer.step()

                    with torch.no_grad():
                        p_final = pred_tensor + noise_opt.detach()
                        r_final = residual_operator(p_final)
                        w_final = torch.abs(r_final) <= qhat_t
                        if joint:
                            traj_ok_final = w_final.reshape(w_final.shape[0], -1).all(dim=1)
                            for _ in range(len(input_shape)):
                                traj_ok_final = traj_ok_final.unsqueeze(-1)
                            w_final = traj_ok_final.expand_as(w_final)

                        opt_pred = p_final[full_slice]
                        opt_mask = w_final[full_slice]
                        opt_min = torch.where(opt_mask, opt_pred,
                                              torch.full_like(opt_pred, float("inf"))).amin(dim=0)
                        opt_max = torch.where(opt_mask, opt_pred,
                                              torch.full_like(opt_pred, float("-inf"))).amax(dim=0)
                        lower = torch.minimum(lower, opt_min)
                        upper = torch.maximum(upper, opt_max)
                        counts = counts + opt_mask.sum(dim=0)

            remaining -= draw
            round_idx += 1

        missing_mask = counts == 0
        last_missing = int(missing_mask.sum().item())
        if last_missing == 0:
            break

    if last_missing > 0:
        if fallback_lower is None or fallback_upper is None:
            if config.use_generator:
                # Generator mode can occasionally miss a few points; fill these
                # conservatively from the valid envelope rather than aborting.
                valid_mask = ~missing_mask
                if not valid_mask.any():
                    raise RuntimeError(
                        f"Perturbation sampling failed at {last_missing} points. Increase n_samples."
                    )
                global_lower = lower[valid_mask].min()
                global_upper = upper[valid_mask].max()
                lower = torch.where(missing_mask, global_lower, lower)
                upper = torch.where(missing_mask, global_upper, upper)
            else:
                raise RuntimeError(f"Perturbation sampling failed at {last_missing} points. Increase n_samples.")
        if fallback_lower is not None and fallback_upper is not None:
            fallback_lower_t = torch.tensor(fallback_lower, dtype=torch.float32, device=pred_tensor.device)
            fallback_upper_t = torch.tensor(fallback_upper, dtype=torch.float32, device=pred_tensor.device)
            lower = torch.where(missing_mask, fallback_lower_t, lower)
            upper = torch.where(missing_mask, fallback_upper_t, upper)

    lower_np = lower.detach().cpu().numpy().astype(float)
    upper_np = upper.detach().cpu().numpy().astype(float)
    return InversionBounds(lower=lower_np, upper=upper_np, width=upper_np - lower_np)


def perturbation_bounds_1d(
    pred_signal: np.ndarray,
    residual_operator,
    qhat,
    *,
    interior_slice: Union[slice, tuple[slice, ...]] = slice(1, -1),
    config: PerturbationSamplingConfig = PerturbationSamplingConfig(),
    fallback_lower: Optional[np.ndarray] = None,
    fallback_upper: Optional[np.ndarray] = None,
    joint: bool = False,
) -> InversionBounds:
    """Wrapper around perturbation_bounds_nd for 1D/2D backward compatibility."""
    if not isinstance(interior_slice, tuple):
        if pred_signal.ndim == 2:
            interior_slice = (slice(None), interior_slice)
        else:
            interior_slice = (interior_slice,)
            
    return perturbation_bounds_nd(
        pred_signal=pred_signal,
        residual_operator=residual_operator,
        qhat=qhat,
        interior_slice=interior_slice,
        config=config,
        fallback_lower=fallback_lower,
        fallback_upper=fallback_upper,
        joint=joint,
    )


def empirical_coverage_curve_nd(
    preds: np.ndarray,
    truths: np.ndarray,
    residual_cal: torch.Tensor,
    operator,
    *,
    alphas: Sequence[float],
    interior_slice: tuple[slice, ...] = (slice(1, -1),),
    perturbation_config: Optional[PerturbationSamplingConfig] = None,
    cp_mode: str = "marginal",
) -> CoverageResult:
    nominal = []
    cov_perturb = []

    preds = np.asarray(preds, dtype=float)
    truths = np.asarray(truths, dtype=float)
    joint = cp_mode == "joint"

    for alpha in tqdm(alphas, desc="Coverage alphas", unit="α"):
        if joint:
            qhat_scalar, modulation = calibrate_qhat_joint_from_residual(residual_cal, alpha=float(alpha))
            qhat = qhat_scalar * modulation
        else:
            qhat = calibrate_qhat_from_residual(residual_cal, alpha=float(alpha))

        if perturbation_config is not None:
            for i in tqdm(range(preds.shape[0]), desc=f"  α={float(alpha):.2f} trajectories", leave=False, unit="traj"):
                truth_i = truths[i][interior_slice]
                perturb_bounds = perturbation_bounds_nd(
                    pred_signal=preds[i],
                    residual_operator=operator,
                    qhat=qhat,
                    interior_slice=interior_slice,
                    config=perturbation_config,
                    joint=joint,
                )
                cov_perturb.append(_trajectory_coverage_nd(truth_i[None, ...], perturb_bounds))

        nominal.append(1.0 - float(alpha))

    if perturbation_config is not None:
        n_traj = preds.shape[0]
        cov_perturb = np.asarray(cov_perturb, dtype=float).reshape(len(alphas), n_traj).mean(axis=1)
        perturb_cov = cov_perturb
    else:
        perturb_cov = None

    return CoverageResult(
        nominal_coverage=np.asarray(nominal, dtype=float),
        empirical_coverage_perturbation=perturb_cov,
    )

# Aliases for backward compatibility and convenience
perturbation_bounds = perturbation_bounds_1d
empirical_coverage_curve = empirical_coverage_curve_nd
empirical_coverage_curve_1d = empirical_coverage_curve_nd
InversionBounds1D = InversionBounds
