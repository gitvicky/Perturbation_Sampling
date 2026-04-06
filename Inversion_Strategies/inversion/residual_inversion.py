from dataclasses import dataclass
from typing import Optional, Sequence
import os
import sys

import numpy as np
import torch
from interval import interval
from scipy.fft import fft, ifft
from tqdm import tqdm

_INTERVALFFT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "intervalFFT"))
if _INTERVALFFT_DIR not in sys.path:
    sys.path.append(_INTERVALFFT_DIR)

from intervalFFT import Real, complex_prod, intervalFFT, inverse_intervalFFT
from Neural_PDE.UQ.inductive_cp import calibrate
from Utils.noise_gen import PDENoiseGenerator1D


@dataclass(frozen=True)
class IntervalFFTSlicing:
    center_start: int = 3
    center_end: int = -1
    n_right_edges: int = 3
    output_offset: int = 2


@dataclass(frozen=True)
class InversionBounds1D:
    lower: np.ndarray
    upper: np.ndarray
    width: np.ndarray


@dataclass(frozen=True)
class CoverageResult:
    nominal_coverage: np.ndarray
    empirical_coverage_pointwise: np.ndarray
    empirical_coverage_intervalfft: np.ndarray
    empirical_coverage_perturbation: Optional[np.ndarray] = None


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
    seed: Optional[int] = None
    std_retry_factors: tuple[float, ...] = (1.0, 0.5, 0.25)


def calibrate_qhat_from_residual(residual_cal: torch.Tensor, alpha: float) -> float:
    scores = np.abs(residual_cal.detach().cpu().numpy().flatten())
    return float(calibrate(scores=scores, n=len(scores), alpha=alpha))


def pointwise_inverse_width(
    operator,
    n_points: int,
    qhat: float,
    *,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    slice_pad: bool = False,
) -> np.ndarray:
    qhat_tensor = torch.full((1, n_points), float(qhat), dtype=dtype, device=device)
    inv_qhat = operator.integrate(qhat_tensor, slice_pad=slice_pad)[0]
    return np.abs(inv_qhat.detach().cpu().numpy())


def pointwise_inversion_bounds_1d(
    pred_signal: np.ndarray,
    inv_width: np.ndarray,
    *,
    interior_slice: slice = slice(1, -1),
) -> InversionBounds1D:
    center = np.asarray(pred_signal[interior_slice], dtype=float)
    width = np.asarray(inv_width[interior_slice], dtype=float)
    lower = center - width
    upper = center + width
    return InversionBounds1D(lower=lower, upper=upper, width=width)


def pointwise_inverse_width_nd(
    operator,
    field_shape: Sequence[int],
    qhat: float,
    *,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    slice_pad: bool = True,
) -> np.ndarray:
    qhat_tensor = torch.full((1, *field_shape), float(qhat), dtype=dtype, device=device)
    inv_qhat = operator.integrate(qhat_tensor, slice_pad=slice_pad)[0]
    return np.abs(inv_qhat.detach().cpu().numpy())


def pointwise_inversion_bounds_nd(pred_field: np.ndarray, inv_width: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(pred_field, dtype=float)
    width = np.asarray(inv_width, dtype=float)
    return center - width, center + width


def _prepare_interval_signal(
    pred_signal: np.ndarray,
    kernel: np.ndarray,
    qhat: float,
    slicing: IntervalFFTSlicing,
) -> tuple[list, np.ndarray]:
    signal_padded = np.concatenate(([0.0], pred_signal, [0.0]))
    kernel_pad = np.concatenate((kernel, np.zeros(len(signal_padded) - len(kernel))))

    signal_fft = fft(signal_padded)
    kernel_fft = fft(kernel_pad)
    convolved = ifft(signal_fft * kernel_fft)

    center = convolved[slicing.center_start : slicing.center_end]
    right_edges = convolved[: slicing.n_right_edges]
    left_edge = convolved[-1]

    center_set = [interval([x.real - qhat, x.real + qhat]) for x in center]
    right_set = [interval([x.real - qhat, x.real + qhat]) for x in right_edges]
    left_set = [interval([left_edge.real - qhat, left_edge.real + qhat])]

    return right_set + center_set + left_set, kernel_fft


def intervalfft_inversion_bounds_1d(
    pred_signal: np.ndarray,
    kernel: np.ndarray,
    qhat: float,
    output_size: int,
    *,
    slicing: IntervalFFTSlicing = IntervalFFTSlicing(),
    eps: float = 1e-16,
) -> InversionBounds1D:
    convolved_set, kernel_fft = _prepare_interval_signal(pred_signal, kernel, qhat, slicing)
    inverse_kernel = 1.0 / (kernel_fft + eps)

    convolved_set_fft = intervalFFT(convolved_set)
    convolved_set_fft_kernel = [complex_prod(z, c) for z, c in zip(convolved_set_fft, inverse_kernel)]
    retrieved_signal = inverse_intervalFFT(convolved_set_fft_kernel)
    signal_bounds = [Real(z) for z in retrieved_signal]

    bounded = signal_bounds[slicing.output_offset : slicing.output_offset + output_size]
    lower = np.array([float(x[0].inf) for x in bounded], dtype=float)
    upper = np.array([float(x[0].sup) for x in bounded], dtype=float)
    width = upper - lower
    return InversionBounds1D(lower=lower, upper=upper, width=width)


def intervalfft_slice_inversion_bounds_1d(
    pred_signal: np.ndarray,
    kernel: np.ndarray,
    qhat: float,
    output_size: Optional[int] = None,
    *,
    prepad_repeat: int = 3,
    postpad_repeat: int = 1,
    output_offset: int = 3,
    eps: float = 1e-2,
) -> InversionBounds1D:
    if output_size is None:
        output_size = len(pred_signal)

    signal_padded = np.concatenate(([0.0], pred_signal, [0.0]))

    kernel_pad = np.zeros(len(signal_padded), dtype=float)
    kernel_pad[: len(kernel)] = kernel
    kernel_pad = np.roll(kernel_pad, -1)

    conv_signal = ifft(fft(signal_padded) * fft(kernel_pad))
    conv_set = [interval([x.real - qhat, x.real + qhat]) for x in conv_signal]

    padded_set = [conv_set[0]] * prepad_repeat + conv_set + [conv_set[-1]] * postpad_repeat
    inverse_kernel = 1.0 / (fft(np.concatenate((kernel, np.zeros(len(padded_set) - len(kernel))))) + eps)

    conv_set_fft = intervalFFT(padded_set)
    conv_set_fft_kernel = [complex_prod(z, c) for z, c in zip(conv_set_fft, inverse_kernel)]
    retrieved_signal = inverse_intervalFFT(conv_set_fft_kernel)
    bounds = [Real(z) for z in retrieved_signal][output_offset : output_offset + output_size]

    lower = np.array([float(x[0].inf) for x in bounds], dtype=float)
    upper = np.array([float(x[0].sup) for x in bounds], dtype=float)
    width = upper - lower
    return InversionBounds1D(lower=lower, upper=upper, width=width)


def invert_residual_bounds_1d(
    pred_signal: np.ndarray,
    kernel: np.ndarray,
    qhat: float,
    operator,
    *,
    interior_slice: slice = slice(1, -1),
    intervalfft_slicing: IntervalFFTSlicing = IntervalFFTSlicing(),
    integrate_slice_pad: bool = False,
) -> tuple[InversionBounds1D, InversionBounds1D]:
    point_width = pointwise_inverse_width(
        operator,
        n_points=len(pred_signal),
        qhat=qhat,
        dtype=torch.float32,
        device=None,
        slice_pad=integrate_slice_pad,
    )
    point_bounds = pointwise_inversion_bounds_1d(
        pred_signal=pred_signal,
        inv_width=point_width,
        interior_slice=interior_slice,
    )
    interval_bounds = intervalfft_inversion_bounds_1d(
        pred_signal=pred_signal,
        kernel=kernel,
        qhat=qhat,
        output_size=len(point_bounds.lower),
        slicing=intervalfft_slicing,
    )
    return point_bounds, interval_bounds


def _trajectory_coverage_1d(truth: np.ndarray, bounds: InversionBounds1D) -> float:
    in_lower = truth >= bounds.lower[None, :]
    in_upper = truth <= bounds.upper[None, :]
    contained = np.logical_and(in_lower, in_upper).all(axis=1)
    return float(contained.mean())


def perturbation_bounds_1d(
    pred_signal: np.ndarray,
    residual_operator,
    qhat: float,
    *,
    interior_slice: slice = slice(1, -1),
    config: PerturbationSamplingConfig = PerturbationSamplingConfig(),
    fallback_lower: Optional[np.ndarray] = None,
    fallback_upper: Optional[np.ndarray] = None,
) -> InversionBounds1D:
    pred_signal = np.asarray(pred_signal, dtype=np.float32)
    pred_tensor = torch.tensor(pred_signal, dtype=torch.float32).unsqueeze(0)
    interior_idx = np.arange(len(pred_signal))[interior_slice]
    n_interior = len(interior_idx)

    noise_gen = PDENoiseGenerator1D(device=pred_tensor.device, dtype=pred_tensor.dtype)

    lower = None
    upper = None
    counts = None
    last_missing = n_interior

    for retry_id, std_factor in enumerate(config.std_retry_factors):
        trial_std = float(config.noise_std) * float(std_factor)
        lower = torch.full((n_interior,), float("inf"), dtype=torch.float32, device=pred_tensor.device)
        upper = torch.full((n_interior,), float("-inf"), dtype=torch.float32, device=pred_tensor.device)
        counts = torch.zeros((n_interior,), dtype=torch.long, device=pred_tensor.device)

        remaining = int(config.n_samples)
        round_idx = 0
        while remaining > 0 and round_idx < config.max_rounds:
            draw = min(config.batch_size, remaining)
            seed = None if config.seed is None else int(config.seed + 100 * retry_id + round_idx)

            if config.noise_type == "spatial":
                noise = noise_gen.spatially_correlated_noise(
                    draw,
                    len(pred_signal),
                    correlation_length=config.correlation_length,
                    std=trial_std,
                    seed=seed,
                )
            elif config.noise_type == "white":
                noise = noise_gen.white_noise(draw, len(pred_signal), std=trial_std, seed=seed)
            elif config.noise_type == "gp":
                noise = noise_gen.gp_noise(
                    draw,
                    len(pred_signal),
                    correlation_length=config.correlation_length,
                    std=trial_std,
                    kernel_type=config.gp_kernel,
                    nu=config.gp_nu,
                    seed=seed,
                )
            else:
                raise ValueError(f"Unknown noise_type: {config.noise_type}")

            perturbed = pred_tensor + noise
            residuals = residual_operator(perturbed)
            within = torch.abs(residuals) <= abs(float(qhat))

            pred_slice = perturbed[:, interior_slice]
            mask_slice = within[:, interior_slice]

            inf_tensor = torch.full_like(pred_slice, float("inf"))
            ninf_tensor = torch.full_like(pred_slice, float("-inf"))
            batch_min = torch.where(mask_slice, pred_slice, inf_tensor).amin(dim=0)
            batch_max = torch.where(mask_slice, pred_slice, ninf_tensor).amax(dim=0)

            lower = torch.minimum(lower, batch_min)
            upper = torch.maximum(upper, batch_max)
            counts = counts + mask_slice.sum(dim=0)

            remaining -= draw
            round_idx += 1

        missing_mask = counts == 0
        last_missing = int(missing_mask.sum().item())
        if last_missing == 0:
            break

    if last_missing > 0:
        missing_mask = (counts == 0)
        if fallback_lower is None or fallback_upper is None:
            raise RuntimeError(
                f"Perturbation sampling produced no accepted samples at {last_missing} interior points "
                f"after std retries {config.std_retry_factors}. Increase n_samples/max_rounds."
            )
        fallback_lower_t = torch.tensor(fallback_lower, dtype=torch.float32, device=pred_tensor.device)
        fallback_upper_t = torch.tensor(fallback_upper, dtype=torch.float32, device=pred_tensor.device)
        lower = torch.where(missing_mask, fallback_lower_t, lower)
        upper = torch.where(missing_mask, fallback_upper_t, upper)

    lower_np = lower.detach().cpu().numpy().astype(float)
    upper_np = upper.detach().cpu().numpy().astype(float)
    return InversionBounds1D(lower=lower_np, upper=upper_np, width=upper_np - lower_np)


def empirical_coverage_curve_1d(
    preds: np.ndarray,
    truths: np.ndarray,
    residual_cal: torch.Tensor,
    kernel: np.ndarray,
    operator,
    *,
    alphas: Sequence[float],
    interior_slice: slice = slice(1, -1),
    intervalfft_slicing: IntervalFFTSlicing = IntervalFFTSlicing(),
    integrate_slice_pad: bool = False,
    perturbation_config: Optional[PerturbationSamplingConfig] = None,
) -> CoverageResult:
    nominal = []
    cov_point = []
    cov_interval = []
    cov_perturb = []

    preds = np.asarray(preds, dtype=float)
    truths = np.asarray(truths, dtype=float)

    for alpha in tqdm(alphas, desc="Coverage alphas", unit="α"):
        qhat = calibrate_qhat_from_residual(residual_cal, alpha=float(alpha))
        point_cover_flags = []
        interval_cover_flags = []

        for i in tqdm(range(preds.shape[0]), desc=f"  α={float(alpha):.2f} trajectories", leave=False, unit="traj"):
            point_bounds, interval_bounds = invert_residual_bounds_1d(
                pred_signal=preds[i],
                kernel=kernel,
                qhat=qhat,
                operator=operator,
                interior_slice=interior_slice,
                intervalfft_slicing=intervalfft_slicing,
                integrate_slice_pad=integrate_slice_pad,
            )
            truth_i = truths[i][interior_slice]
            point_cover_flags.append(_trajectory_coverage_1d(truth_i[None, :], point_bounds))
            interval_cover_flags.append(_trajectory_coverage_1d(truth_i[None, :], interval_bounds))
            if perturbation_config is not None:
                perturb_bounds = perturbation_bounds_1d(
                    pred_signal=preds[i],
                    residual_operator=operator,
                    qhat=qhat,
                    interior_slice=interior_slice,
                    config=perturbation_config,
                )
                cov_perturb.append(_trajectory_coverage_1d(truth_i[None, :], perturb_bounds))

        nominal.append(1.0 - float(alpha))
        cov_point.append(float(np.mean(point_cover_flags)))
        cov_interval.append(float(np.mean(interval_cover_flags)))

    if perturbation_config is not None:
        n_traj = preds.shape[0]
        cov_perturb = np.asarray(cov_perturb, dtype=float).reshape(len(alphas), n_traj).mean(axis=1)
        perturb_cov = cov_perturb
    else:
        perturb_cov = None

    return CoverageResult(
        nominal_coverage=np.asarray(nominal, dtype=float),
        empirical_coverage_pointwise=np.asarray(cov_point, dtype=float),
        empirical_coverage_intervalfft=np.asarray(cov_interval, dtype=float),
        empirical_coverage_perturbation=perturb_cov,
    )
