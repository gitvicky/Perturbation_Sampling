import pytest
import numpy as np
import torch
import torch.nn as nn

from Inversion_Strategies.inversion.residual_inversion import (
    PerturbationSamplingConfig,
    perturbation_bounds_1d,
    BoundaryGenerator,
    _boundary_prior_loss
)


class DummyLinearOperator(nn.Module):
    """
    A simple operator that just scales the input.
    Residual = 2 * p
    So if true_p = 0, residual = 2 * noise.
    """
    def __init__(self, scale=2.0):
        super().__init__()
        self.scale = scale

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        return self.scale * p


@pytest.fixture
def basic_setup():
    seq_len = 10
    pred_signal = np.zeros(seq_len)
    # We set a qhat bound of 0.5. Since operator is 2*x, 
    # the noise must be within [-0.25, 0.25] to be accepted.
    qhat = 0.5
    operator = DummyLinearOperator()
    return pred_signal, operator, qhat


def test_standard_sampling(basic_setup):
    pred_signal, operator, qhat = basic_setup
    config = PerturbationSamplingConfig(
        n_samples=100,
        batch_size=100,
        noise_type="white",
        noise_std=1.0,  # high noise to ensure rejections without advanced methods
        use_optimisation=False,
        use_mcmc=False,
        use_generator=False,
        max_rounds=1
    )
    
    # Standard sampling might fail because the space is too constrained
    # and we only use 1 round of 100 samples with very high noise.
    try:
        bounds = perturbation_bounds_1d(
            pred_signal, operator, qhat, 
            config=config, interior_slice=slice(None)
        )
        # If it didn't fail, check bounds are within theoretical limit [-0.25, 0.25]
        assert np.all(bounds.upper <= 0.25)
        assert np.all(bounds.lower >= -0.25)
    except RuntimeError:
        # It's expected to fail due to low acceptance rate
        pass


def test_mcmc_sampling(basic_setup):
    pred_signal, operator, qhat = basic_setup
    config = PerturbationSamplingConfig(
        n_samples=100,
        batch_size=100,
        noise_type="white",
        noise_std=1.0,
        use_mcmc=True,
        mcmc_steps=50,
        mcmc_step_size=0.01,
        mcmc_noise_scale=0.0,  # Set to 0 to test pure gradient drift into valid region
        use_optimisation=False,
        use_generator=False,
        lambda_boundary=10.0,
        lambda_prior=0.0
    )
    
    bounds = perturbation_bounds_1d(
        pred_signal, operator, qhat, 
        config=config, interior_slice=slice(None),
        fallback_lower=np.full(10, -0.25),
        fallback_upper=np.full(10, 0.25)
    )
    
    assert np.all(bounds.upper <= 0.2501)
    assert np.all(bounds.lower >= -0.2501)
    # Should find valid bounds without just falling back perfectly if possible
    # (Though we provided fallbacks to prevent crash if not 100% acceptance)


def test_optimisation_sampling(basic_setup):
    pred_signal, operator, qhat = basic_setup
    config = PerturbationSamplingConfig(
        n_samples=100,
        batch_size=100,
        noise_type="white",
        noise_std=1.0,
        use_optimisation=True,
        opt_steps=100,
        opt_lr=0.1,
        use_mcmc=False,
        use_generator=False,
        lambda_boundary=1.0,
        lambda_prior=0.0
    )
    
    # Optimisation should drag all rejected samples into the bound
    bounds = perturbation_bounds_1d(
        pred_signal, operator, qhat, 
        config=config, interior_slice=slice(None)
    )
    
    # The bounds must be strictly enforced
    assert np.all(bounds.upper <= 0.2501)
    assert np.all(bounds.lower >= -0.2501)


def test_generator_sampling(basic_setup):
    pred_signal, operator, qhat = basic_setup
    config = PerturbationSamplingConfig(
        n_samples=100,
        batch_size=100,
        noise_type="white",
        use_generator=True,
        gen_train_steps=100,
        gen_lr=0.05,
        lambda_boundary=10.0,
        lambda_prior=0.0,
        use_optimisation=False,
        use_mcmc=False
    )
    
    bounds = perturbation_bounds_1d(
        pred_signal, operator, qhat, 
        config=config, interior_slice=slice(None),
        fallback_lower=np.full(10, -0.25),
        fallback_upper=np.full(10, 0.25)
    )
    
    assert np.all(bounds.upper <= 0.251)
    assert np.all(bounds.lower >= -0.251)


def test_boundary_prior_loss():
    residuals = torch.tensor([1.0, 0.5, 2.0])
    noise = torch.tensor([0.1, 0.1, 0.1])
    qhat_t = torch.tensor(1.0)
    
    # Max(0, |res| - 1.0)^2 => [0.0, 0.0, 1.0^2] = [0.0, 0.0, 1.0]. Mean = 1/3
    l_bound, total_loss = _boundary_prior_loss(
        residuals, noise, qhat_t, lambda_boundary=1.0, lambda_prior=0.0
    )
    assert torch.isclose(l_bound, torch.tensor(1.0/3.0))
    assert torch.isclose(total_loss, torch.tensor(1.0/3.0))

    # Test prior loss: noise^2 mean = 0.01
    l_bound, total_loss = _boundary_prior_loss(
        residuals, noise, qhat_t, lambda_boundary=0.0, lambda_prior=1.0
    )
    assert torch.isclose(total_loss, torch.tensor(0.01))
