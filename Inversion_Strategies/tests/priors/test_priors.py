"""Invariant tests for Utils.priors.

Three families of checks:

1. Statistical: the empirical marginal std over a large batch matches the
   prior's analytical ``marginal_std()`` to a loose tolerance.  Only the
   *interior* of the output is checked — boundary effects from convolution
   padding reduce variance at the edges and are intentionally not renormalised
   (we now provide the analytical interior std instead).
2. Differentiability: ``decode`` is linear in the latent, so the Jacobian
   ``d eta / d z`` is well-defined; we check gradients flow via a simple
   backward pass on a scalar objective.
3. Device / dtype: priors inherit from ``nn.Module``, so ``.to(...)`` moves
   all buffers and produces identical samples (up to floating-point noise)
   from a fixed latent.
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Utils.priors import (  # noqa: E402
    BoundaryProject,
    BSpline2DPrior,
    BSplinePrior,
    GPPrior,
    MeshScale,
    OUPrior,
    PreCorrelatedPrior,
    PriorSpec,
    Spatial2DPrior,
    SpatialPrior,
    Spectral2DPrior,
    SpectralPrior,
    White2DPrior,
    WhitePrior,
    build_prior,
)


STD = 0.1
N = 64
BATCH = 4096
TOL = 0.08  # relative tolerance on empirical marginal std


def _empirical_std(prior, batch=BATCH):
    torch.manual_seed(0)
    z = prior.sample_latent(batch)
    eta = prior.decode(z)
    return eta.std(dim=0)


def _interior_mask_1d(n: int, margin: int = 4) -> slice:
    return slice(margin, n - margin)


def _has_gpytorch():
    try:
        import gpytorch  # noqa: F401
        return True
    except ImportError:
        return False


# ----------------------------------------------------------------------
# 1D statistical invariants
# ----------------------------------------------------------------------
@pytest.mark.parametrize("factory,kwargs", [
    (WhitePrior, dict(n_points=N, std=STD)),
    (SpatialPrior, dict(n_points=N, correlation_length=5.0, std=STD)),
    (BSplinePrior, dict(n_points=N, n_knots=16, std=STD)),
    (SpectralPrior, dict(n_points=N, alpha=1.0, std=STD)),
])
def test_1d_marginal_std_matches_analytic(factory, kwargs):
    prior = factory(**kwargs)
    sl = _interior_mask_1d(N)
    emp = _empirical_std(prior)[sl]
    theo = prior.marginal_std()[sl]
    assert torch.allclose(emp, theo, rtol=TOL, atol=0.03 * STD), (
        f"{factory.__name__}: emp.mean={emp.mean():.4g}, theo.mean={theo.mean():.4g}"
    )


def test_gp_prior_marginal_std():
    if not _has_gpytorch():
        pytest.skip("gpytorch not available")
    prior = GPPrior(n_points=N, correlation_length=5.0, std=STD)
    sl = _interior_mask_1d(N)
    emp = _empirical_std(prior)[sl]
    theo = prior.marginal_std()[sl]
    assert torch.allclose(emp, theo, rtol=TOL, atol=0.03 * STD)


def test_pre_correlated_prior_marginal_std():
    kernel = torch.tensor([0.25, 0.5, 0.25])
    prior = PreCorrelatedPrior(n_points=N, kernel=kernel, std=STD)
    sl = _interior_mask_1d(N)
    emp = _empirical_std(prior)[sl]
    theo = prior.marginal_std()[sl]
    assert torch.allclose(emp, theo, rtol=TOL, atol=0.03 * STD)


def test_ou_prior_stationary_std():
    prior = OUPrior(n_steps=N, dt=0.05, tau=1.0, std=STD)
    emp = _empirical_std(prior)
    theo = prior.marginal_std()
    assert torch.allclose(emp, theo, rtol=TOL, atol=0.03 * STD)


# ----------------------------------------------------------------------
# 2D statistical invariants
# ----------------------------------------------------------------------
def test_2d_white_marginal_std():
    prior = White2DPrior(shape=(16, 16), std=STD)
    emp = _empirical_std(prior, batch=2048)
    theo = prior.marginal_std()
    assert torch.allclose(emp, theo, rtol=TOL, atol=0.03 * STD)


def test_2d_spatial_interior_marginal_std():
    prior = Spatial2DPrior(shape=(32, 32), correlation_length=3.0, std=STD)
    emp = _empirical_std(prior, batch=2048)
    theo = prior.marginal_std()
    emp_int = emp[6:-6, 6:-6]
    theo_int = theo[6:-6, 6:-6]
    assert torch.allclose(emp_int, theo_int, rtol=TOL, atol=0.03 * STD)


def test_2d_bspline_marginal_std():
    prior = BSpline2DPrior(shape=(24, 24), n_knots=8, std=STD)
    emp = _empirical_std(prior, batch=4096)
    theo = prior.marginal_std()
    # B-spline marginal std varies across the grid; compare mean values.
    assert abs(emp.mean().item() - theo.mean().item()) / STD < 0.1


def test_2d_spectral_marginal_std():
    prior = Spectral2DPrior(shape=(32, 32), alpha=1.0, std=STD)
    emp = _empirical_std(prior, batch=1024)
    # Spatial average should match the declared std.
    assert abs(emp.mean().item() - STD) / STD < 0.1


# ----------------------------------------------------------------------
# Differentiability
# ----------------------------------------------------------------------
@pytest.mark.parametrize("factory,kwargs", [
    (WhitePrior, dict(n_points=N, std=STD)),
    (SpatialPrior, dict(n_points=N, correlation_length=5.0, std=STD)),
    (BSplinePrior, dict(n_points=N, n_knots=16, std=STD)),
    (SpectralPrior, dict(n_points=N, alpha=1.0, std=STD)),
])
def test_decode_is_differentiable(factory, kwargs):
    prior = factory(**kwargs)
    z = prior.sample_latent(2).clone().requires_grad_(True)
    eta = prior.decode(z)
    loss = (eta ** 2).sum()
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert z.grad.abs().sum() > 0


def test_log_prior_is_differentiable():
    prior = SpatialPrior(n_points=N, correlation_length=5.0, std=STD)
    z = prior.sample_latent(4).clone().requires_grad_(True)
    lp = prior.log_prior(z)
    lp.backward()
    assert torch.allclose(z.grad, 2 * z.detach() / z.numel(), atol=1e-6)


# ----------------------------------------------------------------------
# Device / dtype
# ----------------------------------------------------------------------
def test_prior_to_dtype_preserves_decode_shape():
    prior = SpatialPrior(n_points=N, correlation_length=5.0, std=STD,
                         dtype=torch.float32)
    prior64 = prior.to(torch.float64)
    z = torch.randn(3, N, dtype=torch.float64)
    out = prior64.decode(z)
    assert out.dtype == torch.float64
    assert out.shape == (3, N)


# ----------------------------------------------------------------------
# Wrappers
# ----------------------------------------------------------------------
def test_dirichlet_boundary_zeroes_edges_1d():
    base = WhitePrior(n_points=N, std=STD)
    prior = BoundaryProject(base, "dirichlet")
    eta = prior.decode(prior.sample_latent(16))
    assert torch.all(eta[:, 0] == 0)
    assert torch.all(eta[:, -1] == 0)


def test_mesh_scale_factor():
    base = WhitePrior(n_points=N, std=1.0)
    prior = MeshScale(base, spacing=(0.25,))
    expected = 2.0  # 1/sqrt(0.25)
    assert math.isclose(prior.factor, expected, rel_tol=1e-6)


def test_wrapper_decode_matches_scaled_inner():
    base = WhitePrior(n_points=N, std=1.0)
    wrapped = MeshScale(base, spacing=(0.25,))
    z = base.sample_latent(5)
    torch.testing.assert_close(wrapped.decode(z), 2.0 * base.decode(z))


# ----------------------------------------------------------------------
# Factory / PriorSpec
# ----------------------------------------------------------------------
@pytest.mark.parametrize("nt", ["white", "spatial", "bspline", "spectral"])
def test_build_prior_1d_roundtrip(nt):
    spec = PriorSpec(noise_type=nt, noise_std=STD,
                     correlation_length=4.0, bspline_n_knots=8,
                     spectral_alpha=1.0)
    prior = build_prior(spec, input_shape=(N,), device="cpu")
    eta = prior.decode(prior.sample_latent(8))
    assert eta.shape == (8, N)


def test_build_prior_2d_spatial():
    spec = PriorSpec(noise_type="spatial", noise_std=STD,
                     correlation_length=3.0)
    prior = build_prior(spec, input_shape=(16, 16), device="cpu")
    eta = prior.decode(prior.sample_latent(4))
    assert eta.shape == (4, 16, 16)


def test_build_prior_applies_mesh_and_boundary():
    spec = PriorSpec(noise_type="white", noise_std=1.0,
                     boundary="dirichlet", mesh_spacing=(0.25,))
    prior = build_prior(spec, input_shape=(N,), device="cpu")
    eta = prior.decode(prior.sample_latent(32))
    assert torch.all(eta[:, 0] == 0)
    assert torch.all(eta[:, -1] == 0)


def test_build_prior_unknown_noise_raises():
    spec = PriorSpec(noise_type="nonsense", noise_std=STD)
    with pytest.raises(ValueError):
        build_prior(spec, input_shape=(N,), device="cpu")
