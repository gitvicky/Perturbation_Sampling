"""
Autoregressive 1D neural surrogates and data utilities for Burgers' equation.

The model interfaces mirror `Expts/Advection/Advection_Model.py`:
`build_model`, `train_model`, and `rollout` are reused directly, while this
module provides Burgers-specific data generation and cached dataset loading.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import qmc
from tqdm import tqdm

from Neural_PDE.Numerical_Solvers.Burgers.Burgers_1D import Burgers_1D
from Expts.Advection.Advection_Model import build_model, train_model, rollout


@dataclass(frozen=True)
class BurgersGrid:
    x: np.ndarray
    dt: float
    dx: float


def _sample_ic_params_lhs(n_sims: int, seed: int) -> np.ndarray:
    """LHS over [-3, 3]^3 for (alpha, beta, gamma), matching Data_Gen.py."""
    sampler = qmc.LatinHypercube(d=3, seed=seed)
    unit = sampler.random(n=n_sims)
    lb = np.array([-3.0, -3.0, -3.0], dtype=np.float64)
    ub = np.array([3.0, 3.0, 3.0], dtype=np.float64)
    return qmc.scale(unit, lb, ub)


def _generate_burgers_dataset(
    n_sims: int,
    seed: int,
    *,
    nx: int = 1000,
    nt: int = 500,
    x_min: float = 0.0,
    x_max: float = 2.0,
    t_end: float = 1.25,
    nu: float = 0.002,
    x_slice: int = 5,
    t_slice: int = 10,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Generate Burgers trajectories using solver settings from Data_Gen.py."""
    params = _sample_ic_params_lhs(n_sims, seed=seed)
    sim = Burgers_1D(nx, nt, x_min, x_max, t_end, nu)

    trajectories = []
    for idx in tqdm(range(n_sims), desc="Generating Burgers data", unit="traj"):
        alpha, beta, gamma = params[idx]
        sim.InitializeU(float(alpha), float(beta), float(gamma))
        u_sol, x, dt = sim.solve()
        trajectories.append(u_sol.astype(np.float32))

    u = np.asarray(trajectories, dtype=np.float32)[:, ::t_slice, ::x_slice]
    x = np.asarray(x, dtype=np.float32)[::x_slice]
    dt_eff = float(dt * t_slice)
    return u, x, dt_eff, params.astype(np.float32)


def load_or_generate_burgers_dataset(
    cache_path: str,
    n_sims: int,
    seed: int = 0,
    *,
    force_regen: bool = False,
) -> tuple[np.ndarray, BurgersGrid]:
    """Load cached Burgers trajectories or generate and save them.

    The cache stores arrays `{u, x, dt, params}`. If the file exists with at
    least `n_sims` trajectories, the first `n_sims` are reused.
    """
    def _load_if_compatible(path: str):
        if not os.path.exists(path):
            return None
        data = np.load(path)
        u = np.asarray(data["u"], dtype=np.float32)
        if u.shape[0] < n_sims:
            return None
        x = np.asarray(data["x"], dtype=np.float32)
        dt = float(data["dt"])
        dx = float(x[1] - x[0])
        return u[:n_sims], BurgersGrid(x=x, dt=dt, dx=dx)

    if not force_regen:
        loaded = _load_if_compatible(cache_path)
        if loaded is not None:
            return loaded

        # Reuse pre-generated Burgers datasets when available.
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        candidates = (
            os.path.join(repo_root, "Neural_PDE", "Data", "Burgers_1d.npz"),
            os.path.join(repo_root, "Neural_PDE", "Numerical_Solvers", "Burgers", "Burgers_1d.npz"),
            os.path.join(repo_root, "Burgers_1d.npz"),
        )
        for candidate in candidates:
            loaded = _load_if_compatible(candidate)
            if loaded is not None:
                return loaded

    u, x, dt, params = _generate_burgers_dataset(n_sims=n_sims, seed=seed)
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    np.savez(cache_path, u=u, x=x, dt=dt, params=params)
    dx = float(x[1] - x[0])
    return u, BurgersGrid(x=x, dt=dt, dx=dx)


def make_train_pairs(trajectories: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert trajectories [N, Nt, Nx] into autoregressive (u_t, u_{t+1}) pairs."""
    traj = np.asarray(trajectories, dtype=np.float32)
    inputs = traj[:, :-1, :]
    targets = traj[:, 1:, :]
    x = torch.from_numpy(inputs).reshape(-1, 1, inputs.shape[-1])
    y = torch.from_numpy(targets).reshape(-1, 1, targets.shape[-1])
    return x, y


@torch.no_grad()
def evaluate_rollouts(model, trajectories: np.ndarray, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Roll out the surrogate from each trajectory initial condition."""
    truth = torch.tensor(np.asarray(trajectories, dtype=np.float32), dtype=torch.float32)
    u0 = truth[:, 0, :]
    pred = rollout(model, u0, n_steps=truth.shape[1] - 1, device=device).cpu()
    return truth, pred
