"""Task adapters for config-driven train/evaluate workflows."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from Neural_PDE.Utils.processing_utils import Normalisation
from Neural_PDE.Numerical_Solvers.Advection.Advection_1D import Advection_1d

from Expts.Advection.Advection_Model import (
    build_model as build_advection_model,
    generate_training_data as generate_advection_training_data,
    train_model as train_advection_model,
    sample_ic_params as sample_advection_ic_params,
    solve_parameterised as solve_advection_parameterised,
)
from Expts.Burgers.Burgers_Model import (
    build_model as build_burgers_model,
    make_train_pairs as make_burgers_pairs,
    train_model as train_burgers_model,
    load_or_generate_burgers_dataset,
)


@dataclass
class DatasetBundle:
    train_inputs: torch.Tensor | None
    train_targets: torch.Tensor | None
    train_reference: torch.Tensor
    eval_truth: torch.Tensor
    eval_init: torch.Tensor
    aux: dict[str, Any]


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _rollout_autoregressive(model: torch.nn.Module, u0: torch.Tensor, n_steps: int) -> torch.Tensor:
    model.eval()
    u = u0.unsqueeze(1)  # [B, 1, Nx]
    frames = [u.squeeze(1)]
    with torch.no_grad():
        for _ in range(n_steps):
            u = model(u)
            frames.append(u.squeeze(1))
    return torch.stack(frames, dim=1)


class BaseAdapter(ABC):
    @abstractmethod
    def train(self, config: dict[str, Any], paths: Any) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, config: dict[str, Any], paths: Any) -> dict[str, Any]:
        raise NotImplementedError


class ODEAdapter(BaseAdapter):
    def __init__(self, task: str):
        self.task = task
        self.module = self._load_module(task)

    @staticmethod
    def _load_module(task: str):
        if task == "ode_sho":
            from Expts.SHO.SHO_NODE import (
                HarmonicOscillator as Oscillator,
                ODEFunc,
                generate_training_data,
                train_neural_ode,
                evaluate,
            )
            return dict(
                Oscillator=Oscillator,
                ODEFunc=ODEFunc,
                generate_training_data=generate_training_data,
                train_neural_ode=train_neural_ode,
                evaluate=evaluate,
            )
        if task == "ode_dho":
            from Expts.DHO.DHO_NODE import (
                DampedHarmonicOscillator as Oscillator,
                ODEFunc,
                generate_training_data,
                train_neural_ode,
                evaluate,
            )
            return dict(
                Oscillator=Oscillator,
                ODEFunc=ODEFunc,
                generate_training_data=generate_training_data,
                train_neural_ode=train_neural_ode,
                evaluate=evaluate,
            )
        if task == "ode_duffing":
            from Expts.Duffing.Duffing_NODE import (
                DuffingOscillator as Oscillator,
                ODEFunc,
                generate_training_data,
                train_neural_ode,
                evaluate,
            )
            return dict(
                Oscillator=Oscillator,
                ODEFunc=ODEFunc,
                generate_training_data=generate_training_data,
                train_neural_ode=train_neural_ode,
                evaluate=evaluate,
            )
        raise ValueError(f"Unsupported ODE task: {task}")

    def _build_oscillator(self, cfg: dict[str, Any]):
        p = cfg["Physics"]
        Oscillator = self.module["Oscillator"]
        if self.task == "ode_sho":
            return Oscillator(p["k"], p["m"])
        if self.task == "ode_dho":
            return Oscillator(p["k"], p["m"], p["c"])
        return Oscillator(p["alpha"], p["beta"], p["delta"])

    def train(self, config: dict[str, Any], paths: Any) -> dict[str, Any]:
        physics = config["Physics"]
        data = config["Data"]
        model_cfg = config["Model"]
        opt = config["Opt"]

        osc = self._build_oscillator(config)
        t_train, states, derivs = self.module["generate_training_data"](
            osc,
            tuple(physics["t_span"]),
            int(data["n_points"]),
            int(data["n_trajectories"]),
        )

        model = self.module["ODEFunc"](hidden_dim=int(model_cfg["hidden_dim"]))
        self.module["train_neural_ode"](
            model,
            t_train,
            states,
            derivs,
            n_epochs=int(opt["epochs"]),
            batch_size=int(opt["batch_size"]),
        )
        paths.run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), paths.model_path)

        normalizer_ctor = Normalisation(data.get("normalisation", "Identity"))
        train_positions = torch.tensor(states[..., 0], dtype=torch.float32)
        normalizer = normalizer_ctor(train_positions.clone())
        from Expts.pipeline.io_utils import save_norms

        save_norms(normalizer, paths.norms_path)
        return self.evaluate(config, paths)

    def evaluate(self, config: dict[str, Any], paths: Any) -> dict[str, Any]:
        physics = config["Physics"]
        data = config["Data"]
        model_cfg = config["Model"]
        eval_cfg = config.get("Evaluate", {})

        osc = self._build_oscillator(config)
        model = self.module["ODEFunc"](hidden_dim=int(model_cfg["hidden_dim"]))
        model.load_state_dict(torch.load(paths.model_path, map_location="cpu", weights_only=False))

        n_solves = int(eval_cfg.get("n_solves", data.get("n_eval_trajectories", 100)))
        t, numerical_sol, neural_sol = self.module["evaluate"](
            osc,
            model,
            tuple(physics["t_span"]),
            int(data["n_points"]),
            x_range=tuple(data.get("x_range", [-2.0, 2.0])),
            v_range=tuple(data.get("v_range", [-2.0, 2.0])),
            n_solves=n_solves,
        )
        truth = torch.tensor(numerical_sol[..., 0], dtype=torch.float32)
        pred = torch.tensor(neural_sol[..., 0], dtype=torch.float32)
        mse = float(torch.mean((pred - truth) ** 2))
        mae = float(torch.mean(torch.abs(pred - truth)))

        return {
            "task": self.task,
            "n_solves": n_solves,
            "n_points": int(data["n_points"]),
            "mse": mse,
            "mae": mae,
            "time_start": float(t[0]),
            "time_end": float(t[-1]),
        }


class Advection1DAdapter(BaseAdapter):
    def _build_dataset(self, config: dict[str, Any]) -> DatasetBundle:
        physics = config["Physics"]
        data = config["Data"]
        sim = Advection_1d(
            int(physics["Nx"]),
            int(physics["Nt"]),
            float(physics["x_min"]),
            float(physics["x_max"]),
            float(physics["t_end"]),
        )
        v = float(physics.get("v", 1.0))
        seed = int(config["Run"].get("seed", 0))

        train_x, train_y, train_traj = generate_advection_training_data(
            sim, int(data["n_train"]), v=v, seed=seed
        )

        n_eval = int(data["n_eval"])
        rng = np.random.default_rng(seed + 1)
        eval_traj = []
        eval_u0 = []
        for _ in range(n_eval):
            xc, width, height = sample_advection_ic_params(rng)
            traj = solve_advection_parameterised(sim, xc, width, height, v)
            eval_traj.append(traj)
            eval_u0.append(traj[0])
        eval_truth = torch.tensor(np.asarray(eval_traj, dtype=np.float32), dtype=torch.float32)
        eval_init = torch.tensor(np.asarray(eval_u0, dtype=np.float32), dtype=torch.float32)

        return DatasetBundle(
            train_inputs=train_x,
            train_targets=train_y,
            train_reference=train_traj,
            eval_truth=eval_truth,
            eval_init=eval_init,
            aux={"dt": sim.dt, "dx": sim.dx},
        )

    def _build_model(self, config: dict[str, Any]) -> torch.nn.Module:
        m = config["Model"]
        return build_advection_model(
            kind=m.get("kind", "unet"),
            features=int(m.get("features", 32)),
            modes=int(m.get("fno_modes", 16)),
            n_layers=int(m.get("fno_layers", 4)),
        )

    def train(self, config: dict[str, Any], paths: Any) -> dict[str, Any]:
        data = config["Data"]
        opt = config["Opt"]
        run = config["Run"]
        device = run.get("device", _default_device())
        ds = self._build_dataset(config)

        normalizer_ctor = Normalisation(data.get("normalisation", "Min-Max"))
        normalizer = normalizer_ctor(ds.train_reference.clone())
        train_x = normalizer.encode(ds.train_inputs.clone())
        train_y = normalizer.encode(ds.train_targets.clone())

        model = self._build_model(config)
        train_advection_model(
            model,
            train_x,
            train_y,
            n_epochs=int(opt["epochs"]),
            batch_size=int(opt["batch_size"]),
            lr=float(opt["learning_rate"]),
            device=device,
        )
        paths.run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), paths.model_path)

        from Expts.pipeline.io_utils import save_norms

        save_norms(normalizer, paths.norms_path)
        return self.evaluate(config, paths)

    def evaluate(self, config: dict[str, Any], paths: Any) -> dict[str, Any]:
        data = config["Data"]
        run = config["Run"]
        device = run.get("device", _default_device())
        ds = self._build_dataset(config)

        model = self._build_model(config)
        model.load_state_dict(torch.load(paths.model_path, map_location=device, weights_only=False))
        model = model.to(device)

        normalizer_ctor = Normalisation(data.get("normalisation", "Min-Max"))
        normalizer = normalizer_ctor(torch.zeros_like(ds.train_reference))
        from Expts.pipeline.io_utils import load_norms_into

        normalizer = load_norms_into(normalizer, paths.norms_path)

        u0_norm = normalizer.encode(ds.eval_init.clone())
        pred_norm = _rollout_autoregressive(model, u0_norm.to(device), n_steps=ds.eval_truth.shape[1] - 1).cpu()
        pred = normalizer.decode(pred_norm.clone())

        mse = float(torch.mean((pred - ds.eval_truth) ** 2))
        mae = float(torch.mean(torch.abs(pred - ds.eval_truth)))
        return {
            "task": "pde_advection_1d",
            "n_eval": int(ds.eval_truth.shape[0]),
            "nt": int(ds.eval_truth.shape[1]),
            "nx": int(ds.eval_truth.shape[2]),
            "mse": mse,
            "mae": mae,
            "dt": float(ds.aux["dt"]),
            "dx": float(ds.aux["dx"]),
        }


class Burgers1DAdapter(BaseAdapter):
    def _build_model(self, config: dict[str, Any]) -> torch.nn.Module:
        m = config["Model"]
        return build_burgers_model(
            kind=m.get("kind", "unet"),
            features=int(m.get("features", 32)),
            modes=int(m.get("fno_modes", 16)),
            n_layers=int(m.get("fno_layers", 4)),
        )

    def _build_dataset(self, config: dict[str, Any]) -> DatasetBundle:
        data = config["Data"]
        run = config["Run"]
        cache_path = data.get("cache_path", "Expts/Burgers/data/Burgers_1d_cached.npz")
        all_traj, grid = load_or_generate_burgers_dataset(
            cache_path=cache_path,
            n_sims=int(data["n_train"]) + int(data["n_eval"]),
            seed=int(run.get("seed", 0)),
            force_regen=bool(data.get("force_regen_data", False)),
        )
        n_train = int(data["n_train"])
        train = all_traj[:n_train]
        eval_traj = all_traj[n_train : n_train + int(data["n_eval"])]
        train_x, train_y = make_burgers_pairs(train)
        eval_truth = torch.tensor(eval_traj, dtype=torch.float32)
        eval_init = eval_truth[:, 0, :]
        return DatasetBundle(
            train_inputs=train_x,
            train_targets=train_y,
            train_reference=torch.tensor(train, dtype=torch.float32),
            eval_truth=eval_truth,
            eval_init=eval_init,
            aux={"dt": grid.dt, "dx": grid.dx},
        )

    def train(self, config: dict[str, Any], paths: Any) -> dict[str, Any]:
        data = config["Data"]
        opt = config["Opt"]
        run = config["Run"]
        device = run.get("device", _default_device())
        ds = self._build_dataset(config)

        normalizer_ctor = Normalisation(data.get("normalisation", "Min-Max"))
        normalizer = normalizer_ctor(ds.train_reference.clone())
        train_x = normalizer.encode(ds.train_inputs.clone())
        train_y = normalizer.encode(ds.train_targets.clone())

        model = self._build_model(config)
        train_burgers_model(
            model,
            train_x,
            train_y,
            n_epochs=int(opt["epochs"]),
            batch_size=int(opt["batch_size"]),
            lr=float(opt["learning_rate"]),
            device=device,
        )
        paths.run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), paths.model_path)

        from Expts.pipeline.io_utils import save_norms

        save_norms(normalizer, paths.norms_path)
        return self.evaluate(config, paths)

    def evaluate(self, config: dict[str, Any], paths: Any) -> dict[str, Any]:
        data = config["Data"]
        run = config["Run"]
        device = run.get("device", _default_device())
        ds = self._build_dataset(config)

        model = self._build_model(config)
        model.load_state_dict(torch.load(paths.model_path, map_location=device, weights_only=False))
        model = model.to(device)

        normalizer_ctor = Normalisation(data.get("normalisation", "Min-Max"))
        normalizer = normalizer_ctor(ds.train_reference.clone())
        from Expts.pipeline.io_utils import load_norms_into

        normalizer = load_norms_into(normalizer, paths.norms_path)

        u0_norm = normalizer.encode(ds.eval_init.clone())
        pred_norm = _rollout_autoregressive(model, u0_norm.to(device), n_steps=ds.eval_truth.shape[1] - 1).cpu()
        pred = normalizer.decode(pred_norm.clone())

        mse = float(torch.mean((pred - ds.eval_truth) ** 2))
        mae = float(torch.mean(torch.abs(pred - ds.eval_truth)))
        return {
            "task": "pde_burgers_1d",
            "n_eval": int(ds.eval_truth.shape[0]),
            "nt": int(ds.eval_truth.shape[1]),
            "nx": int(ds.eval_truth.shape[2]),
            "mse": mse,
            "mae": mae,
            "dt": float(ds.aux["dt"]),
            "dx": float(ds.aux["dx"]),
        }


def get_adapter(task: str) -> BaseAdapter:
    if task in {"ode_sho", "ode_dho", "ode_duffing"}:
        return ODEAdapter(task)
    if task == "pde_advection_1d":
        return Advection1DAdapter()
    if task == "pde_burgers_1d":
        return Burgers1DAdapter()
    raise ValueError(f"Unknown Experiment.task: {task}")

