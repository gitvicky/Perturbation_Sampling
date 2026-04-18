"""IO and utility helpers for config-driven experiment runs."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    model_path: Path
    norms_path: Path
    config_path: Path
    train_metrics_path: Path
    eval_metrics_path: Path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def resolve_run_paths(config: dict[str, Any]) -> RunPaths:
    run_cfg = config.get("Run", {})
    run_name = str(run_cfg.get("name", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    folder = run_cfg.get("folder", "Expts/Weights")
    run_dir = Path(folder) / run_name
    return RunPaths(
        run_dir=run_dir,
        model_path=run_dir / "model.pth",
        norms_path=run_dir / "norms.npz",
        config_path=run_dir / "config.yaml",
        train_metrics_path=run_dir / "train_metrics.yaml",
        eval_metrics_path=run_dir / "eval_metrics.yaml",
    )


def save_norms(normalizer: Any, norms_path: Path) -> None:
    norms_path.parent.mkdir(parents=True, exist_ok=True)
    a = getattr(normalizer, "a", torch.tensor(0.0))
    b = getattr(normalizer, "b", torch.tensor(0.0))
    np.savez(norms_path, a=np.asarray(a.detach().cpu()), b=np.asarray(b.detach().cpu()))


def load_norms_into(normalizer: Any, norms_path: Path) -> Any:
    norms = np.load(norms_path)
    normalizer.a = torch.tensor(norms["a"])
    normalizer.b = torch.tensor(norms["b"])
    return normalizer

