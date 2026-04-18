#!/usr/bin/env python3
"""Config-driven training entrypoint for toy ODE and 1D PDE surrogates."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Expts.pipeline import get_adapter
from Expts.pipeline.io_utils import (
    load_yaml,
    resolve_run_paths,
    save_yaml,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train surrogate model from YAML config.")
    parser.add_argument("--config", required=True, help="Path to training YAML config.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_yaml(args.config)
    task = config["Experiment"]["task"]
    seed = int(config.get("Run", {}).get("seed", 0))
    set_seed(seed)

    paths = resolve_run_paths(config)
    adapter = get_adapter(task)
    metrics = adapter.train(config, paths)

    paths.run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(paths.config_path, config)
    save_yaml(paths.train_metrics_path, metrics)
    print(f"[train] task={task} run_dir={paths.run_dir} mse={metrics.get('mse'):.4e}")


if __name__ == "__main__":
    main()

