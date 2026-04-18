#!/usr/bin/env python3
"""Config-driven evaluation entrypoint for pretrained toy surrogates."""

from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Expts.pipeline import get_adapter
from Expts.pipeline.io_utils import load_yaml, resolve_run_paths, save_yaml, set_seed


def _deep_update(dst: dict, src: dict) -> dict:
    out = deepcopy(dst)
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate surrogate model from YAML config.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to evaluation YAML config containing Run + Experiment.task.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    eval_cfg = load_yaml(args.config)
    paths = resolve_run_paths(eval_cfg)
    if not paths.config_path.exists():
        raise FileNotFoundError(f"Missing trained run config: {paths.config_path}")

    train_cfg = load_yaml(paths.config_path)
    config = _deep_update(train_cfg, eval_cfg)
    seed = int(config.get("Run", {}).get("seed", 0))
    set_seed(seed)

    task = config["Experiment"]["task"]
    adapter = get_adapter(task)
    metrics = adapter.evaluate(config, paths)
    save_yaml(paths.eval_metrics_path, metrics)
    print(f"[evaluate] task={task} run_dir={paths.run_dir} mse={metrics.get('mse'):.4e}")


if __name__ == "__main__":
    main()

