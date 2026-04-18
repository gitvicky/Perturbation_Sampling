import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Expts.pipeline.adapters import get_adapter
from Expts.pipeline.io_utils import resolve_run_paths, save_norms, load_norms_into
from Neural_PDE.Utils.processing_utils import Normalisation


def test_resolve_run_paths():
    cfg = {"Run": {"name": "unit-run", "folder": "Expts/Weights"}}
    paths = resolve_run_paths(cfg)
    assert str(paths.run_dir).endswith("Expts/Weights/unit-run")
    assert str(paths.model_path).endswith("Expts/Weights/unit-run/model.pth")
    assert str(paths.norms_path).endswith("Expts/Weights/unit-run/norms.npz")


def test_norm_roundtrip(tmp_path):
    x = torch.randn(4, 10)
    normalizer = Normalisation("Min-Max")(x.clone())
    norms_path = tmp_path / "norms.npz"
    save_norms(normalizer, norms_path)

    restored = Normalisation("Min-Max")(torch.zeros_like(x))
    restored = load_norms_into(restored, norms_path)
    encoded = normalizer.encode(x.clone())
    encoded_restored = restored.encode(x.clone())
    assert torch.allclose(encoded, encoded_restored)


def test_get_adapter_known_tasks():
    for task in ("ode_sho", "ode_dho", "ode_duffing", "pde_advection_1d", "pde_burgers_1d"):
        assert get_adapter(task) is not None
