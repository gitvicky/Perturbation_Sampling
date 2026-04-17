"""
Autoregressive 1D U-Net surrogate for the linear advection PDE

        u_t + v * u_x = 0

Trained to predict u(x, t_{k+1}) given u(x, t_k) and rolled out
autoregressively to produce full spatiotemporal trajectories of shape
`(Nt, Nx+3)`, matching the numerical solver's output layout.

Mirrors the role of `SHO_NODE.py` / `DHO_NODE.py` / `Duffing_NODE.py`:
provides `generate_training_data`, `train_unet`, and `evaluate` so the
experiment runner can treat this as a drop-in neural surrogate.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class UNet1d(nn.Module):
    """Lightweight 2-level 1D U-Net for next-step prediction.

    Input : `[BS, 1, Nx+3]` — the solution at time `t_k`.
    Output: `[BS, 1, Nx+3]` — the predicted solution at `t_{k+1}`.

    Upsampling uses `F.interpolate` to the skip-connection shape so the
    network handles arbitrary odd spatial dimensions (Nx+3=63 by default).
    A global residual connection (output = input + delta) makes the model
    learn a time-step increment, which is easier than learning the full
    field and keeps the solution stable over many autoregressive steps.
    """

    def __init__(self, features=32):
        super().__init__()
        self.enc1 = self._block(1, features)
        self.enc2 = self._block(features, features * 2)
        self.bottleneck = self._block(features * 2, features * 4)
        self.dec2 = self._block(features * 4 + features * 2, features * 2)
        self.dec1 = self._block(features * 2 + features, features)
        self.head = nn.Conv1d(features, 1, kernel_size=1)

    @staticmethod
    def _block(in_c, out_c):
        return nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(out_c, out_c, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x):
        e1 = self.enc1(x)                                      # [B, F, L]
        e2 = self.enc2(F.avg_pool1d(e1, 2))                    # [B, 2F, L/2]
        b  = self.bottleneck(F.avg_pool1d(e2, 2))              # [B, 4F, L/4]

        d2 = F.interpolate(b, size=e2.shape[-1], mode='linear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(d2, size=e1.shape[-1], mode='linear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        delta = self.head(d1)
        return x + delta  # predict next-step increment

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class SpectralConv1d(nn.Module):
    """1D Fourier layer: rFFT -> truncated complex mult -> irFFT."""

    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x):
        B, C, L = x.shape
        x_ft = torch.fft.rfft(x, n=L)
        out_ft = torch.zeros(B, self.out_channels, L // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        k = min(self.modes, x_ft.shape[-1])
        out_ft[:, :, :k] = torch.einsum("bix,iox->box", x_ft[:, :, :k], self.weights[:, :, :k])
        return torch.fft.irfft(out_ft, n=L)


class FNO1d(nn.Module):
    """Lightweight 1D FNO next-step surrogate.

    Input : `[BS, 1, Nx+3]`. Output: `[BS, 1, Nx+3]`. Uses 4 Fourier
    blocks (spectral conv + pointwise 1x1 conv + GELU), a lifting/projecting
    1x1 conv pair, and a global residual (predicts `Δu`, like `UNet1d`).
    """

    def __init__(self, modes=16, width=32, n_layers=4):
        super().__init__()
        self.lift = nn.Conv1d(1, width, kernel_size=1)
        self.spectral = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(n_layers)])
        self.pointwise = nn.ModuleList([nn.Conv1d(width, width, kernel_size=1) for _ in range(n_layers)])
        self.project = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(width, 1, kernel_size=1),
        )

    def forward(self, x):
        h = self.lift(x)
        for spec, pw in zip(self.spectral, self.pointwise):
            h = F.gelu(spec(h) + pw(h))
        delta = self.project(h)
        return x + delta

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def build_model(kind="unet", features=32, modes=16, n_layers=4):
    """Factory for the supported surrogate architectures.

    Parameters
    ----------
    kind : {"unet", "fno"}
        Architecture to instantiate.
    features : int
        UNet base channel width / FNO lifted channel width.
    modes, n_layers : int
        FNO-only: number of Fourier modes and Fourier blocks.
    """
    if kind == "unet":
        return UNet1d(features=features)
    if kind == "fno":
        return FNO1d(modes=modes, width=features, n_layers=n_layers)
    raise ValueError(f"Unknown model kind: {kind!r}. Choose 'unet' or 'fno'.")


def sample_ic_params(rng, xc_range=(0.25, 1.75), width_range=(30.0, 250.0),
                      height_range=(0.3, 1.5)):
    """Sample a parameterised initial-condition triple `(xc, width, height)`.

    `width` is the Gaussian concentration passed to `sim.solve` as its `amp`
    argument (the solver initialises `u0 = exp(-width * (x - xc)^2)`).
    `height` scales the whole trajectory (valid under linear advection).
    """
    xc = xc_range[0] + (xc_range[1] - xc_range[0]) * rng.random()
    width = width_range[0] + (width_range[1] - width_range[0]) * rng.random()
    height = height_range[0] + (height_range[1] - height_range[0]) * rng.random()
    return float(xc), float(width), float(height)


def solve_parameterised(sim, xc, width, height, v=1.0):
    """Solve the numerical advection PDE for a parameterised Gaussian pulse.

    Only the numerical (Lax-Wendroff) trajectory is kept; the solver's
    analytical return is discarded. The scalar `height` multiplies the
    trajectory — linearity of `u_t + v u_x = 0` makes this exact.
    """
    _, _, u_soln, _ = sim.solve(xc, width, v)
    return (height * u_soln).astype(np.float32)


def generate_training_data(sim, n_trajectories, v=1.0, seed=0,
                            xc_range=(0.25, 1.75), width_range=(30.0, 250.0),
                            height_range=(0.3, 1.5)):
    """Generate `(u_t, u_{t+1})` training pairs from a parameterised IC family.

    Each trajectory samples an independent `(xc, width, height)` triple, so
    the training set covers pulse *location*, *sharpness*, and *amplitude*
    variation — not the near-identical pulses the previous default gave.

    Returns
    -------
    inputs : Tensor `[n_trajectories * (Nt-1), 1, Nx+3]`
    targets : Tensor `[n_trajectories * (Nt-1), 1, Nx+3]`
    trajectories : Tensor `[n_trajectories, Nt, Nx+3]` — raw numerical
        trajectories (useful for downstream evaluation/calibration).
    """
    rng = np.random.default_rng(seed)
    trajs = []
    for _ in range(n_trajectories):
        xc, width, height = sample_ic_params(rng, xc_range, width_range, height_range)
        trajs.append(solve_parameterised(sim, xc, width, height, v))
    trajs = np.stack(trajs, axis=0)                              # [N, Nt, Nx+3]
    inputs = trajs[:, :-1]
    targets = trajs[:, 1:]
    inputs = torch.from_numpy(inputs).reshape(-1, 1, inputs.shape[-1])
    targets = torch.from_numpy(targets).reshape(-1, 1, targets.shape[-1])
    return inputs, targets, torch.from_numpy(trajs)


def train_unet(model, inputs, targets, n_epochs=200, batch_size=64,
                lr=1e-3, device="cpu", verbose=True):
    """Train the U-Net on next-step prediction with MSE loss."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    loss_fn = nn.MSELoss()

    n = inputs.shape[0]
    inputs = inputs.to(device)
    targets = targets.to(device)

    pbar = tqdm(range(n_epochs), desc="Training U-Net", disable=not verbose)
    for _ in pbar:
        perm = torch.randperm(n, device=device)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            x, y = inputs[idx], targets[idx]
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total += loss.item() * x.shape[0]
        scheduler.step()
        pbar.set_postfix(loss=f"{total / n:.3e}")
    return model


@torch.no_grad()
def rollout(model, u0, n_steps, device="cpu"):
    """Autoregressively roll out the U-Net from initial condition `u0`.

    Parameters
    ----------
    u0 : Tensor `[BS, Nx+3]` — initial field at `t = 0`.
    n_steps : int — number of rollout steps (produces `n_steps` frames
        *after* the initial condition, matching solver trajectory length
        when called with `n_steps = Nt - 1` and prepending `u0`).

    Returns
    -------
    Tensor `[BS, n_steps + 1, Nx+3]` — trajectory including `u0`.
    """
    model.eval()
    u = u0.to(device).unsqueeze(1)  # [BS, 1, Nx+3]
    frames = [u.squeeze(1)]
    for _ in range(n_steps):
        u = model(u)
        frames.append(u.squeeze(1))
    return torch.stack(frames, dim=1)


def evaluate(sim, model, n_solves, v=1.0, seed=1, device="cpu",
             xc_range=(0.25, 1.75), width_range=(30.0, 250.0),
             height_range=(0.3, 1.5)):
    """Produce numerical + U-Net rollout trajectories for `n_solves` ICs.

    The ICs are drawn from the same parameterised family used for training
    (`sample_ic_params`) but with an independent RNG seed.

    Returns
    -------
    numerical_sol : Tensor `[n_solves, Nt, Nx+3]` — numerical trajectories.
    neural_sol    : Tensor `[n_solves, Nt, Nx+3]` — U-Net rollouts from the
        same initial conditions.
    """
    rng = np.random.default_rng(seed)
    num_trajs = []
    u0_list = []
    for _ in range(n_solves):
        xc, width, height = sample_ic_params(rng, xc_range, width_range, height_range)
        traj = solve_parameterised(sim, xc, width, height, v)
        num_trajs.append(traj)
        u0_list.append(traj[0])
    num_trajs = torch.tensor(np.stack(num_trajs, axis=0), dtype=torch.float32)
    u0 = torch.tensor(np.stack(u0_list, axis=0), dtype=torch.float32)

    Nt = num_trajs.shape[1]
    neural_sol = rollout(model, u0, n_steps=Nt - 1, device=device).cpu()
    return num_trajs, neural_sol
