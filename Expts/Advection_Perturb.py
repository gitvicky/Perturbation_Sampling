import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Neural_PDE.Numerical_Solvers.Advection.Advection_1D import Advection_1d
from Utils.PRE.ConvOps_1d import ConvOperator
from Inversion_Strategies.inversion.residual_inversion import (
    PerturbationSamplingConfig,
    calibrate_qhat_from_residual,
    perturbation_bounds,
)

def run_advection_experiment(use_optimization=False, use_langevin=False, use_generator=False):
    print(f"Running Advection Experiment (opt={use_optimization}, langevin={use_langevin}, gen={use_generator})...")
    
    # 1. Setup Simulation
    Nx = 100
    Nt = 50
    x_min, x_max = 0.0, 2.0
    t_end = 0.5
    v = 1.0
    sim = Advection_1d(Nx, Nt, x_min, x_max, t_end)
    dt, dx = sim.dt, sim.dx
    
    # Generate calibration and test data
    def get_data(n):
        u_sol = []
        for _ in range(n):
            xc = 0.5 + 0.5 * np.random.rand()
            amp = 50 + 150 * np.random.rand()
            _, _, u_soln, _ = sim.solve(xc, amp, v)
            # u_soln is (Nt, Nx+3)
            u_sol.append(u_soln)
        return torch.tensor(np.array(u_sol), dtype=torch.float32)

    print("  Generating data...")
    u_cal = get_data(20)   # [20, Nt, Nx+3]
    u_test = get_data(5)    # [5, Nt, Nx+3]
    
    # 2. Define Residual Operator: u_t + v * u_x = 0
    D_t = ConvOperator(domain='t', order=1, scale=1.0/(2*dt))
    D_x = ConvOperator(domain='x', order=1, scale=1.0/(2*dx))
    
    def advection_residual(u):
        # u: [BS, Nt, Nx+3]
        return D_t(u) + v * D_x(u)

    # 3. Calibrate qhat
    print("  Calibrating qhat...")
    # Add some surrogate error to calibration data
    u_cal_noisy = u_cal + 0.02 * torch.randn_like(u_cal)
    res_cal = advection_residual(u_cal_noisy)
    qhat = calibrate_qhat_from_residual(res_cal, alpha=0.1)
    print(f"  Calibrated qhat = {qhat:.4f}")
    
    # 4. Invert bounds using perturbation sampling
    print("  Inverting bounds for test trajectory...")
    test_idx = 0
    u_pred = u_test[test_idx] + 0.02 * torch.randn_like(u_test[test_idx])
    
    config = PerturbationSamplingConfig(
        n_samples=500,
        batch_size=100,
        noise_std=0.05,
        use_optimization=use_optimization,
        use_langevin=use_langevin,
        use_generator=use_generator,
        opt_steps=30,
        gen_train_steps=200
    )
    
    # Advection domain is (Nt, Nx+3). Use interior slice for both.
    interior_slice = (slice(1, -1), slice(1, -1))
    
    try:
        bounds = perturbation_bounds(
            pred_signal=u_pred.numpy(),
            residual_operator=advection_residual,
            qhat=qhat,
            config=config,
            interior_slice=interior_slice
        )
        print("  Inversion successful!")
        
        # 5. Plotting
        plt.figure(figsize=(15, 5))
        t_idx = Nt // 2
        # sim.x is (Nx+3,)
        x_interior = sim.x[1:-1]
        
        plt.subplot(1, 2, 1)
        plt.title(f"Advection Profile at t={t_idx*dt:.2f}")
        plt.plot(x_interior, u_test[test_idx, t_idx, 1:-1], 'k-', label='Truth')
        plt.plot(x_interior, u_pred[t_idx, 1:-1], 'b--', label='Prediction')
        plt.fill_between(x_interior, bounds.lower[t_idx-1], bounds.upper[t_idx-1], 
                         color='#81B29A', alpha=0.3, label='Perturbation Bound')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.title("Spatiotemporal Error Bounds (Lower)")
        plt.imshow(bounds.lower, aspect='auto', extent=[x_interior[0], x_interior[-1], t_end, 0])
        plt.colorbar(label='u_lower')
        plt.xlabel('x')
        plt.ylabel('t')
        
        plt.tight_layout()
        plt.savefig("advection_perturb_test.png")
        print("  Plot saved to advection_perturb_test.png")
        
    except Exception as e:
        print(f"  Inversion failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_advection_experiment(use_optimization=True)
