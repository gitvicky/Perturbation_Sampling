'''
Duffing Oscillator:

Eqn:
     x'' + delta*x' + alpha*x + beta*x^3 = 0

First order system:
    dx1/dt = x2
    dx2/dt = -delta*x2 - alpha*x1 - beta*x1^3

where:
    delta: damping coefficient
    alpha: linear stiffness
    beta:  nonlinear (cubic) stiffness

Note: The beta*x^3 term makes this a nonlinear ODE. The linear part
(x'' + delta*x' + alpha*x) has the same structure as the DHO, but the
cubic term means the full residual operator is not a linear convolution.
Only perturbation sampling is valid for bound inversion.
'''

# %%
import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import solve_ivp

import sys
sys.path.append("..")
sys.path.append("../..")
from Utils.PRE.ConvOps_0d import ConvOperator

class DuffingOscillator:
    """Numerical Duffing oscillator implementation."""

    def __init__(self, alpha=1.0, beta=0.5, delta=0.2):
        """
        Initialize the Duffing oscillator.

        Args:
            alpha (float): Linear stiffness coefficient
            beta (float):  Nonlinear (cubic) stiffness coefficient
            delta (float): Damping coefficient
        """
        self.alpha = alpha
        self.beta = beta
        self.delta = delta

    def get_state_derivative(self, t, state):
        """
        Compute the derivative of the state vector [x1, x2].

        Args:
            t (float): Time (unused for autonomous system)
            state (np.ndarray): State vector [x1 (displacement), x2 (velocity)]

        Returns:
            np.ndarray: Derivative of state vector [dx1/dt, dx2/dt]
        """
        x1, x2 = state
        dx1_dt = x2
        dx2_dt = -self.delta * x2 - self.alpha * x1 - self.beta * x1**3
        return np.array([dx1_dt, dx2_dt])

    def solve_ode(self, t_span, initial_state, t_eval=None):
        """
        Solve the ODE numerically using scipy.integrate.solve_ivp

        Args:
            t_span (tuple): Time span (t_start, t_end)
            initial_state (np.ndarray): Initial state [x1_0, x2_0]
            t_eval (np.ndarray, optional): Times at which to evaluate solution

        Returns:
            tuple: Time points and solution array
        """
        solution = solve_ivp(
            fun=self.get_state_derivative,
            t_span=t_span,
            y0=initial_state,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-8,
            atol=1e-8
        )
        return solution.t, solution.y.T

class DuffingResidualOperator:
    """
    Callable residual operator for the Duffing oscillator:
        x'' + delta*x' + alpha*x + beta*x^3 = 0

    Computes the FD residual using the same scaling as the DHO:
        2*D_tt(x) + dt*delta*D_t(x) + 2*dt^2*alpha*x + 2*dt^2*beta*x^3

    The linear part (2*D_tt + dt*delta*D_t + 2*dt^2*alpha*I) is identical to the
    DHO composite kernel. The cubic term 2*dt^2*beta*x^3 is the nonlinear addition.
    """

    def __init__(self, alpha, beta, delta, dt):
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.dt = dt

        # Build the linear part as a ConvOperator (same as DHO)
        D_t = ConvOperator(order=1)
        D_tt = ConvOperator(order=2)
        D_identity = ConvOperator(order=0)
        D_identity.kernel = torch.tensor([0.0, 1.0, 0.0])

        self.D_linear = ConvOperator(conv='direct')
        self.D_linear.kernel = (2 * D_tt.kernel
                                + dt * delta * D_t.kernel
                                + 2 * dt**2 * alpha * D_identity.kernel)

    def __call__(self, x):
        """
        Compute the nonlinear residual.

        Args:
            x (torch.Tensor): Input signal of shape (BS, Nt)

        Returns:
            torch.Tensor: Residual of shape (BS, Nt)
        """
        linear_part = self.D_linear(x)
        nonlinear_part = 2 * self.dt**2 * self.beta * x**3
        return linear_part + nonlinear_part

# Neural network for the Neural ODE
class ODEFunc(nn.Module):
    """Neural network representing the ODE function."""

    def __init__(self, hidden_dim):
        """
        Initialize the neural ODE function.

        Args:
            hidden_dim (int): Number of hidden units
        """
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, t, y):
        """
        Forward pass of the neural network.

        Args:
            t (torch.Tensor): Time point (unused for autonomous system)
            y (torch.Tensor): State vector

        Returns:
            torch.Tensor: Predicted derivative
        """
        return self.net(y)

def generate_training_data(oscillator, t_span, n_points, n_trajectories,
                           x_range=(-2, 2), v_range=(-2, 2)):
    """
    Generate training data using numerical integration.

    Args:
        oscillator (DuffingOscillator): Oscillator instance
        t_span (tuple): Time span (t_start, t_end)
        n_points (int): Number of points per trajectory
        n_trajectories (int): Number of trajectories
        x_range (tuple): Range for initial displacement
        v_range (tuple): Range for initial velocity

    Returns:
        tuple: Times, states, and derivatives arrays
    """
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    states = []
    derivatives = []

    for _ in range(n_trajectories):
        x0 = np.random.uniform(*x_range)
        v0 = np.random.uniform(*v_range)
        initial_state = np.array([x0, v0])

        _, solution = oscillator.solve_ode(t_span, initial_state, t_eval)
        states.append(solution)

        derivs = np.array([oscillator.get_state_derivative(_, state)
                          for state in solution])
        derivatives.append(derivs)

    return (t_eval,
            np.stack(states, axis=0),
            np.stack(derivatives, axis=0))

def train_neural_ode(func, train_t, train_states, train_derivs, n_epochs, batch_size):
    """
    Train the neural ODE.

    Args:
        func (ODEFunc): Neural network instance
        train_t (np.ndarray): Time points
        train_states (np.ndarray): State trajectories
        train_derivs (np.ndarray): State derivatives
        n_epochs (int): Number of training epochs
        batch_size (int): Batch size

    Returns:
        list: Training losses
    """
    optimizer = torch.optim.Adam(func.parameters())
    losses = []

    train_states = torch.FloatTensor(train_states)
    train_derivs = torch.FloatTensor(train_derivs)

    n_samples = train_states.shape[0]

    for epoch in range(n_epochs):
        epoch_loss = 0

        for i in range(0, n_samples, batch_size):
            batch_states = train_states[i:i+batch_size]
            batch_derivs = train_derivs[i:i+batch_size]

            pred_derivs = func(0, batch_states.reshape(-1, 2))
            loss = torch.mean((pred_derivs - batch_derivs.reshape(-1, 2))**2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        losses.append(epoch_loss)
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch+1}, Loss: {epoch_loss:.6f}')

    return losses

def evaluate(oscillator, neural_ode, t_span, n_points, x_range, v_range, n_solves):
    """
    Evaluate numerical and neural ODE solutions for multiple initial conditions.

    Args:
        oscillator (DuffingOscillator): Oscillator instance
        neural_ode (ODEFunc): Trained neural network
        t_span (tuple): Time span (t_start, t_end)
        n_points: Number of time points
        x_range: Domain for initial displacement
        v_range: Domain for initial velocity
        n_solves: Size of the dataset

    Returns:
        tuple: Time points and solutions (numerical and neural)
    """
    t = torch.linspace(t_span[0], t_span[1], n_points)

    num_solns = []
    neural_solns = []

    for ii in tqdm(range(n_solves)):

        x0 = np.random.uniform(*x_range)
        v0 = np.random.uniform(*v_range)
        initial_state = np.array([x0, v0])

        _, numerical_solution = oscillator.solve_ode(
            t_span, initial_state, t.numpy())

        state_0 = torch.FloatTensor(initial_state)
        neural_solution = odeint(neural_ode, state_0, t)

        num_solns.append(numerical_solution)
        neural_solns.append(neural_solution.detach().numpy())

    return (t.numpy(),
            np.asarray(num_solns),
            np.asarray(neural_solns))

def plot_comparison(t, numerical_sol, neural_sol):
    """
    Plot comparison between numerical and neural ODE solutions.

    Args:
        t (np.ndarray): Time points
        numerical_sol (np.ndarray): Numerical solution
        neural_sol (np.ndarray): Neural ODE solution
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(t, numerical_sol[:, 0], 'b-', label='Numerical')
    ax1.plot(t, neural_sol[:, 0], 'r--', label='Neural ODE')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Displacement')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(t, numerical_sol[:, 1], 'b-', label='Numerical')
    ax2.plot(t, neural_sol[:, 1], 'r--', label='Neural ODE')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Velocity')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# %%
if __name__ == "__main__":
    # Initialize system
    alpha, beta, delta = 1.0, 0.5, 0.2
    oscillator = DuffingOscillator(alpha, beta, delta)

    # Generate training data
    t_span = (0, 15)
    n_points = 100
    n_trajectories = 50
    t, states, derivs = generate_training_data(
        oscillator, t_span, n_points, n_trajectories)

    # Initialize and train neural ODE
    func = ODEFunc(hidden_dim=64)
    losses = train_neural_ode(
        func, t, states, derivs, n_epochs=1000, batch_size=16)

    # Evaluate on test set
    t, numerical_sol, neural_sol = evaluate(
        oscillator, func, t_span, n_points,
        x_range=(-2, 2), v_range=(-2, 2), n_solves=5)

    # %%
    # PRE Estimations
    dt = t[1] - t[0]
    pos = torch.tensor(neural_sol[..., 0], dtype=torch.float32)

    residual_op = DuffingResidualOperator(alpha, beta, delta, dt)
    res = residual_op(pos)

    plt.figure()
    plt.plot(t, res[0].detach().numpy(), 'b-', label='Nonlinear Residual')
    plt.xlabel('Time')
    plt.ylabel('Residual')
    plt.title('Duffing: Nonlinear Residual')
    plt.legend()
    plt.grid(True)
# %%
