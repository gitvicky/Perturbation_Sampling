'''
Nonlinear Pendulum:

Eqn:
     x'' + (g/L) sin(x) = 0

First order system:
    dx1/dt = x2
    dx2/dt = -(g/L) sin(x1)

Note: This is a nonlinear ODE — the sin(x1) term means the residual
operator is not a linear convolution, so only perturbation sampling
is valid for bound inversion.
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


class NonlinearPendulum:
    """Numerical nonlinear pendulum implementation."""

    def __init__(self, g=9.81, L=9.81):
        """
        Initialize the nonlinear pendulum.

        Args:
            g (float): Gravitational acceleration
            L (float): Pendulum length
        """
        self.g = g
        self.L = L
        self.omega_0 = np.sqrt(g / L)  # Small-angle natural frequency

    def get_state_derivative(self, t, state):
        """
        Compute the derivative of the state vector [x1, x2].

        Args:
            t (float): Time (unused for autonomous system)
            state (np.ndarray): State vector [x1 (angle), x2 (angular velocity)]

        Returns:
            np.ndarray: Derivative of state vector [dx1/dt, dx2/dt]
        """
        x1, x2 = state
        dx1_dt = x2
        dx2_dt = -(self.g / self.L) * np.sin(x1)
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


class NonlinearPendulumResidualOperator:
    """
    Callable residual operator for the nonlinear pendulum: x'' + (g/L)*sin(x) = 0.

    Computes the FD residual: D_tt(x) + dt^2 * (g/L) * sin(x), where D_tt uses the
    [1, -2, 1] stencil (which approximates dt^2 * x'').

    This operator is used by perturbation sampling, which only needs a callable
    that maps signals -> residuals.
    """

    def __init__(self, g, L, dt):
        self.g = g
        self.L = L
        self.dt = dt
        self.D_tt = ConvOperator(order=2, conv='direct')

    def __call__(self, x):
        """
        Compute the nonlinear residual.

        Args:
            x (torch.Tensor): Input signal of shape (BS, Nt)

        Returns:
            torch.Tensor: Residual of shape (BS, Nt) (boundary values are invalid,
                          handled by interior_slice in the inversion pipeline)
        """
        linear_part = self.D_tt(x)
        nonlinear_part = self.dt ** 2 * (self.g / self.L) * torch.sin(x)
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
                           x_range=(-np.pi/2, np.pi/2), v_range=(-1, 1)):
    """
    Generate training data using numerical integration.

    Args:
        oscillator (NonlinearPendulum): Pendulum instance
        t_span (tuple): Time span (t_start, t_end)
        n_points (int): Number of points per trajectory
        n_trajectories (int): Number of trajectories
        x_range (tuple): Range for initial angle (radians)
        v_range (tuple): Range for initial angular velocity

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
        oscillator (NonlinearPendulum): Pendulum instance
        neural_ode (ODEFunc): Trained neural network
        t_span (tuple): Time span (t_start, t_end)
        n_points: Number of time points
        x_range: Domain for initial angle
        v_range: Domain for initial angular velocity
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
    ax1.set_ylabel('Angle (rad)')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(t, numerical_sol[:, 1], 'b-', label='Numerical')
    ax2.plot(t, neural_sol[:, 1], 'r--', label='Neural ODE')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Angular Velocity')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


# %%
if __name__ == "__main__":
    # Initialize system — g/L = 1.0 for unit natural frequency
    g, L = 9.81, 9.81
    pendulum = NonlinearPendulum(g, L)

    # Generate training data (moderate angles to see nonlinearity)
    t_span = (0, 10)
    n_points = 100
    n_trajectories = 50
    t, states, derivs = generate_training_data(
        pendulum, t_span, n_points, n_trajectories,
        x_range=(-np.pi/2, np.pi/2), v_range=(-1, 1))

    # Initialize and train neural ODE
    func = ODEFunc(hidden_dim=64)
    losses = train_neural_ode(
        func, t, states, derivs, n_epochs=1000, batch_size=16)

    # Evaluate on test set
    t, numerical_sol, neural_sol = evaluate(
        pendulum, func, t_span, n_points,
        x_range=(-np.pi/2, np.pi/2), v_range=(-1, 1), n_solves=5)

    # %%
    # PRE Estimations
    dt = t[1] - t[0]
    pos = torch.tensor(neural_sol[..., 0], dtype=torch.float32)

    residual_op = NonlinearPendulumResidualOperator(g, L, dt)
    res = residual_op(pos)

    plt.figure()
    plt.plot(t, res[0].detach().numpy(), 'b-', label='Nonlinear Residual')
    plt.xlabel('Time')
    plt.ylabel('Residual')
    plt.title('Pendulum: Nonlinear Residual')
    plt.legend()
    plt.grid(True)
# %%
