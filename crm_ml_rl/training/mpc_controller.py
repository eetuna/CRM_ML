"""
Model Predictive Control (MPC) for catheter control.

Uses learned dynamics models for trajectory optimization.
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, Callable, List, Union
from dataclasses import dataclass

try:
    import casadi as ca
    HAS_CASADI = True
except ImportError:
    HAS_CASADI = False
    print("Warning: CasADi not installed. Some MPC features unavailable.")

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
    print("Warning: CVXPY not installed. Some MPC features unavailable.")


@dataclass
class MPCConfig:
    """MPC configuration."""
    horizon: int = 20  # Prediction horizon
    dt: float = 0.02  # Time step
    state_dim: int = 6  # State dimension
    action_dim: int = 3  # Action dimension

    # Weights
    position_weight: float = 10.0
    velocity_weight: float = 1.0
    action_weight: float = 0.1
    action_change_weight: float = 0.5
    terminal_weight: float = 50.0

    # Constraints
    action_min: float = -0.3
    action_max: float = 0.3
    velocity_max: float = 50.0  # mm/s

    # Solver
    max_iter: int = 100
    tol: float = 1e-4
    warmstart: bool = True


class MPCController:
    """
    Model Predictive Controller for catheter navigation.

    Supports both linear and nonlinear MPC with learned dynamics.
    """

    def __init__(
        self,
        dynamics_model: Optional[torch.nn.Module] = None,
        config: Optional[MPCConfig] = None,
        use_gpu: bool = False
    ):
        """
        Initialize MPC controller.

        Args:
            dynamics_model: Learned dynamics model for prediction
            config: MPC configuration
            use_gpu: Use GPU for optimization
        """
        self.config = config or MPCConfig()
        self.dynamics_model = dynamics_model
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

        if dynamics_model is not None:
            self.dynamics_model = dynamics_model.to(self.device)
            self.dynamics_model.eval()

        # Warmstart storage
        self.prev_solution = None

    def set_dynamics_model(self, model: torch.nn.Module):
        """Set or update dynamics model."""
        self.dynamics_model = model.to(self.device)
        self.dynamics_model.eval()

    def _predict_dynamics(
        self,
        state: np.ndarray,
        action: np.ndarray
    ) -> np.ndarray:
        """Predict next state using dynamics model."""
        if self.dynamics_model is None:
            # Simple linear model fallback
            # x_{t+1} = x_t + dt * [v_t; f(u_t)]
            position = state[:3]
            velocity = state[3:6]

            # Simplified dynamics: action affects velocity
            new_velocity = velocity + self.config.dt * action * 100  # Scale factor
            new_position = position + self.config.dt * new_velocity

            return np.concatenate([new_position, new_velocity])

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action_t = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
            next_state = self.dynamics_model(state_t, action_t)
            return next_state.squeeze(0).cpu().numpy()

    def _predict_trajectory(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray
    ) -> np.ndarray:
        """Predict state trajectory given action sequence."""
        horizon = actions.shape[0]
        states = np.zeros((horizon + 1, self.config.state_dim))
        states[0] = initial_state

        for t in range(horizon):
            states[t + 1] = self._predict_dynamics(states[t], actions[t])

        return states

    def _compute_cost(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        target_trajectory: np.ndarray
    ) -> float:
        """Compute trajectory cost."""
        horizon = actions.shape[0]
        cost = 0.0

        for t in range(horizon):
            # Position error
            pos_error = states[t, :3] - target_trajectory[t, :3]
            cost += self.config.position_weight * np.sum(pos_error ** 2)

            # Velocity cost (if target has velocity)
            if target_trajectory.shape[1] >= 6:
                vel_error = states[t, 3:6] - target_trajectory[t, 3:6]
                cost += self.config.velocity_weight * np.sum(vel_error ** 2)

            # Action cost
            cost += self.config.action_weight * np.sum(actions[t] ** 2)

            # Action change cost
            if t > 0:
                action_change = actions[t] - actions[t - 1]
                cost += self.config.action_change_weight * np.sum(action_change ** 2)

        # Terminal cost
        pos_error = states[-1, :3] - target_trajectory[-1, :3]
        cost += self.config.terminal_weight * np.sum(pos_error ** 2)

        return cost

    def solve_shooting(
        self,
        current_state: np.ndarray,
        target_trajectory: np.ndarray,
        prev_action: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Solve MPC using shooting method with gradient descent.

        Args:
            current_state: Current state
            target_trajectory: Target trajectory (horizon+1, state_dim)
            prev_action: Previous action for smoothness

        Returns:
            Tuple of (optimal_action, info_dict)
        """
        horizon = min(self.config.horizon, len(target_trajectory) - 1)

        # Initialize actions
        if self.config.warmstart and self.prev_solution is not None:
            actions = np.zeros((horizon, self.config.action_dim))
            actions[:-1] = self.prev_solution[1:horizon]
            actions[-1] = self.prev_solution[-1]
        else:
            actions = np.zeros((horizon, self.config.action_dim))

        # Simple gradient descent optimization
        learning_rate = 0.1
        best_cost = float('inf')
        best_actions = actions.copy()

        for iteration in range(self.config.max_iter):
            # Compute cost and gradient numerically
            states = self._predict_trajectory(current_state, actions)
            cost = self._compute_cost(states, actions, target_trajectory[:horizon + 1])

            if cost < best_cost:
                best_cost = cost
                best_actions = actions.copy()

            # Numerical gradient
            grad = np.zeros_like(actions)
            eps = 1e-4
            for t in range(horizon):
                for d in range(self.config.action_dim):
                    actions_plus = actions.copy()
                    actions_plus[t, d] += eps
                    states_plus = self._predict_trajectory(current_state, actions_plus)
                    cost_plus = self._compute_cost(states_plus, actions_plus, target_trajectory[:horizon + 1])
                    grad[t, d] = (cost_plus - cost) / eps

            # Gradient step
            actions = actions - learning_rate * grad

            # Clip actions
            actions = np.clip(actions, self.config.action_min, self.config.action_max)

            # Check convergence
            if np.max(np.abs(grad)) < self.config.tol:
                break

        # Store for warmstart
        self.prev_solution = best_actions

        info = {
            'cost': best_cost,
            'iterations': iteration + 1,
            'predicted_trajectory': self._predict_trajectory(current_state, best_actions)
        }

        return best_actions[0], info

    def solve_cem(
        self,
        current_state: np.ndarray,
        target_trajectory: np.ndarray,
        n_samples: int = 100,
        n_elite: int = 10,
        n_iterations: int = 5
    ) -> Tuple[np.ndarray, Dict]:
        """
        Solve MPC using Cross-Entropy Method (CEM).

        Derivative-free optimization good for nonlinear dynamics.

        Args:
            current_state: Current state
            target_trajectory: Target trajectory
            n_samples: Number of samples per iteration
            n_elite: Number of elite samples
            n_iterations: Number of CEM iterations

        Returns:
            Tuple of (optimal_action, info_dict)
        """
        horizon = min(self.config.horizon, len(target_trajectory) - 1)

        # Initialize distribution
        mean = np.zeros((horizon, self.config.action_dim))
        std = np.ones((horizon, self.config.action_dim)) * 0.1

        if self.config.warmstart and self.prev_solution is not None:
            mean[:-1] = self.prev_solution[1:horizon]
            mean[-1] = self.prev_solution[-1]

        best_cost = float('inf')
        best_actions = mean.copy()

        for iteration in range(n_iterations):
            # Sample actions
            samples = np.random.randn(n_samples, horizon, self.config.action_dim)
            samples = mean + std * samples
            samples = np.clip(samples, self.config.action_min, self.config.action_max)

            # Evaluate samples
            costs = np.zeros(n_samples)
            for i in range(n_samples):
                states = self._predict_trajectory(current_state, samples[i])
                costs[i] = self._compute_cost(states, samples[i], target_trajectory[:horizon + 1])

            # Select elite samples
            elite_indices = np.argsort(costs)[:n_elite]
            elite_samples = samples[elite_indices]

            # Update distribution
            mean = np.mean(elite_samples, axis=0)
            std = np.std(elite_samples, axis=0) + 1e-4

            # Track best
            if costs[elite_indices[0]] < best_cost:
                best_cost = costs[elite_indices[0]]
                best_actions = elite_samples[0]

        # Store for warmstart
        self.prev_solution = best_actions

        info = {
            'cost': best_cost,
            'iterations': n_iterations,
            'predicted_trajectory': self._predict_trajectory(current_state, best_actions)
        }

        return best_actions[0], info

    def solve_mppi(
        self,
        current_state: np.ndarray,
        target_trajectory: np.ndarray,
        n_samples: int = 100,
        temperature: float = 0.1
    ) -> Tuple[np.ndarray, Dict]:
        """
        Solve MPC using Model Predictive Path Integral (MPPI).

        Information-theoretic approach for stochastic optimal control.

        Args:
            current_state: Current state
            target_trajectory: Target trajectory
            n_samples: Number of rollout samples
            temperature: Temperature for weighting

        Returns:
            Tuple of (optimal_action, info_dict)
        """
        horizon = min(self.config.horizon, len(target_trajectory) - 1)

        # Nominal trajectory
        if self.config.warmstart and self.prev_solution is not None:
            nominal = np.zeros((horizon, self.config.action_dim))
            nominal[:-1] = self.prev_solution[1:horizon]
            nominal[-1] = self.prev_solution[-1]
        else:
            nominal = np.zeros((horizon, self.config.action_dim))

        # Sample perturbations
        noise_std = 0.1
        perturbations = np.random.randn(n_samples, horizon, self.config.action_dim) * noise_std

        # Evaluate samples
        costs = np.zeros(n_samples)
        for i in range(n_samples):
            actions = nominal + perturbations[i]
            actions = np.clip(actions, self.config.action_min, self.config.action_max)
            states = self._predict_trajectory(current_state, actions)
            costs[i] = self._compute_cost(states, actions, target_trajectory[:horizon + 1])

        # Compute weights
        costs_normalized = costs - np.min(costs)
        weights = np.exp(-costs_normalized / temperature)
        weights = weights / np.sum(weights)

        # Weighted average of perturbations
        weighted_perturbation = np.sum(
            weights[:, np.newaxis, np.newaxis] * perturbations,
            axis=0
        )

        # Update nominal
        optimal_actions = nominal + weighted_perturbation
        optimal_actions = np.clip(optimal_actions, self.config.action_min, self.config.action_max)

        # Store for warmstart
        self.prev_solution = optimal_actions

        # Compute final cost
        states = self._predict_trajectory(current_state, optimal_actions)
        final_cost = self._compute_cost(states, optimal_actions, target_trajectory[:horizon + 1])

        info = {
            'cost': final_cost,
            'iterations': 1,
            'predicted_trajectory': states,
            'weights': weights
        }

        return optimal_actions[0], info

    def control(
        self,
        current_state: np.ndarray,
        target_trajectory: np.ndarray,
        method: str = "cem"
    ) -> Tuple[np.ndarray, Dict]:
        """
        Main control interface.

        Args:
            current_state: Current state
            target_trajectory: Target trajectory
            method: Optimization method ("shooting", "cem", "mppi")

        Returns:
            Tuple of (action, info_dict)
        """
        if method == "shooting":
            return self.solve_shooting(current_state, target_trajectory)
        elif method == "cem":
            return self.solve_cem(current_state, target_trajectory)
        elif method == "mppi":
            return self.solve_mppi(current_state, target_trajectory)
        else:
            raise ValueError(f"Unknown method: {method}")

    def reset(self):
        """Reset controller state."""
        self.prev_solution = None


class LinearMPC:
    """
    Linear MPC using convex optimization.

    Faster than nonlinear MPC but requires linearized dynamics.
    """

    def __init__(self, config: Optional[MPCConfig] = None):
        """Initialize linear MPC."""
        if not HAS_CVXPY:
            raise ImportError("CVXPY required for LinearMPC")

        self.config = config or MPCConfig()

        # Linearized dynamics matrices (to be set)
        self.A = None  # State transition
        self.B = None  # Control matrix

    def set_dynamics(self, A: np.ndarray, B: np.ndarray):
        """Set linearized dynamics matrices."""
        self.A = A
        self.B = B

    def solve(
        self,
        current_state: np.ndarray,
        target: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Solve linear MPC.

        Args:
            current_state: Current state
            target: Target position

        Returns:
            Tuple of (action, info_dict)
        """
        if self.A is None or self.B is None:
            raise ValueError("Dynamics matrices not set")

        horizon = self.config.horizon

        # Decision variables
        x = cp.Variable((horizon + 1, self.config.state_dim))
        u = cp.Variable((horizon, self.config.action_dim))

        # Cost
        cost = 0
        constraints = []

        # Initial condition
        constraints.append(x[0] == current_state)

        for t in range(horizon):
            # Dynamics constraint
            constraints.append(x[t + 1] == self.A @ x[t] + self.B @ u[t])

            # Position cost
            pos_error = x[t, :3] - target[:3]
            cost += self.config.position_weight * cp.sum_squares(pos_error)

            # Action cost
            cost += self.config.action_weight * cp.sum_squares(u[t])

            # Action constraints
            constraints.append(u[t] >= self.config.action_min)
            constraints.append(u[t] <= self.config.action_max)

        # Terminal cost
        pos_error = x[-1, :3] - target[:3]
        cost += self.config.terminal_weight * cp.sum_squares(pos_error)

        # Solve
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.OSQP, warm_start=True)

        if problem.status != cp.OPTIMAL:
            print(f"Warning: MPC solve status: {problem.status}")

        info = {
            'cost': problem.value,
            'status': problem.status,
            'predicted_trajectory': x.value
        }

        return u.value[0], info


if __name__ == "__main__":
    print("Testing MPC Controller...")

    # Create controller
    config = MPCConfig(horizon=10)
    mpc = MPCController(config=config)

    # Test state and target
    current_state = np.array([0.0, 0.0, 80.0, 0.0, 0.0, 0.0])  # position + velocity
    target_trajectory = np.zeros((config.horizon + 1, 6))
    target_trajectory[:, :3] = np.array([10.0, 5.0, 85.0])  # Target position

    # Test different methods
    print("\nTesting shooting method...")
    action, info = mpc.control(current_state, target_trajectory, method="shooting")
    print(f"Action: {action}, Cost: {info['cost']:.4f}")

    mpc.reset()

    print("\nTesting CEM method...")
    action, info = mpc.control(current_state, target_trajectory, method="cem")
    print(f"Action: {action}, Cost: {info['cost']:.4f}")

    mpc.reset()

    print("\nTesting MPPI method...")
    action, info = mpc.control(current_state, target_trajectory, method="mppi")
    print(f"Action: {action}, Cost: {info['cost']:.4f}")

    # Test linear MPC if CVXPY available
    if HAS_CVXPY:
        print("\nTesting Linear MPC...")
        linear_mpc = LinearMPC(config)

        # Simple linear dynamics
        A = np.eye(6)
        A[:3, 3:6] = np.eye(3) * config.dt
        B = np.zeros((6, 3))
        B[3:6, :] = np.eye(3) * config.dt * 100

        linear_mpc.set_dynamics(A, B)
        action, info = linear_mpc.solve(current_state, target_trajectory[0])
        print(f"Action: {action}, Cost: {info['cost']:.4f}")

    print("\nMPC testing complete!")
