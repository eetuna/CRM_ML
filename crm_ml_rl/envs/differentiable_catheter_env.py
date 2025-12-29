"""
Differentiable Catheter Environment.

Extends CatheterEnv to provide gradient computation through physics.
Uses Option C (crm_torch) for differentiable dynamics.
"""

import gymnasium as gym
import numpy as np
import torch
from typing import Optional, Dict, Tuple, Any
from dataclasses import dataclass

from .catheter_env import CatheterEnv, CatheterEnvConfig
from ..wrappers.option_c_physics import OptionCPhysics, OptionCConfig


@dataclass
class DifferentiableCatheterEnvConfig(CatheterEnvConfig):
    """Configuration for differentiable catheter environment."""
    # Override defaults for Option C compatibility
    use_cpp: bool = True
    param_file: str = "data/catheter_params/CatheterParameterSet_1_dyn.txt"
    config_file: str = "data/catheter_params/CatheterSpatialConfiguration_1.txt"
    insertion_length: float = 94.3
    dt: float = 0.05

    # Option C specific
    eps_seed: float = 1e-4


class DifferentiableCatheterEnv(CatheterEnv):
    """
    Catheter environment with differentiable physics.

    This environment provides two modes of operation:

    1. Standard Gym interface (step): Fast, non-differentiable
       - Use for rollouts, data collection, evaluation

    2. Differentiable interface (step_differentiable): Slower, with gradients
       - Use for gradient-based optimization, physics-informed learning

    Example - Gradient-based trajectory optimization:

        env = DifferentiableCatheterEnv()
        obs, _ = env.reset()

        # Optimize action sequence
        actions = torch.randn(10, 3, requires_grad=True)
        optimizer = torch.optim.Adam([actions], lr=0.01)

        for iteration in range(100):
            env.reset()
            loss = 0

            for t in range(10):
                next_state = env.step_differentiable(actions[t])
                loss += torch.norm(next_state[:3] - target)**2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    """

    def __init__(
        self,
        config: Optional[DifferentiableCatheterEnvConfig] = None,
        render_mode: Optional[str] = None,
        device: str = "cpu"
    ):
        # Use differentiable config defaults
        config = config or DifferentiableCatheterEnvConfig()
        super().__init__(config, render_mode)

        self.device = torch.device(device)

        # Create Option C physics wrapper
        option_c_config = OptionCConfig(
            param_file=config.param_file,
            config_file=config.config_file,
            insertion_length=config.insertion_length,
            dt=config.dt,
            eps_seed=config.eps_seed,
            damping=config.damping
        )
        self.differentiable_physics = OptionCPhysics(option_c_config, device)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and differentiable physics."""
        obs, info = super().reset(seed=seed, options=options)

        # Reset differentiable physics with same initial currents
        initial_currents = np.zeros(3)
        self.differentiable_physics.reset(initial_currents)

        return obs, info

    def step_differentiable(self, action: torch.Tensor) -> torch.Tensor:
        """
        Take differentiable step through physics.

        Args:
            action: (3,) or (1, 3) tensor with requires_grad=True

        Returns:
            next_state: (6,) tensor [position(3), velocity(3)]

        Note:
            - Also updates internal state for subsequent steps
            - Use for gradient-based optimization
            - Slower than regular step() due to gradient tracking
        """
        # Differentiable forward pass
        next_state = self.differentiable_physics.step_and_update_differentiable(action)

        # Update env state to stay in sync
        self.tip_position = next_state[:3].detach().cpu().numpy()
        self.tip_velocity = next_state[3:6].detach().cpu().numpy()
        self.current_step += 1

        return next_state

    def compute_differentiable_loss(
        self,
        trajectory: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss over trajectory with gradients.

        Args:
            trajectory: (T, 3) action sequence with requires_grad=True
            targets: (T, 3) target positions

        Returns:
            loss: Scalar tensor with gradients
        """
        self.reset()

        T = trajectory.shape[0]
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        for t in range(T):
            next_state = self.step_differentiable(trajectory[t])
            position = next_state[:3]
            target = targets[t].to(self.device)
            loss = loss + torch.norm(position - target)**2

        return loss / T

    def get_action_jacobian(self, currents: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get local Jacobian d(next_state)/d(currents).

        Useful for MPC/iLQR controllers.

        Args:
            currents: (3,) current action

        Returns:
            next_state: (6,) predicted next state
            jacobian: (6, 3) action Jacobian
        """
        # Create batched tensor directly as leaf
        currents_batched = torch.tensor([currents], dtype=torch.float64, device=self.device)
        currents_batched.requires_grad_(True)

        output = self.differentiable_physics.step_differentiable(currents_batched)

        # Compute Jacobian via backward passes
        jacobian = torch.zeros(6, 3, dtype=torch.float64, device=self.device)
        for i in range(6):
            # Zero out previous gradients
            if currents_batched.grad is not None:
                currents_batched.grad.zero_()

            # Backward for i-th output dimension
            grad_output = torch.zeros_like(output)
            grad_output[0, i] = 1.0
            output.backward(grad_output, retain_graph=(i < 5))  # Only retain for non-final iteration

            # Extract gradient
            if currents_batched.grad is not None:
                jacobian[i] = currents_batched.grad[0].clone()

        return output[0].detach().cpu().numpy(), jacobian.cpu().numpy()

    def optimize_trajectory(
        self,
        target_position: np.ndarray,
        horizon: int = 10,
        num_iterations: int = 100,
        learning_rate: float = 0.01
    ) -> Tuple[np.ndarray, float]:
        """
        Optimize action sequence to reach target using gradients.

        Args:
            target_position: (3,) target position
            horizon: Number of steps to optimize
            num_iterations: Number of optimization iterations
            learning_rate: Learning rate for optimizer

        Returns:
            optimal_actions: (horizon, 3) optimized action sequence
            final_loss: Final loss value
        """
        target = torch.tensor(target_position, dtype=torch.float64, device=self.device)

        # Initialize actions as leaf tensor (no grad_fn)
        actions = torch.randn(horizon, 3, dtype=torch.float64, device=self.device) * 0.05
        actions.requires_grad_(True)
        optimizer = torch.optim.Adam([actions], lr=learning_rate)

        for iteration in range(num_iterations):
            self.reset()

            loss = torch.tensor(0.0, dtype=torch.float64, device=self.device)

            for t in range(horizon):
                next_state = self.step_differentiable(actions[t])
                position = next_state[:3]
                loss = loss + torch.norm(position - target)**2

            loss = loss / horizon

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Clip actions to valid range
            with torch.no_grad():
                actions.clamp_(-self.config.max_current, self.config.max_current)

        return actions.detach().cpu().numpy(), loss.item()

    def plan_mpc(
        self,
        target_position: np.ndarray,
        horizon: int = 5,
        num_iterations: int = 20,
        learning_rate: float = 0.1
    ) -> np.ndarray:
        """
        Model Predictive Control using differentiable physics.

        Args:
            target_position: (3,) target position
            horizon: Planning horizon
            num_iterations: Number of optimization iterations
            learning_rate: Learning rate

        Returns:
            action: (3,) first action of optimal sequence
        """
        actions, _ = self.optimize_trajectory(
            target_position,
            horizon=horizon,
            num_iterations=num_iterations,
            learning_rate=learning_rate
        )

        # Return only first action (MPC principle)
        return actions[0]

    def compute_value_gradients(
        self,
        action: np.ndarray,
        value_function
    ) -> np.ndarray:
        """
        Compute gradients of value function through dynamics.

        Useful for policy gradient methods with learned value function.

        Args:
            action: (3,) action to evaluate
            value_function: Callable that maps state to value

        Returns:
            grad: (3,) gradient of value w.r.t. action
        """
        action_t = torch.tensor(action, dtype=torch.float64, requires_grad=True)

        # Forward through physics
        next_state = self.step_differentiable(action_t)

        # Evaluate value function
        value = value_function(next_state)

        # Backward through physics
        value.backward()

        return action_t.grad.detach().cpu().numpy()

    def enable_differentiable_mode(self):
        """
        Enable differentiable mode.

        Future extension point for mode switching.
        Currently differentiable mode is always available.
        """
        pass

    def disable_differentiable_mode(self):
        """
        Disable differentiable mode for faster execution.

        Future extension point for mode switching.
        Currently just uses regular step() for speed.
        """
        pass
