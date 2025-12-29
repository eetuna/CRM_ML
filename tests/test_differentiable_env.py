"""
Tests for DifferentiableCatheterEnv (crm_ml_rl/envs/differentiable_catheter_env.py).

Tests the differentiable environment that combines standard Gym interface
with gradient-based optimization capabilities.
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from crm_ml_rl.envs.differentiable_catheter_env import (
    DifferentiableCatheterEnv,
    DifferentiableCatheterEnvConfig
)


class TestDifferentiableEnvInitialization:
    """Test initialization and reset functionality."""

    def test_initialization_success(self):
        """Test successful initialization with default config."""
        env = DifferentiableCatheterEnv()
        assert env is not None
        assert hasattr(env, 'differentiable_physics')
        assert env.differentiable_physics is not None

    def test_initialization_with_custom_config(self):
        """Test initialization with custom configuration."""
        config = DifferentiableCatheterEnvConfig(
            dt=0.1,
            max_steps=100,
            max_current=0.3
        )
        env = DifferentiableCatheterEnv(config=config)
        assert env.config.dt == 0.1
        assert env.config.max_steps == 100
        assert env.config.max_current == 0.3

    def test_reset_initializes_physics(self):
        """Test that reset initializes differentiable physics."""
        env = DifferentiableCatheterEnv()
        obs, info = env.reset()

        assert env.differentiable_physics.is_initialized
        assert obs is not None
        assert isinstance(obs, np.ndarray)


class TestStandardGymInterface:
    """Test standard Gymnasium interface compatibility."""

    def test_reset_returns_valid_observation(self):
        """Test that reset returns valid observation."""
        env = DifferentiableCatheterEnv()
        obs, info = env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert np.isfinite(obs).all()

    def test_step_returns_valid_outputs(self):
        """Test that standard step returns valid outputs."""
        env = DifferentiableCatheterEnv()
        env.reset()

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_multiple_steps(self):
        """Test that multiple steps work correctly."""
        env = DifferentiableCatheterEnv()
        env.reset()

        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            assert np.isfinite(obs).all()
            if terminated or truncated:
                break


class TestDifferentiableInterface:
    """Test differentiable step functionality."""

    def test_step_differentiable_returns_tensor(self):
        """Test that step_differentiable returns a tensor."""
        env = DifferentiableCatheterEnv()
        env.reset()

        action = torch.tensor([0.1, 0.0, 0.0], dtype=torch.float64)
        next_state = env.step_differentiable(action)

        assert isinstance(next_state, torch.Tensor)
        assert next_state.shape == (6,)
        assert not torch.isnan(next_state).any()

    def test_step_differentiable_gradients(self):
        """Test that gradients flow through step_differentiable."""
        env = DifferentiableCatheterEnv()
        env.reset()

        action = torch.tensor([0.1, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
        next_state = env.step_differentiable(action)

        loss = next_state[:3].sum()
        loss.backward()

        assert action.grad is not None
        assert not torch.isnan(action.grad).any()
        assert action.grad.abs().sum() > 0

    def test_step_differentiable_updates_state(self):
        """Test that step_differentiable updates environment state."""
        env = DifferentiableCatheterEnv()
        env.reset()

        initial_position = env.tip_position.copy()

        action = torch.tensor([0.1, 0.0, 0.0], dtype=torch.float64)
        next_state = env.step_differentiable(action)

        # State should have changed
        assert not np.allclose(env.tip_position, initial_position)
        # State should match output
        assert np.allclose(env.tip_position, next_state[:3].detach().numpy())


class TestTrajectoryOptimization:
    """Test trajectory optimization functionality."""

    def test_optimize_trajectory_basic(self):
        """Test basic trajectory optimization."""
        env = DifferentiableCatheterEnv()
        env.reset()

        target = np.array([5.0, 10.0, 90.0])

        # Short optimization for speed
        actions, loss = env.optimize_trajectory(
            target,
            horizon=5,
            num_iterations=10,
            learning_rate=0.01
        )

        assert actions.shape == (5, 3)
        assert np.isfinite(actions).all()
        assert np.abs(actions).max() <= env.config.max_current

    def test_optimize_trajectory_is_finite(self):
        """Test that optimization produces finite results."""
        env = DifferentiableCatheterEnv()
        env.reset()

        target = np.array([5.0, 10.0, 90.0])

        # Optimize
        actions, final_loss = env.optimize_trajectory(target, horizon=3, num_iterations=10)

        # Check that results are finite (main correctness check)
        assert np.isfinite(actions).all()
        assert np.isfinite(final_loss)
        assert final_loss >= 0  # Loss should be non-negative

    def test_compute_differentiable_loss(self):
        """Test differentiable loss computation."""
        env = DifferentiableCatheterEnv()
        env.reset()

        # Create trajectory as leaf tensor
        trajectory = torch.randn(5, 3, dtype=torch.float64) * 0.1
        trajectory.requires_grad_(True)
        targets = torch.randn(5, 3, dtype=torch.float64)

        loss = env.compute_differentiable_loss(trajectory, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert not torch.isnan(loss)

        # Test backward pass
        loss.backward()
        assert trajectory.grad is not None
        assert not torch.isnan(trajectory.grad).any()


class TestMPC:
    """Test Model Predictive Control functionality."""

    def test_plan_mpc_returns_action(self):
        """Test that MPC planning returns a valid action."""
        env = DifferentiableCatheterEnv()
        env.reset()

        target = np.array([5.0, 10.0, 90.0])

        action = env.plan_mpc(
            target,
            horizon=3,
            num_iterations=5
        )

        assert action.shape == (3,)
        assert np.isfinite(action).all()
        assert np.abs(action).max() <= env.config.max_current

    def test_plan_mpc_different_targets(self):
        """Test MPC with different targets produces different actions."""
        env = DifferentiableCatheterEnv()
        env.reset()

        target1 = np.array([5.0, 10.0, 90.0])
        action1 = env.plan_mpc(target1, horizon=3, num_iterations=5)

        env.reset()
        target2 = np.array([-5.0, -10.0, 90.0])
        action2 = env.plan_mpc(target2, horizon=3, num_iterations=5)

        # Actions should be different for different targets
        assert not np.allclose(action1, action2)


class TestJacobian:
    """Test Jacobian computation."""

    def test_get_action_jacobian_shape(self):
        """Test that Jacobian has correct shape."""
        env = DifferentiableCatheterEnv()
        env.reset()

        action = np.array([0.1, 0.0, 0.0])
        next_state, jacobian = env.get_action_jacobian(action)

        assert next_state.shape == (6,)
        assert jacobian.shape == (6, 3)
        assert np.isfinite(next_state).all()
        assert np.isfinite(jacobian).all()

    def test_get_action_jacobian_nonzero(self):
        """Test that Jacobian is non-zero."""
        env = DifferentiableCatheterEnv()
        env.reset()

        action = np.array([0.1, 0.0, 0.0])
        _, jacobian = env.get_action_jacobian(action)

        # At least some elements should be non-zero
        assert np.abs(jacobian).sum() > 1e-6


class TestConsistency:
    """Test consistency between interfaces."""

    def test_step_vs_step_differentiable_consistency(self):
        """
        Test that step() and step_differentiable() produce similar results.

        Note: They use different code paths (Option A vs Option C) so
        exact match is not expected, but they should be reasonably close.
        """
        # Create two separate environments to avoid state conflicts
        env1 = DifferentiableCatheterEnv()
        env1.reset(seed=42)

        env2 = DifferentiableCatheterEnv()
        env2.reset(seed=42)

        action_np = np.array([0.1, 0.0, 0.0])
        action_torch = torch.tensor([0.1, 0.0, 0.0], dtype=torch.float64)

        # Standard step
        obs1, _, _, _, _ = env1.step(action_np)
        pos1 = env1.tip_position

        # Differentiable step
        state2 = env2.step_differentiable(action_torch)
        pos2 = state2[:3].detach().numpy()

        # Positions should be reasonably close (within 2mm tolerance)
        pos_error = np.linalg.norm(pos1 - pos2)
        assert pos_error < 2.0, f"Position error {pos_error:.3f} mm exceeds 2mm threshold"


class TestValueGradients:
    """Test value function gradient computation."""

    def test_compute_value_gradients(self):
        """Test computing gradients through value function."""
        env = DifferentiableCatheterEnv()
        env.reset()

        # Simple value function: negative distance from origin
        def value_function(state):
            return -torch.norm(state[:3])

        action = np.array([0.1, 0.0, 0.0])
        grad = env.compute_value_gradients(action, value_function)

        assert grad.shape == (3,)
        assert np.isfinite(grad).all()
        # Gradient should be non-zero
        assert np.abs(grad).sum() > 1e-6


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_action(self):
        """Test with zero action."""
        env = DifferentiableCatheterEnv()
        env.reset()

        action = torch.zeros(3, dtype=torch.float64, requires_grad=True)
        next_state = env.step_differentiable(action)

        assert not torch.isnan(next_state).any()

    def test_max_action(self):
        """Test with maximum allowed action."""
        env = DifferentiableCatheterEnv()
        env.reset()

        action = torch.ones(3, dtype=torch.float64) * env.config.max_current
        action.requires_grad = True
        next_state = env.step_differentiable(action)

        assert not torch.isnan(next_state).any()

    def test_multiple_resets(self):
        """Test that multiple resets work correctly."""
        env = DifferentiableCatheterEnv()

        for _ in range(3):
            obs, info = env.reset()
            assert env.differentiable_physics.is_initialized
            assert np.isfinite(obs).all()


class TestDeviceSupport:
    """Test device (CPU/GPU) support."""

    def test_cpu_device(self):
        """Test with CPU device."""
        env = DifferentiableCatheterEnv(device="cpu")
        env.reset()

        action = torch.tensor([0.1, 0.0, 0.0], dtype=torch.float64)
        next_state = env.step_differentiable(action)

        assert next_state.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self):
        """Test with CUDA device (if available)."""
        env = DifferentiableCatheterEnv(device="cuda")
        env.reset()

        action = torch.tensor([0.1, 0.0, 0.0], dtype=torch.float64, device="cuda")
        next_state = env.step_differentiable(action)

        assert next_state.device.type == "cuda"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
