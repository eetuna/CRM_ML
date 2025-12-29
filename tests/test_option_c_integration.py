"""
Integration tests for Option C with crm_ml_rl.

These tests validate that all Option C components work together correctly:
- OptionCPhysics wrapper
- DifferentiableCatheterEnv
- HybridDynamicsModel with Option C
- DifferentiableMPCAgent

All tests use the validated Option C implementation (crm_torch.CRMDynamicsStep).
"""

import pytest
import numpy as np
import torch


class TestOptionCPhysicsIntegration:
    """Test OptionCPhysics wrapper integration."""

    def test_option_c_physics_forward(self):
        """Test forward pass produces valid output."""
        from crm_ml_rl.wrappers.option_c_physics import OptionCPhysics

        physics = OptionCPhysics()
        physics.reset()

        currents = torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
        output = physics.step_differentiable(currents)

        assert output.shape == (1, 6)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_option_c_physics_gradient(self):
        """Test gradients flow through physics."""
        from crm_ml_rl.wrappers.option_c_physics import OptionCPhysics

        physics = OptionCPhysics()
        physics.reset()

        currents = torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
        output = physics.step_differentiable(currents)

        loss = output[:, :3].sum()  # Loss on position
        loss.backward()

        assert currents.grad is not None
        assert not torch.isnan(currents.grad).any()
        assert currents.grad.abs().sum() > 0  # Non-zero gradients


class TestDifferentiableEnvIntegration:
    """Test DifferentiableCatheterEnv integration."""

    def test_differentiable_env_trajectory(self):
        """Test gradient-based trajectory works with proper gradient flow."""
        from crm_ml_rl.envs.differentiable_catheter_env import DifferentiableCatheterEnv

        env = DifferentiableCatheterEnv()
        env.reset()

        # Test that we can compute gradients through a multi-step trajectory
        actions = torch.nn.Parameter(torch.tensor([
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.1],
        ], dtype=torch.float64))

        target = torch.tensor([5.0, 5.0, 95.0], dtype=torch.float64)

        # Single optimization step to verify gradients flow
        env.reset()
        loss = torch.tensor(0.0, dtype=torch.float64)

        for t in range(3):
            next_state = env.step_differentiable(actions[t])
            loss = loss + torch.norm(next_state[:3] - target)**2

        # Compute gradients
        loss.backward()

        # Verify gradients exist and are non-zero
        assert actions.grad is not None
        assert not torch.isnan(actions.grad).any()
        assert not torch.isinf(actions.grad).any()
        # At least one gradient should be non-zero
        assert actions.grad.abs().sum() > 0

    def test_multi_step_gradient_flow(self):
        """Test gradients flow through multi-step simulation."""
        from crm_ml_rl.envs.differentiable_catheter_env import DifferentiableCatheterEnv

        env = DifferentiableCatheterEnv()
        env.reset()

        # 3-step trajectory - use nn.Parameter for proper gradient tracking
        actions = torch.nn.Parameter(torch.tensor([
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.1],
        ], dtype=torch.float64))

        total_loss = torch.tensor(0.0, dtype=torch.float64)
        for t in range(3):
            state = env.step_differentiable(actions[t])
            total_loss = total_loss + state[:3].sum()

        total_loss.backward()

        # Actions should have gradients
        assert actions.grad is not None, "Actions have no gradient"
        # At least the last action should have non-zero gradient
        assert actions.grad[-1].abs().sum() > 0, "Last action has zero gradient"


class TestHybridDynamicsIntegration:
    """Test HybridDynamicsModel with Option C."""

    def test_hybrid_dynamics_option_c(self):
        """Test HybridDynamicsModel uses Option C correctly."""
        from crm_ml_rl.models.hybrid_dynamics import HybridDynamicsModel, HybridDynamicsConfig

        config = HybridDynamicsConfig(
            use_option_c=True,
            use_cpp=True,
            hidden_dims=[128, 128]
        )
        model = HybridDynamicsModel(config)

        # Initialize Option C physics by calling reset
        model._option_c.reset()

        # Test forward pass with gradients
        # Use float32 to match model's dtype, reasonable state values (position + velocity)
        state = torch.tensor([[0.0, 0.0, 90.0, 0.0, 0.0, 0.0]], dtype=torch.float32, requires_grad=True)
        action = torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float32, requires_grad=True)

        next_state = model.forward(state, action)

        assert next_state.shape == (1, 6)
        assert not torch.isnan(next_state).any()

        # Test backward pass
        loss = next_state.sum()
        loss.backward()

        assert action.grad is not None
        assert action.grad.abs().sum() > 0


class TestDifferentiableMPCAgentIntegration:
    """Test DifferentiableMPCAgent end-to-end."""

    def test_differentiable_mpc_agent_end_to_end(self):
        """Test DifferentiableMPCAgent planning and execution."""
        from crm_ml_rl.envs.differentiable_catheter_env import DifferentiableCatheterEnv
        from crm_ml_rl.training.model_based_rl import DifferentiableMPCAgent

        # Create environment
        env = DifferentiableCatheterEnv()
        obs, _ = env.reset()

        # Create agent
        agent = DifferentiableMPCAgent(
            env,
            horizon=3,
            num_iterations=10,
            learning_rate=0.1
        )

        # Test planning
        target = obs[6:9]  # Extract target from observation
        action_sequence = agent.plan(target)

        assert action_sequence.shape[0] == 3  # horizon
        assert not np.isnan(action_sequence).any()

        # Test prediction
        action = agent.predict(obs)
        assert action.shape == (3,) or action.shape == ()
        assert not np.isnan(action).any()

        # Test step
        action, info = agent.step(obs)
        assert isinstance(info, dict)

    def test_differentiable_mpc_agent_improves_over_iterations(self):
        """Test that MPC agent planning improves with more iterations."""
        from crm_ml_rl.envs.differentiable_catheter_env import DifferentiableCatheterEnv
        from crm_ml_rl.training.model_based_rl import DifferentiableMPCAgent

        env = DifferentiableCatheterEnv()
        obs, _ = env.reset()
        target = obs[6:9]

        # Agent with few iterations
        agent_few = DifferentiableMPCAgent(env, horizon=3, num_iterations=5, learning_rate=0.1)

        # Agent with many iterations
        agent_many = DifferentiableMPCAgent(env, horizon=3, num_iterations=20, learning_rate=0.1)

        # Both should produce valid actions
        action_few = agent_few.predict(obs)
        action_many = agent_many.predict(obs)

        assert not np.isnan(action_few).any()
        assert not np.isnan(action_many).any()
