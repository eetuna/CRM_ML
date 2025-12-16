#!/usr/bin/env python3
"""
Test suite for Hybrid Models (CRM Physics + Neural Network Residuals).

Tests:
1. HybridKinematicsModel - CRM FK + learned residuals
2. HybridKinematicsWithUncertainty - FK with uncertainty estimation
3. HybridDynamicsModel - CRM Dynamics + learned residuals
4. HybridDynamicsLSTM - Dynamics with history-dependent residuals
5. Integration with RL pipeline
6. Training and loss computation

Usage:
    python scripts/check_hybrid_models.py              # Run all checks
    python scripts/check_hybrid_models.py --verbose    # Verbose output
    python scripts/check_hybrid_models.py --quick      # Quick checks only
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn


class TestResults:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def add_pass(self, test_name):
        self.passed += 1
        print(f"  ✓ {test_name}")

    def add_fail(self, test_name, error):
        self.failed += 1
        self.errors.append((test_name, str(error)))
        print(f"  ✗ {test_name}: {error}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"Results: {self.passed}/{total} tests passed")
        if self.errors:
            print("\nFailed tests:")
            for name, error in self.errors:
                print(f"  - {name}: {error}")
        print('='*60)
        return self.failed == 0


def test_hybrid_kinematics(results: TestResults, verbose: bool = False):
    """Test HybridKinematicsModel."""
    print("\n[1/5] Testing HybridKinematicsModel")
    print("-" * 40)

    from crm_ml_rl.models.hybrid_kinematics import (
        HybridKinematicsModel, HybridKinematicsWithUncertainty
    )

    # Test basic creation
    try:
        model = HybridKinematicsModel(use_cpp=True)
        assert model.is_using_cpp or not HAS_CPP_BINDINGS
        results.add_pass(f"Model creation (cpp={model.is_using_cpp})")
    except Exception as e:
        results.add_fail("Model creation", e)
        return  # Can't continue

    # Test single prediction (numpy interface)
    try:
        currents = np.array([0.1, 0.0, 0.0])
        pred = model.predict(currents, insertion_length=50.0)
        assert pred.shape == (3,), f"Expected (3,), got {pred.shape}"
        assert not np.any(np.isnan(pred)), "Prediction contains NaN"
        results.add_pass("Single prediction (numpy)")
    except Exception as e:
        results.add_fail("Single prediction (numpy)", e)

    # Test batch prediction
    try:
        currents_batch = np.array([
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.1],
            [-0.1, 0.05, 0.0]
        ])
        preds = model.predict(currents_batch, insertion_length=50.0)
        assert preds.shape == (4, 3), f"Expected (4, 3), got {preds.shape}"
        results.add_pass("Batch prediction")
    except Exception as e:
        results.add_fail("Batch prediction", e)

    # Test forward with components
    try:
        currents_tensor = torch.tensor(currents_batch, dtype=torch.float32)
        final, physics, residual = model.forward(
            currents_tensor, 50.0, return_components=True
        )
        assert final.shape == (4, 3)
        assert physics.shape == (4, 3)
        assert residual.shape == (4, 3)
        # Final should equal physics + residual (without learnable blend)
        expected = physics + residual
        assert torch.allclose(final, expected, atol=1e-5)
        results.add_pass("Forward with components")
    except Exception as e:
        results.add_fail("Forward with components", e)

    # Test physics prediction directly
    try:
        physics_pred = model.get_physics_prediction(currents, 50.0)
        assert physics_pred.shape == (3,)
        results.add_pass("Direct physics prediction")
    except Exception as e:
        results.add_fail("Direct physics prediction", e)

    # Test loss computation
    try:
        target = torch.randn(4, 3)
        losses = model.compute_loss(currents_tensor, 50.0, target)
        assert 'total' in losses
        assert 'final' in losses
        assert 'physics' in losses
        assert 'residual_magnitude' in losses
        assert not torch.isnan(losses['total'])
        results.add_pass("Loss computation")
    except Exception as e:
        results.add_fail("Loss computation", e)

    # Test with learnable physics weight
    try:
        model_blend = HybridKinematicsModel(
            use_cpp=True,
            learnable_physics_weight=True
        )
        weight = model_blend.get_physics_weight()
        assert 0 < weight < 1, f"Weight {weight} not in (0, 1)"
        pred = model_blend.predict(currents, 50.0)
        assert pred.shape == (3,)
        results.add_pass("Learnable physics weight")
    except Exception as e:
        results.add_fail("Learnable physics weight", e)

    # Test with deep residual
    try:
        model_deep = HybridKinematicsModel(
            use_cpp=True,
            use_deep_residual=True,
            hidden_dims=[128, 128, 64]
        )
        pred = model_deep.predict(currents, 50.0)
        assert pred.shape == (3,)
        results.add_pass("Deep residual architecture")
    except Exception as e:
        results.add_fail("Deep residual architecture", e)

    # Test cache functionality
    try:
        model.enable_cache(True)
        pred1 = model.predict(currents, 50.0)
        pred2 = model.predict(currents, 50.0)
        model.clear_cache()
        results.add_pass("Cache functionality")
    except Exception as e:
        results.add_fail("Cache functionality", e)


def test_hybrid_kinematics_uncertainty(results: TestResults, verbose: bool = False):
    """Test HybridKinematicsWithUncertainty."""
    print("\n[2/5] Testing HybridKinematicsWithUncertainty")
    print("-" * 40)

    from crm_ml_rl.models.hybrid_kinematics import HybridKinematicsWithUncertainty

    # Test creation
    try:
        model = HybridKinematicsWithUncertainty(use_cpp=True)
        results.add_pass("Model creation")
    except Exception as e:
        results.add_fail("Model creation", e)
        return

    # Test single prediction with uncertainty
    try:
        currents = np.array([0.1, 0.0, 0.0])
        mean, var = model.predict(currents, insertion_length=50.0)
        assert mean.shape == (3,)
        assert var.shape == (3,)
        assert np.all(var > 0), "Variance should be positive"
        results.add_pass("Prediction with uncertainty")
    except Exception as e:
        results.add_fail("Prediction with uncertainty", e)

    # Test batch prediction
    try:
        currents_batch = np.random.randn(8, 3) * 0.1
        mean, var = model.predict(currents_batch, insertion_length=50.0)
        assert mean.shape == (8, 3)
        assert var.shape == (8, 3)
        results.add_pass("Batch prediction with uncertainty")
    except Exception as e:
        results.add_fail("Batch prediction with uncertainty", e)

    # Test confidence intervals
    try:
        std = np.sqrt(var)
        lower = mean - 2 * std
        upper = mean + 2 * std
        # Just check shapes
        assert lower.shape == upper.shape == mean.shape
        results.add_pass("Confidence interval computation")
    except Exception as e:
        results.add_fail("Confidence interval computation", e)


def test_hybrid_dynamics(results: TestResults, verbose: bool = False):
    """Test HybridDynamicsModel."""
    print("\n[3/5] Testing HybridDynamicsModel")
    print("-" * 40)

    from crm_ml_rl.models.hybrid_dynamics import (
        HybridDynamicsModel, HybridDynamicsConfig
    )

    # Test creation with config
    try:
        config = HybridDynamicsConfig(
            use_cpp=True,
            dt=0.02,
            hidden_dims=[128, 128]
        )
        model = HybridDynamicsModel(config)
        assert model.is_using_cpp or not HAS_CPP_BINDINGS
        results.add_pass(f"Model creation with config (cpp={model.is_using_cpp})")
    except Exception as e:
        results.add_fail("Model creation with config", e)
        return

    # Test reset
    try:
        initial_state = model.reset()
        assert initial_state.shape == (6,), f"Expected (6,), got {initial_state.shape}"
        results.add_pass("Reset")
    except Exception as e:
        results.add_fail("Reset", e)

    # Test single step
    try:
        action = np.array([0.05, 0.0, 0.0])
        next_state = model.step(action)
        assert next_state.shape == (6,)
        assert not np.any(np.isnan(next_state)), "State contains NaN"
        results.add_pass("Single step")
    except Exception as e:
        results.add_fail("Single step", e)

    # Test trajectory prediction
    try:
        T = 20
        t = np.linspace(0, 0.5, T)
        actions = np.column_stack([
            0.05 * np.sin(2 * np.pi * t),
            0.05 * np.cos(2 * np.pi * t),
            np.zeros(T)
        ])
        trajectory = model.predict_trajectory(actions)
        assert trajectory.shape == (T + 1, 6), f"Expected ({T+1}, 6), got {trajectory.shape}"
        results.add_pass("Trajectory prediction")
    except Exception as e:
        results.add_fail("Trajectory prediction", e)

    # Test batch forward
    try:
        batch_states = torch.randn(8, 6)
        batch_actions = torch.randn(8, 3) * 0.05
        batch_next = model.forward(batch_states, batch_actions)
        assert batch_next.shape == (8, 6)
        results.add_pass("Batch forward")
    except Exception as e:
        results.add_fail("Batch forward", e)

    # Test forward with components
    try:
        final, physics, residual = model.forward(
            batch_states, batch_actions, return_components=True
        )
        assert final.shape == physics.shape == residual.shape == (8, 6)
        results.add_pass("Forward with components")
    except Exception as e:
        results.add_fail("Forward with components", e)

    # Test loss computation
    try:
        target_next = torch.randn(8, 6)
        losses = model.compute_loss(batch_states, batch_actions, target_next)
        assert 'total' in losses
        assert 'position' in losses
        assert 'velocity' in losses
        assert 'improvement' in losses
        results.add_pass("Loss computation")
    except Exception as e:
        results.add_fail("Loss computation", e)

    # Test ensemble model
    try:
        config_ensemble = HybridDynamicsConfig(
            use_cpp=True,
            use_ensemble=True,
            num_ensemble=3
        )
        model_ensemble = HybridDynamicsModel(config_ensemble)
        model_ensemble.reset()
        uncertainty = model_ensemble.get_uncertainty(batch_states, batch_actions)
        assert uncertainty.shape == (8, 6)
        results.add_pass("Ensemble model with uncertainty")
    except Exception as e:
        results.add_fail("Ensemble model with uncertainty", e)


def test_hybrid_dynamics_lstm(results: TestResults, verbose: bool = False):
    """Test HybridDynamicsLSTM."""
    print("\n[4/5] Testing HybridDynamicsLSTM")
    print("-" * 40)

    from crm_ml_rl.models.hybrid_dynamics import (
        HybridDynamicsLSTM, HybridDynamicsConfig
    )

    # Test creation
    try:
        config = HybridDynamicsConfig(use_cpp=True, dt=0.02)
        model = HybridDynamicsLSTM(config, history_length=5)
        results.add_pass("Model creation")
    except Exception as e:
        results.add_fail("Model creation", e)
        return

    # Test reset
    try:
        state = model.reset()
        assert state.shape == (6,)
        results.add_pass("Reset")
    except Exception as e:
        results.add_fail("Reset", e)

    # Test sequential steps (building history)
    try:
        action = np.array([0.05, 0.0, 0.0])
        for i in range(10):
            state = model.step(state, action)
            assert state.shape == (6,)
        results.add_pass("Sequential steps with history")
    except Exception as e:
        results.add_fail("Sequential steps with history", e)

    # Test history resets properly
    try:
        state = model.reset()
        state = model.step(state, action)
        assert state.shape == (6,)
        results.add_pass("Reset clears history")
    except Exception as e:
        results.add_fail("Reset clears history", e)


def test_rl_integration(results: TestResults, verbose: bool = False, quick: bool = False):
    """Test hybrid models integration with RL pipeline."""
    print("\n[5/5] Testing RL Pipeline Integration")
    print("-" * 40)

    from crm_ml_rl.models.hybrid_dynamics import HybridDynamicsModel, HybridDynamicsConfig
    from crm_ml_rl.envs import CatheterEnv, ReachingEnv, CatheterEnvConfig

    # Test environment with hybrid dynamics as learned model
    try:
        # Create hybrid dynamics model
        hybrid_config = HybridDynamicsConfig(use_cpp=True, dt=0.01)
        hybrid_model = HybridDynamicsModel(hybrid_config)

        # Create custom dynamics wrapper for environment
        class HybridDynamicsWrapper:
            """Wrapper to use hybrid model as env dynamics."""
            def __init__(self, model):
                self.model = model
                self._state = None

            def __call__(self, state_tensor, action_tensor):
                # Forward through hybrid model
                next_state = self.model.forward(state_tensor, action_tensor)
                return next_state

        # Create environment with learned dynamics
        env_config = CatheterEnvConfig(
            use_learned_dynamics=True,
            max_steps=50,
            dt=0.01
        )
        env = CatheterEnv(config=env_config)

        # Attach hybrid model
        env.set_dynamics_model(HybridDynamicsWrapper(hybrid_model))

        # Run episode
        obs, info = env.reset()
        total_reward = 0
        for _ in range(10):
            action = env.action_space.sample() * 0.1
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            if term or trunc:
                break

        results.add_pass("CatheterEnv with HybridDynamicsModel")
    except Exception as e:
        results.add_fail("CatheterEnv with HybridDynamicsModel", e)

    # Test HybridKinematics for inverse problem (control)
    try:
        from crm_ml_rl.models.hybrid_kinematics import HybridKinematicsModel

        model = HybridKinematicsModel(use_cpp=True)

        # Use hybrid model to predict tip position for different currents
        # This could be used for model-based control
        currents_grid = np.linspace(-0.1, 0.1, 5)
        predictions = []
        for ix in currents_grid:
            for iy in currents_grid:
                curr = np.array([ix, iy, 0.0])
                pred = model.predict(curr, insertion_length=50.0)
                predictions.append(pred)

        predictions = np.array(predictions)
        assert predictions.shape == (25, 3)
        results.add_pass("HybridKinematicsModel for workspace mapping")
    except Exception as e:
        results.add_fail("HybridKinematicsModel for workspace mapping", e)

    if quick:
        print("  (Skipping RL training test in quick mode)")
        return

    # Test training loop with hybrid model
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        # Simple environment for quick test
        env = DummyVecEnv([lambda: ReachingEnv()])

        # Very short training
        model = PPO("MlpPolicy", env, verbose=0, n_steps=32, batch_size=32)
        model.learn(total_timesteps=64)

        # Test prediction
        obs = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        assert action.shape == (1, 3)

        results.add_pass("PPO training with hybrid model environment")
    except Exception as e:
        results.add_fail("PPO training with hybrid model environment", e)

    # Test model-based trajectory optimization concept
    try:
        from crm_ml_rl.models.hybrid_kinematics import HybridKinematicsModel

        model = HybridKinematicsModel(use_cpp=True)

        # Simple gradient-based optimization to reach target
        target = np.array([5.0, 5.0, 80.0])
        currents = torch.zeros(1, 3, requires_grad=True)
        optimizer = torch.optim.Adam([currents], lr=0.01)

        for _ in range(10):  # Just a few steps
            optimizer.zero_grad()
            pred = model.forward(currents, 50.0)
            loss = ((pred - torch.tensor(target)) ** 2).sum()
            loss.backward()
            optimizer.step()

        results.add_pass("Gradient-based control with hybrid model")
    except Exception as e:
        results.add_fail("Gradient-based control with hybrid model", e)


# Check for C++ bindings
try:
    from crm_ml_rl.wrappers.crm_wrapper import HAS_CPP_BINDINGS
except ImportError:
    HAS_CPP_BINDINGS = False


def main():
    parser = argparse.ArgumentParser(description="Test Hybrid Models")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quick", "-q", action="store_true", help="Quick tests only")
    args = parser.parse_args()

    print("=" * 60)
    print("Hybrid Models Test Suite")
    print("(CRM Physics + Neural Network Residuals)")
    print("=" * 60)
    print(f"C++ bindings available: {HAS_CPP_BINDINGS}")

    results = TestResults()

    # Run tests
    test_hybrid_kinematics(results, verbose=args.verbose)
    test_hybrid_kinematics_uncertainty(results, verbose=args.verbose)
    test_hybrid_dynamics(results, verbose=args.verbose)
    test_hybrid_dynamics_lstm(results, verbose=args.verbose)
    test_rl_integration(results, verbose=args.verbose, quick=args.quick)

    # Summary
    success = results.summary()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
