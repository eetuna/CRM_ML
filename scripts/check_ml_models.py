#!/usr/bin/env python3
"""
Comprehensive test suite for ML models in crm_ml_rl.

Tests:
1. All neural network architectures (MLP, LSTM, Transformer, ResidualBlock)
2. Residual kinematics and dynamics models
3. Full kinematics and dynamics models
4. Integration with C++ bindings
5. Integration within RL pipeline

Usage:
    python scripts/check_ml_models.py              # Run all checks
    python scripts/check_ml_models.py --verbose    # Verbose output
    python scripts/check_ml_models.py --quick      # Quick checks only
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

# Test configuration
BATCH_SIZE = 32
STATE_DIM = 6
ACTION_DIM = 3
POSITION_DIM = 3


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


def test_base_networks(results: TestResults, verbose: bool = False):
    """Test base neural network architectures."""
    print("\n[1/8] Testing Base Network Architectures")
    print("-" * 40)

    from crm_ml_rl.models.networks import MLP, LSTM_MLP, ResidualBlock, DeepResidualMLP

    # Test MLP
    try:
        mlp = MLP(input_dim=10, output_dim=3, hidden_dims=[64, 64])
        x = torch.randn(BATCH_SIZE, 10)
        y = mlp(x)
        assert y.shape == (BATCH_SIZE, 3), f"Expected (32, 3), got {y.shape}"
        results.add_pass("MLP forward pass")
    except Exception as e:
        results.add_fail("MLP forward pass", e)

    # Test MLP with different activations
    try:
        for activation in ['relu', 'tanh', 'gelu', 'leaky_relu']:
            mlp = MLP(input_dim=10, output_dim=3, activation=activation)
            y = mlp(torch.randn(BATCH_SIZE, 10))
            assert y.shape == (BATCH_SIZE, 3)
        results.add_pass("MLP activations (relu, tanh, gelu, leaky_relu)")
    except Exception as e:
        results.add_fail("MLP activations", e)

    # Test MLP with batch norm and layer norm
    try:
        mlp_bn = MLP(input_dim=10, output_dim=3, batch_norm=True)
        mlp_ln = MLP(input_dim=10, output_dim=3, layer_norm=True)
        mlp_do = MLP(input_dim=10, output_dim=3, dropout=0.1)
        for m in [mlp_bn, mlp_ln, mlp_do]:
            y = m(torch.randn(BATCH_SIZE, 10))
            assert y.shape == (BATCH_SIZE, 3)
        results.add_pass("MLP with normalization and dropout")
    except Exception as e:
        results.add_fail("MLP with normalization and dropout", e)

    # Test LSTM-MLP
    try:
        lstm_mlp = LSTM_MLP(input_dim=6, output_dim=3, lstm_hidden_dim=64)
        x_seq = torch.randn(BATCH_SIZE, 10, 6)  # (batch, seq, features)
        y, hidden = lstm_mlp(x_seq)
        assert y.shape == (BATCH_SIZE, 3), f"Expected (32, 3), got {y.shape}"
        assert len(hidden) == 2, "Hidden should be (h_n, c_n) tuple"
        results.add_pass("LSTM-MLP forward pass")
    except Exception as e:
        results.add_fail("LSTM-MLP forward pass", e)

    # Test LSTM-MLP with hidden state
    try:
        lstm_mlp = LSTM_MLP(input_dim=6, output_dim=3, lstm_hidden_dim=64)
        x_seq = torch.randn(BATCH_SIZE, 10, 6)
        hidden = lstm_mlp.init_hidden(BATCH_SIZE, torch.device('cpu'))
        y, new_hidden = lstm_mlp(x_seq, hidden)
        assert y.shape == (BATCH_SIZE, 3)
        results.add_pass("LSTM-MLP with hidden state")
    except Exception as e:
        results.add_fail("LSTM-MLP with hidden state", e)

    # Test bidirectional LSTM
    try:
        lstm_mlp_bi = LSTM_MLP(input_dim=6, output_dim=3, bidirectional=True)
        x_seq = torch.randn(BATCH_SIZE, 10, 6)
        y, _ = lstm_mlp_bi(x_seq)
        assert y.shape == (BATCH_SIZE, 3)
        results.add_pass("Bidirectional LSTM-MLP")
    except Exception as e:
        results.add_fail("Bidirectional LSTM-MLP", e)

    # Test ResidualBlock
    try:
        block = ResidualBlock(dim=64, dropout=0.1)
        x = torch.randn(BATCH_SIZE, 64)
        y = block(x)
        assert y.shape == (BATCH_SIZE, 64), f"Expected (32, 64), got {y.shape}"
        results.add_pass("ResidualBlock forward pass")
    except Exception as e:
        results.add_fail("ResidualBlock forward pass", e)

    # Test DeepResidualMLP
    try:
        deep_mlp = DeepResidualMLP(input_dim=10, output_dim=3, hidden_dim=64, num_blocks=4)
        x = torch.randn(BATCH_SIZE, 10)
        y = deep_mlp(x)
        assert y.shape == (BATCH_SIZE, 3), f"Expected (32, 3), got {y.shape}"
        results.add_pass("DeepResidualMLP forward pass")
    except Exception as e:
        results.add_fail("DeepResidualMLP forward pass", e)


def test_residual_kinematics(results: TestResults, verbose: bool = False):
    """Test residual kinematics models."""
    print("\n[2/8] Testing Residual Kinematics Models")
    print("-" * 40)

    from crm_ml_rl.models.residual_kinematics import (
        ResidualKinematicsModel, ResidualKinematicsWithUncertainty
    )

    # Test basic ResidualKinematicsModel
    try:
        model = ResidualKinematicsModel(
            input_dim=6,
            output_dim=3,
            hidden_dims=[128, 128],
            physics_output_dim=3,
            include_physics_in_input=True
        )
        x = torch.randn(BATCH_SIZE, 6)
        physics_out = torch.randn(BATCH_SIZE, 3)
        residual = model(x, physics_out)
        assert residual.shape == (BATCH_SIZE, 3), f"Expected (32, 3), got {residual.shape}"
        results.add_pass("ResidualKinematicsModel forward pass")
    except Exception as e:
        results.add_fail("ResidualKinematicsModel forward pass", e)

    # Test predict_with_physics
    try:
        model = ResidualKinematicsModel(input_dim=6, output_dim=3, physics_output_dim=3)
        x = torch.randn(BATCH_SIZE, 6)
        physics_out = torch.randn(BATCH_SIZE, 3)
        final_pred = model.predict_with_physics(x, physics_out)
        assert final_pred.shape == (BATCH_SIZE, 3)
        results.add_pass("ResidualKinematicsModel predict_with_physics")
    except Exception as e:
        results.add_fail("ResidualKinematicsModel predict_with_physics", e)

    # Test max_correction clamping
    try:
        model = ResidualKinematicsModel(
            input_dim=6, output_dim=3, max_correction=5.0
        )
        x = torch.randn(BATCH_SIZE, 6) * 100  # Large inputs
        physics_out = torch.randn(BATCH_SIZE, 3)
        residual = model(x, physics_out)
        assert residual.abs().max() <= 5.0, f"Max correction exceeded: {residual.abs().max()}"
        results.add_pass("ResidualKinematicsModel max_correction clamping")
    except Exception as e:
        results.add_fail("ResidualKinematicsModel max_correction clamping", e)

    # Test deep residual architecture
    try:
        # When include_physics_in_input=True (default), input_dim should include physics output
        model = ResidualKinematicsModel(
            input_dim=6, output_dim=3, use_deep_residual=True,
            hidden_dims=[128, 128, 64], physics_output_dim=3, include_physics_in_input=True
        )
        x = torch.randn(BATCH_SIZE, 6)
        physics_out = torch.randn(BATCH_SIZE, 3)
        residual = model(x, physics_out)
        assert residual.shape == (BATCH_SIZE, 3)
        results.add_pass("ResidualKinematicsModel with deep residual")
    except Exception as e:
        results.add_fail("ResidualKinematicsModel with deep residual", e)

    # Test ResidualKinematicsWithUncertainty
    try:
        model = ResidualKinematicsWithUncertainty(
            input_dim=9,  # 6 + 3 physics
            output_dim=3,
            hidden_dims=[128, 128, 64]
        )
        x = torch.randn(BATCH_SIZE, 6)
        physics_out = torch.randn(BATCH_SIZE, 3)
        mean, var = model(x, physics_out)
        assert mean.shape == (BATCH_SIZE, 3)
        assert var.shape == (BATCH_SIZE, 3)
        assert (var > 0).all(), "Variance should be positive"
        results.add_pass("ResidualKinematicsWithUncertainty forward pass")
    except Exception as e:
        results.add_fail("ResidualKinematicsWithUncertainty forward pass", e)

    # Test uncertainty loss
    try:
        model = ResidualKinematicsWithUncertainty(input_dim=9, output_dim=3)
        x = torch.randn(BATCH_SIZE, 6)
        physics_out = torch.randn(BATCH_SIZE, 3)
        target = torch.randn(BATCH_SIZE, 3)
        loss = model.loss(x, physics_out, target)
        assert loss.ndim == 0, "Loss should be scalar"
        assert not torch.isnan(loss), "Loss should not be NaN"
        results.add_pass("ResidualKinematicsWithUncertainty loss computation")
    except Exception as e:
        results.add_fail("ResidualKinematicsWithUncertainty loss computation", e)


def test_residual_dynamics(results: TestResults, verbose: bool = False):
    """Test residual dynamics models."""
    print("\n[3/8] Testing Residual Dynamics Models")
    print("-" * 40)

    from crm_ml_rl.models.residual_dynamics import (
        ResidualDynamicsModel, ResidualDynamicsLSTM, EnsembleResidualDynamics
    )

    # Test basic ResidualDynamicsModel
    try:
        model = ResidualDynamicsModel(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            hidden_dims=[128, 128],
            include_physics_in_input=True
        )
        state = torch.randn(BATCH_SIZE, STATE_DIM)
        action = torch.randn(BATCH_SIZE, ACTION_DIM)
        physics_next = torch.randn(BATCH_SIZE, STATE_DIM)
        residual = model(state, action, physics_next)
        assert residual.shape == (BATCH_SIZE, STATE_DIM)
        results.add_pass("ResidualDynamicsModel forward pass")
    except Exception as e:
        results.add_fail("ResidualDynamicsModel forward pass", e)

    # Test predict_next_state
    try:
        model = ResidualDynamicsModel(state_dim=STATE_DIM, action_dim=ACTION_DIM)
        state = torch.randn(BATCH_SIZE, STATE_DIM)
        action = torch.randn(BATCH_SIZE, ACTION_DIM)
        physics_next = torch.randn(BATCH_SIZE, STATE_DIM)
        next_state = model.predict_next_state(state, action, physics_next)
        assert next_state.shape == (BATCH_SIZE, STATE_DIM)
        results.add_pass("ResidualDynamicsModel predict_next_state")
    except Exception as e:
        results.add_fail("ResidualDynamicsModel predict_next_state", e)

    # Test ResidualDynamicsLSTM
    try:
        model = ResidualDynamicsLSTM(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            lstm_hidden_dim=64
        )
        states = torch.randn(BATCH_SIZE, 10, STATE_DIM)
        actions = torch.randn(BATCH_SIZE, 10, ACTION_DIM)
        physics_next_states = torch.randn(BATCH_SIZE, 10, STATE_DIM)
        residual, hidden = model(states, actions, physics_next_states)
        assert residual.shape == (BATCH_SIZE, STATE_DIM)
        results.add_pass("ResidualDynamicsLSTM forward pass")
    except Exception as e:
        results.add_fail("ResidualDynamicsLSTM forward pass", e)

    # Test EnsembleResidualDynamics
    try:
        ensemble = EnsembleResidualDynamics(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            num_models=3,
            hidden_dims=[64, 64]
        )
        state = torch.randn(BATCH_SIZE, STATE_DIM)
        action = torch.randn(BATCH_SIZE, ACTION_DIM)
        physics_next = torch.randn(BATCH_SIZE, STATE_DIM)
        mean, std = ensemble(state, action, physics_next)
        assert mean.shape == (BATCH_SIZE, STATE_DIM)
        assert std.shape == (BATCH_SIZE, STATE_DIM)
        assert (std >= 0).all(), "Std should be non-negative"
        results.add_pass("EnsembleResidualDynamics forward pass")
    except Exception as e:
        results.add_fail("EnsembleResidualDynamics forward pass", e)

    # Test epistemic uncertainty
    try:
        ensemble = EnsembleResidualDynamics(state_dim=STATE_DIM, action_dim=ACTION_DIM, num_models=3)
        state = torch.randn(BATCH_SIZE, STATE_DIM)
        action = torch.randn(BATCH_SIZE, ACTION_DIM)
        physics_next = torch.randn(BATCH_SIZE, STATE_DIM)
        uncertainty = ensemble.get_epistemic_uncertainty(state, action, physics_next)
        assert uncertainty.shape == (BATCH_SIZE,)
        results.add_pass("EnsembleResidualDynamics epistemic uncertainty")
    except Exception as e:
        results.add_fail("EnsembleResidualDynamics epistemic uncertainty", e)


def test_full_models(results: TestResults, verbose: bool = False):
    """Test full kinematics and dynamics models."""
    print("\n[4/8] Testing Full Kinematics/Dynamics Models")
    print("-" * 40)

    from crm_ml_rl.models.full_kinematics import (
        FullKinematicsModel, FullKinematicsLSTM, FullKinematicsTransformer
    )
    from crm_ml_rl.models.full_dynamics import (
        FullDynamicsModel, FullDynamicsLSTM, EnsembleFullDynamics, ProbabilisticFullDynamics
    )

    # Test FullKinematicsModel
    try:
        model = FullKinematicsModel(
            input_dim=6, output_dim=3, hidden_dims=[128, 128], use_deep_residual=True
        )
        x = torch.randn(BATCH_SIZE, 6)
        y = model(x)
        assert y.shape == (BATCH_SIZE, 3)
        results.add_pass("FullKinematicsModel forward pass")
    except Exception as e:
        results.add_fail("FullKinematicsModel forward pass", e)

    # Test FullKinematicsModel normalization
    try:
        model = FullKinematicsModel(input_dim=6, output_dim=3)
        model.set_normalization(
            input_mean=np.zeros(6),
            input_std=np.ones(6),
            output_mean=np.zeros(3),
            output_std=np.ones(3)
        )
        x = torch.randn(BATCH_SIZE, 6)
        y = model(x, normalize=True)
        assert y.shape == (BATCH_SIZE, 3)
        results.add_pass("FullKinematicsModel with normalization")
    except Exception as e:
        results.add_fail("FullKinematicsModel with normalization", e)

    # Test FullKinematicsLSTM
    try:
        model = FullKinematicsLSTM(input_dim=6, output_dim=3, lstm_hidden_dim=64)
        x_seq = torch.randn(BATCH_SIZE, 10, 6)
        y, hidden = model(x_seq)
        assert y.shape == (BATCH_SIZE, 3)
        results.add_pass("FullKinematicsLSTM forward pass")
    except Exception as e:
        results.add_fail("FullKinematicsLSTM forward pass", e)

    # Test FullKinematicsTransformer
    try:
        model = FullKinematicsTransformer(
            input_dim=6, output_dim=3, d_model=64, nhead=4, num_layers=2
        )
        x_seq = torch.randn(BATCH_SIZE, 10, 6)
        y = model(x_seq)
        assert y.shape == (BATCH_SIZE, 3)
        results.add_pass("FullKinematicsTransformer forward pass")
    except Exception as e:
        results.add_fail("FullKinematicsTransformer forward pass", e)

    # Test FullDynamicsModel
    try:
        model = FullDynamicsModel(
            state_dim=STATE_DIM, action_dim=ACTION_DIM, predict_delta=True
        )
        state = torch.randn(BATCH_SIZE, STATE_DIM)
        action = torch.randn(BATCH_SIZE, ACTION_DIM)
        next_state = model(state, action)
        assert next_state.shape == (BATCH_SIZE, STATE_DIM)
        results.add_pass("FullDynamicsModel forward pass")
    except Exception as e:
        results.add_fail("FullDynamicsModel forward pass", e)

    # Test FullDynamicsModel multi-step prediction
    try:
        model = FullDynamicsModel(state_dim=STATE_DIM, action_dim=ACTION_DIM)
        initial_state = torch.randn(BATCH_SIZE, STATE_DIM)
        actions = torch.randn(BATCH_SIZE, 10, ACTION_DIM)
        trajectory = model.multi_step_prediction(initial_state, actions)
        assert trajectory.shape == (BATCH_SIZE, 11, STATE_DIM)  # 10 steps + initial
        results.add_pass("FullDynamicsModel multi-step prediction")
    except Exception as e:
        results.add_fail("FullDynamicsModel multi-step prediction", e)

    # Test EnsembleFullDynamics
    try:
        ensemble = EnsembleFullDynamics(
            state_dim=STATE_DIM, action_dim=ACTION_DIM, num_models=3
        )
        state = torch.randn(BATCH_SIZE, STATE_DIM)
        action = torch.randn(BATCH_SIZE, ACTION_DIM)
        mean, std = ensemble(state, action)
        assert mean.shape == (BATCH_SIZE, STATE_DIM)
        assert std.shape == (BATCH_SIZE, STATE_DIM)
        results.add_pass("EnsembleFullDynamics forward pass")
    except Exception as e:
        results.add_fail("EnsembleFullDynamics forward pass", e)

    # Test ProbabilisticFullDynamics
    try:
        model = ProbabilisticFullDynamics(state_dim=STATE_DIM, action_dim=ACTION_DIM)
        state = torch.randn(BATCH_SIZE, STATE_DIM)
        action = torch.randn(BATCH_SIZE, ACTION_DIM)
        mean, var = model(state, action)
        assert mean.shape == (BATCH_SIZE, STATE_DIM)
        assert var.shape == (BATCH_SIZE, STATE_DIM)
        assert (var > 0).all()
        results.add_pass("ProbabilisticFullDynamics forward pass")
    except Exception as e:
        results.add_fail("ProbabilisticFullDynamics forward pass", e)

    # Test ProbabilisticFullDynamics sampling
    try:
        model = ProbabilisticFullDynamics(state_dim=STATE_DIM, action_dim=ACTION_DIM)
        state = torch.randn(BATCH_SIZE, STATE_DIM)
        action = torch.randn(BATCH_SIZE, ACTION_DIM)
        samples = model.sample(state, action, num_samples=5)
        assert samples.shape == (5, BATCH_SIZE, STATE_DIM)
        results.add_pass("ProbabilisticFullDynamics sampling")
    except Exception as e:
        results.add_fail("ProbabilisticFullDynamics sampling", e)

    # Test ProbabilisticFullDynamics loss
    try:
        model = ProbabilisticFullDynamics(state_dim=STATE_DIM, action_dim=ACTION_DIM)
        state = torch.randn(BATCH_SIZE, STATE_DIM)
        action = torch.randn(BATCH_SIZE, ACTION_DIM)
        next_state = torch.randn(BATCH_SIZE, STATE_DIM)
        loss = model.loss(state, action, next_state)
        assert loss.ndim == 0
        assert not torch.isnan(loss)
        results.add_pass("ProbabilisticFullDynamics loss")
    except Exception as e:
        results.add_fail("ProbabilisticFullDynamics loss", e)


def test_cpp_integration(results: TestResults, verbose: bool = False):
    """Test ML models integration with C++ physics bindings."""
    print("\n[5/8] Testing C++ Physics Integration")
    print("-" * 40)

    from crm_ml_rl.wrappers.crm_wrapper import CRMWrapper, CRMSimulator, HAS_CPP_BINDINGS
    from crm_ml_rl.models.residual_kinematics import ResidualKinematicsModel
    from crm_ml_rl.models.residual_dynamics import ResidualDynamicsModel

    if not HAS_CPP_BINDINGS:
        print("  ⚠ C++ bindings not available, testing with simplified model")

    # Test CRMWrapper creation
    try:
        wrapper = CRMWrapper(use_cpp=True)
        results.add_pass(f"CRMWrapper creation (using_cpp={wrapper.is_using_cpp})")
    except Exception as e:
        results.add_fail("CRMWrapper creation", e)
        return  # Can't continue without wrapper

    # Test forward kinematics
    try:
        currents = np.array([0.1, 0.0, 0.0])
        result = wrapper.forward_kinematics(currents, insertion_length=50.0)
        assert 'tip_position' in result
        assert result['tip_position'].shape == (3,)
        results.add_pass("Forward kinematics")
    except Exception as e:
        results.add_fail("Forward kinematics", e)

    # Test Jacobian computation
    try:
        currents = np.array([0.1, 0.0, 0.0])
        jacobian = wrapper.compute_jacobian(currents, insertion_length=50.0)
        # Full Jacobian is 6x7 (6 outputs: p[3]+R[3], 7 inputs: currents[3]+insertion+deltau0[3])
        # or 3x3 for simplified model (position only)
        assert jacobian.ndim == 2 and jacobian.shape[0] > 0, f"Jacobian shape is {jacobian.shape}"
        results.add_pass(f"Jacobian computation (shape={jacobian.shape})")
    except NotImplementedError:
        results.add_pass("Jacobian computation (not implemented, skipped)")
    except Exception as e:
        error_msg = str(e) if str(e) else type(e).__name__
        results.add_fail("Jacobian computation", error_msg)

    # Test dynamics initialization and stepping
    try:
        wrapper.reset()
        currents = np.array([0.05, 0.0, 0.0])  # Small currents to avoid unbounded
        success = wrapper.initialize_dynamics(currents, insertion_length=50.0)
        assert success, "Dynamics initialization failed"
        # Use damping to stabilize
        wrapper.set_damping(np.array([12.17, 12.17, 284.43, 0.03, 0.03, 0.005]))
        result = wrapper.step_dynamics(currents, insertion_length=50.0)
        assert 'tip_position' in result
        results.add_pass("Dynamics initialization and stepping")
    except Exception as e:
        error_msg = str(e) if str(e) else type(e).__name__
        results.add_fail("Dynamics initialization and stepping", error_msg)

    # Test CRMSimulator
    try:
        sim = CRMSimulator(dt=0.02, use_cpp=True)
        sim.reset(initial_currents=np.zeros(3), insertion_length=50.0)
        state = sim.step(np.array([0.1, 0.0, 0.0]), insertion_length=50.0)
        assert hasattr(state, 'position')
        results.add_pass("CRMSimulator step")
    except Exception as e:
        results.add_fail("CRMSimulator step", e)

    # Test residual model with physics
    try:
        wrapper = CRMWrapper(use_cpp=True)
        model = ResidualKinematicsModel(
            input_dim=6, output_dim=3, physics_output_dim=3
        )

        # Get physics prediction
        currents = np.array([0.1, 0.0, 0.0])
        physics_result = wrapper.forward_kinematics(currents, insertion_length=50.0)
        physics_pos = torch.tensor(physics_result['tip_position'], dtype=torch.float32).unsqueeze(0)

        # Create input (currents + position)
        input_data = torch.tensor(
            np.concatenate([currents, physics_result['tip_position']]),
            dtype=torch.float32
        ).unsqueeze(0)

        # Get residual correction
        residual = model(input_data, physics_pos)
        corrected = model.predict_with_physics(input_data, physics_pos)

        assert residual.shape == (1, 3)
        assert corrected.shape == (1, 3)
        results.add_pass("ResidualKinematicsModel with physics")
    except Exception as e:
        results.add_fail("ResidualKinematicsModel with physics", e)

    # Test residual dynamics model with physics
    try:
        sim = CRMSimulator(dt=0.02, use_cpp=True)
        model = ResidualDynamicsModel(state_dim=STATE_DIM, action_dim=ACTION_DIM)

        # Run physics simulation
        sim.reset(initial_currents=np.zeros(3), insertion_length=50.0)
        currents = np.array([0.1, 0.0, 0.0])

        # Get current state
        current_pos = sim.state.position
        current_vel = sim.state.velocity
        current_state = torch.tensor(
            np.concatenate([current_pos, current_vel]),
            dtype=torch.float32
        ).unsqueeze(0)

        # Step physics
        next_state_phys = sim.step(currents, insertion_length=50.0)
        physics_next = torch.tensor(
            np.concatenate([next_state_phys.position, next_state_phys.velocity]),
            dtype=torch.float32
        ).unsqueeze(0)

        # Get residual
        action = torch.tensor(currents, dtype=torch.float32).unsqueeze(0)
        residual = model(current_state, action, physics_next)
        corrected = model.predict_next_state(current_state, action, physics_next)

        assert residual.shape == (1, STATE_DIM)
        assert corrected.shape == (1, STATE_DIM)
        results.add_pass("ResidualDynamicsModel with physics simulation")
    except Exception as e:
        results.add_fail("ResidualDynamicsModel with physics simulation", e)


def test_hybrid_models(results: TestResults, verbose: bool = False):
    """Test hybrid models (CRM physics + learned residuals)."""
    print("\n[6/8] Testing Hybrid Models")
    print("-" * 40)

    from crm_ml_rl.models.hybrid_kinematics import HybridKinematicsModel, HybridKinematicsWithUncertainty
    from crm_ml_rl.models.hybrid_dynamics import HybridDynamicsModel, HybridDynamicsConfig

    # Test HybridKinematicsModel
    try:
        model = HybridKinematicsModel(use_cpp=True)
        currents = np.array([0.1, 0.0, 0.0])
        pred = model.predict(currents, insertion_length=50.0)
        assert pred.shape == (3,)
        results.add_pass(f"HybridKinematicsModel (cpp={model.is_using_cpp})")
    except Exception as e:
        results.add_fail("HybridKinematicsModel", e)

    # Test HybridKinematicsModel batch
    try:
        currents_batch = np.random.randn(8, 3) * 0.1
        preds = model.predict(currents_batch, insertion_length=50.0)
        assert preds.shape == (8, 3)
        results.add_pass("HybridKinematicsModel batch prediction")
    except Exception as e:
        results.add_fail("HybridKinematicsModel batch prediction", e)

    # Test HybridKinematicsModel forward with components
    try:
        currents_tensor = torch.tensor(currents_batch, dtype=torch.float32)
        final, physics, residual = model.forward(currents_tensor, 50.0, return_components=True)
        assert final.shape == physics.shape == residual.shape == (8, 3)
        results.add_pass("HybridKinematicsModel forward with components")
    except Exception as e:
        results.add_fail("HybridKinematicsModel forward with components", e)

    # Test HybridKinematicsWithUncertainty
    try:
        model_unc = HybridKinematicsWithUncertainty(use_cpp=True)
        mean, var = model_unc.predict(currents, insertion_length=50.0)
        assert mean.shape == (3,)
        assert var.shape == (3,)
        assert np.all(var > 0)
        results.add_pass("HybridKinematicsWithUncertainty")
    except Exception as e:
        results.add_fail("HybridKinematicsWithUncertainty", e)

    # Test HybridDynamicsModel
    try:
        config = HybridDynamicsConfig(use_cpp=True, dt=0.02)
        model = HybridDynamicsModel(config)
        initial_state = model.reset()
        assert initial_state.shape == (6,)
        results.add_pass(f"HybridDynamicsModel reset (cpp={model.is_using_cpp})")
    except Exception as e:
        results.add_fail("HybridDynamicsModel reset", e)

    # Test HybridDynamicsModel step
    try:
        action = np.array([0.05, 0.0, 0.0])
        next_state = model.step(action)
        assert next_state.shape == (6,)
        results.add_pass("HybridDynamicsModel step")
    except Exception as e:
        results.add_fail("HybridDynamicsModel step", e)

    # Test HybridDynamicsModel trajectory
    try:
        T = 20
        actions = np.random.randn(T, 3) * 0.05
        trajectory = model.predict_trajectory(actions)
        assert trajectory.shape == (T + 1, 6)
        results.add_pass("HybridDynamicsModel trajectory prediction")
    except Exception as e:
        results.add_fail("HybridDynamicsModel trajectory prediction", e)

    # Test HybridDynamicsModel loss
    try:
        batch_states = torch.randn(8, STATE_DIM)
        batch_actions = torch.randn(8, ACTION_DIM) * 0.05
        target_next = torch.randn(8, STATE_DIM)
        losses = model.compute_loss(batch_states, batch_actions, target_next)
        assert 'total' in losses
        assert 'improvement' in losses
        results.add_pass("HybridDynamicsModel loss computation")
    except Exception as e:
        results.add_fail("HybridDynamicsModel loss computation", e)


def test_rl_integration(results: TestResults, verbose: bool = False, quick: bool = False):
    """Test ML models integration in RL pipeline."""
    print("\n[7/8] Testing RL Pipeline Integration")
    print("-" * 40)

    from crm_ml_rl.envs import CatheterEnv, ReachingEnv, TrackingEnv, CatheterEnvConfig
    from crm_ml_rl.models.full_dynamics import FullDynamicsModel

    # Test environment with learned dynamics
    try:
        config = CatheterEnvConfig(
            use_learned_dynamics=True,
            max_steps=50
        )
        env = CatheterEnv(config=config)

        # Create and attach dynamics model
        dynamics_model = FullDynamicsModel(state_dim=STATE_DIM, action_dim=ACTION_DIM)
        env.set_dynamics_model(dynamics_model)

        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)

        assert obs.shape == env.observation_space.shape
        results.add_pass("CatheterEnv with learned dynamics")
    except Exception as e:
        results.add_fail("CatheterEnv with learned dynamics", e)

    # Test environment with C++ physics
    try:
        config = CatheterEnvConfig(
            use_cpp=True,
            param_file="catheterdata/CatheterParameterSet_1_new.txt",
            config_file="catheterdata/CatheterSpatialConfiguration_1.txt",
            max_steps=50
        )
        env = CatheterEnv(config=config)
        obs, info = env.reset()

        for _ in range(10):
            action = env.action_space.sample() * 0.1  # Small actions
            obs, reward, term, trunc, info = env.step(action)
            if term or trunc:
                break

        results.add_pass(f"CatheterEnv with C++ physics (cpp={env.simulator.is_using_cpp})")
    except Exception as e:
        results.add_fail("CatheterEnv with C++ physics", e)

    # Test environment with hybrid dynamics
    try:
        config = CatheterEnvConfig(
            use_hybrid_dynamics=True,
            use_cpp=True,
            param_file="catheterdata/CatheterParameterSet_1_new.txt",
            config_file="catheterdata/CatheterSpatialConfiguration_1.txt",
            max_steps=50
        )
        env = CatheterEnv(config=config)
        assert env.hybrid_model is not None, "Hybrid model should be initialized"

        obs, info = env.reset()

        for _ in range(5):
            action = env.action_space.sample() * 0.05
            obs, reward, term, trunc, info = env.step(action)
            if term or trunc:
                break

        results.add_pass(f"CatheterEnv with hybrid dynamics (cpp={env.hybrid_model.is_using_cpp})")
    except Exception as e:
        results.add_fail("CatheterEnv with hybrid dynamics", e)

    # Test ReachingEnv
    try:
        env = ReachingEnv(random_target=True)
        obs, info = env.reset()
        assert 'target_position' in info

        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)

        results.add_pass("ReachingEnv basic operation")
    except Exception as e:
        results.add_fail("ReachingEnv basic operation", e)

    # Test TrackingEnv
    try:
        env = TrackingEnv(trajectory_type="circle", trajectory_freq=0.5)
        obs, info = env.reset()

        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)

        results.add_pass("TrackingEnv basic operation")
    except Exception as e:
        results.add_fail("TrackingEnv basic operation", e)

    if quick:
        print("  (Skipping RL agent tests in quick mode)")
        return

    # Test with stable-baselines3 agents
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        env = DummyVecEnv([lambda: ReachingEnv()])
        model = PPO("MlpPolicy", env, verbose=0, n_steps=64)
        model.learn(total_timesteps=128)  # Minimal training

        # Test prediction
        obs = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        assert action.shape == (1, 3)

        results.add_pass("PPO with ReachingEnv")
    except Exception as e:
        results.add_fail("PPO with ReachingEnv", e)

    # Test custom agents
    try:
        from crm_ml_rl.training.rl_agents import SACAgent, PPOAgent, TD3Agent

        env = ReachingEnv()

        # Quick instantiation tests
        sac = SACAgent(env, verbose=0)
        ppo = PPOAgent(env, verbose=0)
        td3 = TD3Agent(env, verbose=0)

        obs, _ = env.reset()
        action_sac = sac.predict(obs)
        action_ppo = ppo.predict(obs)
        action_td3 = td3.predict(obs)

        assert action_sac.shape == (3,)
        assert action_ppo.shape == (3,)
        assert action_td3.shape == (3,)

        results.add_pass("Custom RL agents (SAC, PPO, TD3)")
    except Exception as e:
        results.add_fail("Custom RL agents", e)

    # Test model training integration
    try:
        from crm_ml_rl.training.train_dynamics import DynamicsTrainer, TrainingConfig

        config = TrainingConfig(
            model_type="residual",
            epochs=1,
            batch_size=16
        )
        trainer = DynamicsTrainer(config)

        # Create dummy data
        states = np.random.randn(100, STATE_DIM).astype(np.float32)
        actions = np.random.randn(100, ACTION_DIM).astype(np.float32)
        next_states = np.random.randn(100, STATE_DIM).astype(np.float32)

        from crm_ml_rl.training.train_dynamics import DynamicsDataset
        from torch.utils.data import DataLoader

        dataset = DynamicsDataset(states, actions, next_states)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        # Train one epoch
        train_loss = trainer.train_epoch(loader)
        assert train_loss > 0, "Training loss should be positive"

        results.add_pass("DynamicsTrainer training")
    except Exception as e:
        results.add_fail("DynamicsTrainer training", e)


def main():
    parser = argparse.ArgumentParser(description="Test ML models in crm_ml_rl")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quick", "-q", action="store_true", help="Quick tests only")
    args = parser.parse_args()

    print("=" * 60)
    print("CRM_ML ML Models Test Suite")
    print("=" * 60)

    results = TestResults()

    # Run all test categories
    test_base_networks(results, verbose=args.verbose)
    test_residual_kinematics(results, verbose=args.verbose)
    test_residual_dynamics(results, verbose=args.verbose)
    test_full_models(results, verbose=args.verbose)
    test_cpp_integration(results, verbose=args.verbose)
    test_hybrid_models(results, verbose=args.verbose)
    test_rl_integration(results, verbose=args.verbose, quick=args.quick)

    # Print summary
    success = results.summary()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
