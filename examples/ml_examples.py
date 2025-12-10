"""
Lightweight examples for training CRM ML models on synthetic data.

These examples intentionally use small random datasets so they run in a few
seconds and serve as smoke tests for the model implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from crm_ml_rl.models import (
    ResidualDynamicsModel,
    FullDynamicsModel,
    ResidualKinematicsModel,
    FullKinematicsModel,
    TransformerDynamicsModel,
    DiffusionDynamicsModel,
    DiffusionDynamicsConfig,
)

torch.manual_seed(0)
np.random.seed(0)


@dataclass
class SyntheticDynamicsBatch:
    states: torch.Tensor
    actions: torch.Tensor
    next_states: torch.Tensor
    physics_next: torch.Tensor


def _generate_dynamics_batch(num_samples: int = 256) -> SyntheticDynamicsBatch:
    """Generate a batch of synthetic dynamics data."""
    dt = 0.02
    states = torch.randn(num_samples, 6)
    actions = torch.randn(num_samples, 3) * 0.1

    pos = states[:, :3]
    vel = states[:, 3:]
    accel = torch.tanh(actions) * 5.0

    next_vel = vel + dt * accel
    next_pos = pos + dt * next_vel
    next_states = torch.cat([next_pos, next_vel], dim=1)

    # Physics baseline underestimates acceleration
    physics_vel = vel + dt * 0.7 * accel
    physics_pos = pos + dt * physics_vel
    physics_next = torch.cat([physics_pos, physics_vel], dim=1)

    return SyntheticDynamicsBatch(states, actions, next_states, physics_next)


def _generate_sequence_batch(
    batch: SyntheticDynamicsBatch,
    seq_len: int = 4
) -> Dict[str, torch.Tensor]:
    """Create simple rolling windows for transformer training."""
    total = batch.states.size(0)
    usable = total - seq_len
    states_seq = []
    actions_seq = []
    for i in range(usable):
        states_seq.append(batch.states[i:i + seq_len])
        actions_seq.append(batch.actions[i:i + seq_len])
    states_seq = torch.stack(states_seq, dim=0)
    actions_seq = torch.stack(actions_seq, dim=0)
    targets = torch.stack([
        batch.next_states[i:i + seq_len]
        for i in range(usable)
    ], dim=0)
    return {'states': states_seq, 'actions': actions_seq, 'targets': targets}


def _generate_kinematics_batch(num_samples: int = 256) -> Dict[str, torch.Tensor]:
    """Synthetic forward kinematics dataset."""
    currents = torch.randn(num_samples, 3) * 0.2
    insertion = torch.rand(num_samples, 1) * 60 + 20
    inputs = torch.cat([currents, insertion], dim=1)

    physics_fk = torch.stack([
        inputs[:, 0] * 5.0,
        inputs[:, 1] * 5.0,
        insertion.squeeze(-1) * 0.1  # scale down to keep targets well-conditioned
    ], dim=1)
    true_fk = physics_fk + 0.5 * torch.sin(currents)
    return {
        'inputs': inputs,
        'physics_fk': physics_fk,
        'targets': true_fk
    }


def train_residual_dynamics_example(steps: int = 50) -> float:
    batch = _generate_dynamics_batch()
    model = ResidualDynamicsModel(state_dim=6, action_dim=3, hidden_dims=[64, 64])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(steps):
        pred = batch.physics_next + model(batch.states, batch.actions, batch.physics_next)
        loss = F.mse_loss(pred, batch.next_states)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return float(loss.detach())


def train_full_dynamics_example(steps: int = 50) -> float:
    batch = _generate_dynamics_batch()
    model = FullDynamicsModel(state_dim=6, action_dim=3, hidden_dims=[64, 64], predict_delta=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(steps):
        pred = model(batch.states, batch.actions)
        loss = F.mse_loss(pred, batch.next_states)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return float(loss.detach())


def train_transformer_dynamics_example(steps: int = 20) -> float:
    batch = _generate_dynamics_batch()
    seq_data = _generate_sequence_batch(batch, seq_len=4)
    model = TransformerDynamicsModel(
        state_dim=6,
        action_dim=3,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        context_len=4
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    states = seq_data['states']
    actions = seq_data['actions']
    targets = seq_data['targets']

    for _ in range(steps):
        preds = model.predict_sequence(states, actions)
        loss = F.mse_loss(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return float(loss.detach())


def train_diffusion_dynamics_example(steps: int = 20) -> float:
    batch = _generate_dynamics_batch()
    config = DiffusionDynamicsConfig(
        state_dim=6,
        action_dim=3,
        hidden_dim=128,
        num_layers=3,
        num_diffusion_steps=30
    )
    model = DiffusionDynamicsModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(steps):
        loss = model.diffusion_loss(batch.states, batch.actions, batch.next_states)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Deterministic forward for evaluation
    with torch.no_grad():
        preds = model(batch.states, batch.actions, deterministic=True)
        eval_loss = F.mse_loss(preds, batch.next_states)
    return float(eval_loss.detach())


def train_residual_kinematics_example(steps: int = 50) -> float:
    data = _generate_kinematics_batch()
    model = ResidualKinematicsModel(
        input_dim=4,
        output_dim=3,
        hidden_dims=[64, 64],
        physics_output_dim=3,
        include_physics_in_input=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(steps):
        residual = model(data['inputs'], data['physics_fk'])
        preds = data['physics_fk'] + residual
        loss = F.mse_loss(preds, data['targets'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return float(loss.detach())


def train_full_kinematics_example(steps: int = 200) -> float:
    data = _generate_kinematics_batch()
    model = FullKinematicsModel(input_dim=4, output_dim=3, hidden_dims=[128, 64], use_deep_residual=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    for _ in range(steps):
        preds = model(data['inputs'])
        loss = F.mse_loss(preds, data['targets'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return float(loss.detach())


def run_ml_model_examples() -> Dict[str, float]:
    """Run all ML examples and return the final losses."""
    return {
        'residual_dynamics_loss': train_residual_dynamics_example(),
        'full_dynamics_loss': train_full_dynamics_example(),
        'transformer_dynamics_loss': train_transformer_dynamics_example(),
        'diffusion_dynamics_loss': train_diffusion_dynamics_example(),
        'residual_kinematics_loss': train_residual_kinematics_example(),
        'full_kinematics_loss': train_full_kinematics_example(),
    }


if __name__ == "__main__":
    results = run_ml_model_examples()
    for name, value in results.items():
        print(f"{name}: {value:.6f}")
