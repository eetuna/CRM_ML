"""
Examples showcasing RL/MPC integrations with the CRM models.

The examples rely on a tiny synthetic environment so they can run quickly
and serve as smoke tests for the higher-level RL utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from crm_ml_rl.training.model_based_rl import DynaAgent, ModelBasedConfig
from crm_ml_rl.training.mpc_controller import MPCController, MPCConfig
from crm_ml_rl.models.sequence_models import (
    TransformerDynamicsModel,
    DiffusionDynamicsConfig,
    DiffusionDynamicsModel,
)


class ToyDynamicsEnv(gym.Env):
    """A minimal continuous control task with 6D state and 3D actions."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Box(-0.5, 0.5, shape=(3,), dtype=np.float32)
        self.state = np.zeros(6, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        super().reset(seed=seed)
        self.state = self.observation_space.sample()
        return self.state.copy(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        pos = self.state[:3]
        vel = self.state[3:]
        vel = 0.8 * vel + 0.2 * action
        pos = pos + vel
        self.state = np.concatenate([pos, vel]).astype(np.float32)
        reward = -np.linalg.norm(pos)
        done = False
        truncated = False
        return self.state.copy(), float(reward), done, truncated, {}


def _populate_replay_buffer(agent: DynaAgent, env: gym.Env, steps: int = 50):
    """Collect random transitions so the replay buffer and world model have data."""
    obs, _ = env.reset()
    for _ in range(steps):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        agent.agent.model.replay_buffer.add(
            obs.astype(np.float32),
            next_obs.astype(np.float32),
            action.astype(np.float32),
            float(reward),
            done,
            [{}]
        )
        obs = next_obs if not done else env.reset()[0]


def run_dyna_transformer_example(num_start_states: int = 4) -> int:
    """Instantiate a Dyna agent with the transformer world model and run rollouts."""
    env = ToyDynamicsEnv()
    config = ModelBasedConfig(
        dynamics_model_type="transformer",
        transformer_kwargs={
            "context_len": 3,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 1,
            "dim_feedforward": 128,
        },
        use_cpp_physics=False,
        imagined_rollout_horizon=3
    )
    agent = DynaAgent(
        env=env,
        config=config,
        base_algorithm="sac",
        learning_rate=3e-4,
        buffer_size=10_000,
        batch_size=64,
        gamma=0.95,
        device="cpu",
        verbose=0,
        feature_extractor="mlp"
    )
    _populate_replay_buffer(agent, env, steps=100)
    start_states = np.stack([env.reset()[0] for _ in range(num_start_states)], axis=0)
    transitions = agent.generate_imagined_rollouts(start_states, horizon=2)
    return len(transitions)


def run_dyna_diffusion_example(num_start_states: int = 4) -> int:
    """Dyna agent configured with the diffusion world model."""
    env = ToyDynamicsEnv()
    diffusion_config = DiffusionDynamicsConfig(
        state_dim=6,
        action_dim=3,
        hidden_dim=128,
        num_layers=2,
        num_diffusion_steps=20
    )
    config = ModelBasedConfig(
        dynamics_model_type="diffusion",
        diffusion_config=diffusion_config,
        use_cpp_physics=False,
        imagined_rollout_horizon=2
    )
    agent = DynaAgent(
        env=env,
        config=config,
        base_algorithm="sac",
        learning_rate=3e-4,
        buffer_size=5_000,
        batch_size=32,
        gamma=0.95,
        device="cpu",
        verbose=0,
        feature_extractor="mlp"
    )
    _populate_replay_buffer(agent, env, steps=80)
    start_states = np.stack([env.reset()[0] for _ in range(num_start_states)], axis=0)
    transitions = agent.generate_imagined_rollouts(start_states, horizon=2)
    return len(transitions)


def run_mpc_transformer_example() -> np.ndarray:
    """Train a tiny transformer model and run the MPC controller."""
    # Create synthetic training data
    torch.manual_seed(0)
    states = torch.randn(128, 6)
    actions = torch.randn(128, 3) * 0.1
    dt = 0.02
    vel = states[:, 3:]
    pos = states[:, :3]
    next_vel = vel + dt * torch.tanh(actions)
    next_pos = pos + dt * next_vel
    targets = torch.cat([next_pos, next_vel], dim=1)

    model = TransformerDynamicsModel(
        state_dim=6,
        action_dim=3,
        d_model=64,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
        context_len=2
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(30):
        preds = model(states, actions)
        loss = torch.mean((preds - targets) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    controller = MPCController(dynamics_model=model, config=MPCConfig(horizon=5))
    current_state = np.zeros(6)
    target_traj = np.zeros((controller.config.horizon + 1, 6))
    optimal_action, _ = controller.solve_shooting(current_state, target_traj)
    return optimal_action


def run_all_rl_examples() -> Dict[str, float]:
    """Execute all RL/MPC smoke tests and return summary metrics."""
    transitions_transformer = run_dyna_transformer_example()
    transitions_diffusion = run_dyna_diffusion_example()
    optimal_action = run_mpc_transformer_example()
    return {
        'transformer_dyna_transitions': float(transitions_transformer),
        'diffusion_dyna_transitions': float(transitions_diffusion),
        'mpc_action_norm': float(np.linalg.norm(optimal_action)),
    }


if __name__ == "__main__":
    results = run_all_rl_examples()
    for name, value in results.items():
        print(f"{name}: {value:.4f}")
