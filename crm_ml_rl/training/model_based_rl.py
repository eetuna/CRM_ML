"""
Model-Based Reinforcement Learning agents.

This module provides model-based RL agents that use the HybridDynamicsModel
as a world model for planning and imagination.

Agents:
    - DynaAgent: Dyna-style learning with real + imagined experience
    - MBPOAgent: Model-Based Policy Optimization with short-horizon rollouts
    - MPCAgent: Model Predictive Control using learned dynamics
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, List, Tuple, Union, Any
from pathlib import Path
from dataclasses import dataclass
from collections import deque
import gymnasium as gym

from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ..models.hybrid_dynamics import HybridDynamicsModel, HybridDynamicsConfig
from ..models.full_dynamics import FullDynamicsModel
from .rl_agents import SACAgent, PPOAgent, TD3Agent, create_policy_kwargs


@dataclass
class ModelBasedConfig:
    """Configuration for model-based RL."""
    # World model settings
    dynamics_model_type: str = "hybrid"  # "hybrid", "full", "ensemble"
    dynamics_hidden_dims: List[int] = None
    use_cpp_physics: bool = True
    param_file: Optional[str] = None
    config_file: Optional[str] = None

    # Model-based training settings
    imagined_rollout_horizon: int = 10
    imagined_data_ratio: float = 0.5
    model_update_freq: int = 1000
    model_train_epochs: int = 10

    # MPC settings
    mpc_horizon: int = 20
    mpc_num_samples: int = 500
    mpc_temperature: float = 1.0

    # Buffer settings
    model_buffer_size: int = 100_000
    real_buffer_size: int = 1_000_000

    def __post_init__(self):
        if self.dynamics_hidden_dims is None:
            self.dynamics_hidden_dims = [256, 256, 128]


class DynaAgent:
    """
    Dyna-style model-based RL agent.

    Uses a learned world model to generate imagined experience that
    supplements real environment interactions. The policy is trained
    on both real and imagined transitions.

    Reference: Sutton, R. S. (1990). Dyna, an integrated architecture
    for learning, planning, and reacting.
    """

    def __init__(
        self,
        env: gym.Env,
        dynamics_model: Optional[HybridDynamicsModel] = None,
        config: Optional[ModelBasedConfig] = None,
        base_algorithm: str = "sac",
        learning_rate: float = 3e-4,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        gamma: float = 0.99,
        device: str = "auto",
        verbose: int = 1,
        feature_extractor: str = "default",
        features_dim: int = 128,
        extractor_kwargs: Optional[Dict] = None
    ):
        """
        Initialize Dyna agent.

        Args:
            env: Gymnasium environment
            dynamics_model: Pre-trained dynamics model (or None to create new)
            config: Model-based configuration
            base_algorithm: Base RL algorithm ("sac", "ppo", "td3")
            learning_rate: Learning rate
            buffer_size: Replay buffer size
            batch_size: Batch size
            gamma: Discount factor
            device: Device for training
            verbose: Verbosity level
            feature_extractor: Feature extractor type
            features_dim: Feature dimension
            extractor_kwargs: Extra kwargs for feature extractor
        """
        self.env = env
        self.config = config or ModelBasedConfig()
        self.base_algorithm = base_algorithm
        self.device = torch.device(device if device != "auto" else
                                   ("cuda" if torch.cuda.is_available() else "cpu"))
        self.verbose = verbose

        # Create or use provided dynamics model
        if dynamics_model is not None:
            self.dynamics_model = dynamics_model
        else:
            self._create_dynamics_model()

        # Create base RL agent
        if extractor_kwargs is None:
            extractor_kwargs = {}

        if base_algorithm == "sac":
            self.agent = SACAgent(
                env=env,
                learning_rate=learning_rate,
                buffer_size=buffer_size,
                batch_size=batch_size,
                gamma=gamma,
                device=device,
                verbose=verbose,
                feature_extractor=feature_extractor,
                features_dim=features_dim,
                extractor_kwargs=extractor_kwargs
            )
        elif base_algorithm == "ppo":
            self.agent = PPOAgent(
                env=env,
                learning_rate=learning_rate,
                batch_size=batch_size,
                gamma=gamma,
                device=device,
                verbose=verbose,
                feature_extractor=feature_extractor,
                features_dim=features_dim,
                extractor_kwargs=extractor_kwargs
            )
        elif base_algorithm == "td3":
            self.agent = TD3Agent(
                env=env,
                learning_rate=learning_rate,
                buffer_size=buffer_size,
                batch_size=batch_size,
                gamma=gamma,
                device=device,
                verbose=verbose,
                feature_extractor=feature_extractor,
                features_dim=features_dim,
                extractor_kwargs=extractor_kwargs
            )
        else:
            raise ValueError(f"Unknown base algorithm: {base_algorithm}")

        # Imagined experience buffer
        self.imagined_buffer = deque(maxlen=self.config.model_buffer_size)

        # Training statistics
        self.total_real_steps = 0
        self.total_imagined_steps = 0
        self.model_losses = []

    def _create_dynamics_model(self):
        """Create dynamics model based on config."""
        if self.config.dynamics_model_type == "hybrid":
            dynamics_config = HybridDynamicsConfig(
                param_file=self.config.param_file,
                config_file=self.config.config_file,
                use_cpp=self.config.use_cpp_physics,
                hidden_dims=self.config.dynamics_hidden_dims
            )
            self.dynamics_model = HybridDynamicsModel(dynamics_config, device=str(self.device))
        else:
            # Full neural network dynamics
            obs_dim = self.env.observation_space.shape[0]
            action_dim = self.env.action_space.shape[0]
            self.dynamics_model = FullDynamicsModel(
                state_dim=obs_dim,
                action_dim=action_dim,
                hidden_dims=self.config.dynamics_hidden_dims
            ).to(self.device)

    def generate_imagined_rollouts(
        self,
        start_states: np.ndarray,
        horizon: int = None
    ) -> List[Dict]:
        """
        Generate imagined rollouts from given start states.

        Args:
            start_states: Starting states for rollouts (N, state_dim)
            horizon: Rollout horizon (default from config)

        Returns:
            List of transition dictionaries
        """
        if horizon is None:
            horizon = self.config.imagined_rollout_horizon

        transitions = []
        current_states = start_states.copy()

        # Extract only the dynamics-relevant portion of state (first 6 dims)
        dynamics_state_dim = self.dynamics_model.state_dim

        for t in range(horizon):
            # Get actions from policy
            actions = []
            for state in current_states:
                action, _ = self.agent.model.predict(state, deterministic=False)
                actions.append(action)
            actions = np.array(actions)

            # Extract dynamics state (position + velocity) for model
            dynamics_states = current_states[:, :dynamics_state_dim] if current_states.shape[1] > dynamics_state_dim else current_states

            # Predict next states using dynamics model
            states_tensor = torch.tensor(dynamics_states, dtype=torch.float32, device=self.device)
            actions_tensor = torch.tensor(actions, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                next_dynamics_states = self.dynamics_model.forward(states_tensor, actions_tensor)
            next_dynamics_states = next_dynamics_states.cpu().numpy()

            # Reconstruct full states (keep other observation dims unchanged)
            if current_states.shape[1] > dynamics_state_dim:
                next_states = current_states.copy()
                next_states[:, :dynamics_state_dim] = next_dynamics_states
            else:
                next_states = next_dynamics_states

            # Compute rewards (simple distance-based for reaching task)
            # In practice, this should match the environment's reward function
            rewards = self._compute_imagined_rewards(current_states, actions, next_states)

            # Store transitions
            for i in range(len(current_states)):
                transitions.append({
                    'state': current_states[i],
                    'action': actions[i],
                    'reward': rewards[i],
                    'next_state': next_states[i],
                    'done': False  # Assume not done in imagination
                })

            current_states = next_states

        return transitions

    def _compute_imagined_rewards(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray
    ) -> np.ndarray:
        """
        Compute rewards for imagined transitions.

        This is a simplified reward model. For best results, learn a
        separate reward model or use the environment's reward function.
        """
        # Simple distance-based reward (assumes reaching task)
        # Position is first 3 dims, target is next 3 dims after velocity
        if states.shape[-1] >= 12:
            positions = next_states[:, :3]
            targets = states[:, 6:9]  # Assuming target is at indices 6-8
            distances = np.linalg.norm(positions - targets, axis=1)
            rewards = -distances
        else:
            rewards = np.zeros(len(states))

        # Action penalty
        action_penalty = 0.01 * np.sum(actions ** 2, axis=1)
        rewards -= action_penalty

        return rewards

    def train(
        self,
        total_timesteps: int,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 10000,
        save_path: Optional[str] = None,
        log_freq: int = 1000
    ) -> Dict[str, Any]:
        """
        Train the Dyna agent.

        Args:
            total_timesteps: Total training timesteps
            eval_env: Environment for evaluation
            eval_freq: Evaluation frequency
            save_path: Path to save models
            log_freq: Logging frequency

        Returns:
            Training results
        """
        # Collect some initial real experience
        print("Collecting initial experience...")
        obs, _ = self.env.reset()
        initial_steps = min(10000, total_timesteps // 10)

        for _ in range(initial_steps):
            action, _ = self.agent.model.predict(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            # Store in agent's buffer
            self.agent.model.replay_buffer.add(
                obs, next_obs, action, reward,
                terminated or truncated, [info]
            )

            self.total_real_steps += 1
            obs = next_obs if not (terminated or truncated) else self.env.reset()[0]

        # Main training loop
        remaining_steps = total_timesteps - initial_steps
        steps_done = 0

        print(f"Starting Dyna training for {remaining_steps} steps...")

        while steps_done < remaining_steps:
            # Collect real experience
            for _ in range(self.config.model_update_freq):
                action, _ = self.agent.model.predict(obs, deterministic=False)
                next_obs, reward, terminated, truncated, info = self.env.step(action)

                self.agent.model.replay_buffer.add(
                    obs, next_obs, action, reward,
                    terminated or truncated, [info]
                )

                self.total_real_steps += 1
                steps_done += 1

                obs = next_obs if not (terminated or truncated) else self.env.reset()[0]

            # Generate imagined experience
            if self.agent.model.replay_buffer.size() > 1000:
                # Sample start states from real buffer
                real_data = self.agent.model.replay_buffer.sample(100)
                start_states = real_data.observations.cpu().numpy()

                # Generate rollouts
                imagined_transitions = self.generate_imagined_rollouts(
                    start_states,
                    horizon=self.config.imagined_rollout_horizon
                )

                # Add to imagined buffer
                self.imagined_buffer.extend(imagined_transitions)
                self.total_imagined_steps += len(imagined_transitions)

                # Add imagined data to agent's buffer
                for trans in imagined_transitions:
                    self.agent.model.replay_buffer.add(
                        trans['state'], trans['next_state'], trans['action'],
                        trans['reward'], trans['done'], [{}]
                    )

            # Train the policy
            if self.agent.model.replay_buffer.size() > self.agent.model.batch_size:
                self.agent.model.train(gradient_steps=100)

            # Logging
            if steps_done % log_freq == 0 and self.verbose > 0:
                print(f"Step {steps_done}/{remaining_steps}, "
                      f"Real: {self.total_real_steps}, Imagined: {self.total_imagined_steps}")

            # Evaluation
            if eval_env is not None and steps_done % eval_freq == 0:
                self._evaluate(eval_env)

        # Save models
        if save_path:
            self.save(save_path)

        return {
            'total_real_steps': self.total_real_steps,
            'total_imagined_steps': self.total_imagined_steps
        }

    def _evaluate(self, eval_env: gym.Env, n_episodes: int = 10) -> Dict:
        """Evaluate the agent."""
        rewards = []
        for _ in range(n_episodes):
            obs, _ = eval_env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _ = self.agent.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                episode_reward += reward
                done = terminated or truncated
            rewards.append(episode_reward)

        mean_reward = np.mean(rewards)
        if self.verbose > 0:
            print(f"Eval: Mean reward = {mean_reward:.2f}")
        return {'mean_reward': mean_reward, 'std_reward': np.std(rewards)}

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Predict action."""
        action, _ = self.agent.model.predict(observation, deterministic=deterministic)
        return action

    def save(self, path: str):
        """Save agent and dynamics model."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.agent.save(str(path / "policy"))
        torch.save(self.dynamics_model.state_dict(), path / "dynamics_model.pt")

    def load(self, path: str):
        """Load agent and dynamics model."""
        path = Path(path)
        self.agent.load(str(path / "policy"))
        self.dynamics_model.load_state_dict(torch.load(path / "dynamics_model.pt"))


class MBPOAgent:
    """
    Model-Based Policy Optimization agent.

    Uses short-horizon model rollouts starting from real states to
    augment the policy training data. This reduces model error accumulation.

    Reference: Janner et al. (2019). When to Trust Your Model:
    Model-Based Policy Optimization.
    """

    def __init__(
        self,
        env: gym.Env,
        dynamics_model: Optional[HybridDynamicsModel] = None,
        config: Optional[ModelBasedConfig] = None,
        learning_rate: float = 3e-4,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        gamma: float = 0.99,
        device: str = "auto",
        verbose: int = 1,
        feature_extractor: str = "default",
        features_dim: int = 128,
        extractor_kwargs: Optional[Dict] = None
    ):
        """
        Initialize MBPO agent.

        Args:
            env: Gymnasium environment
            dynamics_model: Pre-trained dynamics model
            config: Model-based configuration
            learning_rate: Learning rate
            buffer_size: Replay buffer size
            batch_size: Batch size
            gamma: Discount factor
            device: Device for training
            verbose: Verbosity level
            feature_extractor: Feature extractor type
            features_dim: Feature dimension
            extractor_kwargs: Extra kwargs for feature extractor
        """
        self.env = env
        self.config = config or ModelBasedConfig()
        self.device = torch.device(device if device != "auto" else
                                   ("cuda" if torch.cuda.is_available() else "cpu"))
        self.verbose = verbose

        # Create or use dynamics model
        if dynamics_model is not None:
            self.dynamics_model = dynamics_model
        else:
            self._create_dynamics_model()

        # Create SAC agent (MBPO typically uses SAC)
        if extractor_kwargs is None:
            extractor_kwargs = {}

        self.agent = SACAgent(
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            gamma=gamma,
            device=device,
            verbose=verbose,
            feature_extractor=feature_extractor,
            features_dim=features_dim,
            extractor_kwargs=extractor_kwargs
        )

        # Model buffer for imagined data
        self.model_buffer = deque(maxlen=self.config.model_buffer_size)

        # Dynamics model optimizer
        self.dynamics_optimizer = torch.optim.Adam(
            self.dynamics_model.parameters(),
            lr=1e-3
        )

        # Statistics
        self.rollout_length = 1  # Start with short rollouts

    def _create_dynamics_model(self):
        """Create dynamics model."""
        if self.config.dynamics_model_type == "hybrid":
            dynamics_config = HybridDynamicsConfig(
                param_file=self.config.param_file,
                config_file=self.config.config_file,
                use_cpp=self.config.use_cpp_physics,
                hidden_dims=self.config.dynamics_hidden_dims
            )
            self.dynamics_model = HybridDynamicsModel(dynamics_config, device=str(self.device))
        else:
            obs_dim = self.env.observation_space.shape[0]
            action_dim = self.env.action_space.shape[0]
            self.dynamics_model = FullDynamicsModel(
                state_dim=obs_dim,
                action_dim=action_dim,
                hidden_dims=self.config.dynamics_hidden_dims
            ).to(self.device)

    def train_dynamics_model(self, batch_size: int = 256, epochs: int = 10):
        """Train the dynamics model on real data."""
        if self.agent.model.replay_buffer.size() < batch_size:
            return

        for _ in range(epochs):
            data = self.agent.model.replay_buffer.sample(batch_size)
            states = data.observations
            actions = data.actions
            next_states = data.next_observations

            # Forward pass
            pred_next_states = self.dynamics_model.forward(states, actions)

            # Loss
            loss = nn.functional.mse_loss(pred_next_states, next_states)

            # Backward pass
            self.dynamics_optimizer.zero_grad()
            loss.backward()
            self.dynamics_optimizer.step()

    def generate_model_rollouts(
        self,
        num_rollouts: int,
        rollout_length: int = None
    ) -> List[Dict]:
        """Generate short-horizon rollouts from real states."""
        if rollout_length is None:
            rollout_length = self.rollout_length

        if self.agent.model.replay_buffer.size() < num_rollouts:
            return []

        # Sample start states
        data = self.agent.model.replay_buffer.sample(num_rollouts)
        start_states = data.observations.cpu().numpy()

        transitions = []
        current_states = start_states

        for t in range(rollout_length):
            actions = []
            for state in current_states:
                action, _ = self.agent.model.predict(state, deterministic=False)
                actions.append(action)
            actions = np.array(actions)

            # Predict next states
            states_tensor = torch.tensor(current_states, dtype=torch.float32, device=self.device)
            actions_tensor = torch.tensor(actions, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                next_states = self.dynamics_model.forward(states_tensor, actions_tensor)
            next_states = next_states.cpu().numpy()

            # Simplified rewards
            rewards = -np.linalg.norm(next_states[:, :3], axis=1)  # Distance to origin

            for i in range(len(current_states)):
                transitions.append({
                    'state': current_states[i],
                    'action': actions[i],
                    'reward': rewards[i],
                    'next_state': next_states[i],
                    'done': False
                })

            current_states = next_states

        return transitions

    def train(
        self,
        total_timesteps: int,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 10000,
        save_path: Optional[str] = None
    ) -> Dict:
        """Train MBPO agent."""
        obs, _ = self.env.reset()
        steps = 0
        epoch = 0

        print(f"Starting MBPO training for {total_timesteps} steps...")

        while steps < total_timesteps:
            epoch += 1

            # Collect real data
            for _ in range(1000):
                action, _ = self.agent.model.predict(obs, deterministic=False)
                next_obs, reward, terminated, truncated, info = self.env.step(action)

                self.agent.model.replay_buffer.add(
                    obs, next_obs, action, reward,
                    terminated or truncated, [info]
                )

                steps += 1
                obs = next_obs if not (terminated or truncated) else self.env.reset()[0]

            # Train dynamics model
            self.train_dynamics_model(epochs=self.config.model_train_epochs)

            # Generate model rollouts
            model_transitions = self.generate_model_rollouts(
                num_rollouts=400,
                rollout_length=self.rollout_length
            )

            # Add to buffer
            for trans in model_transitions:
                self.agent.model.replay_buffer.add(
                    trans['state'], trans['next_state'], trans['action'],
                    trans['reward'], trans['done'], [{}]
                )

            # Train policy
            self.agent.model.train(gradient_steps=1000)

            # Increase rollout length over time
            if epoch % 20 == 0 and self.rollout_length < self.config.imagined_rollout_horizon:
                self.rollout_length += 1

            if self.verbose > 0 and epoch % 10 == 0:
                print(f"Epoch {epoch}, Steps: {steps}, Rollout length: {self.rollout_length}")

            # Evaluation
            if eval_env is not None and steps % eval_freq < 1000:
                self._evaluate(eval_env)

        if save_path:
            self.save(save_path)

        return {'total_steps': steps}

    def _evaluate(self, eval_env: gym.Env, n_episodes: int = 10) -> Dict:
        """Evaluate agent."""
        rewards = []
        for _ in range(n_episodes):
            obs, _ = eval_env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _ = self.agent.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                episode_reward += reward
                done = terminated or truncated
            rewards.append(episode_reward)

        mean_reward = np.mean(rewards)
        if self.verbose > 0:
            print(f"Eval: Mean reward = {mean_reward:.2f}")
        return {'mean_reward': mean_reward}

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Predict action."""
        action, _ = self.agent.model.predict(observation, deterministic=deterministic)
        return action

    def save(self, path: str):
        """Save models."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.agent.save(str(path / "policy"))
        torch.save(self.dynamics_model.state_dict(), path / "dynamics_model.pt")

    def load(self, path: str):
        """Load models."""
        path = Path(path)
        self.agent.load(str(path / "policy"))
        self.dynamics_model.load_state_dict(torch.load(path / "dynamics_model.pt"))


class MPCAgent:
    """
    Model Predictive Control agent using learned dynamics.

    Plans actions by optimizing over a planning horizon using
    the learned dynamics model. Uses Cross-Entropy Method (CEM)
    or random shooting for optimization.
    """

    def __init__(
        self,
        env: gym.Env,
        dynamics_model: HybridDynamicsModel,
        config: Optional[ModelBasedConfig] = None,
        device: str = "auto"
    ):
        """
        Initialize MPC agent.

        Args:
            env: Gymnasium environment
            dynamics_model: Learned dynamics model
            config: Configuration
            device: Device for computations
        """
        self.env = env
        self.dynamics_model = dynamics_model
        self.config = config or ModelBasedConfig()
        self.device = torch.device(device if device != "auto" else
                                   ("cuda" if torch.cuda.is_available() else "cpu"))

        self.action_dim = env.action_space.shape[0]
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high

    def plan_actions(
        self,
        current_state: np.ndarray,
        target: Optional[np.ndarray] = None,
        horizon: int = None,
        num_samples: int = None
    ) -> np.ndarray:
        """
        Plan optimal action sequence using CEM.

        Args:
            current_state: Current state
            target: Target state (optional, uses state[6:9] if not provided)
            horizon: Planning horizon
            num_samples: Number of action sequences to sample

        Returns:
            First action from optimal sequence
        """
        if horizon is None:
            horizon = self.config.mpc_horizon
        if num_samples is None:
            num_samples = self.config.mpc_num_samples

        # Extract target from state if not provided
        if target is None and len(current_state) >= 9:
            target = current_state[6:9]  # Assuming target is at indices 6-8
        elif target is None:
            target = np.zeros(3)

        # Initialize action distribution
        action_mean = np.zeros((horizon, self.action_dim))
        action_std = np.ones((horizon, self.action_dim)) * 0.5

        # CEM iterations
        num_elites = num_samples // 10
        for _ in range(5):
            # Sample action sequences
            action_sequences = np.random.normal(
                action_mean, action_std,
                size=(num_samples, horizon, self.action_dim)
            )
            action_sequences = np.clip(action_sequences, self.action_low, self.action_high)

            # Evaluate sequences
            costs = self._evaluate_sequences(current_state, action_sequences, target)

            # Select elites
            elite_indices = np.argsort(costs)[:num_elites]
            elite_actions = action_sequences[elite_indices]

            # Update distribution
            action_mean = elite_actions.mean(axis=0)
            action_std = elite_actions.std(axis=0) + 1e-6

        return action_mean[0]

    def _evaluate_sequences(
        self,
        start_state: np.ndarray,
        action_sequences: np.ndarray,
        target: np.ndarray
    ) -> np.ndarray:
        """Evaluate action sequences."""
        num_samples, horizon, _ = action_sequences.shape
        costs = np.zeros(num_samples)

        # Extract dynamics-relevant state (first 6 dims for position + velocity)
        dynamics_state_dim = self.dynamics_model.state_dim
        dynamics_state = start_state[:dynamics_state_dim] if len(start_state) > dynamics_state_dim else start_state
        states = np.tile(dynamics_state, (num_samples, 1))

        for t in range(horizon):
            actions = action_sequences[:, t, :]

            # Predict next states
            states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
            actions_tensor = torch.tensor(actions, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                next_states = self.dynamics_model.forward(states_tensor, actions_tensor)
            states = next_states.cpu().numpy()

            # Cost: distance to target + action penalty
            positions = states[:, :3]
            distances = np.linalg.norm(positions - target, axis=1)
            action_costs = 0.01 * np.sum(actions ** 2, axis=1)

            costs += distances + action_costs

        return costs

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Predict action using MPC."""
        return self.plan_actions(observation)

    def step(self, observation: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Take a step (for compatibility with other agents)."""
        action = self.predict(observation)
        return action, {}


if __name__ == "__main__":
    print("Testing Model-Based RL agents...")

    # Create test environment
    from crm_ml_rl.envs import ReachingEnv

    env = ReachingEnv()

    # Test DynaAgent
    print("\n1. Testing DynaAgent...")
    config = ModelBasedConfig(use_cpp_physics=False)
    dyna = DynaAgent(
        env,
        config=config,
        base_algorithm="sac",
        verbose=0,
        feature_extractor="mlp"
    )
    obs, _ = env.reset()
    action = dyna.predict(obs)
    print(f"   DynaAgent action shape: {action.shape}")

    # Test MBPOAgent
    print("\n2. Testing MBPOAgent...")
    mbpo = MBPOAgent(
        env,
        config=config,
        verbose=0,
        feature_extractor="deep_residual"
    )
    action = mbpo.predict(obs)
    print(f"   MBPOAgent action shape: {action.shape}")

    # Test MPCAgent (needs dynamics model)
    print("\n3. Testing MPCAgent...")
    dynamics_config = HybridDynamicsConfig(use_cpp=False)
    dynamics = HybridDynamicsModel(dynamics_config)
    mpc = MPCAgent(env, dynamics, config)
    action = mpc.predict(obs)
    print(f"   MPCAgent action shape: {action.shape}")

    print("\nModel-based RL tests complete!")
