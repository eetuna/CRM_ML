"""
RL agent implementations for catheter control.

Uses stable-baselines3 with custom configurations for catheter tasks.
Supports custom feature extractors from crm_ml_rl/models/.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, Type, Union, Any, List
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from .custom_policies import (
    FEATURE_EXTRACTORS,
    create_policy_kwargs,
    MLPFeaturesExtractor,
    DeepResidualFeaturesExtractor,
    LSTMFeaturesExtractor,
    PhysicsInformedExtractor,
    CatheterMLPExtractor
)


class CatheterFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for catheter observations.

    Separates state, target, and other components for better learning.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 128,
        state_dim: int = 6,  # position + velocity
        target_dim: int = 3,  # target position
    ):
        super().__init__(observation_space, features_dim)

        self.state_dim = state_dim
        self.target_dim = target_dim

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Target encoder
        self.target_encoder = nn.Sequential(
            nn.Linear(target_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        # Remaining features encoder (action history, etc.)
        remaining_dim = observation_space.shape[0] - state_dim - target_dim
        if remaining_dim > 0:
            self.has_extra = True
            self.extra_encoder = nn.Sequential(
                nn.Linear(remaining_dim, 32),
                nn.ReLU()
            )
            combined_dim = 64 + 32 + 32
        else:
            self.has_extra = False
            combined_dim = 64 + 32

        # Final combination
        self.combiner = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Split observation
        state = observations[:, :self.state_dim]
        target = observations[:, self.state_dim:self.state_dim + self.target_dim]

        # Encode components
        state_features = self.state_encoder(state)
        target_features = self.target_encoder(target)

        if self.has_extra:
            extra = observations[:, self.state_dim + self.target_dim:]
            extra_features = self.extra_encoder(extra)
            combined = torch.cat([state_features, target_features, extra_features], dim=1)
        else:
            combined = torch.cat([state_features, target_features], dim=1)

        return self.combiner(combined)


class TrainingCallback(BaseCallback):
    """Custom callback for logging during training."""

    def __init__(self, log_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Log episode statistics
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            if 'r' in ep_info:
                self.episode_rewards.append(ep_info['r'])
            if 'l' in ep_info:
                self.episode_lengths.append(ep_info['l'])

        if self.n_calls % self.log_freq == 0 and self.verbose > 0:
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-100:])
                print(f"Step {self.n_calls}: Mean reward (last 100): {mean_reward:.2f}")

        return True


class SACAgent:
    """
    Soft Actor-Critic agent for catheter control.

    SAC is good for continuous control with entropy regularization.
    Supports custom feature extractors from crm_ml_rl/models/.
    """

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 3e-4,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        ent_coef: Union[str, float] = "auto",
        policy_kwargs: Optional[Dict] = None,
        device: str = "auto",
        verbose: int = 1,
        feature_extractor: str = "default",
        features_dim: int = 128,
        extractor_kwargs: Optional[Dict] = None
    ):
        """
        Initialize SAC agent.

        Args:
            env: Gymnasium environment
            learning_rate: Learning rate
            buffer_size: Replay buffer size
            batch_size: Batch size for training
            gamma: Discount factor
            tau: Soft update coefficient
            ent_coef: Entropy coefficient ("auto" for automatic tuning)
            policy_kwargs: Additional policy arguments
            device: Device for training
            verbose: Verbosity level
            feature_extractor: Feature extractor type ("default", "mlp", "deep_residual", "lstm", "physics", "catheter")
            features_dim: Output dimension of feature extractor
            extractor_kwargs: Additional kwargs for feature extractor
        """
        self.env = env
        self.feature_extractor_type = feature_extractor

        # Create policy kwargs with custom feature extractor
        if policy_kwargs is None:
            if extractor_kwargs is None:
                extractor_kwargs = {}

            policy_kwargs = create_policy_kwargs(
                feature_extractor=feature_extractor,
                features_dim=features_dim,
                net_arch=[256, 256],
                activation_fn=nn.ReLU,
                algorithm="sac",
                **extractor_kwargs
            )

        self.model = SAC(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            gamma=gamma,
            tau=tau,
            ent_coef=ent_coef,
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=verbose
        )

    def train(
        self,
        total_timesteps: int,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 10,
        save_path: Optional[str] = None,
        log_freq: int = 1000
    ) -> Dict[str, Any]:
        """Train the agent."""
        callbacks = [TrainingCallback(log_freq=log_freq, verbose=1)]

        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=save_path,
                log_path=save_path,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True
            )
            callbacks.append(eval_callback)

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )

        if save_path:
            self.model.save(Path(save_path) / "final_model")

        return {"model": self.model}

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> np.ndarray:
        """Predict action."""
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def save(self, path: str):
        """Save model."""
        self.model.save(path)

    def load(self, path: str):
        """Load model."""
        self.model = SAC.load(path, env=self.env)


class PPOAgent:
    """
    Proximal Policy Optimization agent.

    PPO is stable and works well for many tasks.
    Supports custom feature extractors from crm_ml_rl/models/.
    """

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        policy_kwargs: Optional[Dict] = None,
        device: str = "auto",
        verbose: int = 1,
        feature_extractor: str = "default",
        features_dim: int = 128,
        extractor_kwargs: Optional[Dict] = None
    ):
        """
        Initialize PPO agent.

        Args:
            env: Gymnasium environment
            learning_rate: Learning rate
            n_steps: Steps per update
            batch_size: Minibatch size
            n_epochs: Number of epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_range: PPO clip range
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Max gradient norm
            policy_kwargs: Additional policy arguments
            device: Device for training
            verbose: Verbosity level
            feature_extractor: Feature extractor type ("default", "mlp", "deep_residual", "lstm", "physics", "catheter")
            features_dim: Output dimension of feature extractor
            extractor_kwargs: Additional kwargs for feature extractor
        """
        self.env = env
        self.feature_extractor_type = feature_extractor

        # Create policy kwargs with custom feature extractor
        if policy_kwargs is None:
            if extractor_kwargs is None:
                extractor_kwargs = {}

            # PPO uses separate policy and value networks
            policy_kwargs = create_policy_kwargs(
                feature_extractor=feature_extractor,
                features_dim=features_dim,
                net_arch=[dict(pi=[256, 256], vf=[256, 256])],
                activation_fn=nn.ReLU,
                **extractor_kwargs
            )

        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=verbose
        )

    def train(
        self,
        total_timesteps: int,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 10,
        save_path: Optional[str] = None,
        log_freq: int = 1000
    ) -> Dict[str, Any]:
        """Train the agent."""
        callbacks = [TrainingCallback(log_freq=log_freq, verbose=1)]

        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=save_path,
                log_path=save_path,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True
            )
            callbacks.append(eval_callback)

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )

        if save_path:
            self.model.save(Path(save_path) / "final_model")

        return {"model": self.model}

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> np.ndarray:
        """Predict action."""
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def save(self, path: str):
        """Save model."""
        self.model.save(path)

    def load(self, path: str):
        """Load model."""
        self.model = PPO.load(path, env=self.env)


class TD3Agent:
    """
    Twin Delayed DDPG agent.

    TD3 is robust for continuous control with less hyperparameter sensitivity.
    Supports custom feature extractors from crm_ml_rl/models/.
    """

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 3e-4,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        action_noise: Optional[str] = "normal",
        noise_std: float = 0.1,
        policy_kwargs: Optional[Dict] = None,
        device: str = "auto",
        verbose: int = 1,
        feature_extractor: str = "default",
        features_dim: int = 128,
        extractor_kwargs: Optional[Dict] = None
    ):
        """
        Initialize TD3 agent.

        Args:
            env: Gymnasium environment
            learning_rate: Learning rate
            buffer_size: Replay buffer size
            batch_size: Batch size
            gamma: Discount factor
            tau: Soft update coefficient
            policy_delay: Delay between policy updates
            target_policy_noise: Noise added to target policy
            target_noise_clip: Clip for target noise
            action_noise: Type of action noise ("normal" or "ou")
            noise_std: Standard deviation of action noise
            policy_kwargs: Additional policy arguments
            device: Device for training
            verbose: Verbosity level
            feature_extractor: Feature extractor type ("default", "mlp", "deep_residual", "lstm", "physics", "catheter")
            features_dim: Output dimension of feature extractor
            extractor_kwargs: Additional kwargs for feature extractor
        """
        self.env = env
        self.feature_extractor_type = feature_extractor

        # Create policy kwargs with custom feature extractor
        if policy_kwargs is None:
            if extractor_kwargs is None:
                extractor_kwargs = {}

            policy_kwargs = create_policy_kwargs(
                feature_extractor=feature_extractor,
                features_dim=features_dim,
                net_arch=[256, 256],
                activation_fn=nn.ReLU,
                algorithm="td3",
                **extractor_kwargs
            )

        # Action noise
        n_actions = env.action_space.shape[0]
        if action_noise == "normal":
            action_noise_obj = NormalActionNoise(
                mean=np.zeros(n_actions),
                sigma=noise_std * np.ones(n_actions)
            )
        elif action_noise == "ou":
            action_noise_obj = OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(n_actions),
                sigma=noise_std * np.ones(n_actions)
            )
        else:
            action_noise_obj = None

        self.model = TD3(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            gamma=gamma,
            tau=tau,
            policy_delay=policy_delay,
            target_policy_noise=target_policy_noise,
            target_noise_clip=target_noise_clip,
            action_noise=action_noise_obj,
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=verbose
        )

    def train(
        self,
        total_timesteps: int,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 10,
        save_path: Optional[str] = None,
        log_freq: int = 1000
    ) -> Dict[str, Any]:
        """Train the agent."""
        callbacks = [TrainingCallback(log_freq=log_freq, verbose=1)]

        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=save_path,
                log_path=save_path,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True
            )
            callbacks.append(eval_callback)

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )

        if save_path:
            self.model.save(Path(save_path) / "final_model")

        return {"model": self.model}

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> np.ndarray:
        """Predict action."""
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def save(self, path: str):
        """Save model."""
        self.model.save(path)

    def load(self, path: str):
        """Load model."""
        self.model = TD3.load(path, env=self.env)


def create_vec_env(
    env_class: Type[gym.Env],
    n_envs: int = 4,
    seed: int = 0,
    use_subprocess: bool = True,
    normalize: bool = True,
    **env_kwargs
) -> VecNormalize:
    """
    Create vectorized environment for parallel training.

    Args:
        env_class: Environment class
        n_envs: Number of parallel environments
        seed: Random seed
        use_subprocess: Use subprocesses for parallelization
        normalize: Apply observation/reward normalization
        **env_kwargs: Environment arguments

    Returns:
        Vectorized environment
    """
    def make_env(rank: int):
        def _init():
            env = env_class(**env_kwargs)
            env.reset(seed=seed + rank)
            return env
        return _init

    if use_subprocess and n_envs > 1:
        vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    else:
        vec_env = DummyVecEnv([make_env(i) for i in range(n_envs)])

    if normalize:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    return vec_env


if __name__ == "__main__":
    print("Testing RL agents...")

    # Create dummy environment for testing
    from crm_ml_rl.envs import ReachingEnv

    env = ReachingEnv()

    print("\nTesting SAC agent...")
    sac = SACAgent(env, verbose=0)
    print(f"SAC model created: {type(sac.model)}")

    print("\nTesting PPO agent...")
    ppo = PPOAgent(env, verbose=0)
    print(f"PPO model created: {type(ppo.model)}")

    print("\nTesting TD3 agent...")
    td3 = TD3Agent(env, verbose=0)
    print(f"TD3 model created: {type(td3.model)}")

    # Quick prediction test
    obs, _ = env.reset()
    action = sac.predict(obs)
    print(f"\nSAC prediction shape: {action.shape}")
