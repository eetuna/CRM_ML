"""
Training script for RL agents on catheter control tasks.

Supports SAC, PPO, and TD3 algorithms with custom environments.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Type
from dataclasses import dataclass, field
import json
from datetime import datetime
import gymnasium as gym

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from crm_ml_rl.envs import CatheterEnv, TrackingEnv, ReachingEnv, CatheterEnvConfig
from crm_ml_rl.training.rl_agents import SACAgent, PPOAgent, TD3Agent


@dataclass
class RLTrainingConfig:
    """RL training configuration."""
    # Algorithm
    algorithm: str = "sac"  # "sac", "ppo", "td3"

    # Environment
    env_type: str = "reaching"  # "reaching", "tracking"
    trajectory_type: str = "circle"  # For tracking env
    trajectory_freq: float = 0.5
    trajectory_radius: float = 15.0

    # Training
    total_timesteps: int = 500_000
    n_envs: int = 4
    eval_freq: int = 10_000
    n_eval_episodes: int = 10
    save_freq: int = 50_000

    # Algorithm hyperparameters
    learning_rate: float = 3e-4
    batch_size: int = 256
    gamma: float = 0.99
    buffer_size: int = 1_000_000

    # PPO specific
    n_steps: int = 2048
    n_epochs: int = 10
    clip_range: float = 0.2

    # TD3 specific
    policy_delay: int = 2
    target_policy_noise: float = 0.2

    # Environment config
    max_steps: int = 200
    dt: float = 0.02
    action_scale: float = 0.3

    # Normalization
    normalize_obs: bool = True
    normalize_reward: bool = True

    # Feature extractor settings
    feature_extractor: str = "default"  # "default", "mlp", "deep_residual", "lstm", "physics", "catheter"
    features_dim: int = 128
    extractor_hidden_dims: List[int] = None  # Hidden dims for MLP-based extractors
    extractor_dropout: float = 0.1
    extractor_num_blocks: int = 4  # For deep_residual extractor

    # Physics-informed extractor settings
    use_cpp_physics: bool = False  # Use C++ bindings for physics features
    param_file: Optional[str] = None
    config_file: Optional[str] = None

    # Model-based RL settings
    use_model_based: bool = False
    dynamics_model_path: Optional[str] = None
    imagined_data_ratio: float = 0.5
    planning_horizon: int = 10

    def __post_init__(self):
        if self.extractor_hidden_dims is None:
            self.extractor_hidden_dims = [256, 256]


def make_env(
    env_type: str,
    config: RLTrainingConfig,
    seed: int = 0,
    monitor: bool = True
):
    """Create environment with optional monitoring."""
    def _init():
        env_config = CatheterEnvConfig(
            max_steps=config.max_steps,
            dt=config.dt,
            action_scale=config.action_scale
        )

        if env_type == "reaching":
            env = ReachingEnv(config=env_config, random_target=True)
        elif env_type == "tracking":
            env = TrackingEnv(
                config=env_config,
                trajectory_type=config.trajectory_type,
                trajectory_freq=config.trajectory_freq,
                trajectory_radius=config.trajectory_radius
            )
        else:
            env = CatheterEnv(config=env_config)

        env.reset(seed=seed)

        if monitor:
            env = Monitor(env)

        return env

    return _init


def create_vec_env(
    config: RLTrainingConfig,
    use_subprocess: bool = True,
    seed: int = 0
) -> VecNormalize:
    """Create vectorized environment."""
    env_fns = [
        make_env(config.env_type, config, seed=seed + i)
        for i in range(config.n_envs)
    ]

    if use_subprocess and config.n_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    if config.normalize_obs or config.normalize_reward:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=config.normalize_obs,
            norm_reward=config.normalize_reward,
            clip_obs=10.0,
            clip_reward=10.0
        )

    return vec_env


def create_agent(
    config: RLTrainingConfig,
    env: gym.Env,
    device: str = "auto"
):
    """Create RL agent based on config."""
    # Build extractor kwargs based on extractor type
    extractor_kwargs = {}
    if config.feature_extractor == "mlp":
        extractor_kwargs = {
            "hidden_dims": config.extractor_hidden_dims,
            "dropout": config.extractor_dropout
        }
    elif config.feature_extractor == "deep_residual":
        extractor_kwargs = {
            "hidden_dim": config.extractor_hidden_dims[0] if config.extractor_hidden_dims else 256,
            "num_blocks": config.extractor_num_blocks,
            "dropout": config.extractor_dropout
        }
    elif config.feature_extractor == "physics":
        extractor_kwargs = {
            "use_cpp": config.use_cpp_physics,
            "param_file": config.param_file,
            "config_file": config.config_file
        }
    elif config.feature_extractor == "catheter":
        extractor_kwargs = {
            "state_dim": 6,
            "target_dim": 3
        }

    if config.algorithm == "sac":
        return SACAgent(
            env=env,
            learning_rate=config.learning_rate,
            buffer_size=config.buffer_size,
            batch_size=config.batch_size,
            gamma=config.gamma,
            device=device,
            feature_extractor=config.feature_extractor,
            features_dim=config.features_dim,
            extractor_kwargs=extractor_kwargs
        )
    elif config.algorithm == "ppo":
        return PPOAgent(
            env=env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            clip_range=config.clip_range,
            device=device,
            feature_extractor=config.feature_extractor,
            features_dim=config.features_dim,
            extractor_kwargs=extractor_kwargs
        )
    elif config.algorithm == "td3":
        return TD3Agent(
            env=env,
            learning_rate=config.learning_rate,
            buffer_size=config.buffer_size,
            batch_size=config.batch_size,
            gamma=config.gamma,
            policy_delay=config.policy_delay,
            target_policy_noise=config.target_policy_noise,
            device=device,
            feature_extractor=config.feature_extractor,
            features_dim=config.features_dim,
            extractor_kwargs=extractor_kwargs
        )
    else:
        raise ValueError(f"Unknown algorithm: {config.algorithm}")


def train_rl_agent(
    config: RLTrainingConfig,
    output_dir: str,
    seed: int = 0,
    device: str = "auto"
) -> Dict:
    """
    Train RL agent.

    Args:
        config: Training configuration
        output_dir: Directory to save results
        seed: Random seed
        device: Device for training

    Returns:
        Training results
    """
    # Setup directories
    output_path = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.algorithm}_{config.env_type}_{timestamp}"
    run_dir = output_path / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training {config.algorithm.upper()} on {config.env_type} environment")
    print(f"Output directory: {run_dir}")

    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create environments
    print("Creating environments...")
    train_env = create_vec_env(config, use_subprocess=True, seed=seed)
    eval_env = create_vec_env(config, use_subprocess=False, seed=seed + 1000)

    # Create agent
    print("Creating agent...")
    agent = create_agent(config, train_env, device=device)

    # Callbacks
    callbacks = []

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(run_dir / "logs"),
        eval_freq=config.eval_freq // config.n_envs,
        n_eval_episodes=config.n_eval_episodes,
        deterministic=True,
        verbose=1
    )
    callbacks.append(eval_callback)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config.save_freq // config.n_envs,
        save_path=str(run_dir / "checkpoints"),
        name_prefix="model"
    )
    callbacks.append(checkpoint_callback)

    # Save config
    with open(run_dir / "config.json", 'w') as f:
        json.dump(config.__dict__, f, indent=2)

    # Train
    print(f"Starting training for {config.total_timesteps} timesteps...")
    try:
        agent.model.learn(
            total_timesteps=config.total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    # Save final model
    agent.save(str(run_dir / "final_model"))

    # Save normalization stats
    if isinstance(train_env, VecNormalize):
        train_env.save(str(run_dir / "vec_normalize.pkl"))

    # Clean up
    train_env.close()
    eval_env.close()

    print(f"\nTraining complete!")
    print(f"Results saved to: {run_dir}")

    return {
        'run_dir': str(run_dir),
        'algorithm': config.algorithm,
        'env_type': config.env_type,
        'total_timesteps': config.total_timesteps
    }


def compare_algorithms(
    output_dir: str,
    env_type: str = "reaching",
    algorithms: List[str] = None,
    total_timesteps: int = 200_000,
    n_seeds: int = 3
) -> Dict:
    """
    Compare different RL algorithms.

    Args:
        output_dir: Directory to save results
        env_type: Environment type
        algorithms: List of algorithms to compare
        total_timesteps: Training timesteps per run
        n_seeds: Number of random seeds

    Returns:
        Comparison results
    """
    if algorithms is None:
        algorithms = ["sac", "ppo", "td3"]

    results = {}

    for algo in algorithms:
        print(f"\n{'='*50}")
        print(f"Training {algo.upper()}")
        print('='*50)

        algo_results = []

        for seed in range(n_seeds):
            print(f"\nSeed {seed + 1}/{n_seeds}")

            config = RLTrainingConfig(
                algorithm=algo,
                env_type=env_type,
                total_timesteps=total_timesteps
            )

            result = train_rl_agent(
                config=config,
                output_dir=output_dir,
                seed=seed
            )
            algo_results.append(result)

        results[algo] = algo_results

    # Summary
    print("\n" + "="*50)
    print("COMPARISON COMPLETE")
    print("="*50)
    for algo, runs in results.items():
        print(f"\n{algo.upper()}: {len(runs)} runs completed")

    return results


def evaluate_agent(
    model_path: str,
    env_type: str = "reaching",
    n_episodes: int = 100,
    render: bool = False
) -> Dict:
    """
    Evaluate trained agent.

    Args:
        model_path: Path to saved model
        env_type: Environment type
        n_episodes: Number of evaluation episodes
        render: Whether to render

    Returns:
        Evaluation results
    """
    from stable_baselines3 import SAC, PPO, TD3

    # Load model
    model_path = Path(model_path)
    if (model_path / "final_model.zip").exists():
        model_file = model_path / "final_model.zip"
    elif (model_path / "best_model" / "best_model.zip").exists():
        model_file = model_path / "best_model" / "best_model.zip"
    else:
        model_file = model_path

    # Try to determine algorithm from config
    config_path = model_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        algorithm = config.get('algorithm', 'sac')
    else:
        algorithm = 'sac'

    # Load appropriate model
    if algorithm == 'sac':
        model = SAC.load(str(model_file))
    elif algorithm == 'ppo':
        model = PPO.load(str(model_file))
    elif algorithm == 'td3':
        model = TD3.load(str(model_file))

    # Create environment
    env_config = CatheterEnvConfig()
    if env_type == "reaching":
        env = ReachingEnv(config=env_config, random_target=True)
    elif env_type == "tracking":
        env = TrackingEnv(config=env_config)
    else:
        env = CatheterEnv(config=env_config)

    # Load normalization if available
    vec_norm_path = model_path / "vec_normalize.pkl"
    if vec_norm_path.exists():
        from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
        env = DummyVecEnv([lambda: env])
        env = VecNormalize.load(str(vec_norm_path), env)
        env.training = False
        env.norm_reward = False

    # Evaluate
    episode_rewards = []
    episode_lengths = []
    successes = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

            if render:
                env.render()

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        successes.append(info.get('success', info.get('reached_target', False)))

    env.close()

    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'success_rate': np.mean(successes),
        'n_episodes': n_episodes
    }

    print(f"\nEvaluation Results ({n_episodes} episodes):")
    print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Mean Length: {results['mean_length']:.1f}")
    print(f"  Success Rate: {results['success_rate']*100:.1f}%")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train RL agent")
    parser.add_argument("--algorithm", type=str, default="sac",
                        choices=["sac", "ppo", "td3"], help="RL algorithm")
    parser.add_argument("--env-type", type=str, default="reaching",
                        choices=["reaching", "tracking"], help="Environment type")
    parser.add_argument("--total-timesteps", type=int, default=500_000,
                        help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Number of parallel environments")
    parser.add_argument("--output-dir", type=str, default="crm_ml_rl/trained_models/rl",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all algorithms")
    parser.add_argument("--evaluate", type=str, default=None,
                        help="Path to model to evaluate")

    args = parser.parse_args()

    if args.evaluate:
        evaluate_agent(
            model_path=args.evaluate,
            env_type=args.env_type,
            n_episodes=100
        )
    elif args.compare:
        compare_algorithms(
            output_dir=args.output_dir,
            env_type=args.env_type,
            total_timesteps=args.total_timesteps
        )
    else:
        config = RLTrainingConfig(
            algorithm=args.algorithm,
            env_type=args.env_type,
            total_timesteps=args.total_timesteps,
            n_envs=args.n_envs
        )

        train_rl_agent(
            config=config,
            output_dir=args.output_dir,
            seed=args.seed
        )
