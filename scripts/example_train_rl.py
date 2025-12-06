#!/usr/bin/env python3
"""
Example RL training script for catheter control.

This script demonstrates how to train an RL agent (PPO, SAC, or TD3)
on the catheter reaching task using the C++ physics bindings.

Usage:
    # Quick training demo (1000 steps)
    python scripts/example_train_rl.py --demo

    # Full training with PPO
    python scripts/example_train_rl.py --algorithm ppo --timesteps 100000

    # Compare algorithms
    python scripts/example_train_rl.py --compare
"""

import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def create_env(env_type="reaching", seed=0):
    """Create a catheter control environment."""
    from crm_ml_rl.envs import CatheterEnv, ReachingEnv, TrackingEnv, CatheterEnvConfig

    config = CatheterEnvConfig(
        dt=0.02,
        max_steps=200,
        max_current=0.3,
        success_threshold=5.0,  # 5mm
        position_reward_scale=1.0,
        action_penalty_scale=0.01,
        smoothness_penalty_scale=0.01,
        success_bonus=10.0
    )

    if env_type == "reaching":
        env = ReachingEnv(config=config, random_target=True)
    elif env_type == "tracking":
        env = TrackingEnv(
            config=config,
            trajectory_type="circle",
            trajectory_freq=0.5,
            trajectory_radius=10.0
        )
    else:
        env = CatheterEnv(config=config)

    env.reset(seed=seed)
    return env


def quick_training_demo():
    """Quick training demonstration with minimal timesteps."""
    print("\n" + "=" * 60)
    print("Quick Training Demo (PPO, 2000 timesteps)")
    print("=" * 60)

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.evaluation import evaluate_policy

    # Create environment
    print("\nCreating environment...")
    env = DummyVecEnv([lambda: create_env("reaching", seed=42)])

    # Create PPO agent
    print("Creating PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        verbose=1
    )

    # Train for a few steps
    print("\nTraining for 2000 timesteps...")
    model.learn(total_timesteps=2000)

    # Evaluate
    print("\nEvaluating trained agent...")
    eval_env = create_env("reaching", seed=100)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5)
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

    # Run a demo episode
    print("\nRunning demo episode...")
    obs, info = eval_env.reset()
    total_reward = 0
    positions = []

    for step in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward
        positions.append(info['tip_position'].copy())

        if terminated or truncated:
            print(f"  Episode ended at step {step+1}")
            print(f"  Final distance to target: {info.get('distance', 'N/A')}mm")
            print(f"  Success: {info.get('success', False)}")
            break

    print(f"  Total reward: {total_reward:.2f}")

    # Cleanup
    env.close()
    eval_env.close()

    print("\n[DEMO COMPLETE]")
    return model


def train_agent(algorithm="ppo", total_timesteps=100000, output_dir="trained_models"):
    """Train an RL agent."""
    print("\n" + "=" * 60)
    print(f"Training {algorithm.upper()} for {total_timesteps} timesteps")
    print("=" * 60)

    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"{algorithm}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {run_dir}")

    # Create environments
    print("\nCreating environments...")

    def make_env(seed):
        def _init():
            env = create_env("reaching", seed=seed)
            return Monitor(env)
        return _init

    n_envs = 4
    train_env = DummyVecEnv([make_env(i) for i in range(n_envs)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

    eval_env = DummyVecEnv([make_env(1000)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    # Create agent
    print(f"Creating {algorithm.upper()} agent...")

    if algorithm == "ppo":
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            verbose=1,
            tensorboard_log=str(run_dir / "logs")
        )
    elif algorithm == "sac":
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            buffer_size=100000,
            batch_size=256,
            gamma=0.99,
            verbose=1,
            tensorboard_log=str(run_dir / "logs")
        )
    elif algorithm == "td3":
        model = TD3(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            buffer_size=100000,
            batch_size=256,
            gamma=0.99,
            verbose=1,
            tensorboard_log=str(run_dir / "logs")
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(run_dir / "eval_logs"),
        eval_freq=max(total_timesteps // 20, 1000) // n_envs,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(total_timesteps // 5, 1000) // n_envs,
        save_path=str(run_dir / "checkpoints"),
        name_prefix="model"
    )

    # Train
    print("\nStarting training...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    # Save final model
    model.save(str(run_dir / "final_model"))
    train_env.save(str(run_dir / "vec_normalize.pkl"))

    print(f"\nTraining complete! Models saved to: {run_dir}")

    # Cleanup
    train_env.close()
    eval_env.close()

    return run_dir


def evaluate_trained_model(model_path, n_episodes=10):
    """Evaluate a trained model."""
    print("\n" + "=" * 60)
    print(f"Evaluating model from: {model_path}")
    print("=" * 60)

    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    model_path = Path(model_path)

    # Find model file
    if (model_path / "final_model.zip").exists():
        model_file = model_path / "final_model.zip"
    elif (model_path / "best_model" / "best_model.zip").exists():
        model_file = model_path / "best_model" / "best_model.zip"
    else:
        model_file = model_path

    # Determine algorithm from path name
    algorithm = "ppo"
    if "sac" in str(model_path).lower():
        algorithm = "sac"
    elif "td3" in str(model_path).lower():
        algorithm = "td3"

    # Load model
    print(f"Loading {algorithm.upper()} model...")
    if algorithm == "ppo":
        model = PPO.load(str(model_file))
    elif algorithm == "sac":
        model = SAC.load(str(model_file))
    elif algorithm == "td3":
        model = TD3.load(str(model_file))

    # Create evaluation environment
    env = create_env("reaching", seed=9999)

    # Load normalization if available
    vec_norm_path = model_path / "vec_normalize.pkl"
    if vec_norm_path.exists():
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize.load(str(vec_norm_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        use_vec = True
    else:
        vec_env = env
        use_vec = False

    # Evaluate
    episode_rewards = []
    episode_lengths = []
    successes = []

    for ep in range(n_episodes):
        if use_vec:
            obs = vec_env.reset()
        else:
            obs, _ = env.reset()

        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if use_vec:
                obs, reward, done, info = vec_env.step(action)
                done = done[0]
                reward = reward[0]
                info = info[0]
            else:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            total_reward += reward
            steps += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        successes.append(info.get('success', False))

    # Results
    print(f"\nResults over {n_episodes} episodes:")
    print(f"  Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Mean length: {np.mean(episode_lengths):.1f}")
    print(f"  Success rate: {np.mean(successes) * 100:.1f}%")

    if use_vec:
        vec_env.close()
    else:
        env.close()

    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'success_rate': np.mean(successes)
    }


def compare_algorithms(total_timesteps=50000):
    """Compare PPO, SAC, and TD3."""
    print("\n" + "=" * 60)
    print("Comparing RL Algorithms")
    print("=" * 60)

    algorithms = ["ppo", "sac", "td3"]
    results = {}

    for algo in algorithms:
        print(f"\n{'='*40}")
        print(f"Training {algo.upper()}")
        print('='*40)

        run_dir = train_agent(
            algorithm=algo,
            total_timesteps=total_timesteps,
            output_dir="trained_models/comparison"
        )

        eval_result = evaluate_trained_model(run_dir, n_episodes=10)
        results[algo] = eval_result

    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"{'Algorithm':<10} {'Mean Reward':<15} {'Success Rate':<15}")
    print("-" * 40)
    for algo, res in results.items():
        print(f"{algo.upper():<10} {res['mean_reward']:>8.2f} ± {res['std_reward']:<4.2f} {res['success_rate']*100:>10.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description="Example RL training for catheter control")
    parser.add_argument("--demo", action="store_true", help="Run quick training demo")
    parser.add_argument("--algorithm", type=str, default="ppo",
                        choices=["ppo", "sac", "td3"], help="RL algorithm")
    parser.add_argument("--timesteps", type=int, default=100000,
                        help="Total training timesteps")
    parser.add_argument("--output-dir", type=str, default="trained_models",
                        help="Output directory for models")
    parser.add_argument("--evaluate", type=str, default=None,
                        help="Path to model to evaluate")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all algorithms")

    args = parser.parse_args()

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if args.demo:
        quick_training_demo()
    elif args.evaluate:
        evaluate_trained_model(args.evaluate)
    elif args.compare:
        compare_algorithms(total_timesteps=args.timesteps)
    else:
        train_agent(
            algorithm=args.algorithm,
            total_timesteps=args.timesteps,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()
