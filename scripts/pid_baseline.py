#!/usr/bin/env python3
"""
Simple PID baseline controller for CatheterEnv reaching and tracking tasks.

Usage:
    python3 scripts/pid_baseline.py --env reaching --episodes 5
    python3 scripts/pid_baseline.py --env tracking --episodes 5

Notes:
- Uses the default CatheterEnv config (CRM or simplified backend depending on availability).
- Gains are heuristic; adjust with flags for better performance.
- Actions are clipped to env action space.
"""

import argparse
import numpy as np
from pathlib import Path

import gymnasium as gym

from crm_ml_rl.envs.catheter_env import CatheterEnv, CatheterEnvConfig


def build_env(env_type: str, seed: int = 0) -> gym.Env:
    cfg = CatheterEnvConfig()
    if env_type == "reaching":
        from crm_ml_rl.envs.reaching_env import ReachingEnv
        env = ReachingEnv(cfg)
    elif env_type == "tracking":
        from crm_ml_rl.envs.tracking_env import TrackingEnv
        env = TrackingEnv(cfg)
    else:
        raise ValueError(f"Unknown env type: {env_type}")
    return env


def pid_controller(error, integral, derivative, kp, ki, kd):
    return kp * error + ki * integral + kd * derivative


def run_pid(env_type: str, episodes: int, kp: float, ki: float, kd: float, max_steps: int):
    env = build_env(env_type)
    returns = []
    lengths = []

    for ep in range(episodes):
        obs, info = env.reset()
        target = info.get("target_position", np.zeros(3))
        tip_pos = obs[:3]
        tip_vel = obs[3:6] if env.unwrapped.config.include_velocity else np.zeros(3)

        integral_err = np.zeros(3)
        prev_err = target - tip_pos

        total_reward = 0.0
        for t in range(max_steps):
            tip_pos = obs[:3]
            tip_vel = obs[3:6] if env.unwrapped.config.include_velocity else np.zeros(3)
            target = info.get("target_position", target)

            error = target - tip_pos
            integral_err += error * env.unwrapped.config.dt
            derivative = (error - prev_err) / env.unwrapped.config.dt

            action = pid_controller(error, integral_err, derivative, kp, ki, kd)
            action = np.clip(action, env.action_space.low, env.action_space.high)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            prev_err = error

            if terminated or truncated:
                break

        returns.append(total_reward)
        lengths.append(t + 1)
        print(f"Episode {ep+1}: return={total_reward:.2f}, length={lengths[-1]}")

    env.close()
    print("\nSummary:")
    print(f"  Episodes: {episodes}")
    print(f"  Mean return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"  Mean length: {np.mean(lengths):.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["reaching", "tracking"], default="reaching")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--kp", type=float, default=0.05)
    parser.add_argument("--ki", type=float, default=0.0)
    parser.add_argument("--kd", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()

    run_pid(args.env, args.episodes, args.kp, args.ki, args.kd, args.max_steps)


if __name__ == "__main__":
    main()
