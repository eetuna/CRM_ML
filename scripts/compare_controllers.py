#!/usr/bin/env python3
"""
Compare controllers: PID baseline, PPO (model-free), and a simple model-based shooter
using the hybrid dynamics model for one-step lookahead.

Usage examples:
  python3 scripts/compare_controllers.py --env reaching --episodes 3
  python3 scripts/compare_controllers.py --env tracking --episodes 3 --ppo-steps 2000

The PPO training here is intentionally short for quick comparisons.
"""

import argparse
import numpy as np
from typing import Callable

from crm_ml_rl.envs.catheter_env import CatheterEnv, CatheterEnvConfig
from crm_ml_rl.envs.reaching_env import ReachingEnv
from crm_ml_rl.envs.tracking_env import TrackingEnv
from crm_ml_rl.models.hybrid_dynamics import HybridDynamicsModel, HybridDynamicsConfig

from stable_baselines3 import PPO


# ---------------------- Controllers ----------------------
def pid_action(error, integral, derivative, kp, ki, kd):
    return kp * error + ki * integral + kd * derivative


def build_env(env_name: str):
    cfg = CatheterEnvConfig()
    if env_name == "reaching":
        return ReachingEnv(cfg)
    elif env_name == "tracking":
        return TrackingEnv(cfg)
    else:
        raise ValueError(f"Unknown env {env_name}")


def eval_controller(env_builder: Callable, policy_fn: Callable, episodes: int, max_steps: int):
    rewards, lengths = [], []
    for _ in range(episodes):
        env = env_builder()
        obs, info = env.reset()
        target = info.get("target_position", np.zeros(3))
        integral = np.zeros(3)
        prev_err = target - obs[:3]
        total = 0.0
        for t in range(max_steps):
            action = policy_fn(obs, info, integral, prev_err, env)
            obs, r, term, trunc, info = env.step(action)
            total += r
            target = info.get("target_position", target)
            err = target - obs[:3]
            dt = env.unwrapped.config.dt
            integral += err * dt
            prev_err = err
            if term or trunc:
                break
        rewards.append(total)
        lengths.append(t + 1)
        env.close()
    return np.mean(rewards), np.std(rewards), np.mean(lengths)


# ---------------------- Policy builders ----------------------
def make_pid_policy(kp: float, ki: float, kd: float):
    def policy(obs, info, integral, prev_err, env):
        target = info.get("target_position", np.zeros(3))
        err = target - obs[:3]
        dt = env.unwrapped.config.dt
        deriv = (err - prev_err) / dt
        act = pid_action(err, integral, deriv, kp, ki, kd)
        return np.clip(act, env.action_space.low, env.action_space.high)
    return policy


def make_model_free_policy(env_name: str, steps: int):
    # Train a small PPO agent quickly
    env = build_env(env_name)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=steps)
    env.close()

    def policy(obs, info, integral, prev_err, env_inst):
        act, _ = model.predict(obs, deterministic=True)
        return act
    return policy


def make_model_based_policy(env_name: str, candidates: int = 16):
    # Use CRM simulator (simplified backend) for one-step lookahead without residual size assumptions
    from crm_ml_rl.wrappers.crm_wrapper import CRMSimulator
    sim = CRMSimulator(dt=0.02, use_cpp=False)

    def policy(obs, info, integral, prev_err, env_inst):
        target = info.get("target_position", np.zeros(3))
        best_score = float("inf")
        best_act = env_inst.action_space.sample()
        for _ in range(candidates):
            act = env_inst.action_space.sample()
            # Clone current state into the simulator
            sim.wrapper._position = obs[:3].copy()
            sim.wrapper._velocity = obs[3:6].copy()
            res = sim.step(act)
            pos_next = res.position
            score = np.linalg.norm(pos_next - target) + 0.01 * np.sum(act ** 2)
            if score < best_score:
                best_score = score
                best_act = act
        return best_act

    return policy


# ---------------------- Main ----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["reaching", "tracking"], default="reaching")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--kp", type=float, default=0.05)
    parser.add_argument("--ki", type=float, default=0.0)
    parser.add_argument("--kd", type=float, default=0.01)
    parser.add_argument("--ppo-steps", type=int, default=2000)
    parser.add_argument("--mb-candidates", type=int, default=16)
    args = parser.parse_args()

    env_builder = lambda: build_env(args.env)

    pid_policy = make_pid_policy(args.kp, args.ki, args.kd)
    mf_policy = make_model_free_policy(args.env, args.ppo_steps)
    mb_policy = make_model_based_policy(args.env, args.mb_candidates)

    print(f"Evaluating PID (kp={args.kp}, ki={args.ki}, kd={args.kd})...")
    pid_mean, pid_std, pid_len = eval_controller(env_builder, pid_policy, args.episodes, args.max_steps)
    print(f"  Return: {pid_mean:.2f} ± {pid_std:.2f}, Length: {pid_len:.1f}")

    print(f"Evaluating PPO (model-free, steps={args.ppo_steps})...")
    mf_mean, mf_std, mf_len = eval_controller(env_builder, mf_policy, args.episodes, args.max_steps)
    print(f"  Return: {mf_mean:.2f} ± {mf_std:.2f}, Length: {mf_len:.1f}")

    print(f"Evaluating Model-Based (hybrid one-step shooting, candidates={args.mb_candidates})...")
    mb_mean, mb_std, mb_len = eval_controller(env_builder, mb_policy, args.episodes, args.max_steps)
    print(f"  Return: {mb_mean:.2f} ± {mb_std:.2f}, Length: {mb_len:.1f}")


if __name__ == "__main__":
    main()
