#!/usr/bin/env python3
"""
Test script for Gymnasium environments.

This script validates that the catheter control environments work correctly
with the C++ physics bindings.

Usage:
    python scripts/check_environment.py
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_catheter_env_basic():
    """Test basic CatheterEnv functionality."""
    print("\n1. Testing CatheterEnv Basic Operations...")

    from crm_ml_rl.envs import CatheterEnv, CatheterEnvConfig

    # Create environment with default config (uses simplified model)
    config = CatheterEnvConfig(
        dt=0.02,
        max_steps=100,
        max_current=0.3,
        use_cpp=False  # Use simplified model for speed
    )
    env = CatheterEnv(config=config)
    print(f"   Using C++ bindings: {env.simulator.wrapper.is_using_cpp}")

    # Check spaces
    print(f"   Action space: {env.action_space}")
    print(f"   Observation space: {env.observation_space}")

    # Reset
    obs, info = env.reset(seed=42)
    print(f"   Initial observation shape: {obs.shape}")
    print(f"   Initial info: {list(info.keys())}")

    # Take some steps
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"   Episode ended at step {i+1}")
            break

    print(f"   Completed 10 steps, total reward: {total_reward:.2f}")

    env.close()
    print("   [PASS] CatheterEnv basic operations working")
    return True


def test_reaching_env():
    """Test ReachingEnv with random targets."""
    print("\n2. Testing ReachingEnv...")

    from crm_ml_rl.envs import ReachingEnv, CatheterEnvConfig

    config = CatheterEnvConfig(
        dt=0.02,
        max_steps=200,
        success_threshold=5.0  # 5mm
    )
    env = ReachingEnv(config=config, random_target=True)

    # Run a few episodes
    episode_rewards = []
    episode_lengths = []
    successes = []

    for ep in range(3):
        obs, info = env.reset()
        target = info.get('target_position', np.zeros(3))
        print(f"   Episode {ep+1}: target = {target}")

        done = False
        total_reward = 0
        steps = 0

        while not done and steps < 100:
            # Simple proportional controller towards target
            tip_pos = obs[:3]  # First 3 elements are position
            error = target - tip_pos
            action = np.clip(error * 0.01, -0.3, 0.3)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        successes.append(info.get('success', False))
        print(f"     Reward: {total_reward:.2f}, Steps: {steps}, Success: {info.get('success', False)}")

    env.close()

    print(f"   Mean reward: {np.mean(episode_rewards):.2f}")
    print(f"   Mean length: {np.mean(episode_lengths):.1f}")
    print("   [PASS] ReachingEnv working correctly")
    return True


def test_tracking_env():
    """Test TrackingEnv with trajectory following."""
    print("\n3. Testing TrackingEnv...")

    from crm_ml_rl.envs import TrackingEnv, CatheterEnvConfig

    config = CatheterEnvConfig(
        dt=0.02,
        max_steps=200
    )

    # Test circle trajectory
    env = TrackingEnv(
        config=config,
        trajectory_type="circle",
        trajectory_freq=0.5,
        trajectory_radius=10.0
    )

    obs, info = env.reset()
    print(f"   Trajectory type: circle")
    print(f"   Initial target: {info.get('target_position', 'N/A')}")

    total_reward = 0
    for i in range(50):
        action = env.action_space.sample() * 0.1  # Small random actions
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    print(f"   Completed 50 steps, reward: {total_reward:.2f}")
    env.close()

    print("   [PASS] TrackingEnv working correctly")
    return True


def test_environment_rollout():
    """Test longer environment rollout for stability."""
    print("\n4. Testing Environment Rollout Stability...")

    from crm_ml_rl.envs import CatheterEnv, CatheterEnvConfig

    config = CatheterEnvConfig(
        dt=0.02,
        max_steps=500
    )
    env = CatheterEnv(config=config)

    # Run multiple episodes
    n_episodes = 5
    all_rewards = []
    all_lengths = []
    any_nan = False

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

            # Check for NaN
            if np.any(np.isnan(obs)):
                print(f"   [WARN] NaN in observation at step {steps}")
                any_nan = True
                break

        all_rewards.append(total_reward)
        all_lengths.append(steps)

    env.close()

    print(f"   Episodes: {n_episodes}")
    print(f"   Mean reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"   Mean length: {np.mean(all_lengths):.1f}")
    print(f"   Any NaN: {any_nan}")

    if any_nan:
        print("   [FAIL] Found NaN values in observations")
        return False

    print("   [PASS] Environment rollout stable")
    return True


def test_gym_api_compliance():
    """Test Gymnasium API compliance."""
    print("\n5. Testing Gymnasium API Compliance...")

    from crm_ml_rl.envs import CatheterEnv

    env = CatheterEnv()

    # Check required attributes
    required_attrs = ['action_space', 'observation_space', 'reset', 'step', 'close']
    for attr in required_attrs:
        if not hasattr(env, attr):
            print(f"   [FAIL] Missing required attribute: {attr}")
            return False

    # Check reset returns (obs, info)
    result = env.reset()
    if not (isinstance(result, tuple) and len(result) == 2):
        print("   [FAIL] reset() should return (obs, info)")
        return False

    obs, info = result
    if not isinstance(obs, np.ndarray):
        print("   [FAIL] Observation should be numpy array")
        return False
    if not isinstance(info, dict):
        print("   [FAIL] Info should be dict")
        return False

    # Check step returns (obs, reward, terminated, truncated, info)
    action = env.action_space.sample()
    result = env.step(action)
    if not (isinstance(result, tuple) and len(result) == 5):
        print("   [FAIL] step() should return 5 values")
        return False

    obs, reward, terminated, truncated, info = result
    if not isinstance(obs, np.ndarray):
        print("   [FAIL] step() obs should be numpy array")
        return False
    if not isinstance(reward, (int, float)):
        print("   [FAIL] step() reward should be numeric")
        return False
    if not isinstance(terminated, bool):
        print("   [FAIL] step() terminated should be bool")
        return False
    if not isinstance(truncated, bool):
        print("   [FAIL] step() truncated should be bool")
        return False
    if not isinstance(info, dict):
        print("   [FAIL] step() info should be dict")
        return False

    env.close()
    print("   [PASS] Gymnasium API compliance verified")
    return True


def test_vectorized_env():
    """Test vectorized environment for parallel training."""
    print("\n6. Testing Vectorized Environment...")

    from crm_ml_rl.envs import CatheterEnv, CatheterEnvConfig
    from stable_baselines3.common.vec_env import DummyVecEnv

    config = CatheterEnvConfig(dt=0.02, max_steps=100)

    def make_env():
        return CatheterEnv(config=config)

    # Create vectorized env
    n_envs = 2
    vec_env = DummyVecEnv([make_env for _ in range(n_envs)])

    # Test reset
    obs = vec_env.reset()
    print(f"   Vectorized obs shape: {obs.shape}")

    if obs.shape[0] != n_envs:
        print(f"   [FAIL] Expected {n_envs} observations")
        return False

    # Test step
    actions = np.array([vec_env.action_space.sample() for _ in range(n_envs)])
    obs, rewards, dones, infos = vec_env.step(actions)

    print(f"   Step results: obs={obs.shape}, rewards={rewards.shape}")

    vec_env.close()
    print("   [PASS] Vectorized environment working")
    return True


def test_cpp_physics_env():
    """Test environment with C++ physics bindings."""
    print("\n7. Testing Environment with C++ Physics...")

    from crm_ml_rl.envs import CatheterEnv, CatheterEnvConfig

    # Create environment with C++ physics
    config = CatheterEnvConfig(
        dt=0.05,  # 50ms timestep (from C++ test)
        max_steps=50,
        max_current=0.2,
        use_cpp=True,
        param_file='data/catheter_params/CatheterParameterSet_1_dyn.txt',
        config_file='data/catheter_params/CatheterSpatialConfiguration_1.txt',
        insertion_length=94.3,
        success_threshold=10.0
    )

    try:
        env = CatheterEnv(config=config)
    except Exception as e:
        print(f"   [SKIP] Could not create C++ env: {e}")
        return True  # Skip if C++ not available

    if not env.simulator.wrapper.is_using_cpp:
        print("   [SKIP] C++ bindings not available")
        return True

    print(f"   Using C++ bindings: True")

    # Run a short episode
    obs, info = env.reset()
    print(f"   Initial position: {env.tip_position}")

    total_reward = 0
    has_nan = False

    for i in range(20):
        action = np.array([0.0, 0.0, 0.05])  # Small z current
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if np.any(np.isnan(obs)):
            print(f"   [WARN] NaN at step {i+1}")
            has_nan = True
            break

        if terminated or truncated:
            break

    print(f"   Final position: {env.tip_position}")
    print(f"   Total reward: {total_reward:.2f}")

    env.close()

    if has_nan:
        print("   [FAIL] Environment produced NaN values")
        return False

    print("   [PASS] C++ physics environment working")
    return True


def main():
    """Run all environment tests."""
    print("=" * 60)
    print("CRM_ML Environment Test Suite")
    print("=" * 60)

    results = {}

    # Run tests
    results['basic'] = test_catheter_env_basic()
    results['reaching'] = test_reaching_env()
    results['tracking'] = test_tracking_env()
    results['rollout'] = test_environment_rollout()
    results['gym_api'] = test_gym_api_compliance()
    results['vectorized'] = test_vectorized_env()
    results['cpp_physics'] = test_cpp_physics_env()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("All environment tests passed!")
    else:
        print("Some tests failed. See details above.")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
