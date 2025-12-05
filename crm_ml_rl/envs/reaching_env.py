"""
Point reaching environment for catheter control.

The agent must navigate the catheter tip to a target position.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any, Dict, Optional, Tuple

from .catheter_env import CatheterEnv, CatheterEnvConfig


class ReachingEnv(CatheterEnv):
    """
    Point reaching environment.

    The agent must move the catheter tip to reach a target position.
    """

    def __init__(
        self,
        config: Optional[CatheterEnvConfig] = None,
        random_target: bool = True,
        target_bounds: Tuple[np.ndarray, np.ndarray] = None,
        fixed_target: np.ndarray = None,
        render_mode: Optional[str] = None
    ):
        """
        Initialize reaching environment.

        Args:
            config: Environment configuration
            random_target: Whether to randomize target on reset
            target_bounds: (min, max) bounds for random target generation
            fixed_target: Fixed target position if not randomizing
            render_mode: Rendering mode
        """
        super().__init__(config, render_mode)

        self.random_target = random_target
        self.fixed_target = fixed_target

        # Default target bounds (mm)
        if target_bounds is None:
            self.target_min = np.array([-20.0, -20.0, 60.0])
            self.target_max = np.array([20.0, 20.0, 100.0])
        else:
            self.target_min, self.target_max = target_bounds

        # Success tracking
        self.reached_target = False
        self.time_at_target = 0.0
        self.required_hold_time = 0.5  # seconds to hold at target

    def _generate_random_target(self) -> np.ndarray:
        """Generate random target within bounds."""
        return np.random.uniform(self.target_min, self.target_max)

    def _compute_reward(self) -> Tuple[float, Dict[str, float]]:
        """Compute reaching reward."""
        # Distance to target
        distance = np.linalg.norm(self.tip_position - self.target_position)

        # Base distance reward (negative distance)
        distance_reward = -distance * self.config.position_reward_scale

        # Bonus for getting close
        proximity_bonus = 0.0
        if distance < self.config.success_threshold * 3:
            proximity_bonus = 0.5
        if distance < self.config.success_threshold * 2:
            proximity_bonus = 1.0
        if distance < self.config.success_threshold:
            proximity_bonus = 2.0
            self.time_at_target += self.config.dt
        else:
            self.time_at_target = 0.0

        # Success bonus for holding position
        hold_bonus = 0.0
        if self.time_at_target >= self.required_hold_time:
            hold_bonus = 10.0
            self.reached_target = True

        # Velocity penalty (encourage stopping at target)
        velocity_magnitude = np.linalg.norm(self.tip_velocity)
        velocity_penalty = 0.0
        if distance < self.config.success_threshold * 2:
            velocity_penalty = -velocity_magnitude * 0.1

        # Action penalties
        action_magnitude = np.linalg.norm(self.current_action)
        action_penalty = -action_magnitude * self.config.action_penalty_scale

        action_change = np.linalg.norm(self.current_action - self.prev_action)
        smoothness_penalty = -action_change * self.config.smoothness_penalty_scale

        # Total reward
        total_reward = (
            distance_reward +
            proximity_bonus +
            hold_bonus +
            velocity_penalty +
            action_penalty +
            smoothness_penalty
        )

        info = {
            'distance': distance,
            'distance_reward': distance_reward,
            'proximity_bonus': proximity_bonus,
            'hold_bonus': hold_bonus,
            'velocity_penalty': velocity_penalty,
            'action_penalty': action_penalty,
            'smoothness_penalty': smoothness_penalty,
            'time_at_target': self.time_at_target,
            'reached_target': self.reached_target
        }

        return total_reward, info

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        # Reset tracking
        self.reached_target = False
        self.time_at_target = 0.0

        # Set target
        if options is None:
            options = {}

        if 'target_position' not in options:
            if self.random_target:
                options['target_position'] = self._generate_random_target()
            elif self.fixed_target is not None:
                options['target_position'] = self.fixed_target.copy()

        obs, info = super().reset(seed=seed, options=options)

        info['random_target'] = self.random_target

        return obs, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take environment step."""
        obs, reward, terminated, truncated, info = super().step(action)

        # Check for success
        if self.reached_target:
            terminated = True
            info['success'] = True

        return obs, reward, terminated, truncated, info


class MultiTargetReachingEnv(CatheterEnv):
    """
    Multi-target reaching environment.

    The agent must visit multiple targets in sequence.
    """

    def __init__(
        self,
        config: Optional[CatheterEnvConfig] = None,
        num_targets: int = 3,
        random_order: bool = False,
        target_bounds: Tuple[np.ndarray, np.ndarray] = None,
        render_mode: Optional[str] = None
    ):
        """
        Initialize multi-target environment.

        Args:
            config: Environment configuration
            num_targets: Number of targets to visit
            random_order: Whether to randomize target order
            target_bounds: Bounds for target generation
            render_mode: Rendering mode
        """
        super().__init__(config, render_mode)

        self.num_targets = num_targets
        self.random_order = random_order

        if target_bounds is None:
            self.target_min = np.array([-20.0, -20.0, 60.0])
            self.target_max = np.array([20.0, 20.0, 100.0])
        else:
            self.target_min, self.target_max = target_bounds

        # Target list and progress
        self.targets = []
        self.current_target_idx = 0
        self.targets_reached = 0

        # Modify observation space to include target info
        base_obs_dim = self.observation_space.shape[0]
        # Add: current_target_idx (1), targets_remaining (1)
        new_obs_dim = base_obs_dim + 2
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(new_obs_dim,),
            dtype=np.float32
        )

    def _generate_targets(self) -> list:
        """Generate list of targets."""
        targets = []
        for _ in range(self.num_targets):
            target = np.random.uniform(self.target_min, self.target_max)
            targets.append(target)
        return targets

    def _get_observation(self) -> np.ndarray:
        """Get observation with target progress info."""
        base_obs = super()._get_observation()

        progress_info = np.array([
            self.current_target_idx / self.num_targets,
            (self.num_targets - self.targets_reached) / self.num_targets
        ], dtype=np.float32)

        return np.concatenate([base_obs, progress_info])

    def _compute_reward(self) -> Tuple[float, Dict[str, float]]:
        """Compute multi-target reward."""
        distance = np.linalg.norm(self.tip_position - self.target_position)

        # Distance reward
        distance_reward = -distance * self.config.position_reward_scale

        # Reach bonus
        reach_bonus = 0.0
        target_reached = False
        if distance < self.config.success_threshold:
            reach_bonus = 5.0
            target_reached = True

        # Completion bonus
        completion_bonus = 0.0
        if self.targets_reached == self.num_targets:
            completion_bonus = 20.0

        # Action penalties
        action_magnitude = np.linalg.norm(self.current_action)
        action_penalty = -action_magnitude * self.config.action_penalty_scale

        total_reward = distance_reward + reach_bonus + completion_bonus + action_penalty

        info = {
            'distance': distance,
            'target_reached': target_reached,
            'targets_reached': self.targets_reached,
            'current_target_idx': self.current_target_idx
        }

        return total_reward, info

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        # Generate targets
        self.targets = self._generate_targets()
        if self.random_order:
            np.random.shuffle(self.targets)

        self.current_target_idx = 0
        self.targets_reached = 0

        # Set first target
        if options is None:
            options = {}
        options['target_position'] = self.targets[0]

        obs, info = super().reset(seed=seed, options=options)

        info['num_targets'] = self.num_targets
        info['targets'] = [t.tolist() for t in self.targets]

        return obs, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take environment step."""
        obs, reward, terminated, truncated, info = super().step(action)

        # Check if current target reached
        distance = np.linalg.norm(self.tip_position - self.target_position)
        if distance < self.config.success_threshold:
            self.targets_reached += 1
            self.current_target_idx += 1

            if self.current_target_idx < self.num_targets:
                # Move to next target
                self.target_position = self.targets[self.current_target_idx].copy()
            else:
                # All targets reached
                terminated = True
                info['all_targets_reached'] = True

        # Update observation
        obs = self._get_observation()

        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    print("Testing ReachingEnv...")

    # Test basic reaching
    env = ReachingEnv(random_target=True)

    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Target: {env.target_position}")

    total_reward = 0
    for i in range(100):
        # Simple proportional controller
        error = env.target_position - env.tip_position
        action = np.clip(error * 0.1, -0.3, 0.3)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode ended at step {i+1}")
            print(f"Reached target: {info.get('reached_target', False)}")
            break

    print(f"Total reward: {total_reward:.2f}")

    print("\nTesting MultiTargetReachingEnv...")

    env = MultiTargetReachingEnv(num_targets=3)
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Num targets: {info['num_targets']}")

    for i in range(300):
        error = env.target_position - env.tip_position
        action = np.clip(error * 0.1, -0.3, 0.3)

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(f"Episode ended at step {i+1}")
            print(f"Targets reached: {info.get('targets_reached', 0)}")
            break
