"""
Trajectory tracking environment for catheter control.

The agent must follow a predefined trajectory (circle, lemniscate, etc.)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any, Dict, Optional, Tuple, List

from .catheter_env import CatheterEnv, CatheterEnvConfig


class TrackingEnv(CatheterEnv):
    """
    Trajectory tracking environment.

    The agent must follow a trajectory defined by waypoints.
    """

    def __init__(
        self,
        config: Optional[CatheterEnvConfig] = None,
        trajectory_type: str = "circle",
        trajectory_freq: float = 1.0,  # Hz
        trajectory_radius: float = 10.0,  # mm
        center_position: np.ndarray = None,
        render_mode: Optional[str] = None
    ):
        """
        Initialize tracking environment.

        Args:
            config: Environment configuration
            trajectory_type: Type of trajectory ('circle', 'lemniscate', 'figure8')
            trajectory_freq: Trajectory frequency in Hz
            trajectory_radius: Trajectory size in mm
            center_position: Center of trajectory
            render_mode: Rendering mode
        """
        super().__init__(config, render_mode)

        self.trajectory_type = trajectory_type
        self.trajectory_freq = trajectory_freq
        self.trajectory_radius = trajectory_radius
        self.center_position = center_position if center_position is not None else np.array([0.0, 0.0, 80.0])

        # Tracking state
        self.trajectory_phase = 0.0
        self.trajectory_waypoints = []
        self.current_waypoint_idx = 0

        # Modify observation space to include trajectory info
        base_obs_dim = self.observation_space.shape[0]
        # Add: current_waypoint (3), next_waypoint (3), trajectory_phase (1)
        new_obs_dim = base_obs_dim + 7
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(new_obs_dim,),
            dtype=np.float32
        )

        # Modify reward config
        if self.config:
            self.config.success_threshold = 3.0  # More lenient for tracking

    def _generate_trajectory(self) -> List[np.ndarray]:
        """Generate trajectory waypoints."""
        waypoints = []
        num_points = int(self.config.max_steps)
        dt = self.config.dt

        for i in range(num_points):
            t = i * dt
            point = self._trajectory_point(t)
            waypoints.append(point)

        return waypoints

    def _trajectory_point(self, t: float) -> np.ndarray:
        """Get trajectory point at time t."""
        omega = 2 * np.pi * self.trajectory_freq
        r = self.trajectory_radius

        if self.trajectory_type == "circle":
            x = r * np.cos(omega * t)
            y = r * np.sin(omega * t)
            z = 0.0

        elif self.trajectory_type == "lemniscate" or self.trajectory_type == "figure8":
            # Lemniscate of Bernoulli
            x = r * np.cos(omega * t)
            y = r * np.sin(omega * t) * np.cos(omega * t)
            z = 0.0

        elif self.trajectory_type == "ellipse":
            x = r * np.cos(omega * t)
            y = r * 0.5 * np.sin(omega * t)
            z = 0.0

        elif self.trajectory_type == "spiral":
            phase = omega * t
            x = r * np.cos(phase) * (1 + 0.1 * phase)
            y = r * np.sin(phase) * (1 + 0.1 * phase)
            z = 5.0 * phase / (2 * np.pi)

        else:
            # Default: stationary
            x, y, z = 0.0, 0.0, 0.0

        return self.center_position + np.array([x, y, z])

    def _get_observation(self) -> np.ndarray:
        """Get observation with trajectory info."""
        # Get base observation
        base_obs = super()._get_observation()

        # Current waypoint
        current_waypoint = self.target_position.copy()

        # Next waypoint
        next_idx = min(self.current_waypoint_idx + 1, len(self.trajectory_waypoints) - 1)
        next_waypoint = self.trajectory_waypoints[next_idx]

        # Phase (normalized)
        phase = np.array([self.trajectory_phase / (2 * np.pi)])

        # Concatenate
        obs = np.concatenate([
            base_obs,
            current_waypoint,
            next_waypoint,
            phase
        ])

        return obs.astype(np.float32)

    def _compute_reward(self) -> Tuple[float, Dict[str, float]]:
        """Compute tracking reward."""
        # Distance to current waypoint
        distance = np.linalg.norm(self.tip_position - self.target_position)

        # Base tracking reward
        tracking_reward = -distance * self.config.position_reward_scale

        # Velocity alignment reward (encourage moving in trajectory direction)
        if self.current_waypoint_idx < len(self.trajectory_waypoints) - 1:
            desired_direction = (
                self.trajectory_waypoints[self.current_waypoint_idx + 1] -
                self.target_position
            )
            desired_direction_norm = np.linalg.norm(desired_direction)
            if desired_direction_norm > 1e-6:
                desired_direction /= desired_direction_norm
                velocity_norm = np.linalg.norm(self.tip_velocity)
                if velocity_norm > 1e-6:
                    alignment = np.dot(self.tip_velocity / velocity_norm, desired_direction)
                    tracking_reward += 0.1 * alignment

        # Progress reward (for advancing along trajectory)
        progress_reward = 0.0
        if distance < self.config.success_threshold * 2:
            progress_reward = 0.1  # Small bonus for staying on track

        # Action penalties
        action_magnitude = np.linalg.norm(self.current_action)
        action_penalty = -action_magnitude * self.config.action_penalty_scale

        action_change = np.linalg.norm(self.current_action - self.prev_action)
        smoothness_penalty = -action_change * self.config.smoothness_penalty_scale

        # Total reward
        total_reward = tracking_reward + progress_reward + action_penalty + smoothness_penalty

        info = {
            'distance': distance,
            'tracking_reward': tracking_reward,
            'progress_reward': progress_reward,
            'action_penalty': action_penalty,
            'smoothness_penalty': smoothness_penalty,
            'waypoint_idx': self.current_waypoint_idx,
            'on_track': distance < self.config.success_threshold
        }

        return total_reward, info

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        # Generate trajectory
        self.trajectory_waypoints = self._generate_trajectory()
        self.current_waypoint_idx = 0
        self.trajectory_phase = 0.0

        # Set first waypoint as target
        if options is None:
            options = {}
        options['target_position'] = self.trajectory_waypoints[0]

        obs, info = super().reset(seed=seed, options=options)

        info['trajectory_type'] = self.trajectory_type
        info['num_waypoints'] = len(self.trajectory_waypoints)

        return obs, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take environment step."""
        obs, reward, terminated, truncated, info = super().step(action)

        # Update trajectory phase
        self.trajectory_phase += 2 * np.pi * self.trajectory_freq * self.config.dt

        # Advance waypoint if close enough
        distance = np.linalg.norm(self.tip_position - self.target_position)
        if distance < self.config.success_threshold:
            if self.current_waypoint_idx < len(self.trajectory_waypoints) - 1:
                self.current_waypoint_idx += 1
                self.target_position = self.trajectory_waypoints[self.current_waypoint_idx]

        # Check if trajectory complete
        if self.current_waypoint_idx >= len(self.trajectory_waypoints) - 1:
            final_distance = np.linalg.norm(self.tip_position - self.target_position)
            if final_distance < self.config.success_threshold:
                terminated = True
                info['trajectory_complete'] = True

        # Update observation with new trajectory info
        obs = self._get_observation()

        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    print("Testing TrackingEnv...")

    # Test circle tracking
    env = TrackingEnv(
        trajectory_type="circle",
        trajectory_freq=0.5,
        trajectory_radius=15.0
    )

    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Trajectory: {info['trajectory_type']}, {info['num_waypoints']} waypoints")

    # Run episode
    total_reward = 0
    on_track_count = 0

    for i in range(200):
        # Simple proportional controller for testing
        error = env.target_position - env.tip_position
        action = np.clip(error * 0.1, -0.3, 0.3)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if info.get('on_track', False):
            on_track_count += 1

        if terminated or truncated:
            print(f"Episode ended at step {i+1}")
            break

    print(f"Total reward: {total_reward:.2f}")
    print(f"On track: {on_track_count}/{i+1} steps ({100*on_track_count/(i+1):.1f}%)")
