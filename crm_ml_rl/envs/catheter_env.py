"""
Base Gymnasium environment for catheter control.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import sys
sys.path.append('..')

from ..wrappers.crm_wrapper import CRMSimulator, CatheterParameters, CatheterState


@dataclass
class CatheterEnvConfig:
    """Configuration for catheter environment."""
    # Simulation parameters
    dt: float = 0.01  # Timestep (seconds)
    max_steps: int = 200  # Maximum episode length

    # Action space
    max_current: float = 0.5  # Maximum current (A)
    action_smoothing: float = 0.0  # Action smoothing factor (0 = no smoothing)

    # Observation space
    include_velocity: bool = True
    include_target: bool = True
    history_length: int = 1  # Number of past states to include

    # Reward parameters
    position_reward_scale: float = 1.0
    action_penalty_scale: float = 0.01
    smoothness_penalty_scale: float = 0.01
    success_bonus: float = 10.0
    success_threshold: float = 2.0  # mm

    # Environment bounds (mm)
    workspace_bounds: np.ndarray = field(default_factory=lambda: np.array([
        [-50, 50],   # x bounds
        [-50, 50],   # y bounds
        [0, 150]     # z bounds
    ]))

    # Noise
    observation_noise: float = 0.0
    action_noise: float = 0.0

    # Use learned model instead of physics
    use_learned_dynamics: bool = False

    # Use hybrid model (CRM physics + learned residuals)
    use_hybrid_dynamics: bool = False

    # C++ physics parameters
    use_cpp: bool = False  # Whether to use C++ bindings (slower but more accurate)
    param_file: Optional[str] = None  # Path to catheter parameter file
    config_file: Optional[str] = None  # Path to catheter configuration file
    insertion_length: float = 50.0  # Default insertion length (mm)

    # Damping coefficients for C++ dynamics (from CRMDYN_test.cpp)
    damping: np.ndarray = field(default_factory=lambda: np.array([
        12.1761626666366, 12.1761626666366, 284.429938756989,
        0.0304776127617393, 0.0304776127617393, 0.00502712804532508
    ]))


class CatheterEnv(gym.Env):
    """
    Base Gymnasium environment for catheter control.

    Observation space:
        - tip_position (3): Current tip position in mm
        - tip_velocity (3): Current tip velocity (optional)
        - target_position (3): Target position (optional)
        - currents (3): Current actuation currents
        - history: Past states (optional)

    Action space:
        - currents (3): Desired coil currents [-max_current, max_current]

    Reward:
        - Negative distance to target
        - Penalties for large actions and jerky motion
        - Bonus for reaching target
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        config: Optional[CatheterEnvConfig] = None,
        render_mode: Optional[str] = None
    ):
        """
        Initialize catheter environment.

        Args:
            config: Environment configuration
            render_mode: Rendering mode
        """
        super().__init__()

        self.config = config or CatheterEnvConfig()
        self.render_mode = render_mode

        # Initialize simulator with proper parameters
        self.simulator = CRMSimulator(
            param_file=self.config.param_file,
            config_file=self.config.config_file,
            dt=self.config.dt,
            use_cpp=self.config.use_cpp,
            damping=self.config.damping
        )

        # Configure damping for C++ dynamics
        if self.config.use_cpp and self.simulator.wrapper.is_using_cpp:
            self.simulator.wrapper.set_damping(self.config.damping)
            self.simulator.wrapper.set_timestep(self.config.dt)

        # Initialize hybrid dynamics model if requested
        self.hybrid_model = None
        if self.config.use_hybrid_dynamics:
            self._init_hybrid_model()

        # Action space: coil currents
        self.action_space = spaces.Box(
            low=-self.config.max_current,
            high=self.config.max_current,
            shape=(3,),
            dtype=np.float32
        )

        # Observation space
        obs_dim = self._compute_observation_dim()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # State variables
        self.current_step = 0
        self.tip_position = np.zeros(3)
        self.tip_velocity = np.zeros(3)
        self.prev_tip_position = np.zeros(3)
        self.current_action = np.zeros(3)
        self.prev_action = np.zeros(3)
        self.target_position = np.zeros(3)
        self.history = []

        # Learned dynamics model (optional)
        self.dynamics_model = None

    def _init_hybrid_model(self):
        """Initialize hybrid dynamics model."""
        from ..models.hybrid_dynamics import HybridDynamicsModel, HybridDynamicsConfig

        hybrid_config = HybridDynamicsConfig(
            param_file=self.config.param_file,
            config_file=self.config.config_file,
            use_cpp=self.config.use_cpp,
            dt=self.config.dt,
            insertion_length=self.config.insertion_length,
            damping=self.config.damping
        )
        self.hybrid_model = HybridDynamicsModel(hybrid_config)

    def _compute_observation_dim(self) -> int:
        """Compute observation dimension based on config."""
        dim = 3  # tip position
        if self.config.include_velocity:
            dim += 3  # tip velocity
        if self.config.include_target:
            dim += 3  # target position
        dim += 3  # current action

        dim *= self.config.history_length

        return dim

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        obs_parts = [self.tip_position.copy()]

        if self.config.include_velocity:
            obs_parts.append(self.tip_velocity.copy())

        if self.config.include_target:
            obs_parts.append(self.target_position.copy())

        obs_parts.append(self.current_action.copy())

        obs = np.concatenate(obs_parts)

        # Add history if needed
        if self.config.history_length > 1:
            full_obs = []
            for _ in range(self.config.history_length - 1):
                if len(self.history) > 0:
                    full_obs.append(self.history[-1])
                else:
                    full_obs.append(obs)
            full_obs.append(obs)
            obs = np.concatenate(full_obs)

        # Add noise
        if self.config.observation_noise > 0:
            obs += np.random.randn(*obs.shape) * self.config.observation_noise

        return obs.astype(np.float32)

    def _compute_reward(self) -> Tuple[float, Dict[str, float]]:
        """Compute reward."""
        # Distance to target
        distance = np.linalg.norm(self.tip_position - self.target_position)

        # Position reward (negative distance)
        position_reward = -distance * self.config.position_reward_scale

        # Action penalty
        action_magnitude = np.linalg.norm(self.current_action)
        action_penalty = -action_magnitude * self.config.action_penalty_scale

        # Smoothness penalty (change in action)
        action_change = np.linalg.norm(self.current_action - self.prev_action)
        smoothness_penalty = -action_change * self.config.smoothness_penalty_scale

        # Success bonus
        success_bonus = 0.0
        if distance < self.config.success_threshold:
            success_bonus = self.config.success_bonus

        # Total reward
        total_reward = position_reward + action_penalty + smoothness_penalty + success_bonus

        info = {
            'distance': distance,
            'position_reward': position_reward,
            'action_penalty': action_penalty,
            'smoothness_penalty': smoothness_penalty,
            'success_bonus': success_bonus,
            'success': distance < self.config.success_threshold
        }

        return total_reward, info

    def _is_terminated(self) -> bool:
        """Check if episode is terminated (goal reached or failure)."""
        distance = np.linalg.norm(self.tip_position - self.target_position)
        if distance < self.config.success_threshold:
            return True

        # Check bounds
        pos = self.tip_position
        bounds = self.config.workspace_bounds
        if not (bounds[0, 0] <= pos[0] <= bounds[0, 1] and
                bounds[1, 0] <= pos[1] <= bounds[1, 1] and
                bounds[2, 0] <= pos[2] <= bounds[2, 1]):
            return True

        return False

    def _is_truncated(self) -> bool:
        """Check if episode is truncated (max steps reached)."""
        return self.current_step >= self.config.max_steps

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment.

        Args:
            seed: Random seed
            options: Reset options (can include 'target_position')

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        # Reset simulator with proper initialization
        initial_currents = np.zeros(3)
        self.simulator.reset(
            initial_currents=initial_currents,
            insertion_length=self.config.insertion_length
        )

        # Reset hybrid model if used
        if self.config.use_hybrid_dynamics and self.hybrid_model is not None:
            self.hybrid_model.reset(
                initial_currents=initial_currents,
                insertion_length=self.config.insertion_length
            )

        # Reset state
        self.current_step = 0
        self.tip_position = self.simulator.state.position.copy()
        self.tip_velocity = np.zeros(3)
        self.prev_tip_position = self.tip_position.copy()
        self.current_action = np.zeros(3)
        self.prev_action = np.zeros(3)
        self.history = []

        # Set target
        if options and 'target_position' in options:
            self.target_position = np.array(options['target_position'])
        else:
            self.target_position = self._sample_target()

        obs = self._get_observation()
        info = {
            'target_position': self.target_position.copy(),
            'initial_distance': np.linalg.norm(self.tip_position - self.target_position)
        }

        return obs, info

    def _sample_target(self) -> np.ndarray:
        """Sample random target position."""
        bounds = self.config.workspace_bounds

        # Sample within workspace, biased towards reachable region
        target = np.array([
            np.random.uniform(bounds[0, 0] * 0.5, bounds[0, 1] * 0.5),
            np.random.uniform(bounds[1, 0] * 0.5, bounds[1, 1] * 0.5),
            np.random.uniform(bounds[2, 0] + 50, bounds[2, 1] - 20)
        ])

        return target

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take environment step.

        Args:
            action: Coil currents

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Clip and process action
        action = np.clip(action, -self.config.max_current, self.config.max_current)

        # Add action noise
        if self.config.action_noise > 0:
            action += np.random.randn(3) * self.config.action_noise

        # Action smoothing
        if self.config.action_smoothing > 0:
            action = (self.config.action_smoothing * self.prev_action +
                     (1 - self.config.action_smoothing) * action)

        # Update state
        self.prev_action = self.current_action.copy()
        self.current_action = action.copy()
        self.prev_tip_position = self.tip_position.copy()

        # Step simulation
        if self.config.use_hybrid_dynamics and self.hybrid_model is not None:
            # Use hybrid model (CRM physics + learned residuals)
            self._step_hybrid_dynamics(action)
        elif self.config.use_learned_dynamics and self.dynamics_model is not None:
            # Use learned model
            self._step_learned_dynamics(action)
        else:
            # Use physics simulation
            state = self.simulator.step(action, insertion_length=self.config.insertion_length)
            self.tip_position = state.position.copy()

        # Compute velocity
        self.tip_velocity = (self.tip_position - self.prev_tip_position) / self.config.dt

        # Update history
        self.history.append(self._get_observation())
        if len(self.history) > self.config.history_length:
            self.history.pop(0)

        self.current_step += 1

        # Compute outputs
        obs = self._get_observation()
        reward, reward_info = self._compute_reward()
        terminated = self._is_terminated()
        truncated = self._is_truncated()

        info = {
            'step': self.current_step,
            'tip_position': self.tip_position.copy(),
            'target_position': self.target_position.copy(),
            'action': action.copy(),
            **reward_info
        }

        return obs, reward, terminated, truncated, info

    def _step_learned_dynamics(self, action: np.ndarray):
        """Step using learned dynamics model."""
        import torch

        # Prepare input
        state = np.concatenate([self.tip_position, self.tip_velocity])
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)

        # Predict
        with torch.no_grad():
            next_state = self.dynamics_model(state_tensor, action_tensor)

        self.tip_position = next_state[0, :3].numpy()

    def _step_hybrid_dynamics(self, action: np.ndarray):
        """Step using hybrid dynamics model (CRM physics + learned residuals)."""
        import torch

        # Prepare current state
        state = np.concatenate([self.tip_position, self.tip_velocity])
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)

        # Step hybrid model (combines physics + learned residual internally)
        with torch.no_grad():
            next_state = self.hybrid_model.forward(
                state_tensor, action_tensor,
                insertion_length=self.config.insertion_length
            )

        # Update position and velocity
        next_state_np = next_state[0].cpu().numpy()
        self.tip_position = next_state_np[:3]
        self.tip_velocity = next_state_np[3:]

    def set_hybrid_model(self, model):
        """Set hybrid dynamics model."""
        self.hybrid_model = model

    def set_dynamics_model(self, model):
        """Set learned dynamics model."""
        self.dynamics_model = model

    def render(self):
        """Render environment."""
        if self.render_mode == "human":
            print(f"Step {self.current_step}: pos={self.tip_position}, "
                  f"target={self.target_position}, dist={np.linalg.norm(self.tip_position - self.target_position):.2f}")
        return None

    def close(self):
        """Close environment."""
        pass


if __name__ == "__main__":
    # Test environment
    print("Testing CatheterEnv...")

    env = CatheterEnv()

    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    # Run episode
    total_reward = 0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode ended at step {i+1}")
            break

    print(f"Total reward: {total_reward:.2f}")
    print(f"Final distance: {info['distance']:.2f}")
