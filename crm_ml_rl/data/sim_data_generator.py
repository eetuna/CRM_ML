"""
Simulated data generator for training dynamics models.

Generates synthetic catheter trajectory data using physics-based models
or simplified dynamics for pre-training.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from pathlib import Path
import json

from crm_ml_rl.wrappers.crm_wrapper import CRMWrapper, CRMSimulator, CatheterParams


@dataclass
class SimulationConfig:
    """Configuration for data simulation."""
    # Time data/simulation_parameters
    dt: float = 0.02  # 50 Hz
    episode_length: float = 10.0  # seconds
    insertion_length: float = 94.3  # mm (matches dynamics parameter set)

    # Trajectory data/simulation_parameters
    trajectory_types: List[str] = None
    trajectory_freqs: List[float] = None
    trajectory_radii: List[float] = None

    # Noise data/simulation_parameters
    action_noise_std: float = 0.01
    observation_noise_std: float = 0.5  # mm
    process_noise_std: float = 0.1

    # Physics data/simulation_parameters
    damping: float = 10.0
    stiffness: float = 100.0
    mass: float = 0.001  # kg

    # Data generation
    n_episodes_per_config: int = 10
    random_initial_states: bool = True

    def __post_init__(self):
        if self.trajectory_types is None:
            self.trajectory_types = ["circle", "lemniscate", "random"]
        if self.trajectory_freqs is None:
            self.trajectory_freqs = [0.1, 0.2, 0.5, 1.0]
        if self.trajectory_radii is None:
            self.trajectory_radii = [5.0, 10.0, 15.0, 20.0]


class SimplifiedDynamics:
    """
    Simplified catheter dynamics model for data generation.

    Models the catheter tip as a damped mass-spring system under magnetic actuation.
    """

    def __init__(
        self,
        mass: float = 0.001,
        damping: float = 10.0,
        stiffness: float = 100.0,
        dt: float = 0.02
    ):
        """
        Initialize simplified dynamics.

        Args:
            mass: Effective mass (kg)
            damping: Damping coefficient (N·s/m)
            stiffness: Stiffness (N/m)
            dt: Time step
        """
        self.mass = mass
        self.damping = damping
        self.stiffness = stiffness
        self.dt = dt

        # Equilibrium position (mm)
        self.equilibrium = np.array([0.0, 0.0, 80.0])

        # Current to force conversion (simplified)
        self.current_to_force = np.array([5.0, 5.0, 3.0])  # N/A (scaled down for stability)

    def step(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        current: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate one step of dynamics.

        Args:
            position: Current position (mm)
            velocity: Current velocity (mm/s)
            current: Applied current (A)

        Returns:
            Tuple of (new_position, new_velocity)
        """
        # Compute forces
        # Magnetic force (proportional to current)
        F_magnetic = current * self.current_to_force

        # Spring force (towards equilibrium)
        displacement = position - self.equilibrium
        F_spring = -self.stiffness * displacement / 1000  # Convert mm to m

        # Damping force
        F_damping = -self.damping * velocity / 1000

        # Total force
        F_total = F_magnetic + F_spring + F_damping

        # Acceleration (mm/s^2)
        acceleration = F_total / self.mass * 1000
        # Clamp to avoid numerical blow-up
        acc_norm = np.linalg.norm(acceleration)
        if acc_norm > 5e4:  # 50 m/s^2 in mm units
            acceleration = acceleration / acc_norm * 5e4

        # Semi-implicit Euler integration
        new_velocity = velocity + acceleration * self.dt
        new_position = position + new_velocity * self.dt

        return new_position, new_velocity


class TrajectoryGenerator:
    """Generator for reference trajectories."""

    def __init__(
        self,
        center: np.ndarray = None,
        dt: float = 0.02
    ):
        """
        Initialize trajectory generator.

        Args:
            center: Center position for trajectories
            dt: Time step
        """
        self.center = center if center is not None else np.array([0.0, 0.0, 80.0])
        self.dt = dt

    def circle(
        self,
        t: float,
        freq: float = 0.5,
        radius: float = 10.0
    ) -> np.ndarray:
        """Generate circular trajectory point."""
        omega = 2 * np.pi * freq
        x = radius * np.cos(omega * t)
        y = radius * np.sin(omega * t)
        z = 0.0
        return self.center + np.array([x, y, z])

    def lemniscate(
        self,
        t: float,
        freq: float = 0.5,
        radius: float = 10.0
    ) -> np.ndarray:
        """Generate lemniscate (figure-8) trajectory point."""
        omega = 2 * np.pi * freq
        x = radius * np.cos(omega * t)
        y = radius * np.sin(omega * t) * np.cos(omega * t)
        z = 0.0
        return self.center + np.array([x, y, z])

    def spiral(
        self,
        t: float,
        freq: float = 0.5,
        radius: float = 10.0
    ) -> np.ndarray:
        """Generate spiral trajectory point."""
        omega = 2 * np.pi * freq
        phase = omega * t
        x = radius * np.cos(phase) * (1 + 0.1 * phase)
        y = radius * np.sin(phase) * (1 + 0.1 * phase)
        z = 5.0 * phase / (2 * np.pi)
        return self.center + np.array([x, y, z])

    def random_walk(
        self,
        t: float,
        step_size: float = 1.0,
        bounds: float = 20.0
    ) -> np.ndarray:
        """Generate random walk (uses saved path for consistency)."""
        # Generate smooth random trajectory using filtered noise
        np.random.seed(int(t * 1000) % 10000)
        noise = np.random.randn(3) * step_size
        return np.clip(
            self.center + noise,
            self.center - bounds,
            self.center + bounds
        )

    def generate_trajectory(
        self,
        trajectory_type: str,
        duration: float,
        freq: float = 0.5,
        radius: float = 10.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate complete trajectory.

        Args:
            trajectory_type: Type of trajectory
            duration: Duration in seconds
            freq: Frequency for periodic trajectories
            radius: Radius for trajectories

        Returns:
            Tuple of (times, positions)
        """
        n_steps = int(duration / self.dt)
        times = np.arange(n_steps) * self.dt
        positions = np.zeros((n_steps, 3))

        for i, t in enumerate(times):
            if trajectory_type == "circle":
                positions[i] = self.circle(t, freq, radius)
            elif trajectory_type == "lemniscate":
                positions[i] = self.lemniscate(t, freq, radius)
            elif trajectory_type == "spiral":
                positions[i] = self.spiral(t, freq, radius)
            elif trajectory_type == "random":
                # Pre-generate smooth random trajectory
                if i == 0:
                    positions[i] = self.center.copy()
                else:
                    # Smooth random walk
                    direction = np.random.randn(3)
                    direction = direction / (np.linalg.norm(direction) + 1e-6)
                    step = direction * radius * 0.1
                    positions[i] = positions[i-1] + step
                    # Bound check
                    offset = positions[i] - self.center
                    if np.linalg.norm(offset) > radius:
                        positions[i] = self.center + offset / np.linalg.norm(offset) * radius
            else:
                positions[i] = self.center.copy()

        return times, positions


class SimDataGenerator:
    """
    Simulated data generator for dynamics model training.
    """

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        use_crm: bool = False,
        dt: Optional[float] = None
    ):
        """
        Initialize data generator.

        Args:
            config: Simulation configuration
            use_crm: Use full CRM physics (requires C++ bindings)
            dt: Optional timestep override (convenience for scripts)
        """
        self.config = config or SimulationConfig()
        if dt is not None:
            # Allow lightweight override without rebuilding the config
            self.config.dt = dt
        self.use_crm = use_crm

        # Initialize dynamics
        self.dynamics = SimplifiedDynamics(
            mass=self.config.mass,
            damping=self.config.damping,
            stiffness=self.config.stiffness,
            dt=self.config.dt
        )
        if use_crm:
            self.simulator = CRMSimulator(
                dt=self.config.dt,
                use_cpp=True,
                damping=np.array([
                    12.1761626666366, 12.1761626666366, 284.429938756989,
                    0.0304776127617393, 0.0304776127617393, 0.00502712804532508
                ])
            )

        # Trajectory generator
        self.traj_gen = TrajectoryGenerator(dt=self.config.dt)

    def generate_trajectory(
        self,
        currents: np.ndarray,
        insertion_length: Optional[float] = None,
        initial_position: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Simulate a trajectory given a current sequence.

        Args:
            currents: Array of currents (T, 3) or (3, T)
            insertion_length: Inserted length (mm) for CRM simulator
            initial_position: Optional initial position for simplified model

        Returns:
            Dictionary with positions, velocities, times, currents, dt, num_steps
        """
        currents_arr = np.array(currents)
        if currents_arr.shape[0] == 3 and currents_arr.shape[1] != 3:
            currents_arr = currents_arr.T

        if self.use_crm:
            if insertion_length is None:
                insertion_length = self.config.insertion_length
            result = self.simulator.simulate_trajectory(
                currents_arr,
                insertion_length=insertion_length,
                initial_position=initial_position
            )
            # If C++ path failed or produced NaNs, fall back to simplified rollout
            if not np.all(np.isfinite(result["positions"])):
                print("Warning: CRM simulation returned invalid values; using simplified dynamics fallback.")
                result = self._simulate_simplified(currents_arr, initial_position)
            return result

        return self._simulate_simplified(currents_arr, initial_position)

    def _simulate_simplified(
        self,
        currents_arr: np.ndarray,
        initial_position: Optional[np.ndarray]
    ) -> Dict:
        """Roll out simplified dynamics given a current sequence."""
        T = len(currents_arr)
        positions = np.zeros((T + 1, 3))
        velocities = np.zeros((T + 1, 3))
        times = np.arange(T + 1) * self.config.dt

        # Initialize state
        position = initial_position.copy() if initial_position is not None else self.dynamics.equilibrium.copy()
        velocity = np.zeros(3)
        positions[0] = position
        velocities[0] = velocity

        for t in range(T):
            position, velocity = self.dynamics.step(position, velocity, currents_arr[t])
            positions[t + 1] = position
            velocities[t + 1] = velocity

        return {
            "positions": positions,
            "velocities": velocities,
            "times": times,
            "currents": currents_arr,
            "dt": self.config.dt,
            "num_steps": T
        }

    def generate_episode(
        self,
        trajectory_type: str = "circle",
        freq: float = 0.5,
        radius: float = 10.0,
        controller: Optional[Callable] = None
    ) -> Dict:
        """
        Generate one episode of data.

        Args:
            trajectory_type: Type of trajectory to follow
            freq: Trajectory frequency
            radius: Trajectory radius
            controller: Optional custom controller function

        Returns:
            Dictionary containing episode data
        """
        n_steps = int(self.config.episode_length / self.config.dt)

        # Generate reference trajectory
        times, ref_positions = self.traj_gen.generate_trajectory(
            trajectory_type,
            self.config.episode_length,
            freq,
            radius
        )

        # Initialize state
        if self.config.random_initial_states:
            initial_offset = np.random.randn(3) * 5.0
            position = ref_positions[0] + initial_offset
        else:
            position = ref_positions[0].copy()

        velocity = np.zeros(3)

        # Reset CRM simulator if using it
        if self.use_crm:
            self.simulator.reset(initial_position=position)

        # Storage
        positions = np.zeros((n_steps, 3))
        velocities = np.zeros((n_steps, 3))
        currents = np.zeros((n_steps, 3))

        # Default controller: proportional-derivative
        if controller is None:
            Kp = 0.1
            Kd = 0.01
            def controller(pos, vel, ref_pos, ref_vel=None):
                error = ref_pos - pos
                if ref_vel is None:
                    ref_vel = np.zeros(3)
                vel_error = ref_vel - vel
                return Kp * error + Kd * vel_error

        # Simulate
        for i in range(n_steps):
            # Store current state
            positions[i] = position.copy()
            velocities[i] = velocity.copy()

            # Reference velocity (finite difference)
            if i < n_steps - 1:
                ref_vel = (ref_positions[i+1] - ref_positions[i]) / self.config.dt
            else:
                ref_vel = np.zeros(3)

            # Compute control action
            current = controller(position, velocity, ref_positions[i], ref_vel)

            # Add noise
            current += np.random.randn(3) * self.config.action_noise_std

            # Clip currents
            current = np.clip(current, -0.3, 0.3)
            currents[i] = current

            # Step dynamics
            if self.use_crm:
                # Use CRM simulator
                catheter_state = self.simulator.step(current)
                position = catheter_state.position.copy()
                velocity = catheter_state.velocity.copy()
            else:
                # Use simplified dynamics
                position, velocity = self.dynamics.step(position, velocity, current)

            # Add process noise
            position += np.random.randn(3) * self.config.process_noise_std

        # Add observation noise
        noisy_positions = positions + np.random.randn(*positions.shape) * self.config.observation_noise_std

        return {
            'times': times,
            'positions': noisy_positions,
            'velocities': velocities,
            'currents': currents,
            'reference_positions': ref_positions,
            'clean_positions': positions,
            'trajectory_type': trajectory_type,
            'frequency': freq,
            'radius': radius,
            'dt': self.config.dt
        }

    def generate_dataset(
        self,
        n_episodes: Optional[int] = None,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Generate full dataset.

        Args:
            n_episodes: Number of episodes (or use config)
            save_path: Path to save dataset

        Returns:
            Dataset dictionary
        """
        if n_episodes is None:
            n_episodes = self.config.n_episodes_per_config

        all_episodes = []
        episode_id = 0

        for traj_type in self.config.trajectory_types:
            for freq in self.config.trajectory_freqs:
                for radius in self.config.trajectory_radii:
                    print(f"Generating: {traj_type}, freq={freq}Hz, radius={radius}mm")

                    for _ in range(n_episodes):
                        episode = self.generate_episode(
                            trajectory_type=traj_type,
                            freq=freq,
                            radius=radius
                        )
                        episode['episode_id'] = episode_id
                        all_episodes.append(episode)
                        episode_id += 1

        # Combine into dataset
        dataset = {
            'episodes': all_episodes,
            'config': self.config.__dict__,
            'n_episodes': len(all_episodes)
        }

        print(f"Generated {len(all_episodes)} episodes")

        if save_path:
            self._save_dataset(dataset, save_path)

        return dataset

    def _save_dataset(self, dataset: Dict, save_path: str):
        """Save dataset to disk."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save episodes as numpy arrays
        n_episodes = len(dataset['episodes'])

        for i, episode in enumerate(dataset['episodes']):
            ep_dir = save_path / f"episode_{i:04d}"
            ep_dir.mkdir(exist_ok=True)

            np.save(ep_dir / "positions.npy", episode['positions'])
            np.save(ep_dir / "velocities.npy", episode['velocities'])
            np.save(ep_dir / "currents.npy", episode['currents'])
            np.save(ep_dir / "reference.npy", episode['reference_positions'])

            # Save metadata
            metadata = {
                'trajectory_type': episode['trajectory_type'],
                'frequency': episode['frequency'],
                'radius': episode['radius'],
                'dt': episode['dt'],
                'episode_id': episode['episode_id']
            }
            with open(ep_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

        # Save config
        config_save = {}
        for k, v in dataset['config'].items():
            if isinstance(v, np.ndarray):
                config_save[k] = v.tolist()
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], np.ndarray):
                config_save[k] = [x.tolist() for x in v]
            else:
                config_save[k] = v

        with open(save_path / "config.json", 'w') as f:
            json.dump(config_save, f, indent=2)

        print(f"Dataset saved to {save_path}")

    def load_dataset(self, load_path: str) -> Dict:
        """Load dataset from disk."""
        load_path = Path(load_path)

        # Load config
        with open(load_path / "config.json") as f:
            config = json.load(f)

        # Load episodes
        episodes = []
        ep_dirs = sorted(load_path.glob("episode_*"))

        for ep_dir in ep_dirs:
            episode = {
                'positions': np.load(ep_dir / "positions.npy"),
                'velocities': np.load(ep_dir / "velocities.npy"),
                'currents': np.load(ep_dir / "currents.npy"),
                'reference_positions': np.load(ep_dir / "reference.npy")
            }

            with open(ep_dir / "metadata.json") as f:
                metadata = json.load(f)

            episode.update(metadata)
            episodes.append(episode)

        return {
            'episodes': episodes,
            'config': config,
            'n_episodes': len(episodes)
        }


def prepare_training_data(
    dataset: Dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare dataset for dynamics model training.

    Args:
        dataset: Dataset from generate_dataset

    Returns:
        Tuple of (states, actions, next_states)
    """
    all_states = []
    all_actions = []
    all_next_states = []

    for episode in dataset['episodes']:
        positions = episode['positions']
        velocities = episode['velocities']
        currents = episode['currents']
        dt = episode['dt']

        # Build states
        states = np.hstack([positions, velocities])

        # Create transitions
        for t in range(len(states) - 1):
            all_states.append(states[t])
            all_actions.append(currents[t])
            all_next_states.append(states[t + 1])

    return (
        np.array(all_states),
        np.array(all_actions),
        np.array(all_next_states)
    )


if __name__ == "__main__":
    print("Testing SimDataGenerator...")

    # Create generator
    config = SimulationConfig(
        trajectory_types=["circle", "lemniscate"],
        trajectory_freqs=[0.5],
        trajectory_radii=[10.0],
        n_episodes_per_config=2
    )

    generator = SimDataGenerator(config=config, use_crm=False)

    # Generate single episode
    print("\nGenerating single episode...")
    episode = generator.generate_episode(
        trajectory_type="circle",
        freq=0.5,
        radius=10.0
    )
    print(f"Episode length: {len(episode['positions'])} steps")
    print(f"Position shape: {episode['positions'].shape}")
    print(f"Current shape: {episode['currents'].shape}")

    # Generate dataset
    print("\nGenerating dataset...")
    dataset = generator.generate_dataset(n_episodes=2)
    print(f"Total episodes: {dataset['n_episodes']}")

    # Prepare training data
    print("\nPreparing training data...")
    states, actions, next_states = prepare_training_data(dataset)
    print(f"States shape: {states.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Next states shape: {next_states.shape}")

    # Save and load test
    print("\nTesting save/load...")
    save_path = "/tmp/test_sim_data"
    dataset = generator.generate_dataset(n_episodes=1, save_path=save_path)

    loaded_dataset = generator.load_dataset(save_path)
    print(f"Loaded {loaded_dataset['n_episodes']} episodes")

    print("\nSimDataGenerator testing complete!")
