"""
Data loader for CRM experimental and simulated data.

Data format from 3D_dynamic_response_data_0124/:
- Output trajectories: timestamp, arduino_state, base_pos(3), coil_pos(3), normal(3), tip_pos(3)
- Input currents: 3 coil currents (axial, side1, side2)
- File naming: {trajectory_type}{sampling_ms}_01.txt (e.g., circle100_01.txt = 100ms sampling)

Current processing (from greybox_modeling.m):
- Keep first column as-is
- Negate third column: u(3) = -u(3)
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.io as sio

# Constants
CAMERA_SAMPLING_RATE_HZ = 60.0
CAMERA_SAMPLING_PERIOD_S = 1.0 / CAMERA_SAMPLING_RATE_HZ  # ~16.7ms

# Available sampling times in milliseconds
SAMPLING_TIMES_MS = [1, 3, 5, 8, 10, 12, 15, 18, 20, 25, 50, 100]

# Sampling times slower than camera (>=16.7ms, so >=20ms to be safe)
SLOW_SAMPLING_TIMES_MS = [20, 25, 50, 100]

# Data directory relative to project root
DATA_DIR = "3D_dynamic_response_data_0124"


def get_project_root() -> Path:
    """Get the project root directory."""
    current = Path(__file__).resolve()
    # Navigate up to find CRM_ML root
    for parent in current.parents:
        if (parent / "CLAUDE.md").exists() or (parent / "src").exists():
            return parent
    return current.parent.parent.parent


def get_available_trajectories(data_dir: Optional[str] = None) -> Dict[str, List[int]]:
    """
    Get available trajectory types and their sampling times.

    Returns:
        Dict mapping trajectory type to list of available sampling times in ms.
    """
    if data_dir is None:
        data_dir = get_project_root() / DATA_DIR / "output_trajectories"
    else:
        data_dir = Path(data_dir)

    trajectories = {}
    pattern = re.compile(r"(\w+?)(\d+)_01\.txt")

    for file in data_dir.glob("*_01.txt"):
        match = pattern.match(file.name)
        if match:
            traj_type = match.group(1)
            sampling_ms = int(match.group(2))
            if traj_type not in trajectories:
                trajectories[traj_type] = []
            trajectories[traj_type].append(sampling_ms)

    # Sort sampling times
    for traj_type in trajectories:
        trajectories[traj_type].sort()

    return trajectories


def process_currents(currents: np.ndarray) -> np.ndarray:
    """
    Process currents according to the convention from greybox_modeling.m:
    - Keep first column (axial coil)
    - Negate third column (side coil 2)

    Args:
        currents: Raw currents array of shape (3, N) or (N, 3)

    Returns:
        Processed currents array
    """
    currents = np.array(currents, dtype=np.float32)

    # Ensure shape is (3, N)
    if currents.shape[0] != 3 and currents.shape[1] == 3:
        currents = currents.T

    # Apply sign corrections
    processed = currents.copy()
    processed[2, :] = -processed[2, :]  # Negate third row

    return processed


def load_currents(
    trajectory_type: str = "circle",
    data_dir: Optional[str] = None,
    process: bool = True
) -> np.ndarray:
    """
    Load input currents for a trajectory type.

    Args:
        trajectory_type: 'circle' or 'lemniscate'
        data_dir: Data directory path
        process: Whether to apply sign corrections

    Returns:
        Currents array of shape (3, N)
    """
    if data_dir is None:
        data_dir = get_project_root() / DATA_DIR / "input_currents"
    else:
        data_dir = Path(data_dir)

    mat_file = data_dir / f"{trajectory_type}Currents.mat"

    if not mat_file.exists():
        raise FileNotFoundError(f"Currents file not found: {mat_file}")

    mat_data = sio.loadmat(str(mat_file))

    # Find the currents variable (usually 'expectedCurrents_traj')
    currents = None
    for key in mat_data:
        if not key.startswith('_'):
            currents = mat_data[key]
            break

    if currents is None:
        raise ValueError(f"No current data found in {mat_file}")

    currents = np.array(currents, dtype=np.float32)

    # Ensure shape is (3, N)
    if currents.shape[0] != 3:
        currents = currents.T

    if process:
        currents = process_currents(currents)

    return currents


def parse_output_trajectory(file_path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """
    Parse output trajectory file.

    Format: timestamp, arduino_state, <base_x,y,z>, <coil_x,y,z>, <normal_x,y,z>, <tip_x,y,z>

    Returns:
        Dict with keys: timestamp, arduino_state, base_position, coil_position, normal, tip_position
    """
    file_path = Path(file_path)

    data = {
        'timestamp': [],
        'arduino_state': [],
        'base_position': [],
        'coil_position': [],
        'normal': [],
        'tip_position': []
    }

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Parse: timestamp, state, <x,y,z>, <x,y,z>, <x,y,z>, <x,y,z>
            # Remove angle brackets and split
            line = line.replace('<', '').replace('>', '')
            parts = [p.strip() for p in line.split(',')]

            if len(parts) >= 14:
                data['timestamp'].append(float(parts[0]))
                data['arduino_state'].append(int(parts[1]))
                data['base_position'].append([float(parts[2]), float(parts[3]), float(parts[4])])
                data['coil_position'].append([float(parts[5]), float(parts[6]), float(parts[7])])
                data['normal'].append([float(parts[8]), float(parts[9]), float(parts[10])])
                data['tip_position'].append([float(parts[11]), float(parts[12]), float(parts[13])])

    # Convert to numpy arrays
    for key in data:
        data[key] = np.array(data[key], dtype=np.float32)

    return data


def load_experimental_data(
    trajectory_type: str = "circle",
    sampling_ms: int = 100,
    data_dir: Optional[str] = None,
    convert_to_mm: bool = True,
    relative_to_base: bool = True
) -> Dict[str, np.ndarray]:
    """
    Load experimental trajectory data.

    Args:
        trajectory_type: 'circle' or 'lemniscate'
        sampling_ms: Sampling time in milliseconds
        data_dir: Data directory path
        convert_to_mm: Convert positions from meters to millimeters
        relative_to_base: Make positions relative to base position

    Returns:
        Dict with processed trajectory data
    """
    if data_dir is None:
        root = get_project_root()
        output_dir = root / DATA_DIR / "output_trajectories"
        input_dir = root / DATA_DIR / "input_currents"
    else:
        data_dir = Path(data_dir)
        output_dir = data_dir / "output_trajectories"
        input_dir = data_dir / "input_currents"

    # Load output trajectory
    output_file = output_dir / f"{trajectory_type}{sampling_ms:02d}_01.txt"
    if not output_file.exists():
        # Try without leading zero
        output_file = output_dir / f"{trajectory_type}{sampling_ms}_01.txt"

    if not output_file.exists():
        raise FileNotFoundError(f"Trajectory file not found: {output_file}")

    data = parse_output_trajectory(output_file)

    # Load currents
    currents = load_currents(trajectory_type, input_dir, process=True)

    # Process positions
    if convert_to_mm:
        scale = 1000.0  # m to mm
        data['base_position'] *= scale
        data['coil_position'] *= scale
        data['tip_position'] *= scale

    if relative_to_base:
        # Make positions relative to base
        base = data['base_position']
        data['coil_position'] = data['coil_position'] - base
        data['tip_position'] = data['tip_position'] - base

    # Add metadata
    data['sampling_ms'] = sampling_ms
    data['sampling_s'] = sampling_ms / 1000.0
    data['trajectory_type'] = trajectory_type
    data['currents'] = currents
    data['num_samples'] = len(data['timestamp'])

    return data


class CRMDataLoader:
    """
    Data loader class for CRM experimental and simulated data.

    Handles loading, preprocessing, and batching of training data.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        trajectory_types: List[str] = ["circle", "lemniscate"],
        sampling_times_ms: Optional[List[int]] = None,
        use_slow_only: bool = True,
        convert_to_mm: bool = True,
        relative_to_base: bool = True
    ):
        """
        Initialize data loader.

        Args:
            data_dir: Data directory path
            trajectory_types: List of trajectory types to load
            sampling_times_ms: List of sampling times to use (None = all available)
            use_slow_only: If True, only use sampling times >= 20ms (slower than camera)
            convert_to_mm: Convert positions to millimeters
            relative_to_base: Make positions relative to base
        """
        self.data_dir = data_dir or str(get_project_root() / DATA_DIR)
        self.trajectory_types = trajectory_types
        self.convert_to_mm = convert_to_mm
        self.relative_to_base = relative_to_base

        if sampling_times_ms is not None:
            self.sampling_times_ms = sampling_times_ms
        elif use_slow_only:
            self.sampling_times_ms = SLOW_SAMPLING_TIMES_MS
        else:
            self.sampling_times_ms = SAMPLING_TIMES_MS

        self.data_cache = {}
        self._load_all_data()

    def _load_all_data(self):
        """Load all available data into cache."""
        available = get_available_trajectories(
            str(Path(self.data_dir) / "output_trajectories")
        )

        for traj_type in self.trajectory_types:
            if traj_type not in available:
                print(f"Warning: trajectory type '{traj_type}' not found")
                continue

            for sampling_ms in self.sampling_times_ms:
                if sampling_ms not in available[traj_type]:
                    continue

                try:
                    key = f"{traj_type}_{sampling_ms}"
                    self.data_cache[key] = load_experimental_data(
                        trajectory_type=traj_type,
                        sampling_ms=sampling_ms,
                        data_dir=self.data_dir,
                        convert_to_mm=self.convert_to_mm,
                        relative_to_base=self.relative_to_base
                    )
                    print(f"Loaded {key}: {self.data_cache[key]['num_samples']} samples")
                except Exception as e:
                    print(f"Error loading {traj_type}_{sampling_ms}: {e}")

    def get_dataset(
        self,
        trajectory_type: Optional[str] = None,
        sampling_ms: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Get dataset for specific trajectory and sampling time.

        Args:
            trajectory_type: Trajectory type (None = first available)
            sampling_ms: Sampling time (None = first available)

        Returns:
            Dataset dict
        """
        if trajectory_type is None and sampling_ms is None:
            # Return first available
            if self.data_cache:
                return list(self.data_cache.values())[0]
            raise ValueError("No data loaded")

        key = f"{trajectory_type}_{sampling_ms}"
        if key in self.data_cache:
            return self.data_cache[key]

        raise ValueError(f"Dataset not found: {key}")

    def get_combined_dataset(self) -> Dict[str, np.ndarray]:
        """
        Combine all loaded datasets into one.

        Returns:
            Combined dataset dict
        """
        if not self.data_cache:
            raise ValueError("No data loaded")

        combined = {
            'coil_position': [],
            'tip_position': [],
            'normal': [],
            'currents': [],
            'sampling_ms': [],
        }

        for key, data in self.data_cache.items():
            n_samples = data['num_samples']
            n_currents = data['currents'].shape[1]

            # Match currents to samples (may need interpolation)
            combined['coil_position'].append(data['coil_position'])
            combined['tip_position'].append(data['tip_position'])
            combined['normal'].append(data['normal'])
            combined['sampling_ms'].extend([data['sampling_ms']] * n_samples)

        # Concatenate arrays
        for key in ['coil_position', 'tip_position', 'normal']:
            combined[key] = np.concatenate(combined[key], axis=0)
        combined['sampling_ms'] = np.array(combined['sampling_ms'])

        return combined

    def get_training_pairs(
        self,
        sequence_length: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get training pairs (inputs, targets) for supervised learning.

        Args:
            sequence_length: Number of timesteps per sample

        Returns:
            Tuple of (currents, states, targets)
        """
        all_currents = []
        all_states = []
        all_targets = []

        for key, data in self.data_cache.items():
            positions = data['coil_position']
            currents = data['currents']

            # Create input-output pairs
            n_samples = min(positions.shape[0], currents.shape[1])

            for i in range(n_samples - sequence_length):
                # Input: current + current state
                curr_seq = currents[:, i:i+sequence_length].T.flatten()
                state_seq = positions[i:i+sequence_length].flatten()

                # Target: next position
                target = positions[i + sequence_length]

                all_currents.append(curr_seq)
                all_states.append(state_seq)
                all_targets.append(target)

        return (
            np.array(all_currents, dtype=np.float32),
            np.array(all_states, dtype=np.float32),
            np.array(all_targets, dtype=np.float32)
        )


if __name__ == "__main__":
    # Test data loading
    print("Testing data loader...")

    available = get_available_trajectories()
    print(f"Available trajectories: {available}")

    loader = CRMDataLoader(use_slow_only=True)
    print(f"Loaded datasets: {list(loader.data_cache.keys())}")

    if loader.data_cache:
        currents, states, targets = loader.get_training_pairs()
        print(f"Training pairs: currents={currents.shape}, states={states.shape}, targets={targets.shape}")
