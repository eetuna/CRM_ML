#!/usr/bin/env python3
"""
Test script for data generation using the C++ simulator.

This script tests the SimDataGenerator and validates that it can
generate trajectory data using the C++ physics bindings.

Usage:
    python scripts/check_data_generation.py
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_sim_data_generator():
    """Test SimDataGenerator with C++ bindings."""
    print("\n1. Testing SimDataGenerator...")

    from crm_ml_rl.data.sim_data_generator import SimDataGenerator

    # Create generator (will use simplified model if C++ not available)
    gen = SimDataGenerator(
        dt=0.02,
        use_crm=True
    )

    print(f"   Using CRM: {gen.use_crm}")

    # Generate a single trajectory
    print("   Generating single trajectory...")

    T = 50
    t = np.linspace(0, 1, T)
    currents = np.column_stack([
        np.zeros(T),
        np.zeros(T),
        0.1 * np.sin(2 * np.pi * 0.5 * t)
    ])

    try:
        trajectory = gen.generate_trajectory(
            currents=currents,
            insertion_length=94.3
        )

        print(f"   Trajectory keys: {list(trajectory.keys())}")
        print(f"   Positions shape: {trajectory['positions'].shape}")
        print(f"   Start position: {trajectory['positions'][0]}")
        print(f"   End position: {trajectory['positions'][-1]}")

        # Check for NaN
        if np.any(np.isnan(trajectory['positions'])):
            print("   [FAIL] Trajectory contains NaN values")
            return False

        print("   [PASS] Single trajectory generation working")

    except Exception as e:
        print(f"   [FAIL] Error generating trajectory: {e}")
        return False

    return True


def test_batch_generation():
    """Test batch trajectory generation."""
    print("\n2. Testing Batch Trajectory Generation...")

    from crm_ml_rl.data.sim_data_generator import SimDataGenerator

    gen = SimDataGenerator(dt=0.02, use_crm=True)

    # Generate multiple trajectories
    n_trajectories = 5
    trajectories = []

    print(f"   Generating {n_trajectories} trajectories...")

    for i in range(n_trajectories):
        T = 30
        t = np.linspace(0, 0.6, T)

        # Random current pattern
        freq = np.random.uniform(0.3, 1.0)
        amplitude = np.random.uniform(0.05, 0.15)
        phase = np.random.uniform(0, 2*np.pi)

        currents = np.column_stack([
            amplitude * np.sin(2 * np.pi * freq * t + phase),
            amplitude * np.cos(2 * np.pi * freq * t + phase),
            np.zeros(T)
        ])

        traj = gen.generate_trajectory(currents, insertion_length=94.3)
        trajectories.append(traj)

        print(f"     Trajectory {i+1}: shape={traj['positions'].shape}, "
              f"range=[{traj['positions'].min():.2f}, {traj['positions'].max():.2f}]")

    # Check diversity
    end_positions = np.array([t['positions'][-1] for t in trajectories])
    position_std = np.std(end_positions, axis=0)
    print(f"   End position std: {position_std}")

    if np.all(position_std < 0.01):
        print("   [WARN] Trajectories are very similar (low diversity)")

    print("   [PASS] Batch generation working")
    return True


def test_dataset_creation():
    """Test creating a dataset from generated trajectories."""
    print("\n3. Testing Dataset Creation...")

    from crm_ml_rl.data.sim_data_generator import SimDataGenerator

    gen = SimDataGenerator(dt=0.02, use_crm=True)

    # Create dataset structure
    dataset = {
        'positions': [],
        'velocities': [],
        'currents': [],
        'times': []
    }

    n_trajectories = 3
    T = 25

    print(f"   Creating dataset with {n_trajectories} trajectories...")

    for i in range(n_trajectories):
        t = np.linspace(0, 0.5, T)

        currents = np.column_stack([
            0.1 * np.sin(2 * np.pi * (i + 1) * 0.5 * t),
            0.1 * np.cos(2 * np.pi * (i + 1) * 0.5 * t),
            np.zeros(T)
        ])

        traj = gen.generate_trajectory(currents, insertion_length=94.3)

        dataset['positions'].append(traj['positions'])
        dataset['velocities'].append(traj['velocities'])
        dataset['currents'].append(currents)
        dataset['times'].append(t)

    # Convert to arrays
    dataset['positions'] = np.array(dataset['positions'])
    dataset['velocities'] = np.array(dataset['velocities'])
    dataset['currents'] = np.array(dataset['currents'])
    dataset['times'] = np.array(dataset['times'])

    print(f"   Dataset shapes:")
    print(f"     positions: {dataset['positions'].shape}")
    print(f"     velocities: {dataset['velocities'].shape}")
    print(f"     currents: {dataset['currents'].shape}")
    print(f"     times: {dataset['times'].shape}")

    # Save example
    output_dir = Path("data/output")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "example_dataset.npz"

    np.savez(output_file, **dataset)
    print(f"   Saved to: {output_file}")

    # Verify loading
    loaded = np.load(output_file)
    print(f"   Loaded keys: {list(loaded.keys())}")

    print("   [PASS] Dataset creation working")
    return True


def test_circle_trajectory():
    """Test generating circle trajectory data."""
    print("\n4. Testing Circle Trajectory Generation...")

    from crm_ml_rl.data.sim_data_generator import SimDataGenerator

    gen = SimDataGenerator(dt=0.02, use_crm=True)

    # Generate circle trajectory currents
    T = 100
    freq = 0.5  # Hz
    t = np.linspace(0, 2, T)  # 2 seconds

    # Circular current pattern
    amplitude = 0.1
    currents = np.column_stack([
        amplitude * np.cos(2 * np.pi * freq * t),
        amplitude * np.sin(2 * np.pi * freq * t),
        np.zeros(T)
    ])

    print(f"   Circle frequency: {freq} Hz")
    print(f"   Duration: 2 seconds")
    print(f"   Time steps: {T}")

    traj = gen.generate_trajectory(currents, insertion_length=94.3)

    # Analyze trajectory
    positions = traj['positions']
    center = positions.mean(axis=0)
    radii = np.linalg.norm(positions - center, axis=1)

    print(f"   Trajectory center: {center}")
    print(f"   Mean radius: {radii.mean():.2f} mm")
    print(f"   Radius std: {radii.std():.2f} mm")
    print(f"   X range: [{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}]")
    print(f"   Y range: [{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}]")
    print(f"   Z range: [{positions[:, 2].min():.2f}, {positions[:, 2].max():.2f}]")

    print("   [PASS] Circle trajectory generation working")
    return True


def test_lemniscate_trajectory():
    """Test generating lemniscate (figure-8) trajectory."""
    print("\n5. Testing Lemniscate Trajectory Generation...")

    from crm_ml_rl.data.sim_data_generator import SimDataGenerator

    gen = SimDataGenerator(dt=0.02, use_crm=True)

    # Generate lemniscate (figure-8) currents
    T = 100
    freq = 0.5
    t = np.linspace(0, 2, T)

    amplitude = 0.1
    currents = np.column_stack([
        amplitude * np.sin(2 * np.pi * freq * t),
        amplitude * np.sin(2 * np.pi * 2 * freq * t),  # 2x frequency for figure-8
        np.zeros(T)
    ])

    print(f"   Generating lemniscate pattern...")

    traj = gen.generate_trajectory(currents, insertion_length=94.3)

    positions = traj['positions']
    print(f"   Trajectory shape: {positions.shape}")
    print(f"   X range: [{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}]")
    print(f"   Y range: [{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}]")

    print("   [PASS] Lemniscate trajectory generation working")
    return True


def main():
    """Run all data generation tests."""
    print("=" * 60)
    print("CRM_ML Data Generation Test Suite")
    print("=" * 60)

    results = {}

    results['sim_data_generator'] = test_sim_data_generator()
    results['batch_generation'] = test_batch_generation()
    results['dataset_creation'] = test_dataset_creation()
    results['circle_trajectory'] = test_circle_trajectory()
    results['lemniscate_trajectory'] = test_lemniscate_trajectory()

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
        print("All data generation tests passed!")
    else:
        print("Some tests failed. See details above.")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
