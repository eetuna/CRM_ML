#!/usr/bin/env python3
"""
Performance benchmark for Phase 4 Task 4.3.

Compares 30-step linearization time using full AD implicit method.
Target: < 10 seconds (down from ~2 minutes with FD-based approach).
"""

import time
import numpy as np
import os

# Enable implicit linearization (AD-based)
os.environ["CRM_DYN_LINEARIZATION_METHOD"] = "implicit"

from crm_ml_rl.wrappers import crm_python


def benchmark_linearization(num_steps=30):
    """
    Benchmark linearization performance over consecutive steps.

    Args:
        num_steps: Number of linearization calls to perform

    Returns:
        Total time in seconds
    """
    print(f"\n{'='*70}")
    print(f"LINEARIZATION PERFORMANCE BENCHMARK")
    print(f"{'='*70}")
    print(f"Method: Implicit (AD-based)")
    print(f"Number of steps: {num_steps}")

    # Initialize dynamics
    dyn = crm_python.CRMDynamics()
    ok = dyn.load_parameters(
        "data/catheter_params/CatheterParameterSet_1_dyn.txt",
        "data/catheter_params/CatheterSpatialConfiguration_1.txt",
    )
    if not ok:
        raise RuntimeError("Failed to load parameters")

    # Set damping and timestep (stable values)
    BASE_DAMPING = np.array([
        12.1761626666366,      # Linear damping X
        12.1761626666366,      # Linear damping Y
        284.429938756989,      # Linear damping Z
        0.0304776127617393,    # Angular damping X
        0.0304776127617393,    # Angular damping Y
        0.00502712804532508    # Angular damping Z
    ], dtype=np.float64)

    dyn.set_damping(BASE_DAMPING)
    dyn.dt = 0.02  # 20ms timestep
    dyn.integration_step_size = 0.1

    # Initialize from kinematics
    insertion_length = 94.3  # mm
    dyn.initialize_from_kinematics([0.0, 0.0, 0.2], insertion_length)

    # Get initial seed state
    seed = dyn.get_seed_state()

    # Small test current
    currents = np.array([0.01, 0.0, 0.0], dtype=np.float64)

    print(f"\nStarting benchmark...")

    # Warm-up call (first call may be slower due to initialization)
    try:
        _ = dyn.linearize_full_seed_action_from_seed_implicit(
            currents,
            insertion_length,
            seed['v'],
            seed['w'],
            seed['p'],
            seed['R'],
            seed['xf'],
            seed['mL'],
            seed['nL']
        )
        print(f"✓ Warm-up linearization completed")
    except Exception as e:
        print(f"⚠️  Warm-up linearization failed: {e}")
        return None

    # Timed benchmark
    start_time = time.time()
    successful_calls = 0
    failed_calls = 0

    for step in range(num_steps):
        try:
            result = dyn.linearize_full_seed_action_from_seed_implicit(
                currents,
                insertion_length,
                seed['v'],
                seed['w'],
                seed['p'],
                seed['R'],
                seed['xf'],
                seed['mL'],
                seed['nL']
            )
            successful_calls += 1

            # Update seed for next iteration (simulate trajectory)
            if 'next_v' in result:
                seed['v'] = result['next_v']
                seed['w'] = result['next_w']
                seed['p'] = result['next_p']
                seed['R'] = result['next_R']
                seed['xf'] = result['next_xf']
                seed['mL'] = result['next_mL']
                seed['nL'] = result['next_nL']
        except Exception as e:
            failed_calls += 1
            print(f"  Step {step + 1}: Linearization failed: {e}")
            break

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Successful linearization calls: {successful_calls}/{num_steps}")
    print(f"Failed calls: {failed_calls}")
    print(f"Total time: {total_time:.2f} seconds")

    if successful_calls > 0:
        avg_time = total_time / successful_calls
        print(f"Average time per linearization: {avg_time*1000:.2f} ms")

    # Check target
    TARGET_TIME = 10.0  # seconds
    if total_time < TARGET_TIME:
        print(f"\n✅ SUCCESS: Completed in {total_time:.2f}s (target: < {TARGET_TIME}s)")
    else:
        print(f"\n⚠️  SLOWER THAN TARGET: {total_time:.2f}s (target: < {TARGET_TIME}s)")
        print(f"   Exceeded by: {total_time - TARGET_TIME:.2f}s")

    return total_time


def main():
    print("\n" + "="*70)
    print("PHASE 4 TASK 4.3: Performance Benchmark")
    print("="*70)
    print("\nComparing full AD implicit linearization performance")
    print("Target: 30 steps in < 10 seconds")

    # Run benchmark
    total_time = benchmark_linearization(num_steps=30)

    if total_time is None:
        print("\n❌ Benchmark failed - linearization not working")
        return 1

    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)

    return 0


if __name__ == "__main__":
    exit(main())
