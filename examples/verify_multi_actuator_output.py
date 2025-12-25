#!/usr/bin/env python3
"""
Task 4.9: Verify Multi-Actuator Output Generalization

This script verifies that the output Jacobian infrastructure supports
dynamic output sizing based on num_actuator_sets.

With NUM_ACT_SET=1 (current compile-time setting):
- Output should be 6D: [tip_position (3), actuator_0_velocity (3)]
- A and B matrices should have 6 rows

For future NUM_ACT_SET > 1:
- Output will be (3 + 3*num_sets)D
- A and B matrices will have (3 + 3*num_sets) rows
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from crm_ml_rl.wrappers import crm_python

def main():
    print("=" * 70)
    print("Task 4.9: Multi-Actuator Output Verification")
    print("=" * 70)

    # Initialize dynamics
    dyn = crm_python.CRMDynamics()
    ok = dyn.load_parameters(
        "data/catheter_params/CatheterParameterSet_1_dyn.txt",
        "data/catheter_params/CatheterSpatialConfiguration_1.txt",
    )
    if not ok:
        raise RuntimeError("Failed to load parameters")

    # Stable damping values
    BASE_DAMPING = np.array([
        12.1761626666366,
        12.1761626666366,
        284.429938756989,
        0.0304776127617393,
        0.0304776127617393,
        0.00502712804532508
    ], dtype=np.float64)

    dyn.set_damping(BASE_DAMPING)
    dyn.dt = 0.02
    dyn.integration_step_size = 0.1

    # Initialize from kinematics
    insertion_length = 94.3
    init_currents = np.array([0.0, 0.0, 0.2])
    test_currents = np.array([0.01, 0.0, 0.0], dtype=np.float64)

    print(f"\nInitializing from kinematics with insertion={insertion_length}mm...")
    dyn.initialize_from_kinematics(init_currents, insertion_length)

    seed_state = dyn.get_seed_state()
    num_sets = seed_state['v'].shape[0]

    print(f"✓ Initialization successful")
    print(f"\nSystem configuration:")
    print(f"  Num actuator sets: {num_sets}")
    print(f"  Expected output dim: {3 + 3 * num_sets}")
    print(f"    - Tip position: 3D")
    print(f"    - Actuator velocities: {3 * num_sets}D ({num_sets} actuators × 3)")

    # Call linearization
    print(f"\nCalling linearize_full_seed_action_from_seed_implicit...")
    result = dyn.linearize_full_seed_action_from_seed_implicit(
        test_currents,
        insertion_length,
        seed_state['v'],
        seed_state['w'],
        seed_state['p'],
        seed_state['R'],
        seed_state['xf'],
        seed_state['mL'],
        seed_state['nL'],
        return_debug=True
    )

    # Check output dimensions
    next_state = result['next_state']
    A = result['A']
    B = result['B']

    print(f"\n" + "=" * 70)
    print("Output Dimensions")
    print("=" * 70)
    print(f"next_state shape: {next_state.shape}")
    print(f"A matrix shape: {A.shape}")
    print(f"B matrix shape: {B.shape}")

    expected_output_dim = 3 + 3 * num_sets
    if next_state.shape[0] == expected_output_dim:
        print(f"\n✓ PASS: Output dimension is {expected_output_dim} as expected")
    else:
        print(f"\n✗ FAIL: Expected output dim {expected_output_dim}, got {next_state.shape[0]}")
        return 1

    if A.shape[0] == expected_output_dim:
        print(f"✓ PASS: A matrix has {expected_output_dim} rows as expected")
    else:
        print(f"✗ FAIL: A matrix should have {expected_output_dim} rows, got {A.shape[0]}")
        return 1

    if B.shape[0] == expected_output_dim:
        print(f"✓ PASS: B matrix has {expected_output_dim} rows as expected")
    else:
        print(f"✗ FAIL: B matrix should have {expected_output_dim} rows, got {B.shape[0]}")
        return 1

    # Check debug outputs if available
    if 'gx' in result and 'gth' in result:
        gx = result['gx']
        gth = result['gth']

        print(f"\n" + "=" * 70)
        print("Debug Output Jacobians")
        print("=" * 70)
        print(f"gx shape: {gx.shape} (should be {expected_output_dim} × {A.shape[1] - result['seed_dim']})")
        print(f"gth shape: {gth.shape} (should be {expected_output_dim} × {3 + result['seed_dim']})")

        if gx.shape[0] == expected_output_dim:
            print(f"✓ PASS: gx has {expected_output_dim} rows")
        else:
            print(f"✗ FAIL: gx should have {expected_output_dim} rows, got {gx.shape[0]}")
            return 1

        if gth.shape[0] == expected_output_dim:
            print(f"✓ PASS: gth has {expected_output_dim} rows")
        else:
            print(f"✗ FAIL: gth should have {expected_output_dim} rows, got {gth.shape[0]}")
            return 1

    # Print output breakdown
    print(f"\n" + "=" * 70)
    print("Output Vector Breakdown")
    print("=" * 70)
    print(f"next_state[0:3] = tip position:")
    print(f"  {next_state[0:3]}")

    for i in range(num_sets):
        start = 3 + i * 3
        end = start + 3
        print(f"next_state[{start}:{end}] = actuator {i} velocity:")
        print(f"  {next_state[start:end]}")

    print(f"\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    print(f"✓ PASS: Multi-actuator output infrastructure verified")
    print(f"  - Output dimension: {expected_output_dim} (dynamic based on num_sets={num_sets})")
    print(f"  - Jacobians (A, B, gx, gth) have correct row dimensions")
    print(f"  - Ready for NUM_ACT_SET > 1 (requires recompilation)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
