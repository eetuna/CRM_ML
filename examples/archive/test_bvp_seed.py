#!/usr/bin/env python3
"""
Minimal test to understand BVP solver failure in step_from_seed.

Tests whether the issue is:
1. The seed state values themselves
2. How the seed is passed to the BVP solver
3. Something about consecutive calls
"""

import numpy as np
from crm_ml_rl.wrappers import crm_python

# Stable damping values
BASE_DAMPING = np.array([
    12.1761626666366,
    12.1761626666366,
    284.429938756989,
    0.0304776127617393,
    0.0304776127617393,
    0.00502712804532508
], dtype=np.float64)

def test_step_from_seed():
    """Test step_from_seed with different seed sources."""

    # Initialize dynamics
    dyn = crm_python.CRMDynamics()
    ok = dyn.load_parameters(
        "data/catheter_params/CatheterParameterSet_1_dyn.txt",
        "data/catheter_params/CatheterSpatialConfiguration_1.txt",
    )
    if not ok:
        raise RuntimeError("Failed to load parameters")

    dyn.set_damping(BASE_DAMPING)
    dyn.dt = 0.02
    dyn.integration_step_size = 0.1

    insertion_length = 94.3
    currents = np.array([0.01, 0.0, 0.0], dtype=np.float64)

    # Test 1: Initialize from kinematics, get seed, step TWICE with SAME seed
    print("="*70)
    print("Test 1: Step twice with SAME kinematic seed")
    print("="*70)

    dyn.initialize_from_kinematics([0.0, 0.0, 0.2], insertion_length)
    seed_kin = dyn.get_seed_state()

    print("\nStep 1 (using kinematic seed):")
    result1 = dyn.step_from_seed(
        currents, insertion_length,
        seed_kin['v'], seed_kin['w'], seed_kin['p'], seed_kin['R'],
        seed_kin['xf'], seed_kin['mL'], seed_kin['nL']
    )
    print(f"  Converged: {result1['converged']}, Diverged: {result1['diverged']}")

    print("\nStep 2 (using SAME kinematic seed again):")
    result2 = dyn.step_from_seed(
        currents, insertion_length,
        seed_kin['v'], seed_kin['w'], seed_kin['p'], seed_kin['R'],
        seed_kin['xf'], seed_kin['mL'], seed_kin['nL']
    )
    print(f"  Converged: {result2['converged']}, Diverged: {result2['diverged']}")

    # Test 2: Use output from step 1 as input to step 2
    print("\n" + "="*70)
    print("Test 2: Step 1 output -> Step 2 input (consecutive stepping)")
    print("="*70)

    dyn.initialize_from_kinematics([0.0, 0.0, 0.2], insertion_length)
    seed_kin = dyn.get_seed_state()

    print("\nStep 1 (using kinematic seed):")
    result1 = dyn.step_from_seed(
        currents, insertion_length,
        seed_kin['v'], seed_kin['w'], seed_kin['p'], seed_kin['R'],
        seed_kin['xf'], seed_kin['mL'], seed_kin['nL']
    )
    print(f"  Converged: {result1['converged']}, Diverged: {result1['diverged']}")

    if result1['converged']:
        seed_from_step1 = {
            'v': result1['next_v'],
            'w': result1['next_w'],
            'p': result1['next_p'],
            'R': result1['next_R'],
            'xf': result1['next_xf'],
            'mL': result1['next_mL'],
            'nL': result1['next_nL']
        }

        print("\nStep 2 (using Step 1 output as seed):")
        print(f"  Input seed v: {seed_from_step1['v'][0]}")
        print(f"  Input seed w: {seed_from_step1['w'][0]}")
        print(f"  Input seed mL: {seed_from_step1['mL'][0]}")
        print(f"  Input seed nL: {seed_from_step1['nL'][0]}")

        result2 = dyn.step_from_seed(
            currents, insertion_length,
            seed_from_step1['v'], seed_from_step1['w'], seed_from_step1['p'],
            seed_from_step1['R'], seed_from_step1['xf'],
            seed_from_step1['mL'], seed_from_step1['nL']
        )
        print(f"  Converged: {result2['converged']}, Diverged: {result2['diverged']}")

    # Test 3: Re-initialize dynamics between calls
    print("\n" + "="*70)
    print("Test 3: Re-initialize dynamics between step 1 and step 2")
    print("="*70)

    dyn.initialize_from_kinematics([0.0, 0.0, 0.2], insertion_length)
    seed_kin = dyn.get_seed_state()

    print("\nStep 1 (using kinematic seed):")
    result1 = dyn.step_from_seed(
        currents, insertion_length,
        seed_kin['v'], seed_kin['w'], seed_kin['p'], seed_kin['R'],
        seed_kin['xf'], seed_kin['mL'], seed_kin['nL']
    )
    print(f"  Converged: {result1['converged']}, Diverged: {result1['diverged']}")

    if result1['converged']:
        # Re-initialize with the result from step 1
        # But we can't directly initialize from v,w,p,R... let me try something else

        # Try just calling step_from_seed again but with zero currents first
        print("\nIntermediate: Step with zero currents using Step 1 output:")
        seed_from_step1 = {
            'v': result1['next_v'],
            'w': result1['next_w'],
            'p': result1['next_p'],
            'R': result1['next_R'],
            'xf': result1['next_xf'],
            'mL': result1['next_mL'],
            'nL': result1['next_nL']
        }

        zero_currents = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        result_zero = dyn.step_from_seed(
            zero_currents, insertion_length,
            seed_from_step1['v'], seed_from_step1['w'], seed_from_step1['p'],
            seed_from_step1['R'], seed_from_step1['xf'],
            seed_from_step1['mL'], seed_from_step1['nL']
        )
        print(f"  Converged: {result_zero['converged']}, Diverged: {result_zero['diverged']}")


def test_exact_debug_script_pattern():
    """Replicate EXACT pattern from debug_consecutive_stepping.py"""
    print("\n" + "="*70)
    print("Test 4: EXACT debug_consecutive_stepping.py pattern")
    print("="*70)

    dyn = crm_python.CRMDynamics()
    dyn.load_parameters('data/catheter_params/CatheterParameterSet_1_dyn.txt', 'data/catheter_params/CatheterSpatialConfiguration_1.txt')
    dyn.set_damping(BASE_DAMPING)
    dyn.dt = 0.02
    dyn.integration_step_size = 0.1

    insertion_length = 94.3
    dyn.initialize_from_kinematics([0.0, 0.0, 0.2], insertion_length)

    currents = np.array([0.01, 0.0, 0.0], dtype=np.float64)

    seed = dyn.get_seed_state()

    # Step 1
    result = dyn.step_from_seed(currents, insertion_length, seed['v'], seed['w'], seed['p'], seed['R'], seed['xf'], seed['mL'], seed['nL'])
    print(f"Step 1: converged={result['converged']}, diverged={result['diverged']}")

    # Update seed EXACTLY as in debug script
    seed = {
        'v': result['next_v'],
        'w': result['next_w'],
        'p': result['next_p'],
        'R': result['next_R'],
        'xf': result['next_xf'],
        'mL': result['next_mL'],
        'nL': result['next_nL']
    }

    # Step 2
    result = dyn.step_from_seed(currents, insertion_length, seed['v'], seed['w'], seed['p'], seed['R'], seed['xf'], seed['mL'], seed['nL'])
    print(f"Step 2: converged={result['converged']}, diverged={result['diverged']}")


if __name__ == "__main__":
    test_step_from_seed()
    test_exact_debug_script_pattern()
