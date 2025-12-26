#!/usr/bin/env python3
"""
Task 3.1: Forward Correctness Tests (CP-C06)

Objective: Verify extension forward matches existing Python bindings exactly.

Test Plan:
1. Compare crm_torch_ext.crm_step() vs crm_python.step_from_seed()
2. At 10+ random stable points
3. Assert allclose with atol=1e-10
4. Test edge cases:
   - Zero currents
   - Maximum safe currents
   - Various insertion lengths

Acceptance Criteria:
- All forward tests pass
- Maximum difference < 1e-10
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import crm_torch_ext._crm_torch_ext as ext
from crm_ml_rl.wrappers import crm_python

# Change to repo root for file paths
os.chdir('/workspaces/catheter/CRM_ML')


def initialize_dynamics():
    """Initialize both C++ extension and Python bindings with matching parameters."""

    # Parameters
    param_file = "data/catheter_params/CatheterParameterSet_1_dyn.txt"
    config_file = "data/catheter_params/CatheterSpatialConfiguration_1.txt"
    damping = [
        12.1761626666366, 12.1761626666366, 284.429938756989,
        0.0304776127617393, 0.0304776127617393, 0.00502712804532508
    ]
    dt = 0.02
    integration_step_size = 0.1
    integrator = "abm4"

    # Initialize extension
    ext.initialize_params(param_file, config_file)
    ext.set_timestep(dt)
    ext.set_integrator(integrator)
    ext.set_integration_step_size(integration_step_size)
    ext.set_damping(damping)

    # Initialize Python bindings
    dyn = crm_python.CRMDynamics()
    dyn.load_parameters(param_file, config_file)
    dyn.set_damping(np.array(damping))
    dyn.dt = dt
    dyn.integration_step_size = integration_step_size
    dyn.set_integrator(integrator)

    return dyn


def get_stable_seed(dyn, currents, insertion_length):
    """
    Get a stable seed state using FK initialization.

    Args:
        dyn: Python dynamics object
        currents: numpy array [3]
        insertion_length: float

    Returns:
        dict with seed state tensors, or None if FK fails
    """
    success = dyn.initialize_from_kinematics(currents, insertion_length)
    if not success:
        return None

    seed = dyn.get_seed_state()

    # Convert to torch tensors
    return {
        'currents': torch.from_numpy(currents).double(),
        'insertion_length': torch.tensor([insertion_length], dtype=torch.float64),
        'seed_v': torch.from_numpy(seed['v']),
        'seed_w': torch.from_numpy(seed['w']),
        'seed_p': torch.from_numpy(seed['p']),
        'seed_R': torch.from_numpy(seed['R']),
        'seed_xf': torch.from_numpy(seed['xf']),
        'seed_mL': torch.from_numpy(seed['mL']),
        'seed_nL': torch.from_numpy(seed['nL']),
    }


def compare_outputs(ext_output, py_output):
    """
    Compare C++ extension output with Python bindings output.

    Args:
        ext_output: torch tensor from crm_step - [tip_pos (3), coil_vel (3)]
        py_output: dict from step_from_seed with keys 'tip_position', 'coil_velocities', etc.

    Returns:
        (max_error, component_errors)
    """
    # Extension returns: [tip_pos (3), coil_velocities (num_sets*3)]
    # For num_sets=1: [tip_pos (3), coil_vel (3)] = 6 elements

    ext_output_np = ext_output.detach().numpy()

    # Extract components from extension
    ext_tip_pos = ext_output_np[:3]
    ext_coil_vel = ext_output_np[3:6]

    # Extract from Python output
    py_tip_pos = py_output['tip_position']
    py_coil_vel = py_output['coil_velocities'][0]  # First actuator set

    # Compute errors
    error_tip = np.max(np.abs(ext_tip_pos - py_tip_pos))
    error_vel = np.max(np.abs(ext_coil_vel - py_coil_vel))

    max_error = max(error_tip, error_vel)

    return max_error, {
        'tip_position': error_tip,
        'coil_velocity': error_vel
    }


def test_zero_currents():
    """Test 1: Zero currents at standard insertion length."""
    print("\n" + "="*70)
    print("Test 1: Zero currents")
    print("="*70)

    dyn = initialize_dynamics()

    currents = np.array([0.0, 0.0, 0.0])
    insertion_length = 94.3

    # Get stable seed from FK
    seed_data = get_stable_seed(dyn, currents, insertion_length)
    if seed_data is None:
        print("✗ FK failed - cannot run test")
        return False

    # Run C++ extension
    try:
        ext_output = ext.crm_step(
            seed_data['currents'],
            seed_data['insertion_length'],
            seed_data['seed_v'],
            seed_data['seed_w'],
            seed_data['seed_p'],
            seed_data['seed_R'],
            seed_data['seed_xf'],
            seed_data['seed_mL'],
            seed_data['seed_nL']
        )
    except Exception as e:
        print(f"✗ Extension failed: {e}")
        return False

    # Run Python bindings
    py_output = dyn.step_from_seed(
        seed_data['currents'].numpy(),
        seed_data['insertion_length'].item(),
        seed_data['seed_v'].numpy(),
        seed_data['seed_w'].numpy(),
        seed_data['seed_p'].numpy(),
        seed_data['seed_R'].numpy(),
        seed_data['seed_xf'].numpy(),
        seed_data['seed_mL'].numpy(),
        seed_data['seed_nL'].numpy()
    )

    # Compare
    max_error, errors = compare_outputs(ext_output, py_output)

    print(f"Currents: {currents}")
    print(f"Insertion: {insertion_length} mm")
    print(f"\nErrors:")
    print(f"  Tip position:   {errors['tip_position']:.2e}")
    print(f"  Coil velocity:  {errors['coil_velocity']:.2e}")
    print(f"  Max: {max_error:.2e}")

    if max_error < 1e-10:
        print(f"\n✓ PASS: Maximum error {max_error:.2e} < 1e-10")
        return True
    else:
        print(f"\n✗ FAIL: Maximum error {max_error:.2e} >= 1e-10")
        return False


def test_small_currents():
    """Test 2: Small non-zero currents."""
    print("\n" + "="*70)
    print("Test 2: Small non-zero currents")
    print("="*70)

    dyn = initialize_dynamics()

    test_cases = [
        np.array([0.1, 0.0, 0.0]),
        np.array([0.0, 0.1, 0.0]),
        np.array([0.0, 0.0, 0.1]),
        np.array([0.05, 0.05, 0.05]),
    ]

    insertion_length = 94.3
    all_passed = True

    for i, currents in enumerate(test_cases):
        print(f"\nCase {i+1}: currents = {currents}")

        # Get stable seed from FK
        seed_data = get_stable_seed(dyn, currents, insertion_length)
        if seed_data is None:
            print("  ✗ FK failed - skipping")
            continue

        # Run C++ extension
        try:
            ext_output = ext.crm_step(
                seed_data['currents'],
                seed_data['insertion_length'],
                seed_data['seed_v'],
                seed_data['seed_w'],
                seed_data['seed_p'],
                seed_data['seed_R'],
                seed_data['seed_xf'],
                seed_data['seed_mL'],
                seed_data['seed_nL']
            )
        except Exception as e:
            print(f"  ✗ Extension failed: {e}")
            all_passed = False
            continue

        # Run Python bindings
        py_output = dyn.step_from_seed(
            seed_data['currents'].numpy(),
            seed_data['insertion_length'].item(),
            seed_data['seed_v'].numpy(),
            seed_data['seed_w'].numpy(),
            seed_data['seed_p'].numpy(),
            seed_data['seed_R'].numpy(),
            seed_data['seed_xf'].numpy(),
            seed_data['seed_mL'].numpy(),
            seed_data['seed_nL'].numpy()
        )

        # Compare
        max_error, errors = compare_outputs(ext_output, py_output)

        print(f"  Errors: tip={errors['tip_position']:.2e}, vel={errors['coil_velocity']:.2e}")
        print(f"  Max error: {max_error:.2e}")

        if max_error < 1e-10:
            print(f"  ✓ PASS")
        else:
            print(f"  ✗ FAIL")
            all_passed = False

    if all_passed:
        print(f"\n✓ All small current tests PASSED")
        return True
    else:
        print(f"\n✗ Some small current tests FAILED")
        return False


def test_various_insertion_lengths():
    """Test 3: Various insertion lengths with zero currents."""
    print("\n" + "="*70)
    print("Test 3: Various insertion lengths")
    print("="*70)

    dyn = initialize_dynamics()

    currents = np.array([0.0, 0.0, 0.0])
    test_lengths = [70.0, 80.0, 90.0, 94.3, 100.0]

    all_passed = True

    for length in test_lengths:
        print(f"\nInsertion length: {length} mm")

        # Get stable seed from FK
        seed_data = get_stable_seed(dyn, currents, length)
        if seed_data is None:
            print("  ✗ FK failed - skipping")
            continue

        # Run C++ extension
        try:
            ext_output = ext.crm_step(
                seed_data['currents'],
                seed_data['insertion_length'],
                seed_data['seed_v'],
                seed_data['seed_w'],
                seed_data['seed_p'],
                seed_data['seed_R'],
                seed_data['seed_xf'],
                seed_data['seed_mL'],
                seed_data['seed_nL']
            )
        except Exception as e:
            print(f"  ✗ Extension failed: {e}")
            all_passed = False
            continue

        # Run Python bindings
        py_output = dyn.step_from_seed(
            seed_data['currents'].numpy(),
            seed_data['insertion_length'].item(),
            seed_data['seed_v'].numpy(),
            seed_data['seed_w'].numpy(),
            seed_data['seed_p'].numpy(),
            seed_data['seed_R'].numpy(),
            seed_data['seed_xf'].numpy(),
            seed_data['seed_mL'].numpy(),
            seed_data['seed_nL'].numpy()
        )

        # Compare
        max_error, errors = compare_outputs(ext_output, py_output)

        print(f"  Errors: tip={errors['tip_position']:.2e}, vel={errors['coil_velocity']:.2e}")
        print(f"  Max error: {max_error:.2e}")

        if max_error < 1e-10:
            print(f"  ✓ PASS")
        else:
            print(f"  ✗ FAIL")
            all_passed = False

    if all_passed:
        print(f"\n✓ All insertion length tests PASSED")
        return True
    else:
        print(f"\n✗ Some insertion length tests FAILED")
        return False


def test_random_stable_points():
    """Test 4: Random stable points (main validation test)."""
    print("\n" + "="*70)
    print("Test 4: Random stable points (10+ configurations)")
    print("="*70)

    dyn = initialize_dynamics()

    # Generate random test cases
    np.random.seed(42)  # Reproducible
    num_tests = 15

    passed = 0
    failed = 0
    max_error_overall = 0.0

    for i in range(num_tests):
        # Random currents (small values for stability)
        currents = np.random.uniform(-0.2, 0.2, size=3)

        # Random insertion length
        insertion_length = np.random.uniform(70.0, 100.0)

        print(f"\nTest {i+1}/{num_tests}:")
        print(f"  Currents: [{currents[0]:.4f}, {currents[1]:.4f}, {currents[2]:.4f}]")
        print(f"  Insertion: {insertion_length:.2f} mm")

        # Get stable seed from FK
        seed_data = get_stable_seed(dyn, currents, insertion_length)
        if seed_data is None:
            print("  ⊘ FK failed - skipping")
            continue

        # Run C++ extension
        try:
            ext_output = ext.crm_step(
                seed_data['currents'],
                seed_data['insertion_length'],
                seed_data['seed_v'],
                seed_data['seed_w'],
                seed_data['seed_p'],
                seed_data['seed_R'],
                seed_data['seed_xf'],
                seed_data['seed_mL'],
                seed_data['seed_nL']
            )
        except Exception as e:
            print(f"  ✗ Extension failed: {e}")
            failed += 1
            continue

        # Run Python bindings
        py_output = dyn.step_from_seed(
            seed_data['currents'].numpy(),
            seed_data['insertion_length'].item(),
            seed_data['seed_v'].numpy(),
            seed_data['seed_w'].numpy(),
            seed_data['seed_p'].numpy(),
            seed_data['seed_R'].numpy(),
            seed_data['seed_xf'].numpy(),
            seed_data['seed_mL'].numpy(),
            seed_data['seed_nL'].numpy()
        )

        # Compare
        max_error, errors = compare_outputs(ext_output, py_output)
        max_error_overall = max(max_error_overall, max_error)

        print(f"  Max error: {max_error:.2e}")

        if max_error < 1e-10:
            print(f"  ✓ PASS")
            passed += 1
        else:
            print(f"  ✗ FAIL (tip={errors['tip_position']:.2e}, vel={errors['coil_velocity']:.2e})")
            failed += 1

    print(f"\n" + "="*70)
    print(f"Random stable points summary:")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Overall maximum error: {max_error_overall:.2e}")
    print("="*70)

    if failed == 0 and passed >= 10:
        print(f"\n✓ Random stable points test PASSED (>10 points tested)")
        return True
    else:
        print(f"\n✗ Random stable points test FAILED")
        return False


def test_maximum_safe_currents():
    """Test 5: Maximum safe current magnitudes."""
    print("\n" + "="*70)
    print("Test 5: Maximum safe currents")
    print("="*70)

    dyn = initialize_dynamics()

    # Safe current range is typically -1.0 to 1.0 A, but FK may not converge at extremes
    # Test with moderately high currents
    test_cases = [
        np.array([0.5, 0.0, 0.0]),
        np.array([0.0, 0.5, 0.0]),
        np.array([0.0, 0.0, 0.5]),
        np.array([0.3, 0.3, 0.3]),
        np.array([-0.3, -0.3, -0.3]),
    ]

    insertion_length = 94.3
    all_passed = True

    for i, currents in enumerate(test_cases):
        print(f"\nCase {i+1}: currents = {currents}")

        # Get stable seed from FK
        seed_data = get_stable_seed(dyn, currents, insertion_length)
        if seed_data is None:
            print("  ⊘ FK failed (expected for high currents) - skipping")
            continue

        # Run C++ extension
        try:
            ext_output = ext.crm_step(
                seed_data['currents'],
                seed_data['insertion_length'],
                seed_data['seed_v'],
                seed_data['seed_w'],
                seed_data['seed_p'],
                seed_data['seed_R'],
                seed_data['seed_xf'],
                seed_data['seed_mL'],
                seed_data['seed_nL']
            )
        except Exception as e:
            print(f"  ✗ Extension failed: {e}")
            all_passed = False
            continue

        # Run Python bindings
        py_output = dyn.step_from_seed(
            seed_data['currents'].numpy(),
            seed_data['insertion_length'].item(),
            seed_data['seed_v'].numpy(),
            seed_data['seed_w'].numpy(),
            seed_data['seed_p'].numpy(),
            seed_data['seed_R'].numpy(),
            seed_data['seed_xf'].numpy(),
            seed_data['seed_mL'].numpy(),
            seed_data['seed_nL'].numpy()
        )

        # Compare
        max_error, errors = compare_outputs(ext_output, py_output)

        print(f"  Errors: tip={errors['tip_position']:.2e}, vel={errors['coil_velocity']:.2e}")
        print(f"  Max error: {max_error:.2e}")

        if max_error < 1e-10:
            print(f"  ✓ PASS")
        else:
            print(f"  ✗ FAIL")
            all_passed = False

    if all_passed:
        print(f"\n✓ Maximum safe currents test PASSED")
        return True
    else:
        print(f"\n✗ Some maximum safe currents tests FAILED")
        return False


def main():
    """Run all forward correctness tests."""
    print("="*70)
    print("TASK 3.1: FORWARD CORRECTNESS TESTS (CP-C06)")
    print("="*70)
    print("\nObjective: Verify C++ extension matches Python bindings exactly")
    print("Acceptance Criteria: All tests pass with max error < 1e-10")

    tests = [
        ("Zero currents", test_zero_currents),
        ("Small non-zero currents", test_small_currents),
        ("Various insertion lengths", test_various_insertion_lengths),
        ("Random stable points (10+)", test_random_stable_points),
        ("Maximum safe currents", test_maximum_safe_currents),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    passed = sum(results)
    total = len(results)

    for i, (name, _) in enumerate(tests):
        status = "✓ PASS" if results[i] else "✗ FAIL"
        print(f"{status} - {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n" + "="*70)
        print("✓ CP-C06 ACCEPTANCE CRITERIA MET")
        print("="*70)
        print("- All forward tests pass")
        print("- Maximum difference < 1e-10")
        print("- C++ extension matches Python bindings exactly")
        return 0
    else:
        print(f"\n✗ CP-C06 INCOMPLETE ({total - passed} tests failed)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
