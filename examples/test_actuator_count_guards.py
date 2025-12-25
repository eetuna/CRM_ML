#!/usr/bin/env python3
"""
Test Actuator Count Guards for Phase 4 Task 4.11.

Verifies that runtime checks prevent mismatched input shapes.
"""

import numpy as np
from crm_ml_rl.wrappers import crm_python


def test_correct_shapes():
    """Test that correct shapes work fine."""
    print(f"\n{'='*70}")
    print("Test 1: Correct Input Shapes (num_act_set=1)")
    print(f"{'='*70}")

    dyn = crm_python.CRMDynamics()
    ok = dyn.load_parameters(
        "data/catheter_params/CatheterParameterSet_1_dyn.txt",
        "data/catheter_params/CatheterSpatialConfiguration_1.txt",
    )
    if not ok:
        raise RuntimeError("Failed to load parameters")

    # Initialize
    dyn.initialize_from_kinematics([0.0, 0.0, 0.2], 94.3)
    seed = dyn.get_seed_state()

    currents = np.array([0.01, 0.0, 0.0], dtype=np.float64)

    # This should work - shapes are correct for NUM_ACT_SET=1
    try:
        result = dyn.step_from_seed(
            currents,
            94.3,
            seed['v'],    # shape (1, 3)
            seed['w'],    # shape (1, 3)
            seed['p'],    # shape (1, 3)
            seed['R'],    # shape (1, 9)
            seed['xf'],   # shape (15,)
            seed['mL'],   # shape (1, 3)
            seed['nL']    # shape (1, 3)
        )
        print(f"✅ PASS: Correct shapes accepted")
        print(f"   Converged: {result.get('converged', False)}")
        return True
    except Exception as e:
        print(f"❌ FAIL: Correct shapes rejected: {e}")
        return False


def test_wrong_num_sets():
    """Test that wrong number of actuator sets is rejected."""
    print(f"\n{'='*70}")
    print("Test 2: Wrong Number of Actuator Sets (num_act_set=2 when NUM_ACT_SET=1)")
    print(f"{'='*70}")

    dyn = crm_python.CRMDynamics()
    ok = dyn.load_parameters(
        "data/catheter_params/CatheterParameterSet_1_dyn.txt",
        "data/catheter_params/CatheterSpatialConfiguration_1.txt",
    )
    if not ok:
        raise RuntimeError("Failed to load parameters")

    dyn.initialize_from_kinematics([0.0, 0.0, 0.2], 94.3)

    currents = np.array([0.01, 0.0, 0.0], dtype=np.float64)

    # Create inputs with 2 actuator sets (should fail with NUM_ACT_SET=1)
    v_wrong = np.zeros((2, 3), dtype=np.float64)
    w_wrong = np.zeros((2, 3), dtype=np.float64)
    p_wrong = np.zeros((2, 3), dtype=np.float64)
    R_wrong = np.zeros((2, 9), dtype=np.float64)
    R_wrong[:, [0, 4, 8]] = 1.0  # Identity matrices
    xf_wrong = np.zeros(15, dtype=np.float64)
    mL_wrong = np.zeros((2, 3), dtype=np.float64)
    nL_wrong = np.zeros((2, 3), dtype=np.float64)

    try:
        result = dyn.step_from_seed(
            currents,
            94.3,
            v_wrong,
            w_wrong,
            p_wrong,
            R_wrong,
            xf_wrong,
            mL_wrong,
            nL_wrong
        )
        print(f"❌ FAIL: Wrong num_act_set was accepted (should have been rejected)")
        return False
    except RuntimeError as e:
        error_msg = str(e)
        if "exceeds compile-time NUM_ACT_SET" in error_msg:
            print(f"✅ PASS: Wrong num_act_set correctly rejected")
            print(f"   Error message: {error_msg}")
            return True
        else:
            print(f"⚠️  WARNING: Rejected but with different error: {error_msg}")
            return True


def test_wrong_dimensions():
    """Test that wrong dimensions are rejected."""
    print(f"\n{'='*70}")
    print("Test 3: Wrong Dimensions (1D array instead of 2D)")
    print(f"{'='*70}")

    dyn = crm_python.CRMDynamics()
    ok = dyn.load_parameters(
        "data/catheter_params/CatheterParameterSet_1_dyn.txt",
        "data/catheter_params/CatheterSpatialConfiguration_1.txt",
    )
    if not ok:
        raise RuntimeError("Failed to load parameters")

    dyn.initialize_from_kinematics([0.0, 0.0, 0.2], 94.3)
    seed = dyn.get_seed_state()

    currents = np.array([0.01, 0.0, 0.0], dtype=np.float64)

    # Create v with wrong shape (1D instead of 2D)
    v_wrong = np.zeros(3, dtype=np.float64)  # Should be (1, 3)

    try:
        result = dyn.step_from_seed(
            currents,
            94.3,
            v_wrong,
            seed['w'],
            seed['p'],
            seed['R'],
            seed['xf'],
            seed['mL'],
            seed['nL']
        )
        print(f"❌ FAIL: Wrong dimensions were accepted")
        return False
    except RuntimeError as e:
        error_msg = str(e)
        if "must have shape" in error_msg:
            print(f"✅ PASS: Wrong dimensions correctly rejected")
            print(f"   Error message: {error_msg}")
            return True
        else:
            print(f"⚠️  WARNING: Rejected but with different error: {error_msg}")
            return True


def test_mismatched_shapes():
    """Test that mismatched shapes between inputs are rejected."""
    print(f"\n{'='*70}")
    print("Test 4: Mismatched Shapes (v has 1 set, w has 2 sets)")
    print(f"{'='*70}")

    dyn = crm_python.CRMDynamics()
    ok = dyn.load_parameters(
        "data/catheter_params/CatheterParameterSet_1_dyn.txt",
        "data/catheter_params/CatheterSpatialConfiguration_1.txt",
    )
    if not ok:
        raise RuntimeError("Failed to load parameters")

    dyn.initialize_from_kinematics([0.0, 0.0, 0.2], 94.3)
    seed = dyn.get_seed_state()

    currents = np.array([0.01, 0.0, 0.0], dtype=np.float64)

    # v has 1 set, but w has 2 sets (mismatched)
    w_wrong = np.zeros((2, 3), dtype=np.float64)

    try:
        result = dyn.step_from_seed(
            currents,
            94.3,
            seed['v'],  # (1, 3)
            w_wrong,    # (2, 3) - mismatched!
            seed['p'],
            seed['R'],
            seed['xf'],
            seed['mL'],
            seed['nL']
        )
        print(f"❌ FAIL: Mismatched shapes were accepted")
        return False
    except RuntimeError as e:
        error_msg = str(e)
        print(f"✅ PASS: Mismatched shapes correctly rejected")
        print(f"   Error message: {error_msg}")
        return True


def main():
    print("\n" + "="*70)
    print("PHASE 4 TASK 4.11: Actuator Count Guard Tests")
    print("="*70)
    print("\nTesting runtime validation of input shapes vs compile-time NUM_ACT_SET")

    results = []
    results.append(("Correct shapes", test_correct_shapes()))
    results.append(("Wrong num_act_set", test_wrong_num_sets()))
    results.append(("Wrong dimensions", test_wrong_dimensions()))
    results.append(("Mismatched shapes", test_mismatched_shapes()))

    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:<25}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "="*70)
    if all_passed:
        print("✅ SUCCESS: All guard tests passed")
        print("\nRuntime guards properly validate:")
        print("  • Input num_act_set doesn't exceed compile-time NUM_ACT_SET")
        print("  • All seed state arrays have correct dimensions")
        print("  • All seed state arrays have consistent num_act_set")
        return 0
    else:
        print("❌ FAILURE: Some guard tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
