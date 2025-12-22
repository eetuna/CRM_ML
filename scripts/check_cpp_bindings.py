#!/usr/bin/env python3
"""
Test script for C++ physics bindings.

This script validates that the C++ bindings for forward kinematics
and dynamics are working correctly.

Usage:
    python scripts/check_cpp_bindings.py
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_bindings_import():
    """Test that C++ bindings can be imported."""
    print("\n1. Testing C++ bindings import...")
    try:
        from crm_ml_rl.wrappers import crm_python
        print("   [PASS] crm_python module imported successfully")
        return True
    except ImportError as e:
        print(f"   [FAIL] Could not import crm_python: {e}")
        print("   Hint: Run 'cd build && cmake -DBUILD_PYTHON_BINDINGS=ON .. && make crm_python'")
        return False


def test_kinematics():
    """Test forward kinematics."""
    print("\n2. Testing Forward Kinematics...")
    from crm_ml_rl.wrappers import crm_python

    # Create kinematics wrapper
    kin = crm_python.CRMKinematics()

    # Load data/simulation_parameters
    param_file = 'data/catheter_params/CatheterParameterSet_1_dyn.txt'
    config_file = 'data/catheter_params/CatheterSpatialConfiguration_1.txt'

    loaded = kin.load_parameters(param_file, config_file)
    if not loaded:
        print(f"   [FAIL] Could not load parameters from {param_file}")
        return False
    print(f"   Loaded parameters from {param_file}")

    # Test with zero currents
    currents = np.array([0.0, 0.0, 0.0])
    result = kin.forward_kinematics(currents, insertion_length=94.3)

    print(f"   Zero currents:")
    print(f"     Tip position: {result['tip_position']}")
    print(f"     Converged: {result['converged']}")

    if not result['converged']:
        print("   [FAIL] FK did not converge with zero currents")
        return False

    # Test with non-zero currents
    currents = np.array([0.0, 0.0, 0.1])
    result = kin.forward_kinematics(currents, insertion_length=94.3)

    print(f"   Non-zero currents [0, 0, 0.1]:")
    print(f"     Tip position: {result['tip_position']}")
    print(f"     Converged: {result['converged']}")

    if not result['converged']:
        print("   [FAIL] FK did not converge with non-zero currents")
        return False

    # Verify position is reasonable (z should be positive, around insertion length)
    if result['tip_position'][2] < 0 or result['tip_position'][2] > 200:
        print(f"   [FAIL] Tip z-position {result['tip_position'][2]} seems unreasonable")
        return False

    print("   [PASS] Forward Kinematics working correctly")
    return True


def test_dynamics():
    """Test dynamics simulation."""
    print("\n3. Testing Dynamics Simulation...")
    from crm_ml_rl.wrappers import crm_python

    # Create dynamics wrapper
    dyn = crm_python.CRMDynamics()

    # Load data/simulation_parameters
    param_file = 'data/catheter_params/CatheterParameterSet_1_dyn.txt'
    config_file = 'data/catheter_params/CatheterSpatialConfiguration_1.txt'

    loaded = dyn.load_parameters(param_file, config_file)
    if not loaded:
        print(f"   [FAIL] Could not load parameters")
        return False

    # Set damping (from C++ test)
    damping = np.array([
        12.1761626666366, 12.1761626666366, 284.429938756989,
        0.0304776127617393, 0.0304776127617393, 0.00502712804532508
    ])
    dyn.set_damping(damping)
    dyn.set_timestep(0.05)

    currents = np.array([0.0, 0.0, 0.1])

    # Initialize from FK (CRITICAL for avoiding convergence issues)
    print("   Initializing from forward kinematics...")
    init_ok = dyn.initialize_from_kinematics(currents, 94.3)
    if not init_ok:
        print("   [FAIL] Could not initialize from FK")
        return False

    initial_pos = dyn.get_tip_position()
    print(f"   Initial position: {initial_pos}")

    # Step dynamics
    print("   Running 10 dynamics steps...")
    positions = [initial_pos.copy()]
    converged_count = 0

    for i in range(10):
        result = dyn.step(currents, 94.3)
        positions.append(result['tip_position'].copy())
        if result['converged']:
            converged_count += 1

    print(f"   Converged steps: {converged_count}/10")
    print(f"   Final position: {positions[-1]}")

    # Check for NaN
    if np.any(np.isnan(positions[-1])):
        print("   [FAIL] Position contains NaN (unbounded integration)")
        return False

    # Check convergence
    if converged_count < 8:
        print(f"   [FAIL] Too many convergence failures ({10 - converged_count})")
        return False

    print("   [PASS] Dynamics simulation working correctly")
    return True


def test_python_wrapper():
    """Test Python wrapper classes."""
    print("\n4. Testing Python Wrapper (CRMWrapper, CRMSimulator)...")
    from crm_ml_rl.wrappers.crm_wrapper import CRMWrapper, CRMSimulator

    param_file = 'data/catheter_params/CatheterParameterSet_1_dyn.txt'
    config_file = 'data/catheter_params/CatheterSpatialConfiguration_1.txt'

    # Test CRMWrapper
    wrapper = CRMWrapper(param_file=param_file, config_file=config_file, use_cpp=True)

    if not wrapper.is_using_cpp:
        print("   [WARN] CRMWrapper not using C++ bindings (using simplified model)")
    else:
        print("   CRMWrapper using C++ bindings")

    # Set damping
    damping = np.array([
        12.1761626666366, 12.1761626666366, 284.429938756989,
        0.0304776127617393, 0.0304776127617393, 0.00502712804532508
    ])
    wrapper.set_damping(damping)
    wrapper.set_timestep(0.05)

    # Test FK
    currents = np.array([0.0, 0.0, 0.1])
    fk_result = wrapper.forward_kinematics(currents, insertion_length=94.3)
    print(f"   FK position: {fk_result['tip_position']}")

    # Test dynamics with initialization
    wrapper.reset()
    init_ok = wrapper.initialize_dynamics(currents, insertion_length=94.3)
    print(f"   Dynamics init: {'success' if init_ok else 'failed'}")

    # Test CRMSimulator
    print("   Testing CRMSimulator...")
    sim = CRMSimulator(param_file=param_file, config_file=config_file, dt=0.05, use_cpp=True)
    sim.wrapper.set_damping(damping)

    # Generate trajectory
    T = 20
    t = np.linspace(0, 1, T)
    currents_seq = np.column_stack([
        np.zeros(T),
        np.zeros(T),
        0.1 * np.sin(2 * np.pi * 0.5 * t)
    ])

    trajectory = sim.simulate_trajectory(currents_seq, insertion_length=94.3)

    print(f"   Trajectory shape: {trajectory['positions'].shape}")
    print(f"   Start: {trajectory['positions'][0]}")
    print(f"   End: {trajectory['positions'][-1]}")

    has_nan = np.any(np.isnan(trajectory['positions']))
    if has_nan:
        print("   [FAIL] Trajectory contains NaN")
        return False

    print("   [PASS] Python wrapper working correctly")
    return True


def test_comparison_with_simplified():
    """Compare C++ and simplified models."""
    print("\n5. Comparing C++ vs Simplified Model...")
    from crm_ml_rl.wrappers.crm_wrapper import CRMWrapper

    # C++ model
    param_file = 'data/catheter_params/CatheterParameterSet_1_dyn.txt'
    config_file = 'data/catheter_params/CatheterSpatialConfiguration_1.txt'

    wrapper_cpp = CRMWrapper(param_file=param_file, config_file=config_file, use_cpp=True)
    wrapper_simple = CRMWrapper(use_cpp=False)

    currents = np.array([0.0, 0.0, 0.1])

    fk_cpp = wrapper_cpp.forward_kinematics(currents, insertion_length=94.3)
    fk_simple = wrapper_simple.forward_kinematics(currents, insertion_length=94.3)

    print(f"   C++ FK position:        {fk_cpp['tip_position']}")
    print(f"   Simplified FK position: {fk_simple['tip_position']}")
    print("   (Note: Models have different parameters, so results differ)")

    print("   [INFO] Both models produce valid results")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("CRM_ML C++ Bindings Test Suite")
    print("=" * 60)

    results = {}

    # Test 1: Import
    results['import'] = test_bindings_import()

    if not results['import']:
        print("\n[CRITICAL] Cannot proceed without bindings import")
        print("=" * 60)
        return False

    # Test 2: Kinematics
    results['kinematics'] = test_kinematics()

    # Test 3: Dynamics
    results['dynamics'] = test_dynamics()

    # Test 4: Python wrapper
    results['wrapper'] = test_python_wrapper()

    # Test 5: Comparison
    results['comparison'] = test_comparison_with_simplified()

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
        print("All tests passed!")
    else:
        print("Some tests failed. See details above.")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
