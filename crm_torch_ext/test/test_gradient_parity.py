#!/usr/bin/env python3
"""
CP-C07: Gradient Validation Test - Parity Check

This test verifies that the C++ extension gradients match the Python bindings
implementation (linearize_full_seed_action_from_seed_implicit) within tight tolerance.

Test Strategy:
1. Parity Test: Compare C++ gradients vs Python wrapper gradients (target: < 1e-7)
2. Finite Difference Test: Cross-check analytical gradients vs FD (target: < 1%)
3. Multiple Test Points: Run at different operating points to ensure robustness
"""

import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import crm_torch_ext
from crm_ml_rl.wrappers import crm_python

os.chdir('/workspaces/catheter/CRM_ML')


def setup_dynamics(dt=0.02, integrator="abm4", step_size=0.1):
    """Initialize both C++ extension and Python wrapper with identical parameters."""
    damping = [
        12.1761626666366, 12.1761626666366, 284.429938756989,
        0.0304776127617393, 0.0304776127617393, 0.00502712804532508
    ]

    # Initialize C++ extension
    crm_torch_ext.initialize_params(
        "data/catheter_params/CatheterParameterSet_1_dyn.txt",
        "data/catheter_params/CatheterSpatialConfiguration_1.txt"
    )
    crm_torch_ext.set_timestep(dt)
    crm_torch_ext.set_integrator(integrator)
    crm_torch_ext.set_integration_step_size(step_size)
    crm_torch_ext.set_damping(damping)

    # Initialize Python wrapper
    dyn = crm_python.CRMDynamics()
    dyn.load_parameters(
        "data/catheter_params/CatheterParameterSet_1_dyn.txt",
        "data/catheter_params/CatheterSpatialConfiguration_1.txt"
    )
    dyn.set_damping(np.array(damping))
    dyn.dt = dt
    dyn.integration_step_size = step_size
    dyn.set_integrator(integrator)

    return dyn


def get_seed_from_fk(dyn, currents, insertion):
    """Get seed state from forward kinematics."""
    success = dyn.initialize_from_kinematics(currents, insertion)
    if not success:
        raise RuntimeError("FK initialization failed")
    return dyn.get_seed_state()


def test_parity_python_wrapper(currents_np, insertion, seed, verbose=True):
    """
    Test 1: Parity with Python Wrapper
    Compare C++ extension gradients against Python bindings implementation.
    """
    if verbose:
        print("\n" + "="*70)
        print("TEST 1: PARITY WITH PYTHON WRAPPER")
        print("="*70)

    # Setup Python wrapper
    dyn = setup_dynamics()

    # Get Python gradients using implicit linearization
    result = dyn.linearize_full_seed_action_from_seed_implicit(
        currents=currents_np,
        insertion_length=insertion,
        v_in=seed['v'],
        w_in=seed['w'],
        p_in=seed['p'],
        R_in=seed['R'],
        xf_in=seed['xf'],
        mL_in=seed['mL'],
        nL_in=seed['nL']
    )

    B_python = result['B']  # (output_dim, 3) - gradients w.r.t. currents
    grad_insertion_python = result.get('grad_insertion', np.zeros(6))  # (output_dim,) - grad w.r.t. insertion

    if verbose:
        print(f"Python wrapper gradients (B matrix):")
        print(f"  Shape: {B_python.shape}")
        print(f"  B[:3, 0] (dy/d_current[0]): {B_python[:3, 0]}")
        print(f"  Norm: {np.linalg.norm(B_python):.6e}")

    # Get C++ extension gradients via PyTorch autograd
    currents_torch = torch.tensor(currents_np, dtype=torch.float64, requires_grad=True)
    insertion_torch = torch.tensor([insertion], dtype=torch.float64, requires_grad=True)

    seed_v = torch.from_numpy(seed['v'])
    seed_w = torch.from_numpy(seed['w'])
    seed_p = torch.from_numpy(seed['p'])
    seed_R = torch.from_numpy(seed['R'])
    seed_xf = torch.from_numpy(seed['xf'])
    seed_mL = torch.from_numpy(seed['mL'])
    seed_nL = torch.from_numpy(seed['nL'])

    output = crm_torch_ext.crm_step(
        currents_torch, insertion_torch,
        seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL
    )

    # Compute gradients for each output component
    output_dim = output.shape[0]
    B_cpp = np.zeros((output_dim, 3))
    grad_insertion_cpp = np.zeros(output_dim)

    for i in range(output_dim):
        if currents_torch.grad is not None:
            currents_torch.grad.zero_()
        if insertion_torch.grad is not None:
            insertion_torch.grad.zero_()

        output[i].backward(retain_graph=True)

        if currents_torch.grad is not None:
            B_cpp[i, :] = currents_torch.grad.detach().numpy()
        if insertion_torch.grad is not None:
            grad_insertion_cpp[i] = insertion_torch.grad.item()

    if verbose:
        print(f"\nC++ extension gradients (B matrix):")
        print(f"  Shape: {B_cpp.shape}")
        print(f"  B[:3, 0] (dy/d_current[0]): {B_cpp[:3, 0]}")
        print(f"  Norm: {np.linalg.norm(B_cpp):.6e}")

    # Compare
    diff = np.abs(B_cpp - B_python)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    # Relative error
    rel_error = diff / (np.abs(B_python) + 1e-15)
    max_rel_error = np.max(rel_error)

    if verbose:
        print(f"\nComparison:")
        print(f"  Max absolute difference: {max_diff:.6e}")
        print(f"  Mean absolute difference: {mean_diff:.6e}")
        print(f"  Max relative error: {max_rel_error:.6e}")

    # Check tolerance
    tolerance = 1e-7
    passed = max_diff < tolerance

    if verbose:
        if passed:
            print(f"  ✓ PASSED: Difference < {tolerance:.0e}")
        else:
            print(f"  ✗ FAILED: Difference {max_diff:.6e} exceeds tolerance {tolerance:.0e}")
            print(f"\nDetailed comparison (first 3 outputs, first current):")
            for i in range(min(3, output_dim)):
                print(f"    Output[{i}]: Python={B_python[i,0]:.6e}, C++={B_cpp[i,0]:.6e}, diff={diff[i,0]:.6e}")

    return passed, max_diff, max_rel_error


def test_finite_difference(currents_np, insertion, seed, verbose=True):
    """
    Test 2: Finite Difference Cross-Check
    Verify analytical gradients against numerical finite differences.
    """
    if verbose:
        print("\n" + "="*70)
        print("TEST 2: FINITE DIFFERENCE CROSS-CHECK")
        print("="*70)

    eps = 1e-7

    # Setup
    currents_torch = torch.tensor(currents_np, dtype=torch.float64, requires_grad=True)
    insertion_torch = torch.tensor([insertion], dtype=torch.float64)

    seed_v = torch.from_numpy(seed['v'])
    seed_w = torch.from_numpy(seed['w'])
    seed_p = torch.from_numpy(seed['p'])
    seed_R = torch.from_numpy(seed['R'])
    seed_xf = torch.from_numpy(seed['xf'])
    seed_mL = torch.from_numpy(seed['mL'])
    seed_nL = torch.from_numpy(seed['nL'])

    # Forward pass
    output = crm_torch_ext.crm_step(
        currents_torch, insertion_torch,
        seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL
    )

    # Compute analytical gradient for first output w.r.t. first current
    output[0].backward()
    analytical_grad = currents_torch.grad[0].item()

    # Compute numerical gradient
    currents_pert = torch.tensor(currents_np.copy(), dtype=torch.float64)
    currents_pert[0] += eps

    output_pert = crm_torch_ext.crm_step(
        currents_pert, insertion_torch,
        seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL
    )

    numerical_grad = (output_pert[0].item() - output[0].item()) / eps

    # Compare
    abs_diff = abs(analytical_grad - numerical_grad)
    rel_error = abs_diff / (abs(numerical_grad) + 1e-15)

    if verbose:
        print(f"Gradient of output[0] w.r.t. currents[0]:")
        print(f"  Analytical (backprop): {analytical_grad:.10e}")
        print(f"  Numerical (FD, eps={eps:.0e}): {numerical_grad:.10e}")
        print(f"  Absolute difference: {abs_diff:.6e}")
        print(f"  Relative error: {rel_error:.6e} ({rel_error*100:.4f}%)")

    tolerance = 0.01  # 1% tolerance
    passed = rel_error < tolerance

    if verbose:
        if passed:
            print(f"  ✓ PASSED: Relative error < {tolerance*100}%")
        else:
            print(f"  ✗ FAILED: Relative error {rel_error*100:.2f}% exceeds {tolerance*100}%")

    return passed, rel_error


def run_validation_suite():
    """Run complete gradient validation test suite."""
    print("="*70)
    print("CP-C07: GRADIENT VALIDATION TEST SUITE")
    print("="*70)
    print("\nThis test validates the implicit differentiation implementation")
    print("by comparing C++ extension gradients against:")
    print("  1. Python wrapper (linearize_full_seed_action_from_seed_implicit)")
    print("  2. Finite difference approximation")

    # Test cases: different operating points
    test_cases = [
        ("Zero currents, equilibrium", np.array([0.0, 0.0, 0.0]), 94.3),
        ("Small currents", np.array([0.1, 0.05, 0.0]), 94.3),
        ("Moderate currents", np.array([0.3, 0.2, 0.1]), 94.3),
    ]

    all_passed = True
    results = []

    for test_name, currents, insertion in test_cases:
        print(f"\n{'='*70}")
        print(f"TEST CASE: {test_name}")
        print(f"  Currents: {currents}")
        print(f"  Insertion: {insertion} mm")
        print(f"{'='*70}")

        try:
            # Get seed state
            dyn = setup_dynamics()
            seed = get_seed_from_fk(dyn, currents, insertion)

            # Test 1: Parity with Python wrapper
            parity_passed, max_diff, max_rel = test_parity_python_wrapper(
                currents, insertion, seed, verbose=True
            )

            # Test 2: Finite difference
            fd_passed, fd_rel_error = test_finite_difference(
                currents, insertion, seed, verbose=True
            )

            case_passed = parity_passed and fd_passed
            results.append({
                'name': test_name,
                'parity_passed': parity_passed,
                'max_diff': max_diff,
                'fd_passed': fd_passed,
                'fd_rel_error': fd_rel_error,
                'overall_passed': case_passed
            })

            if not case_passed:
                all_passed = False

        except Exception as e:
            print(f"\n✗ TEST CASE FAILED WITH ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
            results.append({
                'name': test_name,
                'parity_passed': False,
                'fd_passed': False,
                'overall_passed': False,
                'error': str(e)
            })

    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    for result in results:
        status = "✓ PASS" if result['overall_passed'] else "✗ FAIL"
        print(f"\n{status}: {result['name']}")
        if result['overall_passed']:
            print(f"  Parity test: max diff = {result['max_diff']:.6e}")
            print(f"  FD test: rel error = {result['fd_rel_error']:.6e}")
        elif 'error' in result:
            print(f"  Error: {result['error']}")

    print("\n" + "="*70)
    if all_passed:
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("CP-C07 (Gradient Validation): COMPLETE")
        print("The C++ extension gradients are mathematically correct.")
    else:
        print("✗✗✗ SOME TESTS FAILED ✗✗✗")
        print("CP-C07 (Gradient Validation): INCOMPLETE")
        print("Gradients require debugging.")
    print("="*70)

    return all_passed


if __name__ == "__main__":
    success = run_validation_suite()
    sys.exit(0 if success else 1)
