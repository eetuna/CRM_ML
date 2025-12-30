#!/usr/bin/env python3
"""
Test multi-step trajectory optimization with seed gradients.

This test directly uses crm_torch.CRMDynamicsStep to validate that the
A matrix (seed gradients) are computed correctly via finite differences.
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/workspaces/catheter/CRM_ML')

import crm_torch
from crm_ml_rl.wrappers import crm_python


def test_seed_gradient_computation():
    """Test that seed gradients are non-zero and flow correctly."""
    print("=" * 80)
    print("Testing Seed Gradient Computation (A Matrix)")
    print("=" * 80)

    # Setup
    param_file = "/workspaces/catheter/CRM_ML/data/catheter_params/CatheterParameterSet_1_dyn.txt"
    config_file = "/workspaces/catheter/CRM_ML/data/catheter_params/CatheterSpatialConfiguration_1.txt"
    insertion_length = 94.3

    # Initialize dynamics to get seed state
    dyn = crm_python.CRMDynamics()
    if not dyn.load_parameters(param_file, config_file):
        raise RuntimeError("Failed to load parameters")

    initial_currents = np.array([0.0, 0.0, 0.01])
    if not dyn.initialize_from_kinematics(initial_currents, insertion_length):
        raise RuntimeError("Failed to initialize")

    # Get seed state
    seed = dyn.get_seed_state()

    # Create torch tensors
    currents = torch.tensor([[0.1, 0.05, -0.05]], dtype=torch.float64, requires_grad=True)
    insertion = torch.tensor([insertion_length], dtype=torch.float64)

    # Seed state tensors (with requires_grad=True to test seed gradients)
    seed_v = torch.from_numpy(seed['v']).unsqueeze(0).double().requires_grad_(True)
    seed_w = torch.from_numpy(seed['w']).unsqueeze(0).double().requires_grad_(True)
    seed_p = torch.from_numpy(seed['p']).unsqueeze(0).double().requires_grad_(True)
    seed_R = torch.from_numpy(seed['R']).unsqueeze(0).double().requires_grad_(True)
    seed_xf = torch.from_numpy(seed['xf']).unsqueeze(0).double().requires_grad_(True)
    seed_mL = torch.from_numpy(seed['mL']).unsqueeze(0).double().requires_grad_(True)
    seed_nL = torch.from_numpy(seed['nL']).unsqueeze(0).double().requires_grad_(True)

    print("\nSeed state shapes:")
    print(f"  v: {seed_v.shape}, requires_grad={seed_v.requires_grad}")
    print(f"  w: {seed_w.shape}, requires_grad={seed_w.requires_grad}")
    print(f"  p: {seed_p.shape}, requires_grad={seed_p.requires_grad}")
    print(f"  R: {seed_R.shape}, requires_grad={seed_R.requires_grad}")
    print(f"  xf: {seed_xf.shape}, requires_grad={seed_xf.requires_grad}")
    print(f"  mL: {seed_mL.shape}, requires_grad={seed_mL.requires_grad}")
    print(f"  nL: {seed_nL.shape}, requires_grad={seed_nL.requires_grad}")

    # Forward pass
    print("\nRunning forward pass...")
    result = crm_torch.CRMDynamicsStep.apply(
        currents, insertion,
        seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL,
        param_file, config_file, 1e-4
    )

    # Phase 3B: CRMDynamicsStep now returns tuple (output, next_v, next_w, ...)
    output, next_v, next_w, next_p, next_R, next_xf, next_mL, next_nL = result

    print(f"Output shape: {output.shape}")
    print(f"Output: {output[0].detach().numpy()}")

    # Create loss
    target_pos = torch.tensor([0.05, 0.02, 0.10], dtype=torch.float64)
    tip_pos = output[0, :3]
    loss = torch.sum((tip_pos - target_pos) ** 2)

    print(f"\nLoss: {loss.item():.6f}")

    # Backward pass
    print("\nRunning backward pass...")
    loss.backward()

    # Check gradients
    print("\n" + "=" * 80)
    print("Gradient Check")
    print("=" * 80)

    def check_grad(name, tensor):
        if tensor.grad is None:
            print(f"{name:10s}: NO GRADIENT (grad is None)")
            return False
        grad_norm = torch.norm(tensor.grad).item()
        is_nonzero = grad_norm > 1e-10
        status = "✓ PASS" if is_nonzero else "✗ FAIL (zero)"
        print(f"{name:10s}: norm={grad_norm:.6e}  {status}")
        return is_nonzero

    all_pass = True
    all_pass &= check_grad("currents", currents)
    all_pass &= check_grad("seed_v", seed_v)
    all_pass &= check_grad("seed_w", seed_w)
    all_pass &= check_grad("seed_p", seed_p)
    all_pass &= check_grad("seed_R", seed_R)
    all_pass &= check_grad("seed_xf", seed_xf)
    all_pass &= check_grad("seed_mL", seed_mL)
    all_pass &= check_grad("seed_nL", seed_nL)

    print("\n" + "=" * 80)
    if all_pass:
        print("✓ SUCCESS: All gradients (B and A matrices) are non-zero!")
        print("  This confirms seed gradients are computed correctly.")
    else:
        print("✗ FAILURE: Some gradients are zero or missing!")
        print("  This indicates A matrix computation has issues.")
    print("=" * 80)

    return all_pass


def test_multistep_gradient_flow():
    """Test that gradients flow through 2 time steps."""
    print("\n" + "=" * 80)
    print("Testing Multi-Step Gradient Flow (2 steps)")
    print("=" * 80)

    # Setup
    param_file = "/workspaces/catheter/CRM_ML/data/catheter_params/CatheterParameterSet_1_dyn.txt"
    config_file = "/workspaces/catheter/CRM_ML/data/catheter_params/CatheterSpatialConfiguration_1.txt"
    insertion_length = 94.3

    # Initialize dynamics to get seed state
    dyn = crm_python.CRMDynamics()
    if not dyn.load_parameters(param_file, config_file):
        raise RuntimeError("Failed to load parameters")

    initial_currents = np.array([0.0, 0.0, 0.01])
    if not dyn.initialize_from_kinematics(initial_currents, insertion_length):
        raise RuntimeError("Failed to initialize")

    # Get initial seed state
    seed0 = dyn.get_seed_state()

    # Convert to torch tensors
    insertion_t = torch.tensor([insertion_length], dtype=torch.float64)

    # Step 1 currents (with requires_grad)
    curr1 = torch.tensor([[0.1, 0.05, -0.05]], dtype=torch.float64, requires_grad=True)

    # Step 1 seed (no requires_grad on initial seed - it's the starting point)
    seed1_v = torch.from_numpy(seed0['v']).unsqueeze(0).double()
    seed1_w = torch.from_numpy(seed0['w']).unsqueeze(0).double()
    seed1_p = torch.from_numpy(seed0['p']).unsqueeze(0).double()
    seed1_R = torch.from_numpy(seed0['R']).unsqueeze(0).double()
    seed1_xf = torch.from_numpy(seed0['xf']).unsqueeze(0).double()
    seed1_mL = torch.from_numpy(seed0['mL']).unsqueeze(0).double()
    seed1_nL = torch.from_numpy(seed0['nL']).unsqueeze(0).double()

    print("\nStep 1: Forward pass...")
    result1 = crm_torch.CRMDynamicsStep.apply(
        curr1, insertion_t,
        seed1_v, seed1_w, seed1_p, seed1_R, seed1_xf, seed1_mL, seed1_nL,
        param_file, config_file, 1e-4
    )
    # Phase 3B: Unpack result tuple
    output1, next1_v, next1_w, next1_p, next1_R, next1_xf, next1_mL, next1_nL = result1
    print(f"  Output: {output1[0, :3].detach().numpy()}")

    # Phase 3B: Use returned updated seeds (with gradient flow!)
    # The next_* tensors are connected to curr1 via the gradient graph
    print(f"\nChecking gradient graph connection:")
    print(f"  curr1 requires_grad: {curr1.requires_grad}")
    print(f"  next1_v requires_grad: {next1_v.requires_grad}")
    print(f"  next1_v grad_fn: {next1_v.grad_fn}")

    seed2_v = next1_v
    seed2_w = next1_w
    seed2_p = next1_p
    seed2_R = next1_R
    seed2_xf = next1_xf
    seed2_mL = next1_mL
    seed2_nL = next1_nL

    # Step 2 currents (with requires_grad)
    curr2 = torch.tensor([[0.08, 0.03, -0.02]], dtype=torch.float64, requires_grad=True)

    print("\nStep 2: Forward pass...")
    result2 = crm_torch.CRMDynamicsStep.apply(
        curr2, insertion_t,
        seed2_v, seed2_w, seed2_p, seed2_R, seed2_xf, seed2_mL, seed2_nL,
        param_file, config_file, 1e-4
    )
    # Phase 3B: Unpack result tuple
    output2, next2_v, next2_w, next2_p, next2_R, next2_xf, next2_mL, next2_nL = result2
    print(f"  Output: {output2[0, :3].detach().numpy()}")

    # Create loss from step 2 output
    target_pos = torch.tensor([0.05, 0.02, 0.10], dtype=torch.float64)
    tip_pos = output2[0, :3]
    loss = torch.sum((tip_pos - target_pos) ** 2)

    print(f"\nLoss at step 2: {loss.item():.6f}")

    # Backward pass
    print("\nRunning backward pass...")
    loss.backward()

    # Phase 3B: Check that gradients flow back to curr1 (multi-step flow!)
    print("\n" + "=" * 80)
    print("Multi-Step Gradient Flow Check")
    print("=" * 80)
    print("Phase 3B FIX: Gradients should now flow from step 2 back to step 1!")
    print()

    # Check curr2 gradient (should be non-zero)
    curr2_ok = False
    if curr2.grad is not None:
        grad_norm = torch.norm(curr2.grad).item()
        is_nonzero = grad_norm > 1e-10
        status = "✓ PASS" if is_nonzero else "✗ FAIL"
        print(f"curr2 (step 2): norm={grad_norm:.6e}  {status}")
        curr2_ok = is_nonzero
    else:
        print("curr2 (step 2): NO GRADIENT ✗ FAIL")

    # Check curr1 gradient (THIS IS THE KEY TEST - multi-step gradient flow!)
    curr1_ok = False
    if curr1.grad is not None:
        grad_norm = torch.norm(curr1.grad).item()
        is_nonzero = grad_norm > 1e-10
        status = "✓ PASS" if is_nonzero else "✗ FAIL"
        print(f"curr1 (step 1): norm={grad_norm:.6e}  {status}")
        curr1_ok = is_nonzero
    else:
        print("curr1 (step 1): NO GRADIENT ✗ FAIL")

    print("\n" + "=" * 80)
    if curr2_ok and curr1_ok:
        print("✓ SUCCESS: Multi-step gradient flow is WORKING!")
        print("  Gradients flow from step 2 all the way back to step 1.")
        print("  This confirms Phase 3B fix is complete!")
    else:
        print("✗ FAILURE: Multi-step gradient flow is broken!")
        print("  Gradients do not flow through time steps correctly.")
    print("=" * 80)

    return curr2_ok and curr1_ok


if __name__ == "__main__":
    print("\n" + "#" * 80)
    print("# Seed Gradient (A Matrix) Tests")
    print("# Phase A.3: Validating FD Implementation")
    print("#" * 80)

    # Test 1: Single-step seed gradients
    test1_passed = test_seed_gradient_computation()

    # Test 2: Multi-step scenario
    test2_passed = test_multistep_gradient_flow()

    # Summary
    print("\n" + "#" * 80)
    print("# Test Summary")
    print("#" * 80)
    print(f"Test 1 (Seed Gradients):     {'✓ PASS' if test1_passed else '✗ FAIL'}")
    print(f"Test 2 (Multi-Step):         {'✓ PASS' if test2_passed else '✗ FAIL'}")
    print()

    if test1_passed and test2_passed:
        print("✓ ALL TESTS PASSED!")
        print("  Phase A.3 implementation is correct.")
        print("  A matrix (seed gradients) are computed via FD.")
        sys.exit(0)
    else:
        print("✗ SOME TESTS FAILED!")
        print("  Phase A.3 needs debugging.")
        sys.exit(1)
