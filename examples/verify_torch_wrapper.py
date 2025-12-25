#!/usr/bin/env python3
"""
Verification script for Phase 4 Task 4.4: Torch Wrapper Gradient Audit.

Verifies:
1. CRMDynamicsStepFunction.backward returns None for insertion_length
2. grad_inputs slicing handles state vector correctly
"""

import os
import torch
import numpy as np

# Enable implicit linearization
os.environ["CRM_DYN_LINEARIZATION_METHOD"] = "implicit"

from crm_ml_rl.wrappers.torch_physics import TorchCRMPhysics


def test_insertion_length_gradient_is_none():
    """
    Verify that insertion_length gradient is None (not computed).
    """
    print(f"\n{'='*70}")
    print("Test 1: Insertion Length Gradient Returns None")
    print(f"{'='*70}")

    physics = TorchCRMPhysics(
        param_file="data/catheter_params/CatheterParameterSet_1_dyn.txt",
        config_file="data/catheter_params/CatheterSpatialConfiguration_1.txt",
        device="cpu",
    )

    # Initialize dynamics state
    currents0_np = [0.0, 0.0, 0.0]
    physics.dyn.initialize_from_kinematics(currents0_np, 50.0)
    seed = physics.dyn.get_seed_state()

    seed_v = torch.tensor(seed["v"], dtype=torch.float32).unsqueeze(0)
    seed_w = torch.tensor(seed["w"], dtype=torch.float32).unsqueeze(0)
    seed_p = torch.tensor(seed["p"], dtype=torch.float32).unsqueeze(0)
    seed_R = torch.tensor(seed["R"], dtype=torch.float32).unsqueeze(0)
    seed_xf = torch.tensor(seed["xf"], dtype=torch.float32).unsqueeze(0)
    seed_mL = torch.tensor(seed["mL"], dtype=torch.float32).unsqueeze(0)
    seed_nL = torch.tensor(seed["nL"], dtype=torch.float32).unsqueeze(0)

    currents = torch.tensor([[0.01, 0.0, 0.0]], dtype=torch.float32, requires_grad=True)
    insertion = torch.tensor([50.0], dtype=torch.float32, requires_grad=True)

    nxt = physics.dyn_step(
        currents,
        insertion,
        seed_v,
        seed_w,
        seed_p,
        seed_R,
        seed_xf,
        seed_mL=seed_mL,
        seed_nL=seed_nL,
    )

    loss = nxt.sum()
    loss.backward()

    # Check currents gradient exists
    assert currents.grad is not None, "Currents gradient should exist"
    assert torch.isfinite(currents.grad).all(), "Currents gradient should be finite"

    # Check insertion gradient is None
    assert insertion.grad is None, "Insertion length gradient should be None"

    print("✓ Currents gradient: EXISTS and is FINITE")
    print(f"  Shape: {currents.grad.shape}, Norm: {torch.norm(currents.grad).item():.6f}")
    print("✓ Insertion length gradient: None (as expected)")
    print("\n✅ PASS: backward() correctly returns None for insertion_length")

    return True


def test_seed_state_gradient_slicing():
    """
    Verify that grad_inputs slicing handles state vector correctly.
    """
    print(f"\n{'='*70}")
    print("Test 2: Seed State Gradient Slicing")
    print(f"{'='*70}")

    physics = TorchCRMPhysics(
        param_file="data/catheter_params/CatheterParameterSet_1_dyn.txt",
        config_file="data/catheter_params/CatheterSpatialConfiguration_1.txt",
        device="cpu",
    )

    # Initialize dynamics state
    currents0_np = [0.0, 0.0, 0.0]
    physics.dyn.initialize_from_kinematics(currents0_np, 50.0)
    seed = physics.dyn.get_seed_state()

    # Make all seed tensors require gradients
    seed_v = torch.tensor(seed["v"], dtype=torch.float32).unsqueeze(0).requires_grad_()
    seed_w = torch.tensor(seed["w"], dtype=torch.float32).unsqueeze(0).requires_grad_()
    seed_p = torch.tensor(seed["p"], dtype=torch.float32).unsqueeze(0).requires_grad_()
    seed_R = torch.tensor(seed["R"], dtype=torch.float32).unsqueeze(0).requires_grad_()
    seed_xf = torch.tensor(seed["xf"], dtype=torch.float32).unsqueeze(0).requires_grad_()
    seed_mL = torch.tensor(seed["mL"], dtype=torch.float32).unsqueeze(0).requires_grad_()
    seed_nL = torch.tensor(seed["nL"], dtype=torch.float32).unsqueeze(0).requires_grad_()

    currents = torch.tensor([[0.01, 0.0, 0.0]], dtype=torch.float32, requires_grad=True)
    insertion = torch.tensor([50.0], dtype=torch.float32)

    nxt = physics.dyn_step(
        currents,
        insertion,
        seed_v,
        seed_w,
        seed_p,
        seed_R,
        seed_xf,
        seed_mL=seed_mL,
        seed_nL=seed_nL,
    )

    loss = nxt.sum()
    loss.backward()

    # Verify all seed gradients exist and have correct shapes
    checks = [
        ("seed_v", seed_v.grad, (1, 1, 3)),
        ("seed_w", seed_w.grad, (1, 1, 3)),
        ("seed_p", seed_p.grad, (1, 1, 3)),
        ("seed_R", seed_R.grad, (1, 1, 9)),
        ("seed_xf", seed_xf.grad, (1, 15)),
        ("seed_mL", seed_mL.grad, (1, 1, 3)),
        ("seed_nL", seed_nL.grad, (1, 1, 3)),
    ]

    all_passed = True
    for name, grad, expected_shape in checks:
        if grad is None:
            print(f"❌ {name}: gradient is None (should exist)")
            all_passed = False
        elif grad.shape != expected_shape:
            print(f"❌ {name}: shape {grad.shape} != expected {expected_shape}")
            all_passed = False
        elif not torch.isfinite(grad).all():
            print(f"❌ {name}: contains non-finite values")
            all_passed = False
        else:
            norm = torch.norm(grad).item()
            print(f"✓ {name}: shape {grad.shape}, norm {norm:.6f}")

    if all_passed:
        print("\n✅ PASS: All seed state gradients correctly sliced and shaped")
    else:
        print("\n❌ FAIL: Some seed state gradients have issues")

    return all_passed


def main():
    print("\n" + "="*70)
    print("PHASE 4 TASK 4.4: Torch Wrapper Gradient Audit")
    print("="*70)
    print("\nVerifying torch_physics.py:93-310")

    test1 = test_insertion_length_gradient_is_none()
    test2 = test_seed_state_gradient_slicing()

    print("\n" + "="*70)
    print("AUDIT SUMMARY")
    print("="*70)
    print(f"Test 1 (insertion_length returns None): {'PASS' if test1 else 'FAIL'}")
    print(f"Test 2 (seed state slicing correct): {'PASS' if test2 else 'FAIL'}")

    if test1 and test2:
        print("\n✅ SUCCESS: Torch wrapper gradient implementation verified")
        return 0
    else:
        print("\n❌ FAILURE: Issues found in Torch wrapper")
        return 1


if __name__ == "__main__":
    exit(main())
