#!/usr/bin/env python3
"""
Test multi-step trajectory optimization with seed gradients.

This test verifies that gradients properly flow through multiple time steps,
which requires the A matrix (seed gradients) to be computed correctly.
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/workspaces/catheter/CRM_ML')

from crm_ml_rl.wrappers.option_c_physics import OptionCPhysics


def test_multistep_gradient_flow():
    """Test that gradients flow through 5 time steps."""
    print("=" * 80)
    print("Testing Multi-Step Gradient Flow (5 steps)")
    print("=" * 80)

    # Create physics engine
    physics = OptionCPhysics(
        param_file="/workspaces/catheter/CRM_Parameters/catheter_parameters.json",
        config_file="/workspaces/catheter/CRM_Parameters/config.ini",
        insertion_length=0.1,
        num_sets=1
    )

    # Reset to get initial seed
    output, seed = physics.reset()

    # Create trajectory of currents (5 steps) with requires_grad=True
    num_steps = 5
    currents = []
    for t in range(num_steps):
        # Small random currents
        curr = torch.tensor([0.1, 0.05, -0.05], dtype=torch.float64, requires_grad=True)
        currents.append(curr)

    # Run forward pass through all steps
    print(f"\nRunning forward pass through {num_steps} steps...")
    outputs = []
    seeds = [seed]

    for t in range(num_steps):
        output, seed = physics.step(currents[t], seeds[-1])
        outputs.append(output)
        seeds.append(seed)
        print(f"  Step {t+1}: tip_pos = {output['tip_position'][:3].detach().numpy()}")

    # Create a simple loss: sum of squared distances from a target
    target_pos = torch.tensor([0.05, 0.02, 0.10], dtype=torch.float64)

    loss = 0.0
    for t in range(num_steps):
        tip_pos = output['tip_position'][:3]
        loss = loss + torch.sum((tip_pos - target_pos) ** 2)

    print(f"\nTotal loss: {loss.item():.6f}")

    # Backward pass
    print("\nRunning backward pass...")
    loss.backward()

    # Check that gradients flow all the way back to step 0
    print("\n" + "=" * 80)
    print("Gradient Flow Check")
    print("=" * 80)

    all_grads_nonzero = True
    for t in range(num_steps):
        grad = currents[t].grad
        grad_norm = torch.norm(grad).item()
        is_nonzero = grad_norm > 1e-10

        status = "✓ PASS" if is_nonzero else "✗ FAIL"
        print(f"Step {t}: grad_norm = {grad_norm:.6e}  {status}")

        if not is_nonzero:
            all_grads_nonzero = False

    print("\n" + "=" * 80)
    if all_grads_nonzero:
        print("✓ SUCCESS: Gradients flow through all time steps!")
        print("  This confirms A matrix (seed gradients) are computed correctly.")
    else:
        print("✗ FAILURE: Some gradients are zero!")
        print("  This indicates A matrix (seed gradients) are not working.")
    print("=" * 80)

    return all_grads_nonzero


def test_seed_gradient_fd_validation():
    """Validate seed gradients against finite differences."""
    print("\n" + "=" * 80)
    print("Seed Gradient Finite Difference Validation")
    print("=" * 80)

    # Create physics engine
    physics = OptionCPhysics(
        param_file="/workspaces/catheter/CRM_Parameters/catheter_parameters.json",
        config_file="/workspaces/catheter/CRM_Parameters/config.ini",
        insertion_length=0.1,
        num_sets=1
    )

    # Step 1: Get initial seed
    _, seed0 = physics.reset()

    # Step 2: Take a step with some currents
    curr1 = torch.tensor([0.1, 0.05, -0.05], dtype=torch.float64, requires_grad=True)
    output1, seed1 = physics.step(curr1, seed0)

    # Step 3: Take another step with different currents
    curr2 = torch.tensor([0.08, 0.03, -0.02], dtype=torch.float64, requires_grad=True)
    output2, seed2 = physics.step(curr2, seed1)

    # Create loss from step 2 output only
    target_pos = torch.tensor([0.05, 0.02, 0.10], dtype=torch.float64)
    tip_pos = output2['tip_position'][:3]
    loss = torch.sum((tip_pos - target_pos) ** 2)

    print(f"\nLoss at step 2: {loss.item():.6f}")

    # Get analytical gradient via backprop
    loss.backward()
    grad_curr1_analytical = curr1.grad.clone()

    print(f"Analytical grad w.r.t. curr1: {grad_curr1_analytical.numpy()}")

    # Compute numerical gradient via FD
    eps = 1e-5
    grad_curr1_numerical = torch.zeros(3, dtype=torch.float64)

    for i in range(3):
        # Perturb +eps
        curr1_plus = curr1.detach().clone()
        curr1_plus[i] += eps

        with torch.no_grad():
            _, seed1_plus = physics.step(curr1_plus, seed0)
            output2_plus, _ = physics.step(curr2.detach(), seed1_plus)
            tip_pos_plus = output2_plus['tip_position'][:3]
            loss_plus = torch.sum((tip_pos_plus - target_pos) ** 2)

        # Perturb -eps
        curr1_minus = curr1.detach().clone()
        curr1_minus[i] -= eps

        with torch.no_grad():
            _, seed1_minus = physics.step(curr1_minus, seed0)
            output2_minus, _ = physics.step(curr2.detach(), seed1_minus)
            tip_pos_minus = output2_minus['tip_position'][:3]
            loss_minus = torch.sum((tip_pos_minus - target_pos) ** 2)

        # Central difference
        grad_curr1_numerical[i] = (loss_plus - loss_minus) / (2 * eps)

    print(f"Numerical grad w.r.t. curr1:  {grad_curr1_numerical.numpy()}")

    # Compare
    diff = grad_curr1_analytical - grad_curr1_numerical
    rel_error = torch.norm(diff) / (torch.norm(grad_curr1_numerical) + 1e-10)

    print(f"\nDifference: {diff.numpy()}")
    print(f"Relative error: {rel_error.item():.6e}")

    # Check if error is small
    tolerance = 1e-3  # 0.1% tolerance
    passed = rel_error < tolerance

    print("\n" + "=" * 80)
    if passed:
        print(f"✓ SUCCESS: Seed gradients match FD within {tolerance*100}% tolerance!")
        print("  A matrix computation is correct.")
    else:
        print(f"✗ FAILURE: Seed gradients don't match FD (error: {rel_error.item():.2%})")
        print("  A matrix may have issues.")
    print("=" * 80)

    return passed


if __name__ == "__main__":
    print("\n" + "#" * 80)
    print("# Multi-Step Trajectory Gradient Tests")
    print("# Phase A.3: Validating A Matrix (Seed Gradient) Implementation")
    print("#" * 80)

    # Test 1: Gradient flow through multiple steps
    test1_passed = test_multistep_gradient_flow()

    # Test 2: FD validation of seed gradients
    test2_passed = test_seed_gradient_fd_validation()

    # Summary
    print("\n" + "#" * 80)
    print("# Test Summary")
    print("#" * 80)
    print(f"Test 1 (Gradient Flow):      {'✓ PASS' if test1_passed else '✗ FAIL'}")
    print(f"Test 2 (FD Validation):      {'✓ PASS' if test2_passed else '✗ FAIL'}")
    print()

    if test1_passed and test2_passed:
        print("✓ ALL TESTS PASSED!")
        print("  Phase A.3 is complete - seed gradients are working correctly.")
        print("  Multi-step trajectory optimization is now fully functional.")
        sys.exit(0)
    else:
        print("✗ SOME TESTS FAILED!")
        print("  Phase A.3 needs more work.")
        sys.exit(1)
