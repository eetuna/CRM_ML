"""
Comprehensive PyTorch autograd integration tests.

Tests the CRMDynamicsStep autograd function in various scenarios:
- Single sample gradients
- Batch gradients
- Gradient accumulation
- Multiple backward passes
- Edge cases (zero currents, extreme values)
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import crm_torch
from crm_ml_rl.wrappers import crm_python


def setup_dynamics():
    """Common setup for all tests."""
    param_file = "data/catheter_params/CatheterParameterSet_1_dyn.txt"
    config_file = "data/catheter_params/CatheterSpatialConfiguration_1.txt"

    dyn_py = crm_python.CRMDynamics()
    dyn_py.load_parameters(param_file, config_file)
    dyn_py.initialize_from_kinematics(np.array([0.0, 0.0, 0.01]), 94.3)
    seed = dyn_py.get_seed_state()

    return param_file, config_file, seed


def test_single_sample_gradient():
    """Test gradient computation for a single sample."""
    print("\n" + "="*70)
    print("TEST 1: Single Sample Gradient")
    print("="*70)

    if not crm_torch.is_available():
        print(f"❌ Extension not available: {crm_torch.get_import_error()}")
        return False

    param_file, config_file, seed = setup_dynamics()

    # Single sample input
    currents = torch.tensor([[0.01, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    insertion = torch.tensor([94.3], dtype=torch.float64)

    # Convert seed to tensors
    seed_v = torch.from_numpy(seed['v']).unsqueeze(0).double()
    seed_w = torch.from_numpy(seed['w']).unsqueeze(0).double()
    seed_p = torch.from_numpy(seed['p']).unsqueeze(0).double()
    seed_R = torch.from_numpy(seed['R']).unsqueeze(0).double()
    seed_xf = torch.from_numpy(seed['xf']).unsqueeze(0).double()
    seed_mL = torch.from_numpy(seed['mL']).unsqueeze(0).double()
    seed_nL = torch.from_numpy(seed['nL']).unsqueeze(0).double()

    print(f"\nInput currents: {currents[0].detach().numpy()}")

    # Forward pass
    output = crm_torch.CRMDynamicsStep.apply(
        currents, insertion,
        seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL,
        param_file, config_file, 1e-4
    )

    print(f"Output shape: {output.shape}")
    print(f"Output: {output[0].detach().numpy()}")

    # Backward pass
    loss = output.sum()
    loss.backward()

    print(f"\nGradient shape: {currents.grad.shape}")
    print(f"Gradient: {currents.grad.numpy()}")
    print(f"Gradient norm: {currents.grad.norm().item():.4f}")

    # Validation
    if currents.grad is None:
        print("❌ Gradient is None")
        return False

    if torch.isnan(currents.grad).any() or torch.isinf(currents.grad).any():
        print("❌ Gradient contains NaN/Inf")
        return False

    if currents.grad.norm() < 1e-10:
        print("❌ Gradient is essentially zero")
        return False

    print("✅ PASSED: Single sample gradient computed correctly")
    return True


def test_batch_gradient():
    """Test gradient computation for batched input."""
    print("\n" + "="*70)
    print("TEST 2: Batch Gradient")
    print("="*70)

    if not crm_torch.is_available():
        print(f"❌ Extension not available: {crm_torch.get_import_error()}")
        return False

    param_file, config_file, seed = setup_dynamics()

    batch_size = 4

    # Batch input with different currents
    currents = torch.tensor([
        [0.01, 0.0, 0.0],
        [0.0, 0.01, 0.0],
        [0.0, 0.0, 0.01],
        [0.01, 0.01, 0.0],
    ], dtype=torch.float64, requires_grad=True)
    insertion = torch.tensor([94.3] * batch_size, dtype=torch.float64)

    # Convert seed to tensors (replicate for batch)
    seed_v = torch.from_numpy(seed['v']).unsqueeze(0).repeat(batch_size, 1, 1).double()
    seed_w = torch.from_numpy(seed['w']).unsqueeze(0).repeat(batch_size, 1, 1).double()
    seed_p = torch.from_numpy(seed['p']).unsqueeze(0).repeat(batch_size, 1, 1).double()
    seed_R = torch.from_numpy(seed['R']).unsqueeze(0).repeat(batch_size, 1, 1).double()
    seed_xf = torch.from_numpy(seed['xf']).unsqueeze(0).repeat(batch_size, 1).double()
    seed_mL = torch.from_numpy(seed['mL']).unsqueeze(0).repeat(batch_size, 1, 1).double()
    seed_nL = torch.from_numpy(seed['nL']).unsqueeze(0).repeat(batch_size, 1, 1).double()

    print(f"\nBatch size: {batch_size}")
    print(f"Input shape: {currents.shape}")

    # Forward pass
    output = crm_torch.CRMDynamicsStep.apply(
        currents, insertion,
        seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL,
        param_file, config_file, 1e-4
    )

    print(f"Output shape: {output.shape}")

    # Backward pass
    loss = output.sum()
    loss.backward()

    print(f"\nGradient shape: {currents.grad.shape}")
    print(f"Gradient norms per sample:")
    for i in range(batch_size):
        norm = currents.grad[i].norm().item()
        print(f"  Sample {i}: {norm:.4f}")

    # Validation
    if currents.grad is None:
        print("❌ Gradient is None")
        return False

    if currents.grad.shape != (batch_size, 3):
        print(f"❌ Gradient shape mismatch: {currents.grad.shape} != ({batch_size}, 3)")
        return False

    if torch.isnan(currents.grad).any() or torch.isinf(currents.grad).any():
        print("❌ Gradient contains NaN/Inf")
        return False

    # Check each sample has non-zero gradient
    for i in range(batch_size):
        if currents.grad[i].norm() < 1e-10:
            print(f"❌ Sample {i} gradient is essentially zero")
            return False

    print("✅ PASSED: Batch gradients computed correctly")
    return True


def test_gradient_accumulation():
    """Test gradient accumulation across multiple backward passes."""
    print("\n" + "="*70)
    print("TEST 3: Gradient Accumulation")
    print("="*70)

    if not crm_torch.is_available():
        print(f"❌ Extension not available: {crm_torch.get_import_error()}")
        return False

    param_file, config_file, seed = setup_dynamics()

    # Shared input for multiple backward passes
    currents = torch.tensor([[0.01, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    insertion = torch.tensor([94.3], dtype=torch.float64)

    seed_v = torch.from_numpy(seed['v']).unsqueeze(0).double()
    seed_w = torch.from_numpy(seed['w']).unsqueeze(0).double()
    seed_p = torch.from_numpy(seed['p']).unsqueeze(0).double()
    seed_R = torch.from_numpy(seed['R']).unsqueeze(0).double()
    seed_xf = torch.from_numpy(seed['xf']).unsqueeze(0).double()
    seed_mL = torch.from_numpy(seed['mL']).unsqueeze(0).double()
    seed_nL = torch.from_numpy(seed['nL']).unsqueeze(0).double()

    print("\nPerforming 3 backward passes with gradient accumulation...")

    # First backward pass
    output1 = crm_torch.CRMDynamicsStep.apply(
        currents, insertion,
        seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL,
        param_file, config_file, 1e-4
    )
    loss1 = output1.sum()
    loss1.backward()

    grad_after_1 = currents.grad.clone()
    print(f"Gradient after 1st backward: {grad_after_1.numpy()}")

    # Second backward pass (accumulates)
    output2 = crm_torch.CRMDynamicsStep.apply(
        currents, insertion,
        seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL,
        param_file, config_file, 1e-4
    )
    loss2 = output2.sum()
    loss2.backward()

    grad_after_2 = currents.grad.clone()
    print(f"Gradient after 2nd backward: {grad_after_2.numpy()}")

    # Third backward pass (accumulates)
    output3 = crm_torch.CRMDynamicsStep.apply(
        currents, insertion,
        seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL,
        param_file, config_file, 1e-4
    )
    loss3 = output3.sum()
    loss3.backward()

    grad_after_3 = currents.grad.clone()
    print(f"Gradient after 3rd backward: {grad_after_3.numpy()}")

    # Validation: gradient should accumulate (approximately triple)
    expected_grad = grad_after_1 * 3
    ratio = (grad_after_3 / expected_grad).abs()

    print(f"\nExpected (3x first): {expected_grad.numpy()}")
    print(f"Ratio: {ratio.numpy()}")

    # Check accumulation is working (should be close to 3x)
    if (ratio - 1.0).abs().max() > 0.1:  # 10% tolerance
        print("❌ Gradient accumulation not working as expected")
        return False

    print("✅ PASSED: Gradient accumulation works correctly")
    return True


def test_zero_grad():
    """Test gradient zeroing between backward passes."""
    print("\n" + "="*70)
    print("TEST 4: Gradient Zeroing")
    print("="*70)

    if not crm_torch.is_available():
        print(f"❌ Extension not available: {crm_torch.get_import_error()}")
        return False

    param_file, config_file, seed = setup_dynamics()

    currents = torch.tensor([[0.01, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    insertion = torch.tensor([94.3], dtype=torch.float64)

    seed_v = torch.from_numpy(seed['v']).unsqueeze(0).double()
    seed_w = torch.from_numpy(seed['w']).unsqueeze(0).double()
    seed_p = torch.from_numpy(seed['p']).unsqueeze(0).double()
    seed_R = torch.from_numpy(seed['R']).unsqueeze(0).double()
    seed_xf = torch.from_numpy(seed['xf']).unsqueeze(0).double()
    seed_mL = torch.from_numpy(seed['mL']).unsqueeze(0).double()
    seed_nL = torch.from_numpy(seed['nL']).unsqueeze(0).double()

    print("\nFirst backward pass...")
    output1 = crm_torch.CRMDynamicsStep.apply(
        currents, insertion,
        seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL,
        param_file, config_file, 1e-4
    )
    loss1 = output1.sum()
    loss1.backward()

    grad_before_zero = currents.grad.clone()
    print(f"Gradient before zero: {grad_before_zero.numpy()}")

    # Zero gradient
    currents.grad.zero_()
    print(f"Gradient after zero: {currents.grad.numpy()}")

    # Second backward pass
    print("\nSecond backward pass after zeroing...")
    output2 = crm_torch.CRMDynamicsStep.apply(
        currents, insertion,
        seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL,
        param_file, config_file, 1e-4
    )
    loss2 = output2.sum()
    loss2.backward()

    grad_after_zero = currents.grad.clone()
    print(f"Gradient after 2nd backward: {grad_after_zero.numpy()}")

    # Validation: gradients should be similar (not accumulated)
    ratio = (grad_after_zero / grad_before_zero).abs()
    print(f"\nRatio (should be ~1.0): {ratio.numpy()}")

    if (ratio - 1.0).abs().max() > 0.1:  # 10% tolerance
        print("❌ Gradient zeroing not working correctly")
        return False

    print("✅ PASSED: Gradient zeroing works correctly")
    return True


def test_edge_case_zero_currents():
    """Test with zero currents (edge case)."""
    print("\n" + "="*70)
    print("TEST 5: Edge Case - Zero Currents")
    print("="*70)

    if not crm_torch.is_available():
        print(f"❌ Extension not available: {crm_torch.get_import_error()}")
        return False

    param_file, config_file, seed = setup_dynamics()

    # Zero currents
    currents = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    insertion = torch.tensor([94.3], dtype=torch.float64)

    seed_v = torch.from_numpy(seed['v']).unsqueeze(0).double()
    seed_w = torch.from_numpy(seed['w']).unsqueeze(0).double()
    seed_p = torch.from_numpy(seed['p']).unsqueeze(0).double()
    seed_R = torch.from_numpy(seed['R']).unsqueeze(0).double()
    seed_xf = torch.from_numpy(seed['xf']).unsqueeze(0).double()
    seed_mL = torch.from_numpy(seed['mL']).unsqueeze(0).double()
    seed_nL = torch.from_numpy(seed['nL']).unsqueeze(0).double()

    print("\nInput currents: [0.0, 0.0, 0.0]")

    try:
        output = crm_torch.CRMDynamicsStep.apply(
            currents, insertion,
            seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL,
            param_file, config_file, 1e-4
        )

        loss = output.sum()
        loss.backward()

        print(f"Output: {output[0].detach().numpy()}")
        print(f"Gradient: {currents.grad.numpy()}")

        # Validation
        if torch.isnan(currents.grad).any() or torch.isinf(currents.grad).any():
            print("❌ Gradient contains NaN/Inf")
            return False

        print("✅ PASSED: Zero currents handled correctly")
        return True

    except Exception as e:
        print(f"❌ Failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_edge_case_large_currents():
    """Test with large currents (edge case)."""
    print("\n" + "="*70)
    print("TEST 6: Edge Case - Large Currents")
    print("="*70)

    if not crm_torch.is_available():
        print(f"❌ Extension not available: {crm_torch.get_import_error()}")
        return False

    param_file, config_file, seed = setup_dynamics()

    # Large currents (within reasonable bounds)
    currents = torch.tensor([[0.1, 0.1, 0.1]], dtype=torch.float64, requires_grad=True)
    insertion = torch.tensor([94.3], dtype=torch.float64)

    seed_v = torch.from_numpy(seed['v']).unsqueeze(0).double()
    seed_w = torch.from_numpy(seed['w']).unsqueeze(0).double()
    seed_p = torch.from_numpy(seed['p']).unsqueeze(0).double()
    seed_R = torch.from_numpy(seed['R']).unsqueeze(0).double()
    seed_xf = torch.from_numpy(seed['xf']).unsqueeze(0).double()
    seed_mL = torch.from_numpy(seed['mL']).unsqueeze(0).double()
    seed_nL = torch.from_numpy(seed['nL']).unsqueeze(0).double()

    print("\nInput currents: [0.1, 0.1, 0.1]")

    try:
        output = crm_torch.CRMDynamicsStep.apply(
            currents, insertion,
            seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL,
            param_file, config_file, 1e-4
        )

        loss = output.sum()
        loss.backward()

        print(f"Output: {output[0].detach().numpy()}")
        print(f"Gradient: {currents.grad.numpy()}")

        # Validation
        if torch.isnan(currents.grad).any() or torch.isinf(currents.grad).any():
            print("❌ Gradient contains NaN/Inf")
            return False

        print("✅ PASSED: Large currents handled correctly")
        return True

    except Exception as e:
        print(f"❌ Failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_losses():
    """Test with multiple loss functions."""
    print("\n" + "="*70)
    print("TEST 7: Multiple Loss Functions")
    print("="*70)

    if not crm_torch.is_available():
        print(f"❌ Extension not available: {crm_torch.get_import_error()}")
        return False

    param_file, config_file, seed = setup_dynamics()

    currents = torch.tensor([[0.01, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    insertion = torch.tensor([94.3], dtype=torch.float64)

    seed_v = torch.from_numpy(seed['v']).unsqueeze(0).double()
    seed_w = torch.from_numpy(seed['w']).unsqueeze(0).double()
    seed_p = torch.from_numpy(seed['p']).unsqueeze(0).double()
    seed_R = torch.from_numpy(seed['R']).unsqueeze(0).double()
    seed_xf = torch.from_numpy(seed['xf']).unsqueeze(0).double()
    seed_mL = torch.from_numpy(seed['mL']).unsqueeze(0).double()
    seed_nL = torch.from_numpy(seed['nL']).unsqueeze(0).double()

    # Test different loss functions
    print("\nTesting different loss functions:")

    # 1. Sum loss
    currents.grad = None
    output = crm_torch.CRMDynamicsStep.apply(
        currents, insertion,
        seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL,
        param_file, config_file, 1e-4
    )
    loss_sum = output.sum()
    loss_sum.backward()
    grad_sum = currents.grad.clone()
    print(f"  Sum loss gradient: {grad_sum.numpy()}")

    # 2. Mean loss
    currents.grad = None
    output = crm_torch.CRMDynamicsStep.apply(
        currents, insertion,
        seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL,
        param_file, config_file, 1e-4
    )
    loss_mean = output.mean()
    loss_mean.backward()
    grad_mean = currents.grad.clone()
    print(f"  Mean loss gradient: {grad_mean.numpy()}")

    # 3. L2 norm loss
    currents.grad = None
    output = crm_torch.CRMDynamicsStep.apply(
        currents, insertion,
        seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL,
        param_file, config_file, 1e-4
    )
    loss_l2 = (output ** 2).sum()
    loss_l2.backward()
    grad_l2 = currents.grad.clone()
    print(f"  L2 loss gradient: {grad_l2.numpy()}")

    # Validation: all should be finite
    if torch.isnan(grad_sum).any() or torch.isinf(grad_sum).any():
        print("❌ Sum loss gradient contains NaN/Inf")
        return False

    if torch.isnan(grad_mean).any() or torch.isinf(grad_mean).any():
        print("❌ Mean loss gradient contains NaN/Inf")
        return False

    if torch.isnan(grad_l2).any() or torch.isinf(grad_l2).any():
        print("❌ L2 loss gradient contains NaN/Inf")
        return False

    print("✅ PASSED: Multiple loss functions work correctly")
    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PYTORCH AUTOGRAD INTEGRATION TEST SUITE")
    print("="*70)

    tests = [
        ("Single Sample Gradient", test_single_sample_gradient),
        ("Batch Gradient", test_batch_gradient),
        ("Gradient Accumulation", test_gradient_accumulation),
        ("Gradient Zeroing", test_zero_grad),
        ("Edge Case: Zero Currents", test_edge_case_zero_currents),
        ("Edge Case: Large Currents", test_edge_case_large_currents),
        ("Multiple Loss Functions", test_multiple_losses),
    ]

    results = []
    for name, test_func in tests:
        success = test_func()
        results.append((name, success))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")

    print("\n" + "="*70)
    print(f"TOTAL: {passed}/{total} tests passed")

    if passed == total:
        print("✅ ALL TESTS PASSED")
    else:
        print(f"❌ {total - passed} test(s) failed")

    print("="*70 + "\n")

    sys.exit(0 if passed == total else 1)
