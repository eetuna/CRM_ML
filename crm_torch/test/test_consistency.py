"""
Test Option C (crm_torch) consistency with Option A (torch_physics.py).

This test verifies that the C++ extension produces identical outputs
to the existing Python implementation within tolerance.
"""

import os
import sys
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import crm_torch
from crm_ml_rl.wrappers.torch_physics import TorchCRMPhysics


def test_forward_single_sample():
    """
    Test that Option C forward pass matches Option A for a single sample.
    """
    print("\n" + "="*70)
    print("TEST: Forward Pass Consistency (Single Sample)")
    print("="*70)

    # Configuration files
    param_file = "data/catheter_params/CatheterParameterSet_1_dyn.txt"
    config_file = "data/catheter_params/CatheterSpatialConfiguration_1.txt"

    # Create Option A physics instance
    physics_a = TorchCRMPhysics(param_file, config_file, device="cpu")

    # Initialize from kinematics
    currents_init = torch.tensor([[0.0, 0.0, 0.2]], dtype=torch.float32)
    insertion_init = torch.tensor([94.3], dtype=torch.float32)

    # Get seed state from Option A
    dyn_a = physics_a.dyn
    dyn_a.initialize_from_kinematics(currents_init[0].numpy(), insertion_init[0].item())
    seed_dict = dyn_a.get_seed_state()

    # Convert seed to tensors
    seed_v = torch.from_numpy(seed_dict['v']).unsqueeze(0).float()  # (1, num_sets, 3)
    seed_w = torch.from_numpy(seed_dict['w']).unsqueeze(0).float()
    seed_p = torch.from_numpy(seed_dict['p']).unsqueeze(0).float()
    seed_R = torch.from_numpy(seed_dict['R']).unsqueeze(0).float()
    seed_xf = torch.from_numpy(seed_dict['xf']).unsqueeze(0).float()
    seed_mL = torch.from_numpy(seed_dict['mL']).unsqueeze(0).float()
    seed_nL = torch.from_numpy(seed_dict['nL']).unsqueeze(0).float()

    # Test currents (small perturbation)
    currents_test = torch.tensor([[0.01, 0.0, 0.0]], dtype=torch.float32)
    insertion_test = torch.tensor([94.3], dtype=torch.float32)

    print(f"\nInput shapes:")
    print(f"  currents: {currents_test.shape}")
    print(f"  seed_v: {seed_v.shape}")
    print(f"  seed_xf: {seed_xf.shape}")

    # Option A forward pass
    print(f"\n[Option A] Running forward pass...")
    output_a = physics_a.dyn_step(
        currents_test, insertion_test,
        seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL
    )
    print(f"  Output shape: {output_a.shape}")
    print(f"  Output: {output_a}")

    # Option C forward pass
    print(f"\n[Option C] Running forward pass...")
    output_c = crm_torch.CRMDynamicsStep.apply(
        currents_test, insertion_test,
        seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL,
        param_file, config_file
    )
    print(f"  Output shape: {output_c.shape}")
    print(f"  Output: {output_c}")

    # Compare
    print(f"\n[Comparison]")
    abs_diff = torch.abs(output_a - output_c)
    rel_diff = abs_diff / (torch.abs(output_a) + 1e-10)

    print(f"  Max absolute difference: {abs_diff.max().item():.2e}")
    print(f"  Mean absolute difference: {abs_diff.mean().item():.2e}")
    print(f"  Max relative difference: {rel_diff.max().item():.2e}")
    print(f"  Mean relative difference: {rel_diff.mean().item():.2e}")

    # Check tolerance (1e-6 as per plan)
    tolerance = 1e-6
    if abs_diff.max().item() < tolerance:
        print(f"\n✅ PASS: Max diff {abs_diff.max().item():.2e} < {tolerance:.2e}")
        return True
    else:
        print(f"\n❌ FAIL: Max diff {abs_diff.max().item():.2e} >= {tolerance:.2e}")
        return False


def test_forward_batch():
    """
    Test that Option C forward pass matches Option A for a batch.
    """
    print("\n" + "="*70)
    print("TEST: Forward Pass Consistency (Batch Size 5)")
    print("="*70)

    # Configuration files
    param_file = "data/catheter_params/CatheterParameterSet_1_dyn.txt"
    config_file = "data/catheter_params/CatheterSpatialConfiguration_1.txt"

    # Create Option A physics instance
    physics_a = TorchCRMPhysics(param_file, config_file, device="cpu")

    # Batch size
    batch_size = 5

    # Initialize from kinematics (same for all batch elements)
    currents_init = torch.tensor([[0.0, 0.0, 0.2]], dtype=torch.float32)
    insertion_init = torch.tensor([94.3], dtype=torch.float32)

    dyn_a = physics_a.dyn
    dyn_a.initialize_from_kinematics(currents_init[0].numpy(), insertion_init[0].item())
    seed_dict = dyn_a.get_seed_state()

    # Create batched seeds (replicate across batch)
    seed_v = torch.from_numpy(seed_dict['v']).unsqueeze(0).float().repeat(batch_size, 1, 1)
    seed_w = torch.from_numpy(seed_dict['w']).unsqueeze(0).float().repeat(batch_size, 1, 1)
    seed_p = torch.from_numpy(seed_dict['p']).unsqueeze(0).float().repeat(batch_size, 1, 1)
    seed_R = torch.from_numpy(seed_dict['R']).unsqueeze(0).float().repeat(batch_size, 1, 1)
    seed_xf = torch.from_numpy(seed_dict['xf']).unsqueeze(0).float().repeat(batch_size, 1)
    seed_mL = torch.from_numpy(seed_dict['mL']).unsqueeze(0).float().repeat(batch_size, 1, 1)
    seed_nL = torch.from_numpy(seed_dict['nL']).unsqueeze(0).float().repeat(batch_size, 1, 1)

    # Varying currents for each batch element
    currents_test = torch.tensor([
        [0.01, 0.0, 0.0],
        [0.0, 0.01, 0.0],
        [0.0, 0.0, 0.01],
        [0.005, 0.005, 0.0],
        [0.0, 0.005, 0.005],
    ], dtype=torch.float32)
    insertion_test = torch.tensor([94.3], dtype=torch.float32)

    print(f"\nBatch size: {batch_size}")
    print(f"Input shapes:")
    print(f"  currents: {currents_test.shape}")
    print(f"  seed_v: {seed_v.shape}")

    # Option A forward pass
    print(f"\n[Option A] Running batched forward pass...")
    output_a = physics_a.dyn_step(
        currents_test, insertion_test,
        seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL
    )
    print(f"  Output shape: {output_a.shape}")

    # Option C forward pass
    print(f"\n[Option C] Running batched forward pass...")
    output_c = crm_torch.CRMDynamicsStep.apply(
        currents_test, insertion_test,
        seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL,
        param_file, config_file
    )
    print(f"  Output shape: {output_c.shape}")

    # Compare
    print(f"\n[Comparison]")
    abs_diff = torch.abs(output_a - output_c)
    rel_diff = abs_diff / (torch.abs(output_a) + 1e-10)

    print(f"  Max absolute difference: {abs_diff.max().item():.2e}")
    print(f"  Mean absolute difference: {abs_diff.mean().item():.2e}")
    print(f"  Max relative difference: {rel_diff.max().item():.2e}")
    print(f"  Mean relative difference: {rel_diff.mean().item():.2e}")

    # Check tolerance
    tolerance = 1e-6
    if abs_diff.max().item() < tolerance:
        print(f"\n✅ PASS: Max diff {abs_diff.max().item():.2e} < {tolerance:.2e}")
        return True
    else:
        print(f"\n❌ FAIL: Max diff {abs_diff.max().item():.2e} >= {tolerance:.2e}")
        return False


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("\n" + "="*70)
    print("Option C vs Option A Consistency Tests")
    print("="*70)
    print(f"crm_torch available: {crm_torch.is_available()}")
    if not crm_torch.is_available():
        print(f"Error: {crm_torch.get_import_error()}")
        sys.exit(1)

    results = []

    # Run tests
    results.append(("Single Sample", test_forward_single_sample()))
    results.append(("Batch (n=5)", test_forward_batch()))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name:<20} {status}")

    all_passed = all(p for _, p in results)
    print("="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*70 + "\n")

    sys.exit(0 if all_passed else 1)
