#!/usr/bin/env python3
"""
Finite Difference Validation of Seed Gradients (A Matrix).

This test quantitatively validates that the A matrix gradients computed via FD
in the C++ backward pass match an independent FD computation in Python.
"""

import numpy as np
import torch
import sys
sys.path.insert(0, '/workspaces/catheter/CRM_ML')

import crm_torch
from crm_ml_rl.wrappers import crm_python


def compute_reference_seed_gradients():
    """Compute reference seed gradients via independent Python FD."""
    print("=" * 80)
    print("Computing Reference Seed Gradients via Python FD")
    print("=" * 80)

    # Setup
    param_file = "/workspaces/catheter/CRM_ML/data/catheter_params/CatheterParameterSet_1_dyn.txt"
    config_file = "/workspaces/catheter/CRM_ML/data/catheter_params/CatheterSpatialConfiguration_1.txt"
    insertion_length = 94.3

    # Initialize dynamics
    dyn = crm_python.CRMDynamics()
    if not dyn.load_parameters(param_file, config_file):
        raise RuntimeError("Failed to load parameters")

    initial_currents = np.array([0.0, 0.0, 0.01])
    if not dyn.initialize_from_kinematics(initial_currents, insertion_length):
        raise RuntimeError("Failed to initialize")

    seed = dyn.get_seed_state()

    # Test currents
    currents_np = np.array([0.1, 0.05, 0.02])

    print(f"\nTest parameters:")
    print(f"  Currents: {currents_np}")
    print(f"  Insertion: {insertion_length}")

    # Compute A matrix via FD in Python
    eps = 1e-5
    num_outputs = 6  # tip_position (3) + tip_velocity (3)

    # Count seed components
    num_seed_components = (
        seed['v'].size + seed['w'].size + seed['p'].size +
        seed['R'].size + seed['xf'].size +
        seed['mL'].size + seed['nL'].size
    )

    print(f"\nSeed state components:")
    print(f"  v: {seed['v'].shape} ({seed['v'].size} elements)")
    print(f"  w: {seed['w'].shape} ({seed['w'].size} elements)")
    print(f"  p: {seed['p'].shape} ({seed['p'].size} elements)")
    print(f"  R: {seed['R'].shape} ({seed['R'].size} elements)")
    print(f"  xf: {seed['xf'].shape} ({seed['xf'].size} elements)")
    print(f"  mL: {seed['mL'].shape} ({seed['mL'].size} elements)")
    print(f"  nL: {seed['nL'].shape} ({seed['nL'].size} elements)")
    print(f"  Total: {num_seed_components} components")

    A_fd = np.zeros((num_outputs, num_seed_components))

    component_idx = 0

    # Helper to compute FD for a seed component
    def compute_fd_column(seed_array, flat_idx):
        nonlocal component_idx

        # Save original value
        original = seed_array.flat[flat_idx]

        # Perturb +eps
        seed_array.flat[flat_idx] = original + eps
        result_plus = dyn.step_from_seed(
            currents_np, insertion_length,
            seed['v'], seed['w'], seed['p'], seed['R'], seed['xf'],
            seed['mL'], seed['nL']
        )
        y_plus = np.concatenate([result_plus['tip_position'], result_plus['tip_velocity']])

        # Perturb -eps
        seed_array.flat[flat_idx] = original - eps
        result_minus = dyn.step_from_seed(
            currents_np, insertion_length,
            seed['v'], seed['w'], seed['p'], seed['R'], seed['xf'],
            seed['mL'], seed['nL']
        )
        y_minus = np.concatenate([result_minus['tip_position'], result_minus['tip_velocity']])

        # Restore original
        seed_array.flat[flat_idx] = original

        # Central difference
        A_fd[:, component_idx] = (y_plus - y_minus) / (2 * eps)
        component_idx += 1

    print("\nComputing A matrix via FD...")

    # v components
    for i in range(seed['v'].size):
        compute_fd_column(seed['v'], i)

    # w components
    for i in range(seed['w'].size):
        compute_fd_column(seed['w'], i)

    # p components
    for i in range(seed['p'].size):
        compute_fd_column(seed['p'], i)

    # R components
    for i in range(seed['R'].size):
        compute_fd_column(seed['R'], i)

    # xf components
    for i in range(seed['xf'].size):
        compute_fd_column(seed['xf'], i)

    # mL components
    for i in range(seed['mL'].size):
        compute_fd_column(seed['mL'], i)

    # nL components
    for i in range(seed['nL'].size):
        compute_fd_column(seed['nL'], i)

    print(f"✓ A matrix computed: shape {A_fd.shape}")
    print(f"  Frobenius norm: {np.linalg.norm(A_fd):.6e}")

    return A_fd, seed, currents_np, insertion_length, param_file, config_file


def test_pytorch_seed_gradients(A_fd_ref, seed, currents_np, insertion_length, param_file, config_file):
    """Test that PyTorch seed gradients match the reference FD."""
    print("\n" + "=" * 80)
    print("Computing PyTorch Seed Gradients (via C++ Backward Pass)")
    print("=" * 80)

    # Create torch tensors
    currents = torch.from_numpy(currents_np).unsqueeze(0).double().requires_grad_(True)
    insertion = torch.tensor([insertion_length], dtype=torch.float64)

    # Seed tensors with requires_grad
    seed_v = torch.from_numpy(seed['v']).unsqueeze(0).double().requires_grad_(True)
    seed_w = torch.from_numpy(seed['w']).unsqueeze(0).double().requires_grad_(True)
    seed_p = torch.from_numpy(seed['p']).unsqueeze(0).double().requires_grad_(True)
    seed_R = torch.from_numpy(seed['R']).unsqueeze(0).double().requires_grad_(True)
    seed_xf = torch.from_numpy(seed['xf']).unsqueeze(0).double().requires_grad_(True)
    seed_mL = torch.from_numpy(seed['mL']).unsqueeze(0).double().requires_grad_(True)
    seed_nL = torch.from_numpy(seed['nL']).unsqueeze(0).double().requires_grad_(True)

    # Forward pass
    print("\nRunning forward pass...")
    output = crm_torch.CRMDynamicsStep.apply(
        currents, insertion,
        seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL,
        param_file, config_file, 1e-4
    )

    # Extract A matrix by backpropagating unit vectors through each output
    print("Extracting A matrix via backpropagation...")

    num_outputs = 6
    num_seed_components = A_fd_ref.shape[1]
    A_pytorch = np.zeros((num_outputs, num_seed_components))

    seed_tensors = [seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL]
    seed_sizes = [s.numel() for s in seed_tensors]

    for i in range(num_outputs):
        # Zero out previous gradients
        for tensor in seed_tensors:
            if tensor.grad is not None:
                tensor.grad.zero_()

        # Backprop unit vector for output i
        grad_output = torch.zeros(num_outputs, dtype=torch.float64)
        grad_output[i] = 1.0
        output[0].backward(grad_output, retain_graph=True)

        # Extract gradients and concatenate
        component_idx = 0
        for tensor in seed_tensors:
            if tensor.grad is not None:
                grad_flat = tensor.grad[0].flatten().numpy()
                A_pytorch[i, component_idx:component_idx + len(grad_flat)] = grad_flat
                component_idx += len(grad_flat)

    print(f"✓ A matrix extracted: shape {A_pytorch.shape}")
    print(f"  Frobenius norm: {np.linalg.norm(A_pytorch):.6e}")

    # Compare with reference
    print("\n" + "=" * 80)
    print("Comparison: PyTorch vs Reference FD")
    print("=" * 80)

    diff = A_pytorch - A_fd_ref
    frob_error = np.linalg.norm(diff) / (np.linalg.norm(A_fd_ref) + 1e-10)
    max_abs_error = np.max(np.abs(diff))
    max_rel_error = np.max(np.abs(diff) / (np.abs(A_fd_ref) + 1e-10))

    print(f"\nError metrics:")
    print(f"  Frobenius error: {frob_error:.6e} ({frob_error * 100:.4f}%)")
    print(f"  Max absolute error: {max_abs_error:.6e}")
    print(f"  Max relative error: {max_rel_error:.6e} ({max_rel_error * 100:.4f}%)")

    # Show some sample comparisons
    print(f"\nSample comparisons (first 5 seed components, output 0):")
    print(f"  {'Component':<12} {'Reference':>15} {'PyTorch':>15} {'Diff':>15}")
    print(f"  {'-'*12} {'-'*15} {'-'*15} {'-'*15}")
    for j in range(min(5, num_seed_components)):
        ref_val = A_fd_ref[0, j]
        pt_val = A_pytorch[0, j]
        diff_val = pt_val - ref_val
        print(f"  Seed[{j:2d}]      {ref_val:15.6e} {pt_val:15.6e} {diff_val:15.6e}")

    # Success criteria
    tolerance = 1e-3  # 0.1%
    passed = frob_error < tolerance

    print("\n" + "=" * 80)
    if passed:
        print(f"✓ SUCCESS: Seed gradients match FD within {tolerance * 100}% tolerance!")
        print("  A matrix (seed gradients) implementation is CORRECT.")
        print("  Phase A.3 is COMPLETE.")
    else:
        print(f"✗ FAILURE: Seed gradients don't match FD!")
        print(f"  Error: {frob_error * 100:.4f}% (tolerance: {tolerance * 100}%)")
        print("  A matrix implementation needs debugging.")
    print("=" * 80)

    return passed


if __name__ == "__main__":
    print("\n" + "#" * 80)
    print("# Seed Gradient (A Matrix) Finite Difference Validation")
    print("# Phase A.3: Quantitative Verification")
    print("#" * 80)

    # Compute reference
    A_fd, seed, currents, insertion, param_file, config_file = compute_reference_seed_gradients()

    # Test PyTorch implementation
    passed = test_pytorch_seed_gradients(A_fd, seed, currents, insertion, param_file, config_file)

    # Exit code
    sys.exit(0 if passed else 1)
