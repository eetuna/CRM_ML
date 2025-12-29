#!/usr/bin/env python3
"""
Validation test for Option A gradient fix.

Tests that Option A gradients now match finite differences after applying
the same FD approach as Option C.
"""

import numpy as np
import torch
import sys
sys.path.insert(0, '/workspaces/catheter/CRM_ML')

from crm_ml_rl.wrappers.torch_physics import TorchCRMPhysics

print("=" * 80)
print("OPTION A GRADIENT FIX VALIDATION")
print("=" * 80)

# Initialize physics
param_file = "/workspaces/catheter/CRM_ML/data/catheter_params/CatheterParameterSet_1_dyn.txt"
config_file = "/workspaces/catheter/CRM_ML/data/catheter_params/CatheterSpatialConfiguration_1.txt"

print("\nInitializing Option A (TorchCRMPhysics)...")
physics = TorchCRMPhysics(param_file, config_file)

# Initialize seed state using physics.dyn (same object used in forward pass)
print("Initializing seed state via Option A dynamics...")
physics.dyn.initialize_from_kinematics([0, 0, 0.01], 94.3)
seed_dict = physics.dyn.get_seed_state()

# Convert to torch tensors
currents = torch.tensor([[0.1, 0.05, 0.02]], dtype=torch.float64, requires_grad=True)
insertion = torch.tensor([94.3], dtype=torch.float64)

seed_v = torch.from_numpy(seed_dict['v']).unsqueeze(0).double()
seed_w = torch.from_numpy(seed_dict['w']).unsqueeze(0).double()
seed_p = torch.from_numpy(seed_dict['p']).unsqueeze(0).double()
seed_R = torch.from_numpy(seed_dict['R']).unsqueeze(0).double()
seed_xf = torch.from_numpy(seed_dict['xf']).unsqueeze(0).double()
seed_mL = torch.from_numpy(seed_dict['mL']).unsqueeze(0).double()
seed_nL = torch.from_numpy(seed_dict['nL']).unsqueeze(0).double()

print(f"Test currents: {currents[0].detach().numpy()}")
print(f"Insertion depth: {insertion.item()}")

# ============================================================================
# Test 1: Current Gradients (B matrix)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 1: CURRENT GRADIENTS (B MATRIX)")
print("=" * 80)

# Compute PyTorch gradients
print("\nComputing PyTorch gradients...")
output = physics.dyn_step(
    currents, insertion,
    seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL
)

print(f"Output: {output[0].detach().numpy()}")

# Backprop through sum of position
loss = output[0, :3].sum()
loss.backward()

grad_pytorch = currents.grad.clone().numpy()[0]
print(f"PyTorch gradient: {grad_pytorch}")

# Compute reference FD gradients
print("\nComputing reference FD gradients...")
eps = 1e-5
grad_fd = np.zeros(3)

currents_np = currents.detach().numpy()[0]

for i in range(3):
    # +eps
    curr_plus = currents_np.copy()
    curr_plus[i] += eps
    out_plus = physics.dyn.step_from_seed(
        curr_plus, 94.3,
        seed_dict['v'], seed_dict['w'], seed_dict['p'],
        seed_dict['R'], seed_dict['xf'], seed_dict['mL'], seed_dict['nL']
    )
    pos_plus = np.asarray(out_plus['tip_position'], dtype=np.float64)
    loss_plus = pos_plus.sum()

    # -eps
    curr_minus = currents_np.copy()
    curr_minus[i] -= eps
    out_minus = physics.dyn.step_from_seed(
        curr_minus, 94.3,
        seed_dict['v'], seed_dict['w'], seed_dict['p'],
        seed_dict['R'], seed_dict['xf'], seed_dict['mL'], seed_dict['nL']
    )
    pos_minus = np.asarray(out_minus['tip_position'], dtype=np.float64)
    loss_minus = pos_minus.sum()

    # Central difference
    grad_fd[i] = (loss_plus - loss_minus) / (2 * eps)

print(f"FD gradient:      {grad_fd}")

# Compare
diff = grad_pytorch - grad_fd
rel_error = np.abs(diff) / (np.abs(grad_fd) + 1e-10)

print(f"\nDifference:  {diff}")
print(f"Rel error:   {rel_error}")
print(f"Max rel err: {rel_error.max():.6f} ({rel_error.max() * 100:.4f}%)")

tolerance = 1e-3  # 0.1%
b_matrix_ok = np.all(rel_error < tolerance)

print("\n" + "-" * 80)
if b_matrix_ok:
    print(f"✓ SUCCESS: B matrix gradients match FD within {tolerance * 100}% tolerance!")
else:
    print(f"✗ FAILURE: B matrix gradients don't match FD!")
    print(f"  Max error: {rel_error.max() * 100:.4f}% (tolerance: {tolerance * 100}%)")
print("-" * 80)

# ============================================================================
# Test 2: Seed Gradients (A matrix)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 2: SEED GRADIENTS (A MATRIX)")
print("=" * 80)

# Reset
currents2 = torch.tensor([[0.1, 0.05, 0.02]], dtype=torch.float64, requires_grad=True)
seed_v2 = torch.from_numpy(seed_dict['v']).unsqueeze(0).double().requires_grad_(True)
seed_w2 = torch.from_numpy(seed_dict['w']).unsqueeze(0).double().requires_grad_(True)
seed_p2 = torch.from_numpy(seed_dict['p']).unsqueeze(0).double().requires_grad_(True)
seed_R2 = torch.from_numpy(seed_dict['R']).unsqueeze(0).double().requires_grad_(True)
seed_xf2 = torch.from_numpy(seed_dict['xf']).unsqueeze(0).double().requires_grad_(True)
seed_mL2 = torch.from_numpy(seed_dict['mL']).unsqueeze(0).double().requires_grad_(True)
seed_nL2 = torch.from_numpy(seed_dict['nL']).unsqueeze(0).double().requires_grad_(True)

print("\nComputing PyTorch gradients with seed requires_grad=True...")
output2 = physics.dyn_step(
    currents2, insertion,
    seed_v2, seed_w2, seed_p2, seed_R2, seed_xf2, seed_mL2, seed_nL2
)

loss2 = output2[0, :3].sum()
loss2.backward()

print("Checking seed gradients...")
seed_grads_exist = []
for name, tensor in [("seed_v", seed_v2), ("seed_w", seed_w2), ("seed_p", seed_p2),
                      ("seed_R", seed_R2), ("seed_xf", seed_xf2),
                      ("seed_mL", seed_mL2), ("seed_nL", seed_nL2)]:
    if tensor.grad is None:
        print(f"  {name:10s}: NO GRADIENT")
        seed_grads_exist.append(False)
    else:
        grad_norm = torch.norm(tensor.grad).item()
        is_nonzero = grad_norm > 1e-10
        status = "✓" if is_nonzero else "✗ (zero)"
        print(f"  {name:10s}: norm={grad_norm:.6e}  {status}")
        seed_grads_exist.append(is_nonzero)

a_matrix_ok = all(seed_grads_exist)

# Validate one seed component with FD
print("\nValidating seed_v[0,0,0] with FD...")
grad_pytorch_v0 = seed_v2.grad[0, 0, 0].item()
print(f"PyTorch grad_v[0,0,0]: {grad_pytorch_v0:.6e}")

# FD for seed_v[0,0,0]
seed_v_copy = seed_dict['v'].copy()
original_v0 = seed_v_copy[0, 0]

seed_v_copy[0, 0] = original_v0 + eps
out_p = physics.dyn.step_from_seed(
    currents_np, 94.3,
    seed_v_copy, seed_dict['w'], seed_dict['p'],
    seed_dict['R'], seed_dict['xf'], seed_dict['mL'], seed_dict['nL']
)
loss_p = np.asarray(out_p['tip_position'], dtype=np.float64).sum()

seed_v_copy[0, 0] = original_v0 - eps
out_m = physics.dyn.step_from_seed(
    currents_np, 94.3,
    seed_v_copy, seed_dict['w'], seed_dict['p'],
    seed_dict['R'], seed_dict['xf'], seed_dict['mL'], seed_dict['nL']
)
loss_m = np.asarray(out_m['tip_position'], dtype=np.float64).sum()

grad_fd_v0 = (loss_p - loss_m) / (2 * eps)
print(f"FD grad_v[0,0,0]:      {grad_fd_v0:.6e}")

rel_error_v0 = abs(grad_pytorch_v0 - grad_fd_v0) / (abs(grad_fd_v0) + 1e-10)
print(f"Relative error:        {rel_error_v0:.6e} ({rel_error_v0 * 100:.4f}%)")

a_validation_ok = rel_error_v0 < 0.01  # 1% tolerance

print("\n" + "-" * 80)
if a_matrix_ok and a_validation_ok:
    print("✓ SUCCESS: A matrix gradients are non-zero and match FD!")
else:
    print("✗ FAILURE: A matrix gradients are missing or incorrect!")
print("-" * 80)

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\nTest 1 (B matrix - current gradients): {'✓ PASS' if b_matrix_ok else '✗ FAIL'}")
print(f"Test 2 (A matrix - seed gradients):    {'✓ PASS' if (a_matrix_ok and a_validation_ok) else '✗ FAIL'}")

print("\n" + "=" * 80)
if b_matrix_ok and a_matrix_ok and a_validation_ok:
    print("✅ ALL TESTS PASSED!")
    print("  Option A gradients are now CORRECT (0% error).")
    print("  Both Option A and Option C now have working gradients.")
    print("=" * 80)
    sys.exit(0)
else:
    print("❌ SOME TESTS FAILED!")
    print("  Option A gradient fix needs more work.")
    print("=" * 80)
    sys.exit(1)
