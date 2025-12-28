"""
Smoke test for backward pass implementation.

Quick verification that backward pass runs without errors.
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import crm_torch
from crm_ml_rl.wrappers import crm_python


def test_backward_smoke():
    """Smoke test: backward pass runs without errors."""
    print("\n" + "="*70)
    print("SMOKE TEST: Backward Pass")
    print("="*70)

    if not crm_torch.is_available():
        print(f"❌ Extension not available: {crm_torch.get_import_error()}")
        return False

    param_file = "data/catheter_params/CatheterParameterSet_1_dyn.txt"
    config_file = "data/catheter_params/CatheterSpatialConfiguration_1.txt"

    # Get seed state
    dyn_py = crm_python.CRMDynamics()
    dyn_py.load_parameters(param_file, config_file)
    dyn_py.initialize_from_kinematics(np.array([0.0, 0.0, 0.01]), 94.3)
    seed = dyn_py.get_seed_state()

    # Create input with requires_grad=True
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

    print("\n[Forward Pass]")
    print(f"  Input currents shape: {currents.shape}, requires_grad={currents.requires_grad}")

    # Forward pass
    output = crm_torch.CRMDynamicsStep.apply(
        currents, insertion,
        seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL,
        param_file, config_file, 1e-4  # eps_seed
    )

    print(f"  Output shape: {output.shape}")
    print(f"  Output: {output[0].detach().numpy()}")

    print("\n[Backward Pass]")
    # Create a simple loss: sum of all outputs
    loss = output.sum()
    print(f"  Loss (sum of outputs): {loss.item():.4f}")

    # Backward pass - this should call our implemented backward()
    try:
        loss.backward()
        print(f"  ✅ Backward pass succeeded")
    except Exception as e:
        print(f"  ❌ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n[Gradient Check]")
    if currents.grad is None:
        print(f"  ❌ currents.grad is None!")
        return False

    print(f"  currents.grad shape: {currents.grad.shape}")
    print(f"  currents.grad: {currents.grad.numpy()}")

    # Check for NaN/Inf
    if torch.isnan(currents.grad).any():
        print(f"  ❌ Gradient contains NaN!")
        return False

    if torch.isinf(currents.grad).any():
        print(f"  ❌ Gradient contains Inf!")
        return False

    # Check gradient is non-zero (should have some sensitivity)
    grad_norm = currents.grad.norm().item()
    print(f"  Gradient norm: {grad_norm:.6f}")

    if grad_norm < 1e-10:
        print(f"  ⚠️ Gradient is essentially zero - may indicate issue")
    else:
        print(f"  ✅ Gradient is non-zero")

    print(f"\n✅ SMOKE TEST PASSED")
    return True


if __name__ == "__main__":
    success = test_backward_smoke()
    sys.exit(0 if success else 1)
