#!/usr/bin/env python3
"""
Test backward pass infrastructure (returns zero gradients for now).
"""

import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import crm_torch_ext._crm_torch_ext as ext
from crm_ml_rl.wrappers import crm_python

os.chdir('/workspaces/catheter/CRM_ML')


def test_backward_infrastructure():
    """Test that backward pass infrastructure works (even with zero gradients)."""
    print("="*60)
    print("BACKWARD INFRASTRUCTURE TEST")
    print("="*60)

    # Initialize extension
    ext.initialize_params(
        "data/catheter_params/CatheterParameterSet_1_dyn.txt",
        "data/catheter_params/CatheterSpatialConfiguration_1.txt"
    )
    ext.set_timestep(0.02)
    ext.set_integrator("abm4")
    ext.set_integration_step_size(0.1)
    damping = [
        12.1761626666366, 12.1761626666366, 284.429938756989,
        0.0304776127617393, 0.0304776127617393, 0.00502712804532508
    ]
    ext.set_damping(damping)

    # Get initial seed from FK
    dyn = crm_python.CRMDynamics()
    dyn.load_parameters(
        "data/catheter_params/CatheterParameterSet_1_dyn.txt",
        "data/catheter_params/CatheterSpatialConfiguration_1.txt"
    )
    dyn.set_damping(np.array(damping))
    dyn.dt = 0.02
    dyn.integration_step_size = 0.1
    dyn.set_integrator("abm4")

    success = dyn.initialize_from_kinematics(np.array([0.0, 0.0, 0.0]), 94.3)
    if not success:
        print("✗ FK failed")
        return False

    seed = dyn.get_seed_state()

    # Convert to torch with requires_grad
    currents = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
    insertion_length = torch.tensor([94.3], dtype=torch.float64)
    seed_v = torch.from_numpy(seed['v'])
    seed_w = torch.from_numpy(seed['w'])
    seed_p = torch.from_numpy(seed['p'])
    seed_R = torch.from_numpy(seed['R'])
    seed_xf = torch.from_numpy(seed['xf'])
    seed_mL = torch.from_numpy(seed['mL'])
    seed_nL = torch.from_numpy(seed['nL'])

    print("\nTest 1: Forward pass")
    try:
        output = ext.crm_step(
            currents, insertion_length,
            seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL
        )
        print(f"✓ Forward pass succeeded")
        print(f"  Output shape: {output.shape}")
        print(f"  Output[0:3]: {output[0:3]}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False

    print("\nTest 2: Backward pass (infrastructure check)")
    try:
        # Create a simple loss: sum of first 3 outputs
        loss = output[0:3].sum()
        loss.backward()
        
        print(f"✓ Backward pass succeeded")
        print(f"  Currents gradient: {currents.grad}")
        print(f"  Gradient norm: {currents.grad.norm().item()}")
        
        # Check that gradient is zero (expected for stub implementation)
        if currents.grad.norm().item() < 1e-10:
            print(f"✓ Gradient is zero (as expected for stub implementation)")
            print(f"\n  NOTE: Gradients are not yet implemented.")
            print(f"  The backward pass returns zero gradients.")
            print(f"  Use zero-order optimization methods (CMA-ES, genetic algorithms)")
            print(f"  or implement implicit differentiation for gradient-based training.")
        else:
            print(f"⚠ Warning: Non-zero gradient detected (unexpected)")
            
    except Exception as e:
        print(f"✗ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✓ Backward infrastructure test completed!")
    print("  Forward pass: Working ✓")
    print("  Backward pass: Infrastructure OK (gradients=0)")
    return True


if __name__ == "__main__":
    success = test_backward_infrastructure()
    sys.exit(0 if success else 1)
