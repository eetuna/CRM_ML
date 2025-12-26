#!/usr/bin/env python3
"""
Simple BVP test - just try to solve a static equilibrium with currents.
"""

import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import crm_torch_ext._crm_torch_ext as ext

os.chdir('/workspaces/catheter/CRM_ML')


def test_static_bvp():
    """Test static BVP (zero velocity) with currents."""
    print("="*60)
    print("TEST: Static BVP with Zero Velocity")
    print("="*60)

    # Initialize extension parameters
    ext.initialize_params(
        "data/catheter_params/CatheterParameterSet_1_dyn.txt",
        "data/catheter_params/CatheterSpatialConfiguration_1.txt"
    )
    ext.set_timestep(0.02)
    ext.set_integrator("abm4")
    ext.set_integration_step_size(0.1)

    # Set damping
    damping = [
        12.1761626666366, 12.1761626666366, 284.429938756989,
        0.0304776127617393, 0.0304776127617393, 0.00502712804532508
    ]
    ext.set_damping(damping)

    # Get initial seed from FK (we must use FK to get proper xf state!)
    from crm_ml_rl.wrappers import crm_python

    print("\nInitializing state from FK...")
    dyn = crm_python.CRMDynamics()
    dyn.load_parameters(
        "data/catheter_params/CatheterParameterSet_1_dyn.txt",
        "data/catheter_params/CatheterSpatialConfiguration_1.txt"
    )
    dyn.set_damping(damping)
    dyn.dt = 0.02
    dyn.integration_step_size = 0.1
    dyn.set_integrator("abm4")

    # Initialize from FK with zero currents
    success = dyn.initialize_from_kinematics(np.array([0.0, 0.0, 0.0]), 94.3)
    if not success:
        print("✗ FK initialization failed")
        return False

    seed = dyn.get_seed_state()
    print(f"✓ FK succeeded, xf[0:3] (base curvature): {seed['xf'][0:3]}")

    # Convert to torch tensors
    currents = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    insertion_length = torch.tensor([94.3], dtype=torch.float64)

    seed_v = torch.from_numpy(seed['v'])
    seed_w = torch.from_numpy(seed['w'])
    seed_p = torch.from_numpy(seed['p'])
    seed_R = torch.from_numpy(seed['R'])
    seed_xf = torch.from_numpy(seed['xf'])
    seed_mL = torch.from_numpy(seed['mL'])
    seed_nL = torch.from_numpy(seed['nL'])

    print("\nTest 1: Zero currents (should converge easily)")
    try:
        output = ext.crm_step(
            currents, insertion_length,
            seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL
        )
        print(f"✓ Zero currents succeeded")
        print(f"  Output tip: {output[0:3]}")
    except Exception as e:
        print(f"✗ Zero currents failed: {e}")
        return False

    print("\nTest 2: Small current [0, 0, 0.1]")
    currents = torch.tensor([0.0, 0.0, 0.1], dtype=torch.float64)
    try:
        output = ext.crm_step(
            currents, insertion_length,
            seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL
        )
        print(f"✓ Small current succeeded")
        print(f"  Output tip: {output[0:3]}")
    except Exception as e:
        print(f"✗ Small current failed: {e}")
        return False

    print("\n✓ All tests passed!")
    return True


if __name__ == "__main__":
    os.environ['CRM_DEBUG_BVP'] = '1'
    success = test_static_bvp()
    sys.exit(0 if success else 1)
