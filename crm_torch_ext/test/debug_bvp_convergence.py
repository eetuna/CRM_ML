#!/usr/bin/env python3
"""
Debug BVP convergence issue by comparing Python bindings vs extension.
"""

import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import crm_torch_ext._crm_torch_ext as ext
from crm_ml_rl.wrappers import crm_python

os.chdir('/workspaces/catheter/CRM_ML')


def test_python_bindings_step():
    """Test that Python bindings can step successfully."""
    print("="*60)
    print("TEST: Python Bindings Step")
    print("="*60)

    dyn = crm_python.CRMDynamics()
    dyn.load_parameters(
        "data/catheter_params/CatheterParameterSet_1_dyn.txt",
        "data/catheter_params/CatheterSpatialConfiguration_1.txt",
    )

    # Set damping
    damping = np.array([
        12.1761626666366, 12.1761626666366, 284.429938756989,
        0.0304776127617393, 0.0304776127617393, 0.00502712804532508
    ])
    dyn.set_damping(damping)
    dyn.dt = 0.02
    dyn.integration_step_size = 0.1
    dyn.set_integrator("abm4")

    # Initialize from kinematics
    print("Initializing from kinematics...")
    success = dyn.initialize_from_kinematics(np.array([0.0, 0.0, 0.2]), 94.3)
    if not success:
        print("✗ FK initialization failed")
        return None
    print("✓ FK initialization succeeded")

    # Get initial seed
    seed0 = dyn.get_seed_state()
    print(f"\nInitial seed state:")
    print(f"  v: {seed0['v']}")
    print(f"  w: {seed0['w']}")
    print(f"  xf[0:3]: {seed0['xf'][0:3]}")
    print(f"  mL: {seed0['mL']}")
    print(f"  nL: {seed0['nL']}")

    # Try stepping with same currents
    print("\nAttempting step with currents=[0, 0, 0.2]...")
    try:
        result = dyn.step_from_seed(
            currents=np.array([0.0, 0.0, 0.2]),
            insertion_length=94.3,
            v=seed0['v'],
            w=seed0['w'],
            p=seed0['p'],
            R=seed0['R'],
            xf=seed0['xf'],
            mL=seed0['mL'],
            nL=seed0['nL']
        )
        print("✓ Python bindings step succeeded")
        print(f"  Output keys: {result.keys()}")
        print(f"  Output tip pos: {result['xf'][0:3]}")
        print(f"  Output v: {result['v']}")
        return seed0, result
    except Exception as e:
        print(f"✗ Python bindings step failed: {e}")
        return seed0, None


def test_extension_step(seed):
    """Test extension with same seed."""
    print("\n" + "="*60)
    print("TEST: Extension Step with Same Seed")
    print("="*60)

    # Initialize extension parameters
    ext.initialize_params(
        "data/catheter_params/CatheterParameterSet_1_dyn.txt",
        "data/catheter_params/CatheterSpatialConfiguration_1.txt"
    )
    ext.set_timestep(0.02)
    ext.set_integrator("abm4")
    ext.set_integration_step_size(0.1)  # CRITICAL: Match Python bindings!

    # Set damping (same as Python bindings)
    damping = [
        12.1761626666366, 12.1761626666366, 284.429938756989,
        0.0304776127617393, 0.0304776127617393, 0.00502712804532508
    ]
    ext.set_damping(damping)

    print("Extension parameters initialized")

    # Convert seed to torch
    currents = torch.tensor([0.0, 0.0, 0.2], dtype=torch.float64)
    insertion_length = torch.tensor([94.3], dtype=torch.float64)
    seed_v = torch.from_numpy(seed['v'])
    seed_w = torch.from_numpy(seed['w'])
    seed_p = torch.from_numpy(seed['p'])
    seed_R = torch.from_numpy(seed['R'])
    seed_xf = torch.from_numpy(seed['xf'])
    seed_mL = torch.from_numpy(seed['mL'])
    seed_nL = torch.from_numpy(seed['nL'])

    print(f"\nInput seed state:")
    print(f"  v: {seed_v}")
    print(f"  w: {seed_w}")
    print(f"  xf[0:3]: {seed_xf[0:3]}")
    print(f"  mL: {seed_mL}")
    print(f"  nL: {seed_nL}")

    print("\nAttempting step with currents=[0, 0, 0.2]...")
    try:
        output = ext.crm_step(
            currents, insertion_length,
            seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL
        )
        print("✓ Extension step succeeded")
        print(f"  Output: {output}")
        return output
    except Exception as e:
        print(f"✗ Extension step failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("\n" + "="*60)
    print("BVP CONVERGENCE DEBUG")
    print("="*60)

    # Test Python bindings
    result = test_python_bindings_step()
    if result is None:
        print("\n❌ Python bindings failed, cannot continue")
        return 1

    seed0, py_result = result

    if py_result is None:
        print("\n⚠️  Python bindings FK succeeded but step failed")
        print("This suggests the issue is with stepping, not FK initialization")

    # Test extension with same seed
    ext_result = test_extension_step(seed0)

    if ext_result is not None and py_result is not None:
        print("\n" + "="*60)
        print("COMPARISON")
        print("="*60)
        print(f"Python output tip: {py_result['xf'][0:3]}")
        print(f"Extension output:  {ext_result[0:3]}")
        print("\n🎉 Both succeeded!")
        return 0
    elif ext_result is not None:
        print("\n⚠️  Extension succeeded but Python failed")
        return 1
    else:
        print("\n❌ Extension failed (Python may have succeeded)")
        print("\nLikely causes:")
        print("1. Actuation inertia not set correctly")
        print("2. Damping values not propagated")
        print("3. Integration step size mismatch")
        print("4. Parameter loading issue")
        return 1


if __name__ == "__main__":
    sys.exit(main())
