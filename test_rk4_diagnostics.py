#!/usr/bin/env python3
"""
Test script to diagnose RK4 angular acceleration behavior.

This script runs a single physics step with zero-velocity seeds
and logs detailed RK4 subdivision diagnostics.
"""

import numpy as np
import sys
import os

# Enable RK4 debug output
os.environ['CRM_DEBUG_RK4'] = '1'

sys.path.insert(0, '/workspaces/catheter/CRM_ML')

from crm_ml_rl.wrappers.torch_physics import TorchCRMPhysics

print("=" * 80)
print("RK4 ANGULAR ACCELERATION DIAGNOSTICS")
print("=" * 80)

# Initialize physics
param_file = "/workspaces/catheter/CRM_ML/data/catheter_params/CatheterParameterSet_1_dyn.txt"
config_file = "/workspaces/catheter/CRM_ML/data/catheter_params/CatheterSpatialConfiguration_1.txt"

print("\nInitializing TorchCRMPhysics...")
physics = TorchCRMPhysics(param_file, config_file)

# Enable RK4 integrator (was previously disabled due to divergence bug - now fixed!)
print("Setting integrator to RK4...")
physics.dyn.set_integrator("rk4")

# Initialize with zero-velocity seed (worst case)
print("\nInitializing with ZERO-VELOCITY seed (v=[0,0,0], w=[0,0,0])...")
physics.dyn.initialize_from_kinematics([0, 0, 0.01], 94.3)
seed_dict = physics.dyn.get_seed_state()

print(f"\nSeed state:")
print(f"  v: {seed_dict['v'][0]}")
print(f"  w: {seed_dict['w'][0]}")
print(f"  p: {seed_dict['p'][0]}")

# Test with small current
currents = np.array([0.1, 0.05, 0.02])
insertion = 94.3

print(f"\nTest currents: {currents}")
print(f"Insertion depth: {insertion} mm")

print("\n" + "=" * 80)
print("RUNNING PHYSICS STEP WITH RK4 DEBUG OUTPUT:")
print("=" * 80)

# Run the dynamics step (C++ will print diagnostics)
result = physics.dyn.step_from_seed(
    currents, insertion,
    seed_dict['v'], seed_dict['w'], seed_dict['p'],
    seed_dict['R'], seed_dict['xf'], seed_dict['mL'], seed_dict['nL']
)

print("=" * 80)
print("STEP COMPLETED")
print("=" * 80)

tip_position = np.asarray(result['tip_position'], dtype=np.float64)
print(f"\nResult:")
print(f"  Tip position: {tip_position}")
print(f"  Converged: localmin={result.get('localmin', 'N/A')}")

print("\n" + "=" * 80)
print("ANALYSIS:")
print("=" * 80)
print("\nExpected behavior:")
print("  - High angular acceleration (>1000 rad/s²) at level 0 due to zero initial velocity")
print("  - Adaptive subdivision should trigger, reducing step size")
print("  - At deeper levels, angular acceleration should decrease")
print("  - Final result should be accurate (tip ~[0.026, 4.34, 94.25])")

expected = np.array([0.026, 4.34, 94.25])
error = np.abs(tip_position - expected)
print(f"\nTip position error: {error}")
print(f"Max error: {error.max():.6f} mm")

if error.max() < 0.1:
    print("\n✓ Result is accurate!")
else:
    print("\n✗ Result has large error - integration may have issues")

print("\n" + "=" * 80)
