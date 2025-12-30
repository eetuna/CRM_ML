#!/usr/bin/env python3
"""
Test script to diagnose RK4 angular acceleration behavior in the autodiff path.

This script runs linearization which uses the autodiff RK4 integrator.
"""

import numpy as np
import sys
import os

# Enable RK4 debug output
os.environ['CRM_DEBUG_RK4'] = '1'
os.environ['CRM_DYN_LINEARIZATION_METHOD'] = 'implicit'

sys.path.insert(0, '/workspaces/catheter/CRM_ML')

from crm_ml_rl.wrappers import crm_python

print("=" * 80)
print("RK4 ANGULAR ACCELERATION DIAGNOSTICS (AUTODIFF PATH)")
print("=" * 80)

# Initialize dynamics
dyn = crm_python.CRMDynamics()
ok = dyn.load_parameters(
    "data/catheter_params/CatheterParameterSet_1_dyn.txt",
    "data/catheter_params/CatheterSpatialConfiguration_1.txt",
)

if not ok:
    print("ERROR: Failed to load parameters")
    sys.exit(1)

print("\nSetting up dynamics...")
insertion = 94.3
dyn.dt = 0.05
dyn.integration_step_size = 0.001  # kCoilTStep
dyn.set_integrator("rk4")  # Use RK4 integrator

print(f"  Integrator: {dyn.get_integrator()}")
print(f"  dt: {dyn.dt}")
print(f"  integration_step_size: {dyn.integration_step_size}")

# Initialize with zero-velocity seed (worst case for RK4)
print("\nInitializing with ZERO-VELOCITY seed (v=[0,0,0], w=[0,0,0])...")
dyn.initialize_from_kinematics([0.0, 0.0, 0.01], insertion)
seed = dyn.get_seed_state()

v = np.asarray(seed["v"], dtype=np.float64)
w = np.asarray(seed["w"], dtype=np.float64)
p = np.asarray(seed["p"], dtype=np.float64)
R = np.asarray(seed["R"], dtype=np.float64)
xf = np.asarray(seed["xf"], dtype=np.float64)
mL = np.asarray(seed["mL"], dtype=np.float64)
nL = np.asarray(seed["nL"], dtype=np.float64)

print(f"\nSeed state:")
print(f"  v: {v[0]}")
print(f"  w: {w[0]}")
print(f"  |v|: {np.linalg.norm(v[0]):.6f} m/s")
print(f"  |w|: {np.linalg.norm(w[0]):.6f} rad/s")

# Test with small current
currents = np.array([0.1, 0.05, 0.02], dtype=np.float64)

print(f"\nTest currents: {currents}")
print(f"Insertion depth: {insertion} mm")

print("\n" + "=" * 80)
print("RUNNING LINEARIZATION WITH RK4 DEBUG OUTPUT:")
print("=" * 80)
print("(This will call the autodiff RK4 integrator which has diagnostics)")
print()

# Run linearization - this uses the autodiff path with RK4 diagnostics
try:
    out = dyn.linearize_full_seed_action_from_seed_implicit(
        currents,
        insertion,
        v, w, p, R, xf, mL, nL,
        eps_residual_x=1e-5,
        eps_residual_theta=1e-5,
        eps_g_x=1e-5,
        eps_g_theta=1e-5,
        return_debug=True,
    )

    print("\n" + "=" * 80)
    print("LINEARIZATION COMPLETED")
    print("=" * 80)

    print(f"\nResults:")
    print(f"  Output keys: {list(out.keys())}")

    # Check if Jxx is available
    if 'Jxx' in out:
        J = np.asarray(out["Jxx"], dtype=np.float64)
        print(f"  Jxx shape: {J.shape}")
        print(f"  Jxx finite: {np.isfinite(J).all()}")
        print(f"  Jxx sample:\n{J[:3, :3]}")
        print("\n✓ Linearization succeeded!")

except Exception as e:
    print(f"\n✗ ERROR during linearization: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("ANALYSIS:")
print("=" * 80)
print("\nIf you see debug output above showing:")
print("  - Initial |wdot| > 1000 rad/s² → Subdivision triggered")
print("  - Increasing subdivision levels → Step size reduced")
print("  - Eventually |wdot| < 1000 rad/s² at deeper levels → OK")
print("\nThis confirms the angular acceleration behavior is physical.")
print("\nIf no debug output appeared, the legacy integrator may have been used.")
print("=" * 80)
