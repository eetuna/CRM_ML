#!/usr/bin/env python3
"""Quick gradient test - just check if it runs without hanging."""

import numpy as np
import sys
sys.path.insert(0, '/workspaces/catheter/CRM_ML')

from crm_ml_rl.wrappers import crm_python

print("Initializing...")
dyn = crm_python.CRMDynamics()
ok = dyn.load_parameters(
    "data/catheter_params/CatheterParameterSet_1_dyn.txt",
    "data/catheter_params/CatheterSpatialConfiguration_1.txt",
)

if not ok:
    print("ERROR: Failed to load parameters")
    sys.exit(1)

insertion = 94.3
dyn.dt = 0.05
dyn.integration_step_size = 0.001

dyn.initialize_from_kinematics([0.0, 0.0, 0.01], insertion)
seed = dyn.get_seed_state()

v = np.asarray(seed["v"], dtype=np.float64)
w = np.asarray(seed["w"], dtype=np.float64)
p = np.asarray(seed["p"], dtype=np.float64)
R = np.asarray(seed["R"], dtype=np.float64)
xf = np.asarray(seed["xf"], dtype=np.float64)
mL = np.asarray(seed["mL"], dtype=np.float64)
nL = np.asarray(seed["nL"], dtype=np.float64)

currents = np.array([0.1, 0.05, 0.02], dtype=np.float64)

print("Running forward pass...")
result = dyn.step_from_seed(currents, insertion, v, w, p, R, xf, mL, nL)
print(f"Forward pass OK: converged={result.get('converged', False)}")

print("Running linearization...")
try:
    result_ad = dyn.linearize_full_seed_action_from_seed_implicit(
        currents, insertion, v, w, p, R, xf, mL, nL,
        eps_residual_x=1e-5,
        eps_residual_theta=1e-5,
        eps_g_x=1e-5,
        eps_g_theta=1e-5,
        return_debug=True
    )
    print(f"Linearization OK!")
    B = np.asarray(result_ad["B"], dtype=np.float64)
    print(f"B shape: {B.shape}, norm: {np.linalg.norm(B):.2e}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ Test completed successfully")
