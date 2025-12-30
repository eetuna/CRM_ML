#!/usr/bin/env python3
"""Test to find if the hang is in FD loop"""
import numpy as np
import sys
sys.path.insert(0, '/workspaces/catheter/CRM_ML')
from crm_ml_rl.wrappers import crm_python

dyn = crm_python.CRMDynamics()
ok = dyn.load_parameters(
    "data/catheter_params/CatheterParameterSet_1_dyn.txt",
    "data/catheter_params/CatheterSpatialConfiguration_1.txt",
)

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

print("Test 1: First forward pass")
result = dyn.step_from_seed(currents, insertion, v, w, p, R, xf, mL, nL)
print(f"  OK: {np.asarray(result['tip_position'])}")

print("\nTest 2: Multiple forward passes (FD loop simulation)")
eps = 1e-5
for i in range(3):
    print(f"  Forward pass {i+1}/6 (curr +eps)")
    curr_plus = currents.copy()
    curr_plus[i] += eps
    out_plus = dyn.step_from_seed(curr_plus, insertion, v, w, p, R, xf, mL, nL)
    print(f"    OK: converged={out_plus.get('converged')}")

    print(f"  Forward pass {i+2}/6 (curr -eps)")
    curr_minus = currents.copy()
    curr_minus[i] -= eps
    out_minus = dyn.step_from_seed(curr_minus, insertion, v, w, p, R, xf, mL, nL)
    print(f"    OK: converged={out_minus.get('converged')}")

print("\n✅ All 7 forward passes completed successfully!")
