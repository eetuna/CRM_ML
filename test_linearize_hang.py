#!/usr/bin/env python3
import numpy as np
import sys
sys.path.insert(0, '/workspaces/catheter/CRM_ML')
from crm_ml_rl.wrappers import crm_python

print("Test 1: Load and initialize")
dyn = crm_python.CRMDynamics()
ok = dyn.load_parameters(
    "data/catheter_params/CatheterParameterSet_1_dyn.txt",
    "data/catheter_params/CatheterSpatialConfiguration_1.txt",
)
print(f"  Load: {ok}")

dyn.dt = 0.05
dyn.integration_step_size = 0.001
dyn.initialize_from_kinematics([0.0, 0.0, 0.01], 94.3)
seed = dyn.get_seed_state()
print(f"  Initialize: OK")

print("\nTest 2: Forward pass (step_from_seed)")
v = np.asarray(seed["v"], dtype=np.float64)
w = np.asarray(seed["w"], dtype=np.float64)
p = np.asarray(seed["p"], dtype=np.float64)
R = np.asarray(seed["R"], dtype=np.float64)
xf = np.asarray(seed["xf"], dtype=np.float64)
mL = np.asarray(seed["mL"], dtype=np.float64)
nL = np.asarray(seed["nL"], dtype=np.float64)
currents = np.array([0.1, 0.05, 0.02], dtype=np.float64)

result = dyn.step_from_seed(currents, 94.3, v, w, p, R, xf, mL, nL)
print(f"  Converged: {result.get('converged', False)}")
print(f"  Tip pos: {np.asarray(result['tip_position'])}")

print("\n✅ Forward pass works!")

print("\nTest 3: Linearization (WITH DEBUG OUTPUT)")
print("This will show exactly where it hangs...")
try:
    result = dyn.linearize_full_seed_action_from_seed_implicit(
        currents, 94.3, v, w, p, R, xf, mL, nL)
    print("✅ LINEARIZATION COMPLETED!")
    print(f"  B shape: {result['B'].shape}")
    print(f"  A shape: {result['A'].shape}")
except Exception as e:
    print(f"❌ ERROR: {e}")
