#!/usr/bin/env python3
"""
Test to quantify the AD forward/backward mismatch.

This script compares:
1. Forward pass: explicit integration step_from_seed()
2. Backward pass: implicit AD differentiation of residual F(x)=0

Expected result: Large gradient mismatch (50-200%)
"""

import numpy as np
import sys
sys.path.insert(0, '/workspaces/catheter/CRM_ML')

from crm_ml_rl.wrappers import crm_python

print("=" * 80)
print("AD FORWARD/BACKWARD MISMATCH DIAGNOSTIC")
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

insertion = 94.3
dyn.dt = 0.05
dyn.integration_step_size = 0.001

# Initialize from kinematics to get a good seed
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

print("\n" + "=" * 80)
print("METHOD 1: FORWARD PASS (step_from_seed)")
print("=" * 80)

# Forward pass: explicit integration
print("\nRunning explicit integration...")
result_forward = dyn.step_from_seed(currents, insertion, v, w, p, R, xf, mL, nL)
tip_pos_forward = np.asarray(result_forward["tip_position"], dtype=np.float64)
tip_vel_forward = np.asarray(result_forward["tip_velocity"], dtype=np.float64)

print(f"Tip position: {tip_pos_forward}")
print(f"Tip velocity: {tip_vel_forward}")
print(f"Converged: {result_forward.get('converged', 'N/A')}")

print("\n" + "=" * 80)
print("METHOD 2: BACKWARD PASS (linearize_implicit with AD)")
print("=" * 80)

# Backward pass: compute gradients via implicit differentiation
print("\nRunning implicit linearization with AD...")
try:
    result_ad = dyn.linearize_full_seed_action_from_seed_implicit(
        currents, insertion, v, w, p, R, xf, mL, nL,
        eps_residual_x=1e-5,
        eps_residual_theta=1e-5,
        eps_g_x=1e-5,
        eps_g_theta=1e-5,
        return_debug=True
    )

    # Extract next state from linearization
    next_state = np.asarray(result_ad["next_state"], dtype=np.float64)
    tip_pos_ad = next_state[:3]
    tip_vel_ad = next_state[3:6] if len(next_state) >= 6 else np.zeros(3)

    print(f"Tip position: {tip_pos_ad}")
    print(f"Tip velocity: {tip_vel_ad}")

    # Get gradients
    B = np.asarray(result_ad["B"], dtype=np.float64)
    A = np.asarray(result_ad["A"], dtype=np.float64)

    print(f"\nB matrix shape: {B.shape}")
    print(f"A matrix shape: {A.shape}")
    print(f"B matrix (currents): \n{B[:3, :]}")  # First 3 rows for position

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("METHOD 3: REFERENCE (Finite Differences)")
print("=" * 80)

# Compute reference gradients via finite differences
print("\nComputing finite difference gradients...")
eps = 1e-5

# B matrix (∂output/∂currents)
B_fd = np.zeros((6, 3))
for i in range(3):
    curr_plus = currents.copy()
    curr_plus[i] += eps
    out_plus = dyn.step_from_seed(curr_plus, insertion, v, w, p, R, xf, mL, nL)
    pos_plus = np.asarray(out_plus["tip_position"], dtype=np.float64)
    vel_plus = np.asarray(out_plus["tip_velocity"], dtype=np.float64)
    state_plus = np.concatenate([pos_plus, vel_plus])

    curr_minus = currents.copy()
    curr_minus[i] -= eps
    out_minus = dyn.step_from_seed(curr_minus, insertion, v, w, p, R, xf, mL, nL)
    pos_minus = np.asarray(out_minus["tip_position"], dtype=np.float64)
    vel_minus = np.asarray(out_minus["tip_velocity"], dtype=np.float64)
    state_minus = np.concatenate([pos_minus, vel_minus])

    B_fd[:, i] = (state_plus - state_minus) / (2 * eps)

print(f"B_fd matrix (currents): \n{B_fd[:3, :]}")  # First 3 rows for position

print("\n" + "=" * 80)
print("COMPARISON: AD vs FD")
print("=" * 80)

# Compare B matrices (position only for clarity)
B_ad_pos = B[:3, :]
B_fd_pos = B_fd[:3, :]

diff = B_ad_pos - B_fd_pos
rel_error = np.abs(diff) / (np.abs(B_fd_pos) + 1e-10)

print(f"\nB matrix (AD):\n{B_ad_pos}")
print(f"\nB matrix (FD):\n{B_fd_pos}")
print(f"\nAbsolute difference:\n{diff}")
print(f"\nRelative error:\n{rel_error}")
print(f"\nMax relative error: {rel_error.max():.2%}")

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

if rel_error.max() > 0.5:
    print(f"\n❌ LARGE MISMATCH DETECTED: {rel_error.max():.1%} error")
    print("\nThis confirms the forward/backward mismatch:")
    print("  - Forward: Explicit integration (IVP)")
    print("  - Backward: Implicit differentiation of BVP residual")
    print("  - These compute DIFFERENT mathematical operations")
elif rel_error.max() > 0.01:
    print(f"\n⚠️  MODERATE MISMATCH: {rel_error.max():.1%} error")
    print("  AD gradients are in the right ballpark but not exact")
elif rel_error.max() < 0.01:
    print(f"\n✓ GOOD MATCH: {rel_error.max():.2%} error")
    print("  AD gradients match FD - the mismatch may have been fixed!")

print("\n" + "=" * 80)
