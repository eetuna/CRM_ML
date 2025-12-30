#!/usr/bin/env python3
"""
Verify gx = ∂y/∂x via finite differences.

gx should represent: how does the output y = [tip_pos, tip_vel] change
when we perturb the BVP solution x = [mL*, nL*]?

Challenge: The current API doesn't allow us to run forward pass with
arbitrary (mL, nL) - it always re-solves the BVP.

Workaround: We'll need to call the lower-level DYNSolverIVP function directly.
"""

import numpy as np
import sys
sys.path.insert(0, '/workspaces/catheter/CRM_ML')

from crm_ml_rl.wrappers import crm_python

def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

print_section("GX VERIFICATION: ∂y/∂x via Finite Differences")

# Initialize dynamics
print("\nInitializing dynamics...")
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

# Initialize from kinematics
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

print_section("STEP 1: Get BVP Solution (mL*, nL*)")

# Run forward pass to get the BVP solution
result = dyn.step_from_seed(currents, insertion, v, w, p, R, xf, mL, nL)
tip_pos_base = np.asarray(result["tip_position"], dtype=np.float64)
tip_vel_base = np.asarray(result["tip_velocity"], dtype=np.float64)
mL_star = np.asarray(result["next_mL"], dtype=np.float64)
nL_star = np.asarray(result["next_nL"], dtype=np.float64)

print(f"Converged: {result.get('converged', False)}")
print(f"mL* shape: {mL_star.shape}, values: {mL_star}")
print(f"nL* shape: {nL_star.shape}, values: {nL_star}")
print(f"Tip position (base): {tip_pos_base}")
print(f"Tip velocity (base): {tip_vel_base}")

# Flatten x* = [mL*, nL*]
x_star = np.concatenate([mL_star.flatten(), nL_star.flatten()])
print(f"\nx* = [mL*, nL*] (flattened): {x_star}")
print(f"x* shape: {x_star.shape}")

print_section("STEP 2: Compute gx via Finite Differences")

print("""
CHALLENGE: We need to evaluate y = f(x, θ) for perturbed x values,
but step_from_seed() ALWAYS re-solves the BVP to find x.

ATTEMPTED WORKAROUNDS:
1. Use step_from_seed with perturbed mL_guess, nL_guess
   → Problem: BVP solver uses these as initial guess but solves to different x*

2. Call DYNSolverIVP directly
   → Problem: Not exposed in Python API

3. Modify step_from_seed to accept skip_bvp=True flag
   → Requires C++ modification

For now, we'll try approach #1 (perturb initial guess) and see if the
BVP solver converges to nearby solutions.
""")

print("\nAttempting FD via perturbed initial guess...")

# We'll perturb x and see if we can get different outputs
eps = 1e-4  # Perturbation size
num_actuators = mL_star.shape[0]
x_dim = 6  # 3 for mL + 3 for nL per actuator

gx_fd = np.zeros((6, x_dim))  # 6 outputs (3 pos + 3 vel) × 6 states

print(f"\nComputing ∂y/∂x via FD (this may not work perfectly)...")

for i in range(x_dim):
    # Perturb x[i]
    x_pert = x_star.copy()
    x_pert[i] += eps

    # Unpack back to mL, nL arrays
    mL_pert = x_pert[:3].reshape(1, 3)
    nL_pert = x_pert[3:6].reshape(1, 3)

    # Try to run forward pass with perturbed initial guess
    # Note: This will re-solve BVP, so we may not get exactly x_pert
    result_pert = dyn.step_from_seed(currents, insertion, v, w, p, R, xf, mL_pert, nL_pert)

    # Check if BVP converged to a different solution
    mL_solved = np.asarray(result_pert["next_mL"], dtype=np.float64)
    nL_solved = np.asarray(result_pert["next_nL"], dtype=np.float64)
    x_solved = np.concatenate([mL_solved.flatten(), nL_solved.flatten()])

    actual_dx = np.linalg.norm(x_solved - x_star)

    if actual_dx < 1e-10:
        print(f"  x[{i}]: BVP converged back to x* (dx={actual_dx:.2e}) - can't compute FD")
        continue

    # Get output
    tip_pos_pert = np.asarray(result_pert["tip_position"], dtype=np.float64)
    tip_vel_pert = np.asarray(result_pert["tip_velocity"], dtype=np.float64)
    y_pert = np.concatenate([tip_pos_pert, tip_vel_pert])
    y_base = np.concatenate([tip_pos_base, tip_vel_base])

    # Compute derivative
    dy = y_pert - y_base
    gx_fd[:, i] = dy / (x_solved[i] - x_star[i]) if abs(x_solved[i] - x_star[i]) > 1e-15 else 0.0

    print(f"  x[{i}]: dx_actual={actual_dx:.2e}, dy_norm={np.linalg.norm(dy):.2e}")

print(f"\ngx_fd shape: {gx_fd.shape}")
print(f"gx_fd norm: {np.linalg.norm(gx_fd):.6e}")
print(f"\ngx_fd matrix:\n{gx_fd}")

print_section("STEP 3: Compare with AD gx")

print("""
⚠️  WARNING: This comparison may not be valid because:
1. The BVP solver likely converged back to x* for all perturbations
2. We can't actually evaluate y at arbitrary x without BVP re-solving
3. The finite difference approximation requires holding θ fixed and only varying x

CONCLUSION:
To properly verify gx, we need one of:
1. C++ function that evaluates output given (x, θ) WITHOUT solving BVP
2. Expose DYNSolverIVP to Python with fixed mL, nL inputs
3. Analytical verification of the AD gradient computation

The debug output showed:
  gx norm: 4.50346e+07  (extremely large)

If this is correct, it suggests the output is VERY sensitive to the BVP solution.
If incorrect, the bug is in DYNNLEquationOutputJacobianEigenAD().
""")

print_section("STEP 4: Alternative Verification Strategy")

print("""
Since we can't directly verify gx via FD, let's check the components:

gx represents: ∂[tip_pos, tip_vel]/∂[mL, nL]

Physical intuition:
- mL, nL are internal moments and forces at the actuator location
- These affect the catheter shape and dynamics
- Large gx means small changes in forces → large changes in tip motion
- This could be physically plausible if the system is near a bifurcation

Next steps:
1. Examine the AD code for gx computation (DYNNLEquationOutputJacobianEigenAD)
2. Check if gx has the right structure (not NaN, not all zeros)
3. Verify gth independently
4. Check if the chain rule application is correct
""")

print(f"\nFrom C++ debug output, we know:")
print(f"  gx norm: 4.50346e+07")
print(f"  gth norm: 178.059")
print(f"  gx * dx/dθ norm: 3422.52")
print(f"  Final gradient norm: 3420.13")
print(f"\nThe large gx is DOMINATING the gradient computation.")
print(f"Need to investigate if this is physically correct or a bug.")
