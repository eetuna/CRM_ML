#!/usr/bin/env python3
"""
Component-wise gradient diagnostic for implicit linearization.

This script breaks down the gradient computation dy/dθ = gth + gx * dx/dθ
and verifies each component against finite differences.

Tests:
1. gth = ∂y/∂θ|ₓ (partial derivative with x fixed)
2. gx = ∂y/∂x (output sensitivity to BVP solution)
3. Jxx = ∂F/∂x (BVP residual Jacobian w.r.t. state)
4. Jxθ = ∂F/∂θ (BVP residual Jacobian w.r.t. parameters)
5. dx/dθ = -Jxx⁻¹ * Jxθ (implicit function theorem)
6. Full chain rule: dy/dθ = gth + gx * dx/dθ
"""

import numpy as np
import sys
sys.path.insert(0, '/workspaces/catheter/CRM_ML')

from crm_ml_rl.wrappers import crm_python

def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def print_matrix(name, mat, max_rows=6, max_cols=6):
    """Print matrix with size limits for readability."""
    rows, cols = mat.shape if mat.ndim == 2 else (mat.shape[0], 1)
    print(f"\n{name} (shape {mat.shape}):")
    if rows <= max_rows and cols <= max_cols:
        print(mat)
    else:
        print(f"  Showing first {min(rows, max_rows)}x{min(cols, max_cols)} block:")
        if mat.ndim == 2:
            print(mat[:max_rows, :max_cols])
        else:
            print(mat[:max_rows])
    print(f"  Norm: {np.linalg.norm(mat):.6e}")
    print(f"  Max abs: {np.abs(mat).max():.6e}")

def compute_fd_jacobian(func, x, eps=1e-6):
    """Compute Jacobian via finite differences."""
    x = np.asarray(x, dtype=np.float64)
    y0 = func(x)
    y0 = np.asarray(y0, dtype=np.float64).flatten()

    m = len(y0)
    n = len(x)
    J = np.zeros((m, n))

    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += eps
        y_plus = np.asarray(func(x_plus), dtype=np.float64).flatten()

        x_minus = x.copy()
        x_minus[i] -= eps
        y_minus = np.asarray(func(x_minus), dtype=np.float64).flatten()

        J[:, i] = (y_plus - y_minus) / (2 * eps)

    return J

print_section("COMPONENT-WISE GRADIENT DIAGNOSTIC")

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

print(f"Initial state: v shape={v.shape}, currents={currents}")

print_section("STEP 1: Forward Pass")

# Run forward pass to get baseline
result = dyn.step_from_seed(currents, insertion, v, w, p, R, xf, mL, nL)
tip_pos = np.asarray(result["tip_position"], dtype=np.float64)
tip_vel = np.asarray(result["tip_velocity"], dtype=np.float64)
mL_star = np.asarray(result["next_mL"], dtype=np.float64)
nL_star = np.asarray(result["next_nL"], dtype=np.float64)

print(f"Converged: {result.get('converged', False)}")
print(f"Tip position: {tip_pos}")
print(f"Tip velocity: {tip_vel}")
print(f"Solved mL* shape: {mL_star.shape}")
print(f"Solved nL* shape: {nL_star.shape}")

# Flatten x* = [mL*, nL*] for later use
x_star_full = np.concatenate([mL_star.flatten(), nL_star.flatten()])
print(f"x* = [mL*, nL*] shape: {x_star_full.shape}")

print_section("STEP 2: Run Implicit Linearization (AD)")

try:
    result_ad = dyn.linearize_full_seed_action_from_seed_implicit(
        currents, insertion, v, w, p, R, xf, mL, nL,
        eps_residual_x=1e-5,
        eps_residual_theta=1e-5,
        eps_g_x=1e-5,
        eps_g_theta=1e-5,
        return_debug=True
    )

    B_ad = np.asarray(result_ad["B"], dtype=np.float64)
    A_ad = np.asarray(result_ad["A"], dtype=np.float64)

    print_matrix("B_ad (∂y/∂currents)", B_ad)
    print_matrix("A_ad (∂y/∂seed)", A_ad, max_cols=12)

    # Check if debug info is available
    has_debug = "debug" in result_ad
    if has_debug:
        debug = result_ad["debug"]
        print("\nDebug info available:")
        for key in debug.keys():
            print(f"  - {key}")
    else:
        print("\n⚠️  No debug info returned (need to modify C++ to return gth, gx, etc.)")

except Exception as e:
    print(f"ERROR in linearization: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print_section("STEP 3: Compute Reference Gradients (Finite Differences)")

print("\nComputing dy/d(currents) via FD on full forward pass...")

def forward_pass(curr):
    """Forward pass: currents → tip state."""
    res = dyn.step_from_seed(curr, insertion, v, w, p, R, xf, mL, nL)
    pos = np.asarray(res["tip_position"], dtype=np.float64)
    vel = np.asarray(res["tip_velocity"], dtype=np.float64)
    return np.concatenate([pos, vel])

B_fd = compute_fd_jacobian(forward_pass, currents, eps=1e-6)
print_matrix("B_fd (∂y/∂currents)", B_fd)

print_section("STEP 4: Compare AD vs FD Gradients")

diff_B = B_ad - B_fd
rel_error_B = np.abs(diff_B) / (np.abs(B_fd) + 1e-10)

print_matrix("B_ad - B_fd (difference)", diff_B)
print_matrix("Relative error", rel_error_B)

max_rel_error = rel_error_B.max()
print(f"\n{'='*80}")
print(f"MAXIMUM RELATIVE ERROR: {max_rel_error:.2%}")
print(f"{'='*80}")

if max_rel_error > 0.5:
    print(f"\n❌ LARGE MISMATCH: {max_rel_error:.1%} error")
    print("\nThis indicates a bug in the AD gradient computation.")
elif max_rel_error > 0.05:
    print(f"\n⚠️  MODERATE MISMATCH: {max_rel_error:.1%} error")
    print("\nGradients are approximately correct but need refinement.")
else:
    print(f"\n✅ GOOD MATCH: {max_rel_error:.2%} error")
    print("\nAD gradients match FD within numerical tolerance!")

print_section("STEP 5: Component Analysis (requires C++ debug output)")

print("""
To diagnose which component is wrong, we need to verify:

1. gth = ∂y/∂θ|ₓ (partial derivative, x fixed)
   - This should be computed with mL*, nL* held constant

2. gx = ∂y/∂x (how y depends on BVP solution)
   - This shows how tip state changes if we perturb the solved forces

3. Jxx = ∂F/∂x (BVP residual Jacobian)
   - Should be invertible (full rank)
   - Residual F(x*, θ) should be ≈ 0

4. Jxθ = ∂F/∂θ (how residual changes with parameters)

5. dx/dθ = -Jxx⁻¹ * Jxθ (implicit differentiation)

6. Full chain: dy/dθ = gth + gx * dx/dθ

NEXT STEP: Add C++ debug output to print these intermediate values.
Set environment variable: export CRM_DEBUG_GRADIENT=1
""")

print_section("STEP 6: Manual Component Verification (Partial)")

# We can manually verify some components even without C++ debug output

print("\nVerifying gx = ∂y/∂x by perturbing mL*, nL*...")
print("(This requires modifying step_from_seed to accept arbitrary mL, nL)")
print("⚠️  Current step_from_seed recomputes mL, nL via BVP, so we can't test this directly")

print("\nVerifying gth = ∂y/∂θ|ₓ...")
print("(This requires a function that computes y given (x, θ) without re-solving BVP)")
print("⚠️  This function doesn't exist in the current API")

print("""
CONCLUSION:
To properly diagnose the issue, we need to modify crm_bindings.cpp to:
1. Return intermediate values (gth, gx, Jxx, Jxθ, dx/dθ) in return_debug mode
2. Or add CRM_DEBUG_GRADIENT environment variable to print these values

Without this, we can only compare the final gradient (which shows {:.1%} error).
""".format(max_rel_error))

print_section("SUMMARY")

print(f"""
Test Configuration:
  Currents: {currents}
  Insertion: {insertion} mm

Results:
  AD Gradient B shape: {B_ad.shape}
  FD Gradient B shape: {B_fd.shape}
  Max relative error: {max_rel_error:.2%}

Status: {'❌ FAILED' if max_rel_error > 0.05 else '✅ PASSED'}

Next Action:
  {'Modify C++ to add debug output, then re-run diagnostics' if max_rel_error > 0.05 else 'Gradient computation is correct!'}
""")
