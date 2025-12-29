"""
Verification experiments to confirm gradient issues in Option C.
Based on plan in /home/vscode/.claude/plans/indexed-skipping-map.md
"""

import numpy as np
import sys
sys.path.insert(0, '/workspaces/catheter/CRM_ML')

from crm_ml_rl.wrappers import crm_python

print("=" * 80)
print("VERIFICATION EXPERIMENTS FOR OPTION C GRADIENT ISSUES")
print("=" * 80)

# Initialize dynamics
print("\nInitializing CRM dynamics...")
dyn = crm_python.CRMDynamics()

param_file = "/workspaces/catheter/CRM_ML/data/catheter_params/CatheterParameterSet_1_dyn.txt"
config_file = "/workspaces/catheter/CRM_ML/data/catheter_params/CatheterSpatialConfiguration_1.txt"

dyn.load_parameters(param_file, config_file)
dyn.initialize_from_kinematics([0, 0, 0.01], 94.3)
seed = dyn.get_seed_state()

currents = np.array([0.1, 0.05, 0.02])

print("Currents:", currents)
print("Insertion depth:", 94.3)

# ============================================================================
# EXPERIMENT 1: Confirm forward/backward mismatch
# ============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 1: Confirm forward/backward mismatch")
print("=" * 80)

print("\nRunning forward pass (step_from_seed)...")
fwd = dyn.step_from_seed(currents, 94.3, seed['v'], seed['w'], seed['p'],
                          seed['R'], seed['xf'], seed['mL'], seed['nL'])

print("\nForward pass result:")
print("  tip_position:", fwd['tip_position'])
print("  tip_velocity:", fwd['tip_velocity'])

print("\nRunning implicit linearization...")
impl = dyn.linearize_full_seed_action_from_seed_implicit(
    currents, 94.3, seed['v'], seed['w'], seed['p'],
    seed['R'], seed['xf'], seed['mL'], seed['nL'])

print("\nImplicit linearization result:")
print("  next_state (first 6):", impl['next_state'][:6])
print("  B matrix shape:", impl['B'].shape)
print("  B matrix:\n", impl['B'])

fwd_state = np.concatenate([fwd['tip_position'], fwd['tip_velocity']])
impl_state = impl['next_state'][:6]

print("\nComparison:")
print("  Forward state:", fwd_state)
print("  Implicit state:", impl_state)
print("  Difference:", np.abs(fwd_state - impl_state))
print("  Match?", np.allclose(fwd_state, impl_state, rtol=1e-6, atol=1e-8))

# ============================================================================
# EXPERIMENT 2: Confirm explicit method returns zeros
# ============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 2: Confirm explicit method returns zeros")
print("=" * 80)

print("\nRunning explicit linearization...")
try:
    expl = dyn.linearize_full_seed_action_from_seed(
        currents, 94.3, seed['v'], seed['w'], seed['p'],
        seed['R'], seed['xf'], seed['mL'], seed['nL'])

    print("\nExplicit linearization result:")
    print("  B matrix shape:", expl['B'].shape)
    print("  B matrix:\n", expl['B'])
    print("  All zeros?", np.allclose(expl['B'], 0))
    print("  A matrix norm:", np.linalg.norm(expl['A']))

except Exception as e:
    print(f"\nExplicit linearization FAILED with error: {e}")

# ============================================================================
# EXPERIMENT 3: Manual FD comparison
# ============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 3: Manual FD comparison")
print("=" * 80)

print("\nComputing finite difference gradients...")
eps = 1e-5
B_fd = np.zeros((6, 3))

for i in range(3):
    print(f"\n  Perturbing current {i}...")

    # Positive perturbation
    c_plus = currents.copy()
    c_plus[i] += eps
    fwd_plus = dyn.step_from_seed(c_plus, 94.3, seed['v'], seed['w'], seed['p'],
                                   seed['R'], seed['xf'], seed['mL'], seed['nL'])
    y_plus = np.concatenate([fwd_plus['tip_position'], fwd_plus['tip_velocity']])

    # Negative perturbation
    c_minus = currents.copy()
    c_minus[i] -= eps
    fwd_minus = dyn.step_from_seed(c_minus, 94.3, seed['v'], seed['w'], seed['p'],
                                    seed['R'], seed['xf'], seed['mL'], seed['nL'])
    y_minus = np.concatenate([fwd_minus['tip_position'], fwd_minus['tip_velocity']])

    # Central difference
    B_fd[:, i] = (y_plus - y_minus) / (2 * eps)
    print(f"    FD gradient for current {i}:", B_fd[:, i])

print("\n" + "-" * 80)
print("RESULTS:")
print("-" * 80)

print("\nB from FD:\n", B_fd)
print("\nB from implicit:\n", impl['B'])

diff = np.abs(B_fd - impl['B'])
rel_error = diff / (np.abs(B_fd) + 1e-8)

print("\nAbsolute difference:\n", diff)
print("\nRelative error (%):\n", rel_error * 100)

print("\nError metrics:")
print(f"  Mean absolute error: {np.mean(diff):.6e}")
print(f"  Max absolute error: {np.max(diff):.6e}")
print(f"  Mean relative error: {np.mean(rel_error) * 100:.2f}%")
print(f"  Max relative error: {np.max(rel_error) * 100:.2f}%")
print(f"  Frobenius norm error: {np.linalg.norm(diff) / np.linalg.norm(B_fd) * 100:.2f}%")

print("\n" + "=" * 80)
print("EXPERIMENTS COMPLETE")
print("=" * 80)
