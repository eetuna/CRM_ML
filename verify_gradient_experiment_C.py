"""
Experiment C: Cross half-plane test
Tests positive initialization with negative currents (sign mismatch).
This tests whether mismatched initialization worsens gradient errors.
"""

import numpy as np
import sys
sys.path.insert(0, '/workspaces/catheter/CRM_ML')

from crm_ml_rl.wrappers import crm_python

print("=" * 80)
print("EXPERIMENT C: CROSS HALF-PLANE (POSITIVE INIT, NEGATIVE CURRENTS)")
print("=" * 80)

# Initialize dynamics
print("\nInitializing CRM dynamics...")
dyn = crm_python.CRMDynamics()

param_file = "/workspaces/catheter/CRM_ML/data/catheter_params/CatheterParameterSet_1_dyn.txt"
config_file = "/workspaces/catheter/CRM_ML/data/catheter_params/CatheterSpatialConfiguration_1.txt"

dyn.load_parameters(param_file, config_file)
dyn.initialize_from_kinematics([0, 0, 0.01], 94.3)  # POSITIVE initialization
seed = dyn.get_seed_state()

currents = np.array([0.1, 0.05, -0.02])  # NEGATIVE c3 (MISMATCH!)

print("Initialization currents: [0, 0, 0.01]  (POSITIVE)")
print("Test currents:", currents, " (NEGATIVE c3)")
print(">>> SIGN MISMATCH - Testing cross half-plane <<<")
print("Insertion depth:", 94.3)

# ============================================================================
# Run same tests as Experiments A and B
# ============================================================================

print("\n" + "=" * 80)
print("FORWARD PASS")
print("=" * 80)

fwd = dyn.step_from_seed(currents, 94.3, seed['v'], seed['w'], seed['p'],
                          seed['R'], seed['xf'], seed['mL'], seed['nL'])

print("\nForward pass result:")
print("  tip_position:", fwd['tip_position'])
print("  tip_velocity:", fwd['tip_velocity'])

print("\n" + "=" * 80)
print("IMPLICIT LINEARIZATION")
print("=" * 80)

impl = dyn.linearize_full_seed_action_from_seed_implicit(
    currents, 94.3, seed['v'], seed['w'], seed['p'],
    seed['R'], seed['xf'], seed['mL'], seed['nL'])

print("\nImplicit linearization result:")
print("  next_state (first 6):", impl['next_state'][:6])
print("  B matrix shape:", impl['B'].shape)
print("  B matrix:\n", impl['B'])

fwd_state = np.concatenate([fwd['tip_position'], fwd['tip_velocity']])
impl_state = impl['next_state'][:6]

print("\nForward vs Implicit comparison:")
print("  Forward state:", fwd_state)
print("  Implicit state:", impl_state)
print("  Difference:", np.abs(fwd_state - impl_state))
print("  Match?", np.allclose(fwd_state, impl_state, rtol=1e-6, atol=1e-8))

print("\n" + "=" * 80)
print("EXPLICIT LINEARIZATION")
print("=" * 80)

try:
    expl = dyn.linearize_full_seed_action_from_seed(
        currents, 94.3, seed['v'], seed['w'], seed['p'],
        seed['R'], seed['xf'], seed['mL'], seed['nL'])

    print("\nExplicit linearization result:")
    print("  B matrix shape:", expl['B'].shape)
    print("  B matrix:\n", expl['B'])
    print("  All zeros?", np.allclose(expl['B'], 0))

except Exception as e:
    print(f"\nExplicit linearization FAILED: {e}")

print("\n" + "=" * 80)
print("FINITE DIFFERENCE VALIDATION")
print("=" * 80)

print("\nComputing finite difference gradients...")
eps = 1e-5
B_fd = np.zeros((6, 3))

for i in range(3):
    print(f"\n  Perturbing current {i}...")

    c_plus = currents.copy()
    c_plus[i] += eps
    fwd_plus = dyn.step_from_seed(c_plus, 94.3, seed['v'], seed['w'], seed['p'],
                                   seed['R'], seed['xf'], seed['mL'], seed['nL'])
    y_plus = np.concatenate([fwd_plus['tip_position'], fwd_plus['tip_velocity']])

    c_minus = currents.copy()
    c_minus[i] -= eps
    fwd_minus = dyn.step_from_seed(c_minus, 94.3, seed['v'], seed['w'], seed['p'],
                                    seed['R'], seed['xf'], seed['mL'], seed['nL'])
    y_minus = np.concatenate([fwd_minus['tip_position'], fwd_minus['tip_velocity']])

    B_fd[:, i] = (y_plus - y_minus) / (2 * eps)
    print(f"    FD gradient for current {i}:", B_fd[:, i])

print("\n" + "-" * 80)
print("RESULTS")
print("-" * 80)

print("\nB from FD (ground truth):\n", B_fd)
print("\nB from implicit:\n", impl['B'])

diff = np.abs(B_fd - impl['B'])
rel_error = diff / (np.abs(B_fd) + 1e-8)

print("\nAbsolute difference:\n", diff)
print("\nRelative error (%):\n", rel_error * 100)

print("\n" + "=" * 80)
print("ERROR METRICS")
print("=" * 80)

print(f"  Mean absolute error: {np.mean(diff):.6e}")
print(f"  Max absolute error: {np.max(diff):.6e}")
print(f"  Mean relative error: {np.mean(rel_error) * 100:.2f}%")
print(f"  Max relative error: {np.max(rel_error) * 100:.2f}%")
print(f"  Frobenius norm error: {np.linalg.norm(diff) / np.linalg.norm(B_fd) * 100:.2f}%")

print("\n" + "=" * 80)
print("EXPERIMENT C COMPLETE")
print("=" * 80)
