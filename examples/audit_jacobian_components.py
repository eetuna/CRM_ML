#!/usr/bin/env python3
"""
Granular Jacobian Component Audit for Phase 4 Task 4.5.

Compares AD-computed Jacobians (Jxx_ad) with FD-computed Jacobians (Jxx_fd).
Target: relative error < 1e-7
"""

import numpy as np
import os

# Enable implicit linearization
os.environ["CRM_DYN_LINEARIZATION_METHOD"] = "implicit"

from crm_ml_rl.wrappers import crm_python


def audit_jacobian_components():
    """
    Compare AD vs FD Jacobians with return_debug=True.
    """
    print(f"\n{'='*70}")
    print("GRANULAR JACOBIAN COMPONENT AUDIT")
    print(f"{'='*70}")

    # Initialize dynamics
    dyn = crm_python.CRMDynamics()
    ok = dyn.load_parameters(
        "data/catheter_params/CatheterParameterSet_1_dyn.txt",
        "data/catheter_params/CatheterSpatialConfiguration_1.txt",
    )
    if not ok:
        raise RuntimeError("Failed to load parameters")

    # Set damping and timestep
    BASE_DAMPING = np.array([
        12.1761626666366,
        12.1761626666366,
        284.429938756989,
        0.0304776127617393,
        0.0304776127617393,
        0.00502712804532508
    ], dtype=np.float64)

    dyn.set_damping(BASE_DAMPING)
    dyn.dt = 0.02
    dyn.integration_step_size = 0.1

    # Initialize from kinematics
    insertion_length = 94.3
    dyn.initialize_from_kinematics([0.0, 0.0, 0.2], insertion_length)

    # Get initial seed state
    seed = dyn.get_seed_state()

    # Small test current
    currents = np.array([0.01, 0.0, 0.0], dtype=np.float64)

    print(f"\nCalling linearize_full_seed_action_from_seed_implicit with return_debug=True...")

    # Call with return_debug=True
    try:
        result = dyn.linearize_full_seed_action_from_seed_implicit(
            currents,
            insertion_length,
            seed['v'],
            seed['w'],
            seed['p'],
            seed['R'],
            seed['xf'],
            seed['mL'],
            seed['nL'],
            eps_residual_x=1e-5,
            eps_residual_theta=1e-5,
            eps_g_x=1e-5,
            eps_g_theta=1e-5,
            return_debug=True
        )
    except Exception as e:
        print(f"\n❌ Linearization failed: {e}")
        return False

    # Check what debug info is available
    print(f"\nDebug keys available: {[k for k in result.keys() if k.startswith('Jxx') or k.startswith('have_')]}")

    # Check if we have both AD and FD Jacobians
    have_ad_jxx = result.get('have_ad_jxx', False)
    have_ad_jxu = result.get('have_ad_jxu', False)

    print(f"\nAD Status:")
    print(f"  have_ad_jxx (state Jacobian): {have_ad_jxx}")
    print(f"  have_ad_jxu (control Jacobian): {have_ad_jxu}")

    # Check condition number
    if 'Jxx_condition_number' in result:
        cond_num = result['Jxx_condition_number']
        print(f"\nJxx condition number: {cond_num:.2e}")
        if cond_num > 1e12:
            print(f"  ⚠️  WARNING: Condition number > 10¹² indicates ill-conditioned Jacobian!")

    # Get Jacobians - 'Jxx' is the AD version when have_ad_jxx=True
    if 'Jxx' not in result or 'Jxx_fd' not in result:
        print(f"\n⚠️  WARNING: Jxx or Jxx_fd not in result")
        print(f"Available keys: {list(result.keys())}")
        return False

    # Get Jacobians (Jxx is the AD version when have_ad_jxx=True)
    Jxx_ad = np.array(result['Jxx'])
    Jxx_fd = np.array(result['Jxx_fd'])

    print(f"\nJacobian shapes:")
    print(f"  Jxx_ad: {Jxx_ad.shape}")
    print(f"  Jxx_fd: {Jxx_fd.shape}")

    # Compute relative error
    abs_diff = np.linalg.norm(Jxx_ad - Jxx_fd)
    norm_fd = np.linalg.norm(Jxx_fd)
    rel_err = abs_diff / norm_fd if norm_fd > 0 else float('inf')

    print(f"\n{'='*70}")
    print("JACOBIAN COMPARISON")
    print(f"{'='*70}")
    print(f"||Jxx_ad - Jxx_fd||: {abs_diff:.6e}")
    print(f"||Jxx_fd||:          {norm_fd:.6e}")
    print(f"Relative error:      {rel_err:.6e}")

    # Target threshold
    TARGET_REL_ERR = 1e-7
    print(f"\nTarget: rel_err < {TARGET_REL_ERR:.0e}")

    if rel_err < TARGET_REL_ERR:
        print(f"✅ PASS: Relative error {rel_err:.2e} < {TARGET_REL_ERR:.0e}")
        return True
    else:
        print(f"❌ FAIL: Relative error {rel_err:.2e} >= {TARGET_REL_ERR:.0e}")
        print(f"   Exceeded by factor: {rel_err / TARGET_REL_ERR:.2f}x")

        # Additional diagnostics
        print(f"\n{'='*70}")
        print("DETAILED DIAGNOSTICS")
        print(f"{'='*70}")

        # Element-wise max error
        elem_abs_diff = np.abs(Jxx_ad - Jxx_fd)
        max_elem_err = np.max(elem_abs_diff)
        max_idx = np.unravel_index(np.argmax(elem_abs_diff), elem_abs_diff.shape)
        print(f"Max element-wise error: {max_elem_err:.6e} at index {max_idx}")
        print(f"  Jxx_ad[{max_idx}] = {Jxx_ad[max_idx]:.6e}")
        print(f"  Jxx_fd[{max_idx}] = {Jxx_fd[max_idx]:.6e}")

        # Check for NaNs
        nan_ad = np.isnan(Jxx_ad).any()
        nan_fd = np.isnan(Jxx_fd).any()
        print(f"\nNaN check:")
        print(f"  Jxx_ad has NaN: {nan_ad}")
        print(f"  Jxx_fd has NaN: {nan_fd}")

        # Distribution of errors
        elem_rel_err = elem_abs_diff / (np.abs(Jxx_fd) + 1e-15)
        print(f"\nElement-wise relative error statistics:")
        print(f"  Mean: {np.mean(elem_rel_err):.6e}")
        print(f"  Median: {np.median(elem_rel_err):.6e}")
        print(f"  Max: {np.max(elem_rel_err):.6e}")
        print(f"  Std: {np.std(elem_rel_err):.6e}")

        return False


def main():
    print("\n" + "="*70)
    print("PHASE 4 TASK 4.5: Granular Jacobian Component Audit")
    print("="*70)
    print("\nComparing AD-computed vs FD-computed Jacobians")
    print("Target: Relative error < 1e-7")

    success = audit_jacobian_components()

    print("\n" + "="*70)
    print("AUDIT SUMMARY")
    print("="*70)

    if success:
        print("✅ SUCCESS: AD Jacobians match FD within tolerance")
        return 0
    else:
        print("⚠️  NEEDS INVESTIGATION: AD vs FD mismatch or debug info unavailable")
        print("\nPossible causes:")
        print("  1. return_debug implementation may not expose Jxx_ad/Jxx_fd")
        print("  2. AD implementation needs refinement")
        print("  3. FD epsilon too large/small for system stiffness")
        return 1


if __name__ == "__main__":
    exit(main())
