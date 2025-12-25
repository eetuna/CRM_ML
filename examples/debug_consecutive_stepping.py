#!/usr/bin/env python3
"""
Debug script for consecutive stepping analysis.

Tests numerical stability of dynamics stepping by performing consecutive
forward integration steps and logging all state variables.

Purpose: Isolate whether iLQR divergence is caused by:
1. Integrator choice (ABM4 vs RK4)
2. Derivative history initialization in step_from_seed API
3. Orthonormality drift in rotation matrix
4. Numerical blow-up in state variables
"""

import argparse
import numpy as np
from typing import Dict
from crm_ml_rl.wrappers import crm_python


# Stable damping values (from integrator stability analysis)
BASE_DAMPING = np.array([
    12.1761626666366,      # Linear damping X
    12.1761626666366,      # Linear damping Y
    284.429938756989,      # Linear damping Z
    0.0304776127617393,    # Angular damping X
    0.0304776127617393,    # Angular damping Y
    0.00502712804532508    # Angular damping Z
], dtype=np.float64)


def log_state(step_num: int, result: Dict, currents: np.ndarray) -> Dict[str, float]:
    """
    Log all state variables from step result.

    Returns dict of key metrics for analysis.
    """
    # Extract state variables (assuming single actuator set, index 0)
    v = result['next_v'][0]
    w = result['next_w'][0]
    p = result['next_p'][0]
    R = result['next_R'][0].reshape(3, 3)
    xf = result['next_xf'][0]
    mL = result['next_mL'][0]
    nL = result['next_nL'][0]

    # Convergence flags
    converged = result.get('converged', True)
    diverged = result.get('diverged', False)

    # Compute diagnostics
    v_norm = np.linalg.norm(v)
    w_norm = np.linalg.norm(w)
    mL_norm = np.linalg.norm(mL)
    nL_norm = np.linalg.norm(nL)

    # Orthonormality check: R^T @ R should equal I
    ortho_error = np.linalg.norm(R.T @ R - np.eye(3))

    # Check for NaNs
    has_nan = (
        np.isnan(v).any() or np.isnan(w).any() or
        np.isnan(p).any() or np.isnan(R).any() or
        np.isnan(xf).any() or np.isnan(mL).any() or np.isnan(nL).any()
    )

    print(f"\n{'='*70}")
    print(f"Step {step_num}")
    print(f"{'='*70}")
    print(f"Input currents:  [{currents[0]:8.5f}, {currents[1]:8.5f}, {currents[2]:8.5f}] A")
    print(f"Converged: {converged:5}  |  Diverged: {diverged:5}  |  Has NaN: {has_nan:5}")
    print(f"\nState Variables:")
    print(f"  v (lin vel):   [{v[0]:10.5f}, {v[1]:10.5f}, {v[2]:10.5f}]  |v| = {v_norm:.5f} mm/s")
    print(f"  w (ang vel):   [{w[0]:10.5f}, {w[1]:10.5f}, {w[2]:10.5f}]  |w| = {w_norm:.5f} rad/s")
    print(f"  p (position):  [{p[0]:10.3f}, {p[1]:10.3f}, {p[2]:10.3f}] mm")
    print(f"  xf (fwd dist): {xf:10.5f} mm")
    print(f"\nForces:")
    print(f"  mL (moment):   [{mL[0]:10.5f}, {mL[1]:10.5f}, {mL[2]:10.5f}]  |mL| = {mL_norm:.5f} N·mm")
    print(f"  nL (force):    [{nL[0]:10.5f}, {nL[1]:10.5f}, {nL[2]:10.5f}]  |nL| = {nL_norm:.5f} N")
    print(f"\nDiagnostics:")
    print(f"  Orthonormality error: {ortho_error:.2e}")
    print(f"  R matrix determinant: {np.linalg.det(R):.6f} (expect ≈ 1.0)")

    # Expected ranges (sanity checks)
    if w_norm > 100:
        print(f"  ⚠️  WARNING: |w| = {w_norm:.2f} rad/s exceeds expected range (< 100)")
    if nL_norm > 10:
        print(f"  ⚠️  WARNING: |nL| = {nL_norm:.2f} N exceeds expected range (< 10)")
    if ortho_error > 1e-6:
        print(f"  ⚠️  WARNING: Orthonormality error {ortho_error:.2e} exceeds tolerance (1e-6)")

    return {
        'step': step_num,
        'converged': converged,
        'diverged': diverged,
        'has_nan': has_nan,
        'v_norm': v_norm,
        'w_norm': w_norm,
        'ortho_error': ortho_error,
        'det_R': np.linalg.det(R),
        'mL_norm': mL_norm,
        'nL_norm': nL_norm,
    }


def run_consecutive_steps(integrator: str = "abm4", num_steps: int = 10):
    """
    Execute consecutive forward steps with minimal current input.

    Args:
        integrator: "abm4" or "rk4"
        num_steps: Number of consecutive steps to perform
    """
    print("\n" + "="*70)
    print(f"CONSECUTIVE STEPPING DEBUG - Integrator: {integrator.upper()}")
    print("="*70)

    # Initialize dynamics
    dyn = crm_python.CRMDynamics()
    ok = dyn.load_parameters(
        "data/catheter_params/CatheterParameterSet_1_dyn.txt",
        "data/catheter_params/CatheterSpatialConfiguration_1.txt",
    )
    if not ok:
        raise RuntimeError("Failed to load parameters")

    # Set damping and timestep
    dyn.set_damping(BASE_DAMPING)
    dyn.dt = 0.02  # 20ms timestep
    dyn.integration_step_size = 0.1

    # Set integrator if RK4 requested
    if integrator == "rk4":
        dyn.set_integrator("rk4")
        print("Integrator set to: RK4")
    else:
        print("Integrator: ABM4 (default)")

    # Initialize from kinematics
    insertion_length = 94.3  # mm
    dyn.initialize_from_kinematics([0.0, 0.0, 0.2], insertion_length)

    print(f"Insertion length: {insertion_length} mm")
    print(f"Timestep dt: {dyn.dt} s")
    print(f"Integration step size: {dyn.integration_step_size}")

    # Get initial seed state
    seed = dyn.get_seed_state()

    # Small test current (conservative)
    currents = np.array([0.01, 0.0, 0.0], dtype=np.float64)

    # Storage for metrics
    metrics = []

    # Perform consecutive steps
    for step_num in range(1, num_steps + 1):
        # Step from current seed
        result = dyn.step_from_seed(
            currents,
            insertion_length,
            seed['v'],
            seed['w'],
            seed['p'],
            seed['R'],
            seed['xf'],
            seed['mL'],
            seed['nL']
        )

        # Log state
        step_metrics = log_state(step_num, result, currents)
        metrics.append(step_metrics)

        # Check for failure
        if step_metrics['has_nan']:
            print(f"\n❌ FAILURE: NaN detected at step {step_num}")
            break

        if step_metrics['diverged']:
            print(f"\n❌ FAILURE: Divergence detected at step {step_num}")

            # Try to compute Jacobian condition number to diagnose the divergence
            try:
                print(f"\n🔍 Computing Jacobian condition number for diagnostic...")
                lin_result = dyn.linearize_full_seed_action_from_seed_implicit(
                    currents,
                    insertion_length,
                    seed['v'],
                    seed['w'],
                    seed['p'],
                    seed['R'],
                    seed['xf'],
                    seed['mL'],
                    seed['nL'],
                    return_debug=True
                )
                if 'Jxx_condition_number' in lin_result:
                    cond_num = lin_result['Jxx_condition_number']
                    print(f"Jacobian condition number: {cond_num:.2e}")
                    if cond_num > 1e12:
                        print(f"⚠️  WARNING: Condition number > 10¹² indicates ill-conditioned Jacobian!")
                else:
                    print("Jacobian condition number not available (linearization may have failed)")
            except Exception as e:
                print(f"Could not compute Jacobian: {e}")

            break

        if not step_metrics['converged']:
            print(f"\n⚠️  WARNING: Step {step_num} did not converge")

        # Update seed for next step (use outputs as inputs)
        seed = {
            'v': result['next_v'],
            'w': result['next_w'],
            'p': result['next_p'],
            'R': result['next_R'],
            'xf': result['next_xf'],
            'mL': result['next_mL'],
            'nL': result['next_nL']
        }

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    successful_steps = sum(1 for m in metrics if not m['diverged'] and not m['has_nan'])
    print(f"Successful steps: {successful_steps}/{num_steps}")

    if successful_steps == num_steps:
        print("✅ All steps completed successfully!")
    else:
        failure_step = next((m['step'] for m in metrics if m['diverged'] or m['has_nan']), None)
        print(f"❌ First failure at step: {failure_step}")

    # Check for trend in orthonormality error
    ortho_errors = [m['ortho_error'] for m in metrics]
    if len(ortho_errors) > 1:
        ortho_increase = ortho_errors[-1] / ortho_errors[0] if ortho_errors[0] > 0 else float('inf')
        print(f"\nOrthonormality drift: {ortho_errors[0]:.2e} → {ortho_errors[-1]:.2e} (×{ortho_increase:.2f})")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Debug consecutive dynamics stepping for numerical stability analysis"
    )
    parser.add_argument(
        "--integrator",
        choices=["abm4", "rk4"],
        default="abm4",
        help="Integration method (default: abm4)"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=10,
        help="Number of consecutive steps to perform (default: 10)"
    )

    args = parser.parse_args()

    # Run debug test
    run_consecutive_steps(
        integrator=args.integrator,
        num_steps=args.num_steps
    )

    return 0


if __name__ == "__main__":
    exit(main())
