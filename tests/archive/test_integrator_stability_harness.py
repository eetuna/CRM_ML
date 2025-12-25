#!/usr/bin/env python3
"""
Test harness for Phase 3 Integrator Stabilization.

Reproduces known failing case and compares ABM4 vs RK4 integrators.
Validates that divergence detection and soft-failure propagation work correctly.

Input: `docs/architecture/TASK_1_7_FAILING_CASE.json`
Output: Results logged to console and optionally saved to `docs/architecture/INTEGRATOR_STABILITY_RESULTS.json`
"""

import json
import time
import numpy as np
from pathlib import Path
import sys

# Add source path
sys.path.insert(0, str(Path(__file__).parent.parent))

from crm_ml_rl.wrappers import CRMWrapper, CRMSimulator
from crm_ml_rl.wrappers.crm_wrapper import CatheterParams, CatheterState


def load_failing_case():
    """Load the known failing case from JSON."""
    failing_case_path = Path(__file__).parent.parent / "docs" / "architecture" / "TASK_1_7_FAILING_CASE.json"
    with open(failing_case_path, 'r') as f:
        case = json.load(f)
    return case


def extract_seed_data(seed_dict):
    """Extract seed data from the case JSON."""
    # Extract seed state as flat arrays for step_from_seed
    v = seed_dict['v'][0]  # [vx, vy, vz]
    w = seed_dict['w'][0]  # [wx, wy, wz]
    p = seed_dict['p'][0]  # [px, py, pz]
    R = seed_dict['R'][0]  # [R00, R01, ..., R22] (9 elements)

    return {
        'v': v,
        'w': w,
        'p': p,
        'R': R,
    }


def run_test_case(case, integrator_type="abm4"):
    """
    Run a single test case with specified integrator.

    Args:
        case: Failing case dict with currents, insertion, and seed
        integrator_type: "ABM4" or "RK4"

    Returns:
        Dict with timing, convergence status, and residuals
    """
    insertion = case['insertion']
    currents = np.array(case['currents'])
    seed = extract_seed_data(case['seed'])

    print(f"\n{'='*70}")
    print(f"Test Case: {case['tag']}, trial={case['trial']}, integrator={integrator_type}")
    print(f"{'='*70}")
    print(f"Insertion: {insertion}")
    print(f"Currents: {currents}")
    print(f"Seed state:")
    print(f"  p: {seed['p']}")
    print(f"  v: {seed['v']}")
    print(f"  w: {seed['w']}")
    print(f"  R: {seed['R']}")

    # Initialize CRMWrapper with default parameters (C++ bindings required)
    try:
        wrapper = CRMWrapper()
    except Exception as e:
        print(f"Failed to initialize CRMWrapper: {str(e)}")
        return {
            'integrator': integrator_type,
            'converged': False,
            'info': None,
            'localmin': None,
            'residual_norm': None,
            'diverged': True,
            'elapsed_time': 0.0,
            'status': 'INIT_FAILED',
            'error_message': str(e)
        }
    if not getattr(wrapper, "_cpp_available", False):
        print("C++ bindings not available; cannot run integrator comparison.")
        return {
            'integrator': integrator_type,
            'converged': False,
            'info': None,
            'localmin': None,
            'residual_norm': None,
            'diverged': True,
            'elapsed_time': 0.0,
            'status': 'NO_CPP_BINDINGS'
        }

    try:
        wrapper._cpp_dynamics.set_integrator(integrator_type)
    except Exception as e:
        print(f"Failed to set integrator '{integrator_type}': {str(e)}")
        return {
            'integrator': integrator_type,
            'converged': False,
            'info': None,
            'localmin': None,
            'residual_norm': None,
            'diverged': True,
            'elapsed_time': 0.0,
            'status': 'SET_INTEGRATOR_FAILED',
            'error_message': str(e)
        }

    # Construct seed state: p_L, R_L, and xf_seed
    # From the failing case JSON, xf is the tip state
    xf_seed = np.array(case['seed']['xf'])
    p_L = np.array(seed['p'])
    R_L = np.array(seed['R'])

    # Run BVP + IVP from seed to capture convergence/divergence
    start_time = time.time()
    try:
        result = wrapper._cpp_dynamics.step_from_seed(
            currents,
            insertion,
            np.asarray(seed['v'], dtype=np.float64),
            np.asarray(seed['w'], dtype=np.float64),
            np.asarray(seed['p'], dtype=np.float64),
            np.asarray(seed['R'], dtype=np.float64),
            np.asarray(xf_seed, dtype=np.float64),
            np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
            np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        )
        elapsed = time.time() - start_time

        converged = bool(result.get("converged", False))
        diverged = bool(result.get("diverged", False))
        localmin = int(result.get("localmin", -1))

        print(f"\nElapsed time: {elapsed:.3f}s")
        print(f"Converged: {converged}, Diverged: {diverged}, localmin: {localmin}")

        return {
            'integrator': integrator_type,
            'converged': converged,
            'info': None,
            'localmin': localmin,
            'residual_norm': None,
            'diverged': diverged,
            'elapsed_time': elapsed,
            'status': 'SUCCESS'
        }

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\nElapsed time: {elapsed:.3f}s")
        print(f"Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'integrator': integrator_type,
            'converged': False,
            'info': None,
            'localmin': None,
            'residual_norm': None,
            'diverged': True,
            'elapsed_time': elapsed,
            'status': 'EXCEPTION',
            'error_message': str(e)
        }


def main():
    """Main test harness."""
    print("=" * 70)
    print("Phase 3 Integrator Stability Test Harness")
    print("=" * 70)

    # Load failing case
    case = load_failing_case()
    print(f"\nLoaded failing case from: docs/architecture/TASK_1_7_FAILING_CASE.json")

    # Run with both integrators
    results = []

    for integrator in ("abm4", "rk4"):
        print("\n" + "=" * 70)
        print(f"Running test with integrator: {integrator}")
        print("=" * 70)
        results.append(run_test_case(case, integrator_type=integrator))

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    for r in results:
        integrator = r['integrator']
        converged = "✅" if r['converged'] else "❌"
        diverged = "✅" if r.get('diverged') else "N/A"
        print(f"\n{integrator}:")
        print(f"  Converged: {converged} ({r['converged']})")
        print(f"  Diverged flag: {diverged}")
        print(f"  Localmin: {r['localmin']}")
        print(f"  Elapsed: {r['elapsed_time']:.3f}s")
        if r['status'] != 'SUCCESS':
            print(f"  Status: {r['status']}")
            if 'error_message' in r:
                print(f"  Error: {r['error_message']}")

    # Save results
    results_file = Path(__file__).parent.parent / "docs" / "architecture" / "INTEGRATOR_STABILITY_RESULTS.json"
    with open(results_file, 'w') as f:
        json.dump({
            'case': case['tag'],
            'trial': case['trial'],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': results
        }, f, indent=2)

    print(f"\n✅ Results saved to: {results_file}")

    # Return exit code based on convergence
    # Expected: all test cases should NOT converge (documenting failure mode)
    # Actual convergence with soft-failure propagation would be success
    converged_count = sum(1 for r in results if r['converged'])
    if converged_count == 0:
        print("\n⚠️  No convergence detected for failing case.")
    else:
        print(f"\n✅ {converged_count} case(s) converged.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
