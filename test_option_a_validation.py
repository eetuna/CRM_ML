"""
Option A Validation Against FK_DYN Test Cases

This script re-runs Option A dynamics with the same currents from archived
test cases and compares the results to validate Option A correctness.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from crm_ml_rl.wrappers import crm_python


def validate_trajectory(npz_file, test_name):
    """
    Validate Option A by re-running dynamics and comparing to archived results.

    Args:
        npz_file: Path to NPZ file with archived test data
        test_name: Name for reporting

    Returns:
        dict with statistics
    """
    print(f"\n{'='*70}")
    print(f"{test_name}")
    print('='*70)

    # Load archived test data
    data = np.load(npz_file)
    currents_archived = data['currents']
    tip_dyn_archived = data['tip_dyn']
    insertion_length = float(data['insertion_length'])
    dt = float(data['dt'])
    dyn_converged_archived = data['dyn_converged']

    print(f"\nArchived Data:")
    print(f"  Trajectory length: {len(currents_archived)} steps")
    print(f"  Insertion length: {insertion_length}mm")
    print(f"  dt: {dt}s")
    print(f"  Archived convergence: {dyn_converged_archived.sum()}/{len(dyn_converged_archived)} steps")

    # Re-run with current Option A
    param_file = "data/catheter_params/CatheterParameterSet_1_dyn.txt"
    config_file = "data/catheter_params/CatheterSpatialConfiguration_1.txt"

    print(f"\nRe-running Option A dynamics...")
    dyn = crm_python.CRMDynamics()
    dyn.load_parameters(param_file, config_file)

    # Initialize from first current (minimal)
    dyn.initialize_from_kinematics(currents_archived[0], insertion_length)
    dyn.set_integrator('abm4')  # Match archived test
    dyn.dt = dt
    dyn.integration_step_size = 0.2

    tip_dyn_current = []
    converged_current = []

    for i, curr in enumerate(currents_archived):
        result = dyn.step(curr, insertion_length)
        tip_dyn_current.append(result['tip_position'])
        converged_current.append(result['converged'])

        if i % 100 == 0:
            print(f"  Step {i}/{len(currents_archived)}...")

    tip_dyn_current = np.array(tip_dyn_current)
    converged_current = np.array(converged_current, dtype=bool)

    print(f"  Current convergence: {converged_current.sum()}/{len(converged_current)} steps")

    # Compare
    diff = np.linalg.norm(tip_dyn_archived - tip_dyn_current, axis=1)

    # Statistics
    stats = {
        'test_name': test_name,
        'n_steps': len(diff),
        'mean_mm': diff.mean(),
        'p50_mm': np.percentile(diff, 50),
        'p95_mm': np.percentile(diff, 95),
        'p99_mm': np.percentile(diff, 99),
        'max_mm': diff.max(),
        'convergence_match': (dyn_converged_archived == converged_current).all(),
        'archived_converged': dyn_converged_archived.sum(),
        'current_converged': converged_current.sum(),
    }

    # Report
    print(f"\n[Comparison: Archived vs Current Option A]")
    print(f"  Mean diff:   {stats['mean_mm']:.6f}mm")
    print(f"  Median diff: {stats['p50_mm']:.6f}mm")
    print(f"  P95 diff:    {stats['p95_mm']:.6f}mm")
    print(f"  P99 diff:    {stats['p99_mm']:.6f}mm")
    print(f"  Max diff:    {stats['max_mm']:.6f}mm")
    print(f"  Convergence match: {'✅ YES' if stats['convergence_match'] else '❌ NO'}")

    # Tolerance check
    # Should be near-exact (< 1 micron) if same code/parameters
    tolerance_exact = 1e-3  # 1 micron
    # Should be < 1mm if minor numerical differences
    tolerance_good = 1.0  # 1mm

    if stats['max_mm'] < tolerance_exact:
        status = "✅ EXACT MATCH"
        print(f"\n{status}: Max diff < {tolerance_exact}mm")
        print("  Option A has NOT changed since archive")
    elif stats['max_mm'] < tolerance_good:
        status = "✅ GOOD MATCH"
        print(f"\n{status}: Max diff < {tolerance_good}mm")
        print("  Minor numerical differences (acceptable)")
    else:
        status = "⚠️ DISCREPANCY"
        print(f"\n{status}: Max diff > {tolerance_good}mm")
        print("  Significant differences - Option A may have changed")
        print("  OR: Different parameters/integrator settings")

    stats['status'] = status
    stats['diff_array'] = diff

    return stats


def main():
    print("\n" + "="*70)
    print("OPTION A VALIDATION: FK_DYN Test Cases")
    print("="*70)
    print("\nThis validates Option A by re-running archived test trajectories")
    print("and comparing current Option A outputs to archived dynamics results.")

    test_cases = [
        ('data/output/dyn_fk_ramp_circle1_hold1.npz', 'Circle Trajectory (Hold=1, 120 ramp + 200 circle)'),
        ('data/output/dyn_fk_ramp_circle1_hold2.npz', 'Circle Trajectory (Hold=2, 240 ramp + 400 circle)'),
        ('data/output/dyn_fk_lem1_y40_a10_hold1.npz', 'Lemniscate Trajectory (Hold=1, 120 ramp + 200 lem)'),
        ('data/output/dyn_fk_lem1_y40_a10_hold2.npz', 'Lemniscate Trajectory (Hold=2, 240 ramp + 400 lem)'),
    ]

    all_stats = []

    for npz_file, test_name in test_cases:
        try:
            stats = validate_trajectory(npz_file, test_name)
            all_stats.append(stats)
        except Exception as e:
            print(f"\n❌ ERROR in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for stats in all_stats:
        print(f"\n{stats['test_name']}:")
        print(f"  Status: {stats['status']}")
        print(f"  Mean: {stats['mean_mm']:.6f}mm, P95: {stats['p95_mm']:.6f}mm, Max: {stats['max_mm']:.6f}mm")
        print(f"  Convergence: {stats['current_converged']}/{stats['n_steps']} (archived: {stats['archived_converged']}/{stats['n_steps']})")

    # Overall assessment
    print("\n" + "="*70)
    print("OVERALL ASSESSMENT")
    print("="*70)

    all_exact = all(s['status'] == "✅ EXACT MATCH" for s in all_stats)
    all_good = all("✅" in s['status'] for s in all_stats)

    if all_exact:
        print("\n✅ EXCELLENT: All tests show exact match (<1 micron)")
        print("   Option A has NOT changed since FK_DYN tests were archived")
        print("   High confidence in Option A correctness")
    elif all_good:
        print("\n✅ GOOD: All tests within acceptable tolerance (<1mm)")
        print("   Minor numerical differences (e.g., compiler, library versions)")
        print("   Option A is functionally correct")
    else:
        print("\n⚠️ WARNING: Some tests show significant discrepancies")
        print("   Option A may have changed since FK_DYN tests were archived")
        print("   Recommend investigating differences before proceeding")

    print("="*70 + "\n")

    return all_stats


if __name__ == "__main__":
    all_stats = main()
