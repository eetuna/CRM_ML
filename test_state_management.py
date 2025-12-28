"""
Quick test: Is step_from_seed() stateful?

Test if calling step_from_seed() multiple times with the same inputs
produces the same output, or if internal state accumulates.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from crm_ml_rl.wrappers import crm_python

def test_stateful_behavior():
    print("\n" + "="*70)
    print("TEST: Is step_from_seed() stateful?")
    print("="*70)

    param_file = "data/catheter_params/CatheterParameterSet_1_dyn.txt"
    config_file = "data/catheter_params/CatheterSpatialConfiguration_1.txt"

    # Create ONE instance
    dyn = crm_python.CRMDynamics()
    dyn.load_parameters(param_file, config_file)
    dyn.initialize_from_kinematics(np.array([0.0, 0.0, 0.2]), 94.3)
    seed = dyn.get_seed_state()

    currents = np.array([0.01, 0.0, 0.0])
    insertion = 94.3

    # Call step_from_seed THREE TIMES with SAME inputs
    print("\nCalling step_from_seed() 3 times with SAME inputs...")
    outputs = []
    for i in range(3):
        result = dyn.step_from_seed(
            currents, insertion,
            seed['v'], seed['w'], seed['p'], seed['R'], seed['xf'],
            seed['mL'], seed['nL']
        )
        output = np.concatenate([result['tip_position'], result['tip_velocity']])
        outputs.append(output)
        print(f"  Call {i+1}: {output}")

    # Check if all outputs are identical
    print("\nChecking if outputs are identical...")
    for i in range(1, 3):
        diff = np.abs(outputs[i] - outputs[0])
        print(f"  Call {i+1} vs Call 1: max diff = {diff.max():.2e}")

    all_identical = all(np.allclose(outputs[i], outputs[0], atol=1e-15) for i in range(1, 3))

    if all_identical:
        print("\n✅ STATELESS: All outputs identical (step_from_seed does NOT accumulate state)")
    else:
        print("\n⚠️ STATEFUL: Outputs differ (step_from_seed DOES accumulate state)")
        print("   This means step_from_seed() modifies internal state even though")
        print("   it takes 'seed' as input parameter!")

    return all_identical


def test_fresh_instances():
    print("\n" + "="*70)
    print("TEST: Do fresh instances produce same output?")
    print("="*70)

    param_file = "data/catheter_params/CatheterParameterSet_1_dyn.txt"
    config_file = "data/catheter_params/CatheterSpatialConfiguration_1.txt"

    # Get seed state from first instance
    dyn0 = crm_python.CRMDynamics()
    dyn0.load_parameters(param_file, config_file)
    dyn0.initialize_from_kinematics(np.array([0.0, 0.0, 0.2]), 94.3)
    seed = dyn0.get_seed_state()

    currents = np.array([0.01, 0.0, 0.0])
    insertion = 94.3

    # Create THREE SEPARATE instances and call step_from_seed once each
    print("\nCreating 3 fresh instances, calling step_from_seed once each...")
    outputs = []
    for i in range(3):
        dyn = crm_python.CRMDynamics()
        dyn.load_parameters(param_file, config_file)
        result = dyn.step_from_seed(
            currents, insertion,
            seed['v'], seed['w'], seed['p'], seed['R'], seed['xf'],
            seed['mL'], seed['nL']
        )
        output = np.concatenate([result['tip_position'], result['tip_velocity']])
        outputs.append(output)
        print(f"  Instance {i+1}: {output}")

    # Check if all outputs are identical
    print("\nChecking if outputs are identical...")
    for i in range(1, 3):
        diff = np.abs(outputs[i] - outputs[0])
        print(f"  Instance {i+1} vs Instance 1: max diff = {diff.max():.2e}")

    all_identical = all(np.allclose(outputs[i], outputs[0], atol=1e-15) for i in range(1, 3))

    if all_identical:
        print("\n✅ All fresh instances produce identical output")
    else:
        print("\n⚠️ Fresh instances produce different outputs!")

    return all_identical


if __name__ == "__main__":
    print("\n" + "="*70)
    print("CRMDynamics State Management Investigation")
    print("="*70)

    stateless = test_stateful_behavior()
    fresh_identical = test_fresh_instances()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  step_from_seed() stateless: {'✅ YES' if stateless else '❌ NO (accumulates state)'}")
    print(f"  Fresh instances identical:  {'✅ YES' if fresh_identical else '❌ NO'}")
    print("="*70)

    if not stateless:
        print("\n💡 INSIGHT:")
        print("  The C++ extension reuses one instance for all batch elements.")
        print("  The Python test reuses one instance for all batch elements.")
        print("  BUT if step_from_seed() is stateful, this causes divergence!")
        print("\n💡 SOLUTION:")
        print("  Either:")
        print("  1. Create fresh CRMDynamics() instance per batch element (C++)")
        print("  2. Create fresh instance per batch element in Python test")
        print("  Both should then match!")
    print("="*70 + "\n")
