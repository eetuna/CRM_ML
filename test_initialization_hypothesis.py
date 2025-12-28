"""
Hypothesis: C++ extension doesn't call initialize_from_kinematics()

The Python test calls:
  dyn.initialize_from_kinematics(np.array([0.0, 0.0, 0.2]), 94.3)

But the C++ code only calls:
  dyn.load_parameters(param_file, config_file)

Test if this is the issue!
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from crm_ml_rl.wrappers import crm_python

def test_with_and_without_initialization():
    print("\n" + "="*70)
    print("TEST: Does initialize_from_kinematics() matter?")
    print("="*70)

    param_file = "data/catheter_params/CatheterParameterSet_1_dyn.txt"
    config_file = "data/catheter_params/CatheterSpatialConfiguration_1.txt"

    # Get seed from initialized instance
    dyn_init = crm_python.CRMDynamics()
    dyn_init.load_parameters(param_file, config_file)
    dyn_init.initialize_from_kinematics(np.array([0.0, 0.0, 0.2]), 94.3)
    seed = dyn_init.get_seed_state()

    currents = np.array([0.01, 0.0, 0.0])
    insertion = 94.3

    # Test 1: WITH initialization
    print("\n[Test 1] WITH initialize_from_kinematics():")
    dyn_with = crm_python.CRMDynamics()
    dyn_with.load_parameters(param_file, config_file)
    dyn_with.initialize_from_kinematics(np.array([0.0, 0.0, 0.2]), 94.3)
    result_with = dyn_with.step_from_seed(
        currents, insertion,
        seed['v'], seed['w'], seed['p'], seed['R'], seed['xf'],
        seed['mL'], seed['nL']
    )
    output_with = np.concatenate([result_with['tip_position'], result_with['tip_velocity']])
    print(f"  Output: {output_with}")

    # Test 2: WITHOUT initialization (like C++ extension)
    print("\n[Test 2] WITHOUT initialize_from_kinematics() (like C++ extension):")
    dyn_without = crm_python.CRMDynamics()
    dyn_without.load_parameters(param_file, config_file)
    # NO initialize_from_kinematics() call!
    result_without = dyn_without.step_from_seed(
        currents, insertion,
        seed['v'], seed['w'], seed['p'], seed['R'], seed['xf'],
        seed['mL'], seed['nL']
    )
    output_without = np.concatenate([result_without['tip_position'], result_without['tip_velocity']])
    print(f"  Output: {output_without}")

    # Compare
    print("\n[Comparison]")
    diff = np.abs(output_with - output_without)
    print(f"  Difference: {diff}")
    print(f"  Max diff: {diff.max():.2e}")

    if diff.max() < 1e-10:
        print("\n✅ IDENTICAL - initialize_from_kinematics() doesn't matter for step_from_seed()")
    else:
        print("\n⚠️ DIFFERENT - initialize_from_kinematics() DOES affect step_from_seed()!")
        print("   But wait... step_from_seed() takes 'seed' as input, so it shouldn't")
        print("   need initialization. This suggests there's hidden state being used.")


if __name__ == "__main__":
    test_with_and_without_initialization()
