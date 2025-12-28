"""
Test: Check if C++ is passing data correctly to step_from_seed()

Hypothesis: Maybe there's a precision loss or data corruption when
converting from torch tensors -> C++ -> numpy -> Python?
"""

import numpy as np
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(__file__))
from crm_ml_rl.wrappers import crm_python
import crm_torch

def test_single_sample_detailed():
    """
    Compare Python vs C++ for a single sample with detailed output.
    """
    print("\n" + "="*70)
    print("SINGLE SAMPLE: Detailed comparison")
    print("="*70)

    param_file = "data/catheter_params/CatheterParameterSet_1_dyn.txt"
    config_file = "data/catheter_params/CatheterSpatialConfiguration_1.txt"

    # Initialize and get seed
    dyn_py = crm_python.CRMDynamics()
    dyn_py.load_parameters(param_file, config_file)
    dyn_py.initialize_from_kinematics(np.array([0.0, 0.0, 0.2]), 94.3)
    seed = dyn_py.get_seed_state()

    currents_np = np.array([0.01, 0.0, 0.0])
    insertion = 94.3

    # Python call
    print("\n[Python] Calling step_from_seed...")
    print(f"  currents: {currents_np}")
    print(f"  insertion: {insertion}")
    print(f"  seed['v'] shape: {seed['v'].shape}, dtype: {seed['v'].dtype}")
    result_py = dyn_py.step_from_seed(
        currents_np, insertion,
        seed['v'], seed['w'], seed['p'], seed['R'], seed['xf'],
        seed['mL'], seed['nL']
    )
    output_py = np.concatenate([result_py['tip_position'], result_py['tip_velocity']])
    print(f"  Output: {output_py}")

    # C++ call
    print("\n[C++] Calling dynamics_forward...")
    currents_t = torch.from_numpy(currents_np).unsqueeze(0).double()
    insertion_t = torch.tensor([insertion]).double()
    seed_v_t = torch.from_numpy(seed['v']).unsqueeze(0).double()
    seed_w_t = torch.from_numpy(seed['w']).unsqueeze(0).double()
    seed_p_t = torch.from_numpy(seed['p']).unsqueeze(0).double()
    seed_R_t = torch.from_numpy(seed['R']).unsqueeze(0).double()
    seed_xf_t = torch.from_numpy(seed['xf']).unsqueeze(0).double()
    seed_mL_t = torch.from_numpy(seed['mL']).unsqueeze(0).double()
    seed_nL_t = torch.from_numpy(seed['nL']).unsqueeze(0).double()

    print(f"  currents_t: {currents_t}")
    print(f"  seed_v_t shape: {seed_v_t.shape}, dtype: {seed_v_t.dtype}")

    output_cpp = crm_torch.dynamics_forward(
        currents_t, insertion_t,
        seed_v_t, seed_w_t, seed_p_t, seed_R_t, seed_xf_t, seed_mL_t, seed_nL_t,
        param_file, config_file
    )
    output_cpp_np = output_cpp[0].numpy()
    print(f"  Output: {output_cpp_np}")

    # Detailed comparison
    print("\n[Detailed Comparison]")
    diff = np.abs(output_py - output_cpp_np)
    for i, (py_val, cpp_val, d) in enumerate(zip(output_py, output_cpp_np, diff)):
        print(f"  [{i}] Python: {py_val:15.8f}, C++: {cpp_val:15.8f}, Diff: {d:.2e}")

    print(f"\n  Max absolute difference: {diff.max():.2e}")
    return diff.max()


def test_third_sample():
    """
    Test sample 2 (third sample, currents=[0, 0, 0.01]) which has the largest error.
    """
    print("\n" + "="*70)
    print("THIRD SAMPLE: currents = [0, 0, 0.01] (largest error)")
    print("="*70)

    param_file = "data/catheter_params/CatheterParameterSet_1_dyn.txt"
    config_file = "data/catheter_params/CatheterSpatialConfiguration_1.txt"

    # Initialize and get seed
    dyn_py = crm_python.CRMDynamics()
    dyn_py.load_parameters(param_file, config_file)
    dyn_py.initialize_from_kinematics(np.array([0.0, 0.0, 0.2]), 94.3)
    seed = dyn_py.get_seed_state()

    currents_np = np.array([0.0, 0.0, 0.01])
    insertion = 94.3

    # Python call
    print("\n[Python]")
    result_py = dyn_py.step_from_seed(
        currents_np, insertion,
        seed['v'], seed['w'], seed['p'], seed['R'], seed['xf'],
        seed['mL'], seed['nL']
    )
    output_py = np.concatenate([result_py['tip_position'], result_py['tip_velocity']])
    print(f"  Output: {output_py}")

    # C++ call
    print("\n[C++]")
    currents_t = torch.from_numpy(currents_np).unsqueeze(0).double()
    insertion_t = torch.tensor([insertion]).double()
    seed_v_t = torch.from_numpy(seed['v']).unsqueeze(0).double()
    seed_w_t = torch.from_numpy(seed['w']).unsqueeze(0).double()
    seed_p_t = torch.from_numpy(seed['p']).unsqueeze(0).double()
    seed_R_t = torch.from_numpy(seed['R']).unsqueeze(0).double()
    seed_xf_t = torch.from_numpy(seed['xf']).unsqueeze(0).double()
    seed_mL_t = torch.from_numpy(seed['mL']).unsqueeze(0).double()
    seed_nL_t = torch.from_numpy(seed['nL']).unsqueeze(0).double()

    output_cpp = crm_torch.dynamics_forward(
        currents_t, insertion_t,
        seed_v_t, seed_w_t, seed_p_t, seed_R_t, seed_xf_t, seed_mL_t, seed_nL_t,
        param_file, config_file
    )
    output_cpp_np = output_cpp[0].numpy()
    print(f"  Output: {output_cpp_np}")

    # Detailed comparison
    print("\n[Detailed Comparison]")
    diff = np.abs(output_py - output_cpp_np)
    for i, (py_val, cpp_val, d) in enumerate(zip(output_py, output_cpp_np, diff)):
        print(f"  [{i}] Python: {py_val:15.8f}, C++: {cpp_val:15.8f}, Diff: {d:.2e}")

    print(f"\n  Max absolute difference: {diff.max():.2e}")
    return diff.max()


if __name__ == "__main__":
    diff1 = test_single_sample_detailed()
    diff2 = test_third_sample()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Sample 0 (currents=[0.01, 0, 0]):  max diff = {diff1:.2e}")
    print(f"  Sample 2 (currents=[0, 0, 0.01]): max diff = {diff2:.2e}")
    print("="*70)
