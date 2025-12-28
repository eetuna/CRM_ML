"""
Investigate: Why do batch results differ?

We know step_from_seed() is stateless, so reusing instances is fine.
But the batch test shows 2.26 difference. Let's trace exactly what's happening.
"""

import numpy as np
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(__file__))
from crm_ml_rl.wrappers import crm_python
import crm_torch

def test_batch_detailed():
    print("\n" + "="*70)
    print("DETAILED BATCH TEST: Trace exactly what's happening")
    print("="*70)

    param_file = "data/catheter_params/CatheterParameterSet_1_dyn.txt"
    config_file = "data/catheter_params/CatheterSpatialConfiguration_1.txt"

    # Initialize
    dyn_py = crm_python.CRMDynamics()
    dyn_py.load_parameters(param_file, config_file)
    dyn_py.initialize_from_kinematics(np.array([0.0, 0.0, 0.2]), 94.3)
    seed = dyn_py.get_seed_state()

    # Batch of 3 different currents
    batch_size = 3
    currents_batch = np.array([
        [0.01, 0.0, 0.0],
        [0.0, 0.01, 0.0],
        [0.0, 0.0, 0.01],
    ])

    print("\n[Phase 1] Python: Compute each sample separately with REUSED instance")
    outputs_py_reused = []
    for i in range(batch_size):
        print(f"\n  Sample {i}: currents = {currents_batch[i]}")
        result = dyn_py.step_from_seed(
            currents_batch[i], 94.3,
            seed['v'], seed['w'], seed['p'], seed['R'], seed['xf'],
            seed['mL'], seed['nL']
        )
        output = np.concatenate([result['tip_position'], result['tip_velocity']])
        outputs_py_reused.append(output)
        print(f"    Output: {output}")
    outputs_py_reused = np.array(outputs_py_reused)

    print("\n[Phase 2] Python: Compute each sample with FRESH instances")
    outputs_py_fresh = []
    for i in range(batch_size):
        print(f"\n  Sample {i}: currents = {currents_batch[i]}")
        # Create fresh instance
        dyn_fresh = crm_python.CRMDynamics()
        dyn_fresh.load_parameters(param_file, config_file)
        result = dyn_fresh.step_from_seed(
            currents_batch[i], 94.3,
            seed['v'], seed['w'], seed['p'], seed['R'], seed['xf'],
            seed['mL'], seed['nL']
        )
        output = np.concatenate([result['tip_position'], result['tip_velocity']])
        outputs_py_fresh.append(output)
        print(f"    Output: {output}")
    outputs_py_fresh = np.array(outputs_py_fresh)

    print("\n[Phase 3] C++: Compute batched")
    currents_t = torch.from_numpy(currents_batch).double()
    insertion_t = torch.tensor([94.3]).double()
    seed_v_t = torch.from_numpy(seed['v']).unsqueeze(0).repeat(batch_size, 1, 1).double()
    seed_w_t = torch.from_numpy(seed['w']).unsqueeze(0).repeat(batch_size, 1, 1).double()
    seed_p_t = torch.from_numpy(seed['p']).unsqueeze(0).repeat(batch_size, 1, 1).double()
    seed_R_t = torch.from_numpy(seed['R']).unsqueeze(0).repeat(batch_size, 1, 1).double()
    seed_xf_t = torch.from_numpy(seed['xf']).unsqueeze(0).repeat(batch_size, 1).double()
    seed_mL_t = torch.from_numpy(seed['mL']).unsqueeze(0).repeat(batch_size, 1, 1).double()
    seed_nL_t = torch.from_numpy(seed['nL']).unsqueeze(0).repeat(batch_size, 1, 1).double()

    outputs_cpp = crm_torch.dynamics_forward(
        currents_t, insertion_t,
        seed_v_t, seed_w_t, seed_p_t, seed_R_t, seed_xf_t, seed_mL_t, seed_nL_t,
        param_file, config_file
    )
    outputs_cpp_np = outputs_cpp.numpy()

    for i in range(batch_size):
        print(f"\n  Sample {i}: currents = {currents_batch[i]}")
        print(f"    Output: {outputs_cpp_np[i]}")

    # Comparisons
    print("\n" + "="*70)
    print("COMPARISON 1: Python Reused vs Python Fresh")
    print("="*70)
    diff1 = np.abs(outputs_py_reused - outputs_py_fresh)
    print(f"Max difference: {diff1.max():.2e}")
    if diff1.max() < 1e-15:
        print("✅ IDENTICAL - Reusing instances is fine")
    else:
        print("⚠️ DIFFERENT - Reusing instances causes divergence!")

    print("\n" + "="*70)
    print("COMPARISON 2: Python Reused vs C++")
    print("="*70)
    diff2 = np.abs(outputs_py_reused - outputs_cpp_np)
    print(f"Max difference: {diff2.max():.2e}")
    print(f"Mean difference: {diff2.mean():.2e}")
    for i in range(batch_size):
        sample_diff = np.abs(outputs_py_reused[i] - outputs_cpp_np[i])
        print(f"  Sample {i}: max diff = {sample_diff.max():.2e}")

    print("\n" + "="*70)
    print("COMPARISON 3: Python Fresh vs C++")
    print("="*70)
    diff3 = np.abs(outputs_py_fresh - outputs_cpp_np)
    print(f"Max difference: {diff3.max():.2e}")
    print(f"Mean difference: {diff3.mean():.2e}")
    for i in range(batch_size):
        sample_diff = np.abs(outputs_py_fresh[i] - outputs_cpp_np[i])
        print(f"  Sample {i}: max diff = {sample_diff.max():.2e}")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    if diff1.max() < 1e-15:
        print("✅ Python reused vs fresh: No difference")
        print("   => Reusing instances is NOT the problem")

    if diff2.max() < 1e-3:
        print("✅ Python vs C++ difference is small (<1e-3)")
    else:
        print("⚠️ Python vs C++ difference is large (>1e-3)")
        print("   => Need to investigate C++ implementation details")


if __name__ == "__main__":
    test_batch_detailed()
