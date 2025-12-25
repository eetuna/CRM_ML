#!/usr/bin/env python3
"""
FD Epsilon Sensitivity Check for Phase 4 Task 4.6.

Tests gradcheck with varying eps values: 1e-4, 1e-5, 1e-6
If gradients change significantly → system too stiff for FD

Also investigates IVALUE_SCALE_M/N application in Jacobian computation.
"""

import os
import numpy as np
import torch
from torch.autograd import gradcheck

# Enable implicit linearization
os.environ["CRM_DYN_LINEARIZATION_METHOD"] = "implicit"

from crm_ml_rl.wrappers.torch_physics import TorchCRMPhysics
from crm_ml_rl.wrappers import crm_python


def test_fd_epsilon_gradcheck(eps_values=[1e-4, 1e-5, 1e-6]):
    """
    Test gradcheck with varying FD epsilon values.
    """
    print(f"\n{'='*70}")
    print("FD EPSILON SENSITIVITY - GRADCHECK")
    print(f"{'='*70}")

    physics = TorchCRMPhysics(
        param_file="data/catheter_params/CatheterParameterSet_1_dyn.txt",
        config_file="data/catheter_params/CatheterSpatialConfiguration_1.txt",
        device="cpu",
    )

    # Initialize dynamics state
    currents0_np = [0.0, 0.0, 0.0]
    physics.dyn.initialize_from_kinematics(currents0_np, 50.0)
    seed = physics.dyn.get_seed_state()

    # Convert seed to tensors
    seed_v = torch.tensor(seed["v"], dtype=torch.float64).unsqueeze(0)
    seed_w = torch.tensor(seed["w"], dtype=torch.float64).unsqueeze(0)
    seed_p = torch.tensor(seed["p"], dtype=torch.float64).unsqueeze(0)
    seed_R = torch.tensor(seed["R"], dtype=torch.float64).unsqueeze(0)
    seed_xf = torch.tensor(seed["xf"], dtype=torch.float64).unsqueeze(0)
    seed_mL = torch.tensor(seed["mL"], dtype=torch.float64).unsqueeze(0)
    seed_nL = torch.tensor(seed["nL"], dtype=torch.float64).unsqueeze(0)
    insertion = torch.tensor([50.0], dtype=torch.float64)

    torch.manual_seed(42)
    currents = torch.randn(1, 3, dtype=torch.float64, requires_grad=True) * 0.01

    def dyn_wrapper(c):
        return physics.dyn_step(
            c,
            insertion,
            seed_v,
            seed_w,
            seed_p,
            seed_R,
            seed_xf,
            seed_mL=seed_mL,
            seed_nL=seed_nL,
            eps_u=1e-4,
            eps_seed=1e-4,
        )

    results = []
    for eps in eps_values:
        print(f"\nTesting with eps = {eps:.0e}...")
        result = gradcheck(
            dyn_wrapper,
            (currents,),
            eps=eps,
            atol=1e-3,
            rtol=1e-3,
            raise_exception=False,
        )
        results.append((eps, result))
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  Result: {status}")

    print(f"\n{'='*70}")
    print("FD EPSILON SENSITIVITY SUMMARY")
    print(f"{'='*70}")
    for eps, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"eps = {eps:.0e}: {status}")

    # Check if results are inconsistent
    pass_count = sum(1 for _, p in results if p)
    if pass_count == 0:
        print("\n⚠️  All epsilon values failed - system may be too stiff for FD")
    elif pass_count == len(results):
        print("\n✅ All epsilon values passed - gradients are consistent")
    else:
        print(f"\n⚠️  Inconsistent results ({pass_count}/{len(results)} passed) - suggests epsilon sensitivity")

    return results


def investigate_jacobian_scaling():
    """
    Investigate if IVALUE_SCALE_M/N is correctly applied in Jacobian computation.
    """
    print(f"\n{'='*70}")
    print("JACOBIAN SCALING INVESTIGATION")
    print(f"{'='*70}")
    print("\nChecking if IVALUE_SCALE_M/N affects Jacobian discrepancy...")

    dyn = crm_python.CRMDynamics()
    ok = dyn.load_parameters(
        "data/catheter_params/CatheterParameterSet_1_dyn.txt",
        "data/catheter_params/CatheterSpatialConfiguration_1.txt",
    )
    if not ok:
        raise RuntimeError("Failed to load parameters")

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

    insertion_length = 94.3
    dyn.initialize_from_kinematics([0.0, 0.0, 0.2], insertion_length)
    seed = dyn.get_seed_state()
    currents = np.array([0.01, 0.0, 0.0], dtype=np.float64)

    # Test with different epsilon values to see scaling effects
    epsilon_values = [1e-4, 1e-5, 1e-6]

    print(f"\nComparing Jxx (AD) vs Jxx_fd for different FD epsilon values:")
    print(f"IVALUE_SCALE_M = IVALUE_SCALE_N = 10000.0")

    for eps in epsilon_values:
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
            eps_residual_x=eps,
            eps_residual_theta=eps,
            eps_g_x=eps,
            eps_g_theta=eps,
            return_debug=True
        )

        Jxx_ad = np.array(result['Jxx'])
        Jxx_fd = np.array(result['Jxx_fd'])

        abs_diff = np.linalg.norm(Jxx_ad - Jxx_fd)
        norm_fd = np.linalg.norm(Jxx_fd)
        rel_err = abs_diff / norm_fd if norm_fd > 0 else float('inf')

        print(f"\neps = {eps:.0e}:")
        print(f"  ||Jxx_ad||:          {np.linalg.norm(Jxx_ad):.6e}")
        print(f"  ||Jxx_fd||:          {norm_fd:.6e}")
        print(f"  ||Jxx_ad - Jxx_fd||: {abs_diff:.6e}")
        print(f"  Relative error:      {rel_err:.6e}")

        # Check if error changes significantly with epsilon
        if eps == epsilon_values[0]:
            first_rel_err = rel_err
        else:
            ratio = rel_err / first_rel_err if first_rel_err > 0 else 0
            print(f"  Error ratio vs eps={epsilon_values[0]:.0e}: {ratio:.2f}x")

    # Check scaling in mL and nL values
    print(f"\n{'='*70}")
    print("SEED STATE SCALING CHECK")
    print(f"{'='*70}")
    mL = seed['mL'][0]  # First actuator set
    nL = seed['nL'][0]
    print(f"\nSeed state forces (physical units):")
    print(f"  mL: {mL}")
    print(f"  nL: {nL}")
    print(f"\nNote: These are used in scaled form (×{10000.0}) in BVP solver")
    print(f"  Scaled mL: {mL * 10000.0}")
    print(f"  Scaled nL: {nL * 10000.0}")


def main():
    print("\n" + "="*70)
    print("PHASE 4 TASK 4.6: FD Epsilon Sensitivity Check")
    print("="*70)
    print("\nTesting gradient sensitivity to FD epsilon")
    print("Investigating IVALUE_SCALE_M/N impact on Jacobian computation")

    # Test 1: Gradcheck with different epsilon values
    gradcheck_results = test_fd_epsilon_gradcheck()

    # Test 2: Investigate scaling impact
    investigate_jacobian_scaling()

    print("\n" + "="*70)
    print("TASK 4.6 SUMMARY")
    print("="*70)

    print("\nKey findings will help diagnose the AD vs FD discrepancy found in Task 4.5")
    print("If errors change significantly with epsilon → system too stiff for FD")
    print("If errors remain constant → likely AD implementation issue")

    return 0


if __name__ == "__main__":
    exit(main())
