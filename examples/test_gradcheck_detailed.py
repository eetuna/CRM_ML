#!/usr/bin/env python3
"""
Detailed gradcheck analysis for Phase 4 Task 4.2.

Quantifies the relative error between AD gradients and FD approximations
to determine if error < 10^-5 target is met.
"""

import os
import numpy as np
import torch
from torch.autograd import gradcheck

# Enable implicit linearization
os.environ["CRM_DYN_LINEARIZATION_METHOD"] = "implicit"

from crm_ml_rl.wrappers.torch_physics import TorchCRMPhysics


def test_gradcheck_currents(eps=1e-4, atol=1e-5, rtol=1e-5):
    """
    Test gradients w.r.t. currents with specified tolerances.
    """
    print(f"\n{'='*70}")
    print(f"Testing Gradient Accuracy w.r.t. Currents")
    print(f"{'='*70}")
    print(f"FD epsilon: {eps:.0e}")
    print(f"Absolute tolerance: {atol:.0e}")
    print(f"Relative tolerance: {rtol:.0e}")

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

    # Small perturbation for stable gradients
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

    result = gradcheck(
        dyn_wrapper,
        (currents,),
        eps=eps,
        atol=atol,
        rtol=rtol,
        raise_exception=False,
    )

    status = "✅ PASS" if result else "❌ FAIL"
    print(f"Result: {status}")
    print(f"Gradcheck passed: {result}")

    return result


def test_gradcheck_seed_state(eps=1e-4, atol=1e-5, rtol=1e-5):
    """
    Test gradients w.r.t. seed state with specified tolerances.
    """
    print(f"\n{'='*70}")
    print(f"Testing Gradient Accuracy w.r.t. Seed State (v)")
    print(f"{'='*70}")
    print(f"FD epsilon: {eps:.0e}")
    print(f"Absolute tolerance: {atol:.0e}")
    print(f"Relative tolerance: {rtol:.0e}")

    physics = TorchCRMPhysics(
        param_file="data/catheter_params/CatheterParameterSet_1_dyn.txt",
        config_file="data/catheter_params/CatheterSpatialConfiguration_1.txt",
        device="cpu",
    )

    # Initialize dynamics state
    currents0_np = [0.0, 0.0, 0.0]
    physics.dyn.initialize_from_kinematics(currents0_np, 50.0)
    seed = physics.dyn.get_seed_state()

    # Test gradient w.r.t. seed_v (linear velocity)
    seed_v = torch.tensor(seed["v"], dtype=torch.float64).unsqueeze(0).requires_grad_()
    seed_w = torch.tensor(seed["w"], dtype=torch.float64).unsqueeze(0)
    seed_p = torch.tensor(seed["p"], dtype=torch.float64).unsqueeze(0)
    seed_R = torch.tensor(seed["R"], dtype=torch.float64).unsqueeze(0)
    seed_xf = torch.tensor(seed["xf"], dtype=torch.float64).unsqueeze(0)
    seed_mL = torch.tensor(seed["mL"], dtype=torch.float64).unsqueeze(0)
    seed_nL = torch.tensor(seed["nL"], dtype=torch.float64).unsqueeze(0)

    currents = torch.tensor([[0.01, 0.0, 0.0]], dtype=torch.float64)
    insertion = torch.tensor([50.0], dtype=torch.float64)

    def dyn_wrapper(v):
        return physics.dyn_step(
            currents,
            insertion,
            v,
            seed_w,
            seed_p,
            seed_R,
            seed_xf,
            seed_mL=seed_mL,
            seed_nL=seed_nL,
            eps_u=1e-4,
            eps_seed=1e-4,
        )

    result = gradcheck(
        dyn_wrapper,
        (seed_v,),
        eps=eps,
        atol=atol,
        rtol=rtol,
        raise_exception=False,
    )

    status = "✅ PASS" if result else "❌ FAIL"
    print(f"Result: {status}")
    print(f"Gradcheck passed: {result}")

    return result


def main():
    print("\n" + "="*70)
    print("PHASE 4 TASK 4.2: Gradient Verification")
    print("="*70)
    print("\nTarget: Relative error < 10^-5")
    print("\nTesting with various tolerance levels...")

    # Test current gradients with different tolerances
    print("\n" + "="*70)
    print("Test 1: Currents Gradient - Strict Tolerance (target)")
    print("="*70)
    result1 = test_gradcheck_currents(eps=1e-5, atol=1e-5, rtol=1e-5)

    if not result1:
        print("\n" + "="*70)
        print("Test 2: Currents Gradient - Relaxed Tolerance")
        print("="*70)
        result2 = test_gradcheck_currents(eps=1e-4, atol=1e-3, rtol=1e-3)

    # Test seed state gradients
    print("\n" + "="*70)
    print("Test 3: Seed State Gradient - Strict Tolerance (target)")
    print("="*70)
    result3 = test_gradcheck_seed_state(eps=1e-5, atol=1e-5, rtol=1e-5)

    if not result3:
        print("\n" + "="*70)
        print("Test 4: Seed State Gradient - Relaxed Tolerance")
        print("="*70)
        result4 = test_gradcheck_seed_state(eps=1e-4, atol=1e-2, rtol=1e-2)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Currents gradient (strict 1e-5 tolerance): {'PASS' if result1 else 'FAIL'}")
    print(f"Seed state gradient (strict 1e-5 tolerance): {'PASS' if result3 else 'FAIL'}")

    if result1 and result3:
        print("\n✅ SUCCESS: All gradients meet < 10^-5 error target!")
    else:
        print("\n⚠️  NEEDS TUNING: Gradients computed but precision needs improvement")
        print("\nPossible causes:")
        print("  1. IVALUE_SCALE_M/N mismatch between forward and AD paths")
        print("  2. System too stiff for chosen FD epsilon")
        print("  3. Need higher precision in AD implementation")


if __name__ == "__main__":
    main()
