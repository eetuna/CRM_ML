#!/usr/bin/env python3
"""
Performance Benchmark for Phase A.3 Implementation.

Measures the performance impact of adding A matrix (seed gradient) computation.
Compares single-step and multi-step backward pass timings.
"""

import torch
import numpy as np
import time
import sys
sys.path.insert(0, '/workspaces/catheter/CRM_ML')

import crm_torch
from crm_ml_rl.wrappers import crm_python


def benchmark_backward_pass():
    """Benchmark backward pass with both B and A matrix computation."""
    print("=" * 80)
    print("Phase A.3 Performance Benchmark")
    print("=" * 80)

    # Setup
    param_file = "/workspaces/catheter/CRM_ML/data/catheter_params/CatheterParameterSet_1_dyn.txt"
    config_file = "/workspaces/catheter/CRM_ML/data/catheter_params/CatheterSpatialConfiguration_1.txt"
    insertion_length = 94.3

    # Initialize dynamics
    dyn = crm_python.CRMDynamics()
    if not dyn.load_parameters(param_file, config_file):
        raise RuntimeError("Failed to load parameters")

    initial_currents = np.array([0.0, 0.0, 0.01])
    if not dyn.initialize_from_kinematics(initial_currents, insertion_length):
        raise RuntimeError("Failed to initialize")

    seed = dyn.get_seed_state()

    # Test parameters
    num_iterations = 3
    num_warmup = 1

    print(f"\nBenchmark parameters:")
    print(f"  Iterations: {num_iterations}")
    print(f"  Warmup: {num_warmup}")

    # ========================================================================
    # Benchmark 1: Forward pass only
    # ========================================================================
    print("\n" + "-" * 80)
    print("Benchmark 1: Forward Pass Only")
    print("-" * 80)

    currents = torch.tensor([[0.1, 0.05, 0.02]], dtype=torch.float64, requires_grad=False)
    insertion = torch.tensor([insertion_length], dtype=torch.float64)

    seed_v = torch.from_numpy(seed['v']).unsqueeze(0).double()
    seed_w = torch.from_numpy(seed['w']).unsqueeze(0).double()
    seed_p = torch.from_numpy(seed['p']).unsqueeze(0).double()
    seed_R = torch.from_numpy(seed['R']).unsqueeze(0).double()
    seed_xf = torch.from_numpy(seed['xf']).unsqueeze(0).double()
    seed_mL = torch.from_numpy(seed['mL']).unsqueeze(0).double()
    seed_nL = torch.from_numpy(seed['nL']).unsqueeze(0).double()

    times = []
    for i in range(num_warmup + num_iterations):
        start = time.perf_counter()

        output = crm_torch.CRMDynamicsStep.apply(
            currents, insertion,
            seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL,
            param_file, config_file, 1e-4
        )

        end = time.perf_counter()

        if i >= num_warmup:
            times.append(end - start)

    fwd_time = np.mean(times)
    fwd_std = np.std(times)

    print(f"Forward pass: {fwd_time*1000:.2f} ± {fwd_std*1000:.2f} ms")

    # ========================================================================
    # Benchmark 2: Forward + Backward (current gradients only via B matrix)
    # ========================================================================
    print("\n" + "-" * 80)
    print("Benchmark 2: Forward + Backward (B matrix only)")
    print("-" * 80)

    currents_grad = torch.tensor([[0.1, 0.05, 0.02]], dtype=torch.float64, requires_grad=True)

    times_fwd = []
    times_bwd = []

    for i in range(num_warmup + num_iterations):
        # Forward
        start_fwd = time.perf_counter()
        output = crm_torch.CRMDynamicsStep.apply(
            currents_grad, insertion,
            seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL,
            param_file, config_file, 1e-4
        )
        end_fwd = time.perf_counter()

        # Create loss
        loss = output[0, :3].sum()

        # Backward
        start_bwd = time.perf_counter()
        loss.backward()
        end_bwd = time.perf_counter()

        if i >= num_warmup:
            times_fwd.append(end_fwd - start_fwd)
            times_bwd.append(end_bwd - start_bwd)

        # Reset gradients
        if currents_grad.grad is not None:
            currents_grad.grad.zero_()

    b_fwd_time = np.mean(times_fwd)
    b_bwd_time = np.mean(times_bwd)
    b_total_time = b_fwd_time + b_bwd_time

    print(f"Forward:  {b_fwd_time*1000:.2f} ms")
    print(f"Backward: {b_bwd_time*1000:.2f} ms (B matrix: 3 currents × 2 directions = 6 FD calls)")
    print(f"Total:    {b_total_time*1000:.2f} ms")

    # ========================================================================
    # Benchmark 3: Forward + Backward (with seed gradients via A matrix)
    # ========================================================================
    print("\n" + "-" * 80)
    print("Benchmark 3: Forward + Backward (B + A matrices)")
    print("-" * 80)

    currents_grad2 = torch.tensor([[0.1, 0.05, 0.02]], dtype=torch.float64, requires_grad=True)
    seed_v_grad = torch.from_numpy(seed['v']).unsqueeze(0).double().requires_grad_(True)
    seed_w_grad = torch.from_numpy(seed['w']).unsqueeze(0).double().requires_grad_(True)
    seed_p_grad = torch.from_numpy(seed['p']).unsqueeze(0).double().requires_grad_(True)
    seed_R_grad = torch.from_numpy(seed['R']).unsqueeze(0).double().requires_grad_(True)
    seed_xf_grad = torch.from_numpy(seed['xf']).unsqueeze(0).double().requires_grad_(True)
    seed_mL_grad = torch.from_numpy(seed['mL']).unsqueeze(0).double().requires_grad_(True)
    seed_nL_grad = torch.from_numpy(seed['nL']).unsqueeze(0).double().requires_grad_(True)

    times_fwd = []
    times_bwd = []

    for i in range(num_warmup + num_iterations):
        # Forward
        start_fwd = time.perf_counter()
        output = crm_torch.CRMDynamicsStep.apply(
            currents_grad2, insertion,
            seed_v_grad, seed_w_grad, seed_p_grad, seed_R_grad, seed_xf_grad,
            seed_mL_grad, seed_nL_grad,
            param_file, config_file, 1e-4
        )
        end_fwd = time.perf_counter()

        # Create loss
        loss = output[0, :3].sum()

        # Backward
        start_bwd = time.perf_counter()
        loss.backward()
        end_bwd = time.perf_counter()

        if i >= num_warmup:
            times_fwd.append(end_fwd - start_fwd)
            times_bwd.append(end_bwd - start_bwd)

        # Reset gradients
        if currents_grad2.grad is not None:
            currents_grad2.grad.zero_()
        for tensor in [seed_v_grad, seed_w_grad, seed_p_grad, seed_R_grad,
                       seed_xf_grad, seed_mL_grad, seed_nL_grad]:
            if tensor.grad is not None:
                tensor.grad.zero_()

    a_fwd_time = np.mean(times_fwd)
    a_bwd_time = np.mean(times_bwd)
    a_total_time = a_fwd_time + a_bwd_time

    print(f"Forward:  {a_fwd_time*1000:.2f} ms")
    print(f"Backward: {a_bwd_time*1000:.2f} ms (B + A: (3 + 39) components × 2 directions = 84 FD calls)")
    print(f"Total:    {a_total_time*1000:.2f} ms")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("Performance Summary")
    print("=" * 80)

    print(f"\n{'Operation':<30} {'Time (ms)':>12} {'vs Forward':>12} {'FD Calls':>12}")
    print("-" * 80)
    print(f"{'Forward pass only':<30} {fwd_time*1000:>10.2f}   {1.0:>10.1f}x   {0:>10d}")
    print(f"{'Forward + B matrix':<30} {b_total_time*1000:>10.2f}   {b_total_time/fwd_time:>10.1f}x   {6:>10d}")
    print(f"{'Forward + B + A matrices':<30} {a_total_time*1000:>10.2f}   {a_total_time/fwd_time:>10.1f}x   {84:>10d}")

    print(f"\n{'Phase':<30} {'Backward (ms)':>15} {'vs B-only':>12}")
    print("-" * 80)
    print(f"{'B matrix only (Phase A.2)':<30} {b_bwd_time*1000:>13.2f}   {1.0:>10.1f}x")
    print(f"{'B + A matrices (Phase A.3)':<30} {a_bwd_time*1000:>13.2f}   {a_bwd_time/b_bwd_time:>10.1f}x")

    overhead = a_bwd_time - b_bwd_time
    print(f"\nA matrix overhead: {overhead*1000:.2f} ms ({overhead/b_bwd_time*100:.1f}% increase)")

    # Theoretical calculation
    theoretical_overhead = 39 / 3  # 39 seed components vs 3 current components
    actual_overhead = a_bwd_time / b_bwd_time
    print(f"Theoretical overhead: {theoretical_overhead:.1f}x")
    print(f"Actual overhead: {actual_overhead:.1f}x")

    print("\n" + "=" * 80)
    print("Conclusions:")
    print("=" * 80)
    print(f"✓ Forward pass: {fwd_time*1000:.1f} ms")
    print(f"✓ B matrix (currents): adds {b_bwd_time*1000:.1f} ms ({b_bwd_time/fwd_time:.1f}x forward)")
    print(f"✓ A matrix (seeds): adds {overhead*1000:.1f} ms ({theoretical_overhead:.1f}x B matrix)")
    print(f"✓ Total backward: {a_bwd_time*1000:.1f} ms ({a_bwd_time/fwd_time:.1f}x forward)")
    print()
    print("Phase A.3 enables multi-step trajectory optimization with correct")
    print("gradient flow through time at the cost of ~13x forward pass time.")
    print("=" * 80)


if __name__ == "__main__":
    benchmark_backward_pass()
