"""
Performance benchmark: Option C vs Option A.

Compares performance of:
1. Option C: PyTorch extension (forward + backward)
2. Option A: Direct Python calls to implicit linearization

Measures:
- Forward pass time
- Backward pass time
- Total time (forward + backward)
- Overhead from extension
"""

import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import crm_torch
from crm_ml_rl.wrappers import crm_python


def setup_test_case(batch_size=1):
    """Setup test case with specified batch size."""
    param_file = "data/catheter_params/CatheterParameterSet_1_dyn.txt"
    config_file = "data/catheter_params/CatheterSpatialConfiguration_1.txt"

    # Initialize dynamics
    dyn_py = crm_python.CRMDynamics()
    dyn_py.load_parameters(param_file, config_file)
    dyn_py.initialize_from_kinematics(np.array([0.0, 0.0, 0.01]), 94.3)
    seed = dyn_py.get_seed_state()

    # Create batch inputs
    currents_np = np.tile([0.01, 0.0, 0.0], (batch_size, 1))
    insertion_np = np.full(batch_size, 94.3)

    # Convert to torch tensors
    currents = torch.from_numpy(currents_np).double().requires_grad_(True)
    insertion = torch.from_numpy(insertion_np).double()

    # Replicate seed for batch
    seed_v = torch.from_numpy(seed['v']).unsqueeze(0).repeat(batch_size, 1, 1).double()
    seed_w = torch.from_numpy(seed['w']).unsqueeze(0).repeat(batch_size, 1, 1).double()
    seed_p = torch.from_numpy(seed['p']).unsqueeze(0).repeat(batch_size, 1, 1).double()
    seed_R = torch.from_numpy(seed['R']).unsqueeze(0).repeat(batch_size, 1, 1).double()
    seed_xf = torch.from_numpy(seed['xf']).unsqueeze(0).repeat(batch_size, 1).double()
    seed_mL = torch.from_numpy(seed['mL']).unsqueeze(0).repeat(batch_size, 1, 1).double()
    seed_nL = torch.from_numpy(seed['nL']).unsqueeze(0).repeat(batch_size, 1, 1).double()

    return {
        'dyn_py': dyn_py,
        'seed': seed,
        'currents': currents,
        'currents_np': currents_np,
        'insertion': insertion,
        'insertion_np': insertion_np,
        'seed_v': seed_v,
        'seed_w': seed_w,
        'seed_p': seed_p,
        'seed_R': seed_R,
        'seed_xf': seed_xf,
        'seed_mL': seed_mL,
        'seed_nL': seed_nL,
        'param_file': param_file,
        'config_file': config_file,
    }


def benchmark_option_a_linearization(test_case, num_iterations=100):
    """Benchmark Option A: Direct call to implicit linearization."""
    dyn_py = test_case['dyn_py']
    seed = test_case['seed']
    currents_np = test_case['currents_np']
    insertion_np = test_case['insertion_np']
    batch_size = len(currents_np)

    print(f"\n[Option A: Direct Linearization]")
    print(f"  Batch size: {batch_size}")
    print(f"  Iterations: {num_iterations}")

    # Warmup
    for _ in range(5):
        for i in range(batch_size):
            result = dyn_py.linearize_full_seed_action_from_seed_implicit(
                currents_np[i], insertion_np[i],
                seed['v'], seed['w'], seed['p'], seed['R'], seed['xf'],
                seed['mL'], seed['nL']
            )

    # Benchmark
    start = time.time()
    for _ in range(num_iterations):
        for i in range(batch_size):
            result = dyn_py.linearize_full_seed_action_from_seed_implicit(
                currents_np[i], insertion_np[i],
                seed['v'], seed['w'], seed['p'], seed['R'], seed['xf'],
                seed['mL'], seed['nL']
            )
            # Extract output and gradients (simulate full usage)
            next_state = result['next_state']
            A = result['A']  # Seed Jacobian
            B = result['B']  # Current Jacobian
    end = time.time()

    total_time = end - start
    time_per_iter = total_time / num_iterations
    time_per_sample = total_time / (num_iterations * batch_size)

    print(f"  Total time: {total_time:.4f}s")
    print(f"  Time per iteration: {time_per_iter*1000:.2f}ms")
    print(f"  Time per sample: {time_per_sample*1000:.2f}ms")

    return {
        'total_time': total_time,
        'time_per_iter': time_per_iter,
        'time_per_sample': time_per_sample,
    }


def benchmark_option_c_forward_only(test_case, num_iterations=100):
    """Benchmark Option C: Forward pass only."""
    currents = test_case['currents'].detach().requires_grad_(False)
    insertion = test_case['insertion']
    batch_size = len(currents)

    print(f"\n[Option C: Forward Pass Only]")
    print(f"  Batch size: {batch_size}")
    print(f"  Iterations: {num_iterations}")

    # Warmup
    for _ in range(5):
        output = crm_torch.CRMDynamicsStep.apply(
            currents, insertion,
            test_case['seed_v'], test_case['seed_w'], test_case['seed_p'],
            test_case['seed_R'], test_case['seed_xf'], test_case['seed_mL'],
            test_case['seed_nL'], test_case['param_file'], test_case['config_file'], 1e-4
        )

    # Benchmark
    start = time.time()
    for _ in range(num_iterations):
        output = crm_torch.CRMDynamicsStep.apply(
            currents, insertion,
            test_case['seed_v'], test_case['seed_w'], test_case['seed_p'],
            test_case['seed_R'], test_case['seed_xf'], test_case['seed_mL'],
            test_case['seed_nL'], test_case['param_file'], test_case['config_file'], 1e-4
        )
    end = time.time()

    total_time = end - start
    time_per_iter = total_time / num_iterations
    time_per_sample = total_time / (num_iterations * batch_size)

    print(f"  Total time: {total_time:.4f}s")
    print(f"  Time per iteration: {time_per_iter*1000:.2f}ms")
    print(f"  Time per sample: {time_per_sample*1000:.2f}ms")

    return {
        'total_time': total_time,
        'time_per_iter': time_per_iter,
        'time_per_sample': time_per_sample,
    }


def benchmark_option_c_forward_backward(test_case, num_iterations=100):
    """Benchmark Option C: Forward + Backward pass."""
    batch_size = len(test_case['currents'])

    print(f"\n[Option C: Forward + Backward Pass]")
    print(f"  Batch size: {batch_size}")
    print(f"  Iterations: {num_iterations}")

    # Warmup
    for _ in range(5):
        currents = test_case['currents'].detach().requires_grad_(True)
        output = crm_torch.CRMDynamicsStep.apply(
            currents, test_case['insertion'],
            test_case['seed_v'], test_case['seed_w'], test_case['seed_p'],
            test_case['seed_R'], test_case['seed_xf'], test_case['seed_mL'],
            test_case['seed_nL'], test_case['param_file'], test_case['config_file'], 1e-4
        )
        loss = output.sum()
        loss.backward()

    # Benchmark
    start = time.time()
    for _ in range(num_iterations):
        currents = test_case['currents'].detach().requires_grad_(True)
        output = crm_torch.CRMDynamicsStep.apply(
            currents, test_case['insertion'],
            test_case['seed_v'], test_case['seed_w'], test_case['seed_p'],
            test_case['seed_R'], test_case['seed_xf'], test_case['seed_mL'],
            test_case['seed_nL'], test_case['param_file'], test_case['config_file'], 1e-4
        )
        loss = output.sum()
        loss.backward()
    end = time.time()

    total_time = end - start
    time_per_iter = total_time / num_iterations
    time_per_sample = total_time / (num_iterations * batch_size)

    print(f"  Total time: {total_time:.4f}s")
    print(f"  Time per iteration: {time_per_iter*1000:.2f}ms")
    print(f"  Time per sample: {time_per_sample*1000:.2f}ms")

    return {
        'total_time': total_time,
        'time_per_iter': time_per_iter,
        'time_per_sample': time_per_sample,
    }


def run_benchmark(batch_size=1, num_iterations=100):
    """Run complete benchmark for given batch size."""
    print("\n" + "="*70)
    print(f"PERFORMANCE BENCHMARK: Batch Size = {batch_size}")
    print("="*70)

    if not crm_torch.is_available():
        print(f"❌ Extension not available: {crm_torch.get_import_error()}")
        return None

    # Setup
    test_case = setup_test_case(batch_size)

    # Benchmark Option A
    result_a = benchmark_option_a_linearization(test_case, num_iterations)

    # Benchmark Option C (forward only)
    result_c_fwd = benchmark_option_c_forward_only(test_case, num_iterations)

    # Benchmark Option C (forward + backward)
    result_c_fwd_bwd = benchmark_option_c_forward_backward(test_case, num_iterations)

    # Analysis
    print("\n" + "="*70)
    print("PERFORMANCE ANALYSIS")
    print("="*70)

    print(f"\n[Time per Sample]")
    print(f"  Option A (linearization):        {result_a['time_per_sample']*1000:.2f}ms")
    print(f"  Option C (forward only):         {result_c_fwd['time_per_sample']*1000:.2f}ms")
    print(f"  Option C (forward + backward):   {result_c_fwd_bwd['time_per_sample']*1000:.2f}ms")

    # Compute overhead
    overhead_fwd = (result_c_fwd['time_per_sample'] / result_a['time_per_sample'] - 1.0) * 100
    overhead_fwd_bwd = (result_c_fwd_bwd['time_per_sample'] / result_a['time_per_sample'] - 1.0) * 100

    print(f"\n[Overhead vs Option A]")
    print(f"  Option C forward:         {overhead_fwd:+.1f}%")
    print(f"  Option C forward+backward: {overhead_fwd_bwd:+.1f}%")

    # Speedup/slowdown
    if result_c_fwd_bwd['time_per_sample'] < result_a['time_per_sample']:
        speedup = result_a['time_per_sample'] / result_c_fwd_bwd['time_per_sample']
        print(f"\n✅ Option C is {speedup:.2f}x FASTER than Option A")
    else:
        slowdown = result_c_fwd_bwd['time_per_sample'] / result_a['time_per_sample']
        print(f"\n⚠️ Option C is {slowdown:.2f}x SLOWER than Option A")

    print("\n[Notes]")
    print("  - Both options use same Python linearization (GIL-limited)")
    print("  - Expected similar performance due to sequential processing")
    print("  - Extension overhead from tensor conversions and C++ wrapper")
    print("  - Phase 2B (native C++) would remove GIL for parallelization")

    return {
        'batch_size': batch_size,
        'option_a': result_a,
        'option_c_fwd': result_c_fwd,
        'option_c_fwd_bwd': result_c_fwd_bwd,
        'overhead_fwd': overhead_fwd,
        'overhead_fwd_bwd': overhead_fwd_bwd,
    }


def main():
    """Run benchmarks for different batch sizes."""
    print("\n" + "="*70)
    print("OPTION C PERFORMANCE BENCHMARK")
    print("="*70)
    print("\nComparing:")
    print("  - Option A: Direct Python linearization calls")
    print("  - Option C: PyTorch extension (Phase 2A + 3A)")

    batch_sizes = [1, 4, 8]
    num_iterations = 50  # Reduced for faster benchmarking

    results = []
    for batch_size in batch_sizes:
        result = run_benchmark(batch_size, num_iterations)
        if result:
            results.append(result)

    # Summary
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)

    print("\n{:<12} {:<20} {:<20} {:<15}".format(
        "Batch Size", "Option A (ms/sample)", "Option C (ms/sample)", "Overhead"
    ))
    print("-" * 70)

    for r in results:
        print("{:<12} {:<20.2f} {:<20.2f} {:<15.1f}%".format(
            r['batch_size'],
            r['option_a']['time_per_sample'] * 1000,
            r['option_c_fwd_bwd']['time_per_sample'] * 1000,
            r['overhead_fwd_bwd']
        ))

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    avg_overhead = np.mean([r['overhead_fwd_bwd'] for r in results])

    print(f"\nAverage overhead: {avg_overhead:+.1f}%")

    if avg_overhead < 20:
        print("\n✅ EXCELLENT: Option C performance is comparable to Option A")
        print("   Extension overhead is minimal (<20%)")
    elif avg_overhead < 50:
        print("\n✅ GOOD: Option C performance is acceptable")
        print("   Extension overhead is moderate (20-50%)")
    else:
        print("\n⚠️ NOTICE: Option C has significant overhead (>50%)")
        print("   This is expected for Phase 2A (Python bindings)")
        print("   Phase 2B (native C++) would significantly improve performance")

    print("\n[Key Findings]")
    print("  1. Both options call same Python linearization (GIL-limited)")
    print("  2. Extension adds overhead from tensor conversions")
    print("  3. PyTorch autograd integration provides convenience")
    print("  4. For production: Consider Phase 2B for native C++ implementation")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
