import torch
import numpy as np
import time
import os
import sys

# Add repository root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import crm_torch_ext
from crm_ml_rl.wrappers.torch_physics import TorchCRMPhysics

def benchmark():
    print("Initializing benchmark...")
    
    # Paths
    param_file = "data/catheter_params/CatheterParameterSet_1_dyn.txt"
    config_file = "data/catheter_params/CatheterSpatialConfiguration_1.txt"
    
    if not os.path.exists(param_file):
        print(f"Error: {param_file} not found")
        return
    if not os.path.exists(config_file):
        print(f"Error: {config_file} not found")
        return

    # Initialize Extension
    print("Initializing crm_torch_ext...")
    crm_torch_ext.initialize_params(param_file, config_file)
    crm_torch_ext.set_integrator("rk4")
    crm_torch_ext.set_timestep(0.01) # Default
    # Set damping to match default/stable values if needed, but defaults in file should be ok for benchmark
    
    # Initialize Python Wrapper
    print("Initializing TorchCRMPhysics...")
    wrapper = TorchCRMPhysics(param_file, config_file)
    wrapper.dyn.set_integrator("rk4") # Ensure match

    # Inputs
    num_sets = 1
    currents = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    insertion_length = torch.tensor([0.0], dtype=torch.float64)
    
    # Seed state (zeros)
    seed_v = torch.zeros((num_sets, 3), dtype=torch.float64)
    seed_w = torch.zeros((num_sets, 3), dtype=torch.float64)
    seed_p = torch.zeros((num_sets, 3), dtype=torch.float64)
    seed_R = torch.zeros((num_sets, 9), dtype=torch.float64)
    # R should be identity? Usually seed is differential state, so maybe 0 is fine for "seed" if it means perturbation or initial guess?
    # In crm_step_op.cpp, R_L is passed to BVPParams.
    # Usually R initial guess might be Identity.
    # Let's check typical usage. In examples/ilqr_catheter_demo.py it uses state from previous step.
    # For benchmark, let's use Identity for R just in case.
    seed_R = torch.eye(3, dtype=torch.float64).reshape(1, 9)

    seed_xf = torch.zeros(15, dtype=torch.float64)
    # xf[3:12] is R, so should be identity?
    seed_xf[3] = 1.0; seed_xf[7] = 1.0; seed_xf[11] = 1.0; 

    seed_mL = torch.zeros((num_sets, 3), dtype=torch.float64)
    seed_nL = torch.zeros((num_sets, 3), dtype=torch.float64)
    
    # Make inputs require grad for backward benchmark
    currents.requires_grad_(True)
    insertion_length.requires_grad_(True)
    
    # Prepare batch inputs for Wrapper (TorchCRMPhysics expects batch dim)
    currents_batch = currents.unsqueeze(0) # [1, 3]
    ins_batch = insertion_length # [1]
    
    # Get a valid initial state using Python bindings directly
    print("Finding valid initial state...")
    try:
        # Need numpy inputs for step_from_seed
        curr_np = currents.detach().numpy()
        ins_np = insertion_length.item()
        v_np = seed_v.detach().numpy()
        w_np = seed_w.detach().numpy()
        p_np = seed_p.detach().numpy()
        R_np = seed_R.detach().numpy()
        xf_np = seed_xf.detach().numpy()
        mL_np = seed_mL.detach().numpy()
        nL_np = seed_nL.detach().numpy()
        
        # Call step_from_seed to converge from zeros
        res = wrapper.dyn.step_from_seed(
            curr_np, ins_np, v_np, w_np, p_np, R_np, xf_np, mL_np, nL_np
        )
        
        if not res["converged"]:
            print("Warning: Initial step did not converge, using result anyway.")
            
        # Update seeds with converged values
        seed_v = torch.from_numpy(res["next_v"])
        seed_w = torch.from_numpy(res["next_w"])
        seed_p = torch.from_numpy(res["next_p"])
        seed_R = torch.from_numpy(res["next_R"])
        seed_xf = torch.from_numpy(res["next_xf"])
        seed_mL = torch.from_numpy(res["next_mL"])
        seed_nL = torch.from_numpy(res["next_nL"])
        
        print("Valid state found.")
        
    except Exception as e:
        print(f"Failed to get valid state: {e}")
        # Proceed with zeros and hope for best (or crash)
    
    seed_v.requires_grad_(True)
    seed_w.requires_grad_(True)
    seed_p.requires_grad_(True)
    seed_R.requires_grad_(True)
    seed_xf.requires_grad_(True)
    seed_mL.requires_grad_(True)
    seed_nL.requires_grad_(True)
    
    seed_v_batch = seed_v.unsqueeze(0)
    seed_w_batch = seed_w.unsqueeze(0)
    seed_p_batch = seed_p.unsqueeze(0)
    seed_R_batch = seed_R.unsqueeze(0)
    seed_xf_batch = seed_xf.unsqueeze(0)
    seed_mL_batch = seed_mL.unsqueeze(0)
    seed_nL_batch = seed_nL.unsqueeze(0)

    # Warmup
    print("Warming up...")
    try:
        # Wrapper First
        print("  Testing Wrapper...")
        out_wrap = wrapper.dyn_step(
            currents_batch, ins_batch,
            seed_v_batch, seed_w_batch, seed_p_batch, seed_R_batch, seed_xf_batch, seed_mL_batch, seed_nL_batch
        )
        loss = out_wrap.sum()
        loss.backward()
        print("  Wrapper OK.")
        
        # Zero grads
        currents.grad = None
        insertion_length.grad = None
        
        # Extension
        print("  Testing Extension...")
        out_ext = crm_torch_ext.crm_step(
            currents, insertion_length, 
            seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL
        )
        loss = out_ext.sum()
        loss.backward()
        print("  Extension OK.")
        
        # Zero grads
        currents.grad = None
        insertion_length.grad = None
        
    except Exception as e:
        print(f"Warmup failed: {e}")
        return

    # Warmup Loop
    print("Running warmup loop...")
    for _ in range(4):
        # Extension
        out_ext = crm_torch_ext.crm_step(
            currents, insertion_length, 
            seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL
        )
        loss = out_ext.sum()
        loss.backward()
        
        # Wrapper
        out_wrap = wrapper.dyn_step(
            currents_batch, ins_batch,
            seed_v_batch, seed_w_batch, seed_p_batch, seed_R_batch, seed_xf_batch, seed_mL_batch, seed_nL_batch
        )
        loss = out_wrap.sum()
        loss.backward()
        
        # Zero grads
        currents.grad = None
        insertion_length.grad = None

    N = 100
    print(f"Benchmarking {N} iterations...")
    
    # Benchmark Extension Forward
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    for _ in range(N):
        out_ext = crm_torch_ext.crm_step(
            currents, insertion_length, 
            seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL
        )
    t1 = time.time()
    ext_fwd_avg = (t1 - t0) / N
    
    # Benchmark Wrapper Forward
    t0 = time.time()
    for _ in range(N):
        out_wrap = wrapper.dyn_step(
            currents_batch, ins_batch,
            seed_v_batch, seed_w_batch, seed_p_batch, seed_R_batch, seed_xf_batch, seed_mL_batch, seed_nL_batch
        )
    t1 = time.time()
    wrap_fwd_avg = (t1 - t0) / N
    
    print(f"Forward Pass Average Time (ms):")
    print(f"  Extension: {ext_fwd_avg*1000:.3f} ms")
    print(f"  Wrapper:   {wrap_fwd_avg*1000:.3f} ms")
    print(f"  Speedup:   {wrap_fwd_avg/ext_fwd_avg:.2f}x")
    
    # Benchmark Extension Backward
    # Note: we need to run forward again to build graph
    t0 = time.time()
    for _ in range(N):
        out_ext = crm_torch_ext.crm_step(
            currents, insertion_length, 
            seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL
        )
        loss = out_ext.sum()
        loss.backward()
        currents.grad = None
        insertion_length.grad = None
    t1 = time.time()
    ext_total_avg = (t1 - t0) / N
    ext_bwd_avg = ext_total_avg - ext_fwd_avg
    
    # Benchmark Wrapper Backward
    t0 = time.time()
    for _ in range(N):
        out_wrap = wrapper.dyn_step(
            currents_batch, ins_batch,
            seed_v_batch, seed_w_batch, seed_p_batch, seed_R_batch, seed_xf_batch, seed_mL_batch, seed_nL_batch
        )
        loss = out_wrap.sum()
        loss.backward()
        currents.grad = None
        insertion_length.grad = None
    t1 = time.time()
    wrap_total_avg = (t1 - t0) / N
    wrap_bwd_avg = wrap_total_avg - wrap_fwd_avg

    print(f"Backward Pass Average Time (ms):")
    print(f"  Extension: {ext_bwd_avg*1000:.3f} ms")
    print(f"  Wrapper:   {wrap_bwd_avg*1000:.3f} ms")
    print(f"  Speedup:   {wrap_bwd_avg/ext_bwd_avg:.2f}x")
    
    print(f"Total Iteration Time (ms):")
    print(f"  Extension: {ext_total_avg*1000:.3f} ms")
    print(f"  Wrapper:   {wrap_total_avg*1000:.3f} ms")
    print(f"  Speedup:   {wrap_total_avg/ext_total_avg:.2f}x")

if __name__ == "__main__":
    benchmark()
