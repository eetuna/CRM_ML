import torch
import numpy as np
import pandas as pd
import time
import os
import sys

# Add repository root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from crm_ml_rl.wrappers.torch_physics import TorchCRMPhysics

def run_stress_test():
    print("========================================================")
    print("STRESS TEST: STATIC WORKSPACE MAPPING")
    print("Target: crm_torch_ext (C++ PyTorch Extension)")
    print("========================================================")

    # 1. Setup Physics
    param_file = "data/catheter_params/CatheterParameterSet_1_dyn.txt"
    config_file = "data/catheter_params/CatheterSpatialConfiguration_1.txt"
    
    try:
        physics = TorchCRMPhysics(param_file, config_file, use_cpp_extension=True)
        if not physics.use_cpp_extension:
            print("CRITICAL ERROR: Failed to load C++ extension. Aborting stress test.")
            return
    except Exception as e:
        print(f"CRITICAL ERROR: Initialization crashed: {e}")
        return

    # 2. Define Grid
    # Currents: 0 to 25 mA (covering safe 20mA and boundary 25mA)
    current_steps = np.linspace(0, 25, 26) # 1mA steps
    # Insertion: 0 to 100 mm
    ins_steps = np.linspace(0, 100, 11) # 10mm steps
    
    print(f"Grid Size: {len(current_steps)} currents x {len(ins_steps)} insertions")
    print(f"Total Points: {len(current_steps) * len(ins_steps)}")
    
    results = []
    
    # Pre-allocate tensors
    seed_v = torch.zeros((1, 1, 3), dtype=torch.float64)
    seed_w = torch.zeros((1, 1, 3), dtype=torch.float64)
    seed_p = torch.zeros((1, 1, 3), dtype=torch.float64)
    seed_R = torch.eye(3, dtype=torch.float64).reshape(1, 1, 9)
    seed_xf = torch.zeros((1, 15), dtype=torch.float64)
    seed_xf[0, 3] = 1.0; seed_xf[0, 7] = 1.0; seed_xf[0, 11] = 1.0
    seed_mL = torch.zeros((1, 1, 3), dtype=torch.float64)
    seed_nL = torch.zeros((1, 1, 3), dtype=torch.float64)

    start_time = time.time()
    
    for ins in ins_steps:
        # Reset seeds for new insertion length (start from zero current)
        seed_mL_curr = torch.zeros((1, 1, 3), dtype=torch.float64)
        seed_nL_curr = torch.zeros((1, 1, 3), dtype=torch.float64)
        
        # Also need to reset v, w, p, R, xf for the new insertion length
        #Ideally we'd solve I=0, L=new_L first.
        # Let's assume I=0 is easy enough to cold start.
        
        seed_v_curr = seed_v.clone()
        seed_w_curr = seed_w.clone()
        seed_p_curr = seed_p.clone()
        seed_R_curr = seed_R.clone()
        seed_xf_curr = seed_xf.clone()

        for curr in current_steps:
            # Test Case: All actuators at 'curr'
            current_vec = torch.tensor([[curr, curr, curr]], dtype=torch.float64)
            ins_vec = torch.tensor([ins], dtype=torch.float64)
            
            status = "OK"
            error_msg = ""
            tip_pos = [np.nan, np.nan, np.nan]
            
            try:
                # Forward Pass
                t0 = time.time()
                # Use current seeds (warm start from previous current step)
                out = physics.dyn_step(
                    current_vec, ins_vec, 
                    seed_v_curr, seed_w_curr, seed_p_curr, seed_R_curr, seed_xf_curr, seed_mL_curr, seed_nL_curr
                )
                dt = time.time() - t0
                
                # Check Validity
                out_np = out.detach().numpy()[0]
                tip_pos = out_np[:3]
                
                if np.any(np.isnan(out_np)) or np.any(np.isinf(out_np)):
                    status = "NAN_INF"
                    error_msg = "Output contains NaN or Inf"
                
                # Update seeds for next step (I+1)
                # The extension outputs [tip_pos(3), coil_vel(3)]. 
                # Wait, the extension forward DOES NOT return the full seed state (mL, nL, etc).
                # It only returns the observation.
                # To warm start, we need the internal state (mL, nL) exposed.
                
                # CRITICAL LIMITATION: The current Torch extension API only returns 'next_state' (observation).
                # It does NOT return the updated seeds (mL_star, nL_star) needed for warm-starting.
                # The Python wrapper 'step_from_seed' returns a dictionary with everything.
                # The Torch 'dyn_step' returns only a Tensor.
                
                # Workaround: For this stress test, we can't easily warm-start via the Torch API 
                # unless we change the API to return the seeds. 
                # BUT, crm_step_op.cpp logic includes a "Continuation Ramp".
                # If that ramp is failing for I=1.0 (very small step), something is wrong with the implementation.
                
                pass # Can't update seeds with current API
                
                # Check Gradient
                current_vec.requires_grad_(True)
                out_grad = physics.dyn_step(
                    current_vec, ins_vec, 
                    seed_v_curr, seed_w_curr, seed_p_curr, seed_R_curr, seed_xf_curr, seed_mL_curr, seed_nL_curr
                )
                loss = out_grad.sum()
                loss.backward()
                
                if current_vec.grad is None:
                     status = "NO_GRAD"
                     error_msg = "Backward pass returned None"
                elif torch.isnan(current_vec.grad).any():
                     status = "GRAD_NAN"
                     error_msg = "Gradient contains NaN"
                
                current_vec.requires_grad_(False) # Reset
                
            except RuntimeError as e:
                status = "CRASH"
                error_msg = str(e)
            except Exception as e:
                status = "ERROR"
                error_msg = str(e)
            
            # Log Result
            results.append({
                "current": curr,
                "insertion": ins,
                "status": status,
                "tip_x": tip_pos[0],
                "tip_y": tip_pos[1],
                "tip_z": tip_pos[2],
                "error": error_msg,
                "time_ms": dt * 1000
            })
            
            # Progress bar
            sys.stdout.write(f"\rTesting I={curr:.1f}, L={ins:.1f} -> {status}   ")
            sys.stdout.flush()

    print("\n\nStress Test Complete!")
    print(f"Total Duration: {time.time() - start_time:.2f}s")
    
    # Analysis
    df = pd.DataFrame(results)
    
    # Save Report
    os.makedirs("crm_torch_ext/test/reports", exist_ok=True)
    report_path = "crm_torch_ext/test/reports/stress_test_grid_results.csv"
    df.to_csv(report_path, index=False)
    print(f"Detailed results saved to: {report_path}")
    
    # Summary
    print("\nSummary Statistics:")
    print(df['status'].value_counts())
    
    failures = df[df['status'] != "OK"]
    if not failures.empty:
        print("\nFailures detected:")
        print(failures[['current', 'insertion', 'status', 'error']])
    else:
        print("\nSUCCESS: All points converged and produced valid gradients.")

if __name__ == "__main__":
    run_stress_test()
