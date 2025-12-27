import torch
import numpy as np
import time
import os
import sys

# Add repository root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from crm_ml_rl.wrappers.torch_physics import TorchCRMPhysics

def run_gradient_consistency_test():
    print("========================================================")
    print("STRESS TEST: GRADIENT CONSISTENCY (AD vs FD)")
    print("Target: crm_torch_ext (C++ PyTorch Extension)")
    print("========================================================")

    # 1. Setup Physics
    param_file = "data/catheter_params/CatheterParameterSet_1_dyn.txt"
    config_file = "data/catheter_params/CatheterSpatialConfiguration_1.txt"
    physics = TorchCRMPhysics(param_file, config_file, use_cpp_extension=True)
    
    import crm_torch_ext
    crm_torch_ext.set_integrator("rk4")
    crm_torch_ext.set_integration_step_size(0.1)
    crm_torch_ext.set_timestep(0.001)
    
    # 2. Test Parameters
    num_samples = 20 # 50 is slow, let's start with 20
    fd_eps = 1e-4
    
    print(f"Sampling {num_samples} points...")
    
    errors = []
    
    # Seeds (Zero start)
    num_sets = 1
    sv = torch.zeros((1, num_sets, 3), dtype=torch.float64)
    sw = torch.zeros((1, num_sets, 3), dtype=torch.float64)
    sp = torch.zeros((1, num_sets, 3), dtype=torch.float64)
    sR = torch.eye(3, dtype=torch.float64).reshape(1, num_sets, 9)
    sxf = torch.zeros((1, 15), dtype=torch.float64)
    sxf[0, 3] = 1.0; sxf[0, 7] = 1.0; sxf[0, 11] = 1.0
    smL = torch.zeros((1, num_sets, 3), dtype=torch.float64)
    snL = torch.zeros((1, num_sets, 3), dtype=torch.float64)

    for i in range(num_samples):
        # 1. Random Valid Point (small currents for convergence)
        currents = torch.rand(3, dtype=torch.float64) * 5.0 # 0-5mA
        insertion = torch.tensor([0.0], dtype=torch.float64)
        
        currents.requires_grad_(True)
        
        try:
            # Analytical Gradient
            out = physics.dyn_step(currents.unsqueeze(0), insertion, sv, sw, sp, sR, sxf, smL, snL)
            
            # Loss = sum of tip position components
            loss = out[0, :3].sum()
            loss.backward()
            grad_ad = currents.grad.clone()
            
            # Finite Difference
            grad_fd = torch.zeros_like(grad_ad)
            for k in range(3):
                curr_p = currents.detach().clone()
                curr_m = currents.detach().clone()
                curr_p[k] += fd_eps
                curr_m[k] -= fd_eps
                
                out_p = physics.dyn_step(curr_p.unsqueeze(0), insertion, sv, sw, sp, sR, sxf, smL, snL)
                out_m = physics.dyn_step(curr_m.unsqueeze(0), insertion, sv, sw, sp, sR, sxf, smL, snL)
                
                loss_p = out_p[0, :3].sum()
                loss_m = out_m[0, :3].sum()
                
                grad_fd[k] = (loss_p - loss_m) / (2 * fd_eps)
            
            # Error Calculation
            abs_err = torch.norm(grad_ad - grad_fd).item()
            rel_err = abs_err / (torch.norm(grad_fd).item() + 1e-9)
            
            errors.append(rel_err)
            
            status = "PASS" if rel_err < 0.05 else "FAIL" # Allow 5% for FD sensitivity
            sys.stdout.write(f"\rPoint {i} - Rel Err: {rel_err:.2%}, Status: {status}   ")
            sys.stdout.flush()
            
        except Exception as e:
            print(f"\nPoint {i} - ERROR: {e}")
            continue

    print("\n\nGradient Consistency Report:")
    if errors:
        avg_err = np.mean(errors)
        max_err = np.max(errors)
        print(f"Average Relative Error: {avg_err:.4%}")
        print(f"Maximum Relative Error: {max_err:.4%}")
        
        if avg_err < 0.01:
            print("RESULT: SUCCESS - Analytical gradients match Finite Differences.")
        else:
            print("RESULT: WARNING - Large discrepancy detected. Check FD epsilon or physics scaling.")
    else:
        print("RESULT: No successful points tested.")

if __name__ == "__main__":
    run_gradient_consistency_test()
