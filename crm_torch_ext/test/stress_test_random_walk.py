import torch
import numpy as np
import time
import os
import sys

# Add repository root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from crm_ml_rl.wrappers.torch_physics import TorchCRMPhysics

def run_random_walk():
    print("========================================================")
    print("STRESS TEST: RANDOM WALK DYNAMICS")
    print("Target: crm_torch_ext (C++ PyTorch Extension)")
    print("========================================================")

    # 1. Setup Physics
    param_file = "data/catheter_params/CatheterParameterSet_1_dyn.txt"
    config_file = "data/catheter_params/CatheterSpatialConfiguration_1.txt"
    physics = TorchCRMPhysics(param_file, config_file, use_cpp_extension=True)
    
    # Configure Extension Singleton (Important!)
    import crm_torch_ext
    crm_torch_ext.set_integrator("rk4")
    crm_torch_ext.set_integration_step_size(0.1)
    crm_torch_ext.set_timestep(0.001) # 1ms for high stability
    stable_damping = [12.1761626666366, 12.1761626666366, 284.429938756989, 
                      0.0304776127617393, 0.0304776127617393, 0.00502712804532508]
    crm_torch_ext.set_damping(stable_damping)
    
    # 2. Parameters
    num_episodes = 5
    steps_per_episode = 1000
    max_current = 20.0 # mA
    delta_u_std = 0.05 # Smaller changes for stability mapping
    
    print(f"Running {num_episodes} episodes x {steps_per_episode} steps...")
    
    total_failures = 0
    total_steps = 0
    
    start_time = time.time()

    for ep in range(num_episodes):
        print(f"\nEpisode {ep+1}/{num_episodes}")
        
        # Initial State
        currents = torch.zeros(3, dtype=torch.float64)
        insertion = torch.tensor([0.0], dtype=torch.float64)
        
        # Seeds
        num_sets = 1
        sv = torch.zeros((1, num_sets, 3), dtype=torch.float64)
        sw = torch.zeros((1, num_sets, 3), dtype=torch.float64)
        sp = torch.zeros((1, num_sets, 3), dtype=torch.float64)
        sR = torch.eye(3, dtype=torch.float64).reshape(1, num_sets, 9)
        sxf = torch.zeros((1, 15), dtype=torch.float64)
        sxf[0, 3] = 1.0; sxf[0, 7] = 1.0; sxf[0, 11] = 1.0
        smL = torch.zeros((1, num_sets, 3), dtype=torch.float64)
        snL = torch.zeros((1, num_sets, 3), dtype=torch.float64)

        for s in range(steps_per_episode):
            total_steps += 1
            
            # 1. Action
            if s > 0:
                # Random Action after establishing converged Step 0
                delta_u = torch.randn(3, dtype=torch.float64) * delta_u_std
                currents = torch.clamp(currents + delta_u, 0, max_current)
            else:
                # Step 0: Force Zero Currents to establish seed
                currents = torch.zeros(3, dtype=torch.float64)
            
            # 2. Step Physics
            try:
                curr_batch = currents.unsqueeze(0)
                
                # Monitor inputs
                sv_norm = torch.norm(sv).item()
                smL_norm = torch.norm(smL).item()
                
                # Perform perfect warm start using full returned state
                res_tuple = physics.dyn_step(
                    curr_batch, insertion, 
                    sv, sw, sp, sR, sxf, smL, snL,
                    return_full_state=True
                )
                
                # Unpack tuple
                # [next_state, next_v, next_w, next_p, next_R, next_xf, next_mL, next_nL, localmin]
                out, next_v, next_w, next_p, next_R, next_xf, next_mL, next_nL, localmin = res_tuple
                
                # Convergence Check
                converged = (localmin.item() == 0)
                
                if converged:
                    # ONLY update seeds if converged
                    sv, sw, sp, sR, sxf, smL, snL = next_v, next_w, next_p, next_R, next_xf, next_mL, next_nL
                else:
                    # If not converged, we keep old seeds (sv, sw, etc. remain unchanged)
                    # This mimics the Python wrapper's robust state management
                    pass

                # 3. Check for NaNs or Divergence
                out_np = out.detach().numpy()[0]
                tip_pos = out_np[:3]
                
                # Progress
                sys.stdout.write(f"\rStep {s}/{steps_per_episode} - Pos Norm: {np.linalg.norm(tip_pos):.2e}, Conv: {converged}   ")
                sys.stdout.flush()

                if torch.isnan(out).any():
                    print(f"\n[STEP {s}] FAILURE: NaN detected!")
                    total_failures += 1
                    break
                
                # Physical Bounds Check
                if np.any(np.abs(tip_pos) > 1000.0):
                    print(f"\n[STEP {s}] FAILURE: Divergence detected! Pos: {tip_pos}")
                    total_failures += 1
                    break
                
            except Exception as e:
                print(f"\n[STEP {s}] CRASH: {e}")
                total_failures += 1
                break
            
            if s % 100 == 0:
                sys.stdout.write(f"\rStep {s}/{steps_per_episode} - Pos: {out_np[:3]}   ")
                sys.stdout.flush()

    duration = time.time() - start_time
    print(f"\n\nStress Test Complete!")
    print(f"Total Steps: {total_steps}")
    print(f"Total Failures: {total_failures}")
    print(f"Success Rate: {(1.0 - total_failures/num_episodes)*100:.1f}%")
    print(f"Average Step Time: {duration/total_steps*1000:.3f} ms")

if __name__ == "__main__":
    run_random_walk()
