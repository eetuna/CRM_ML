import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import time
from pathlib import Path

# Add repository root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from crm_ml_rl.wrappers.torch_physics import TorchCRMPhysics
import crm_torch_ext
print(f"DEBUG: crm_torch_ext file: {crm_torch_ext.__file__}")
print(f"DEBUG: crm_torch_ext dir: {dir(crm_torch_ext)}")

def run_trajectory_stress_test(csv_path, name):
    print(f"\n--- Stress Testing Trajectory: {name} ---")
    
    # 1. Setup Physics
    param_file = "data/catheter_params/CatheterParameterSet_1_dyn.txt"
    config_file = "data/catheter_params/CatheterSpatialConfiguration_1.txt"
    physics = TorchCRMPhysics(param_file, config_file, use_cpp_extension=True)
    
    # 2. Load Currents & Prepend Ramp
    df = pd.read_csv(csv_path)
    target_path = df[['c1', 'c2', 'c3']].values
    
    # Generate 120-step ramp from 0 to first target point
    # This matches the Python wrapper's robust startup strategy
    ramp_steps = 120
    start_target = target_path[0]
    ramp = np.linspace(np.zeros(3), start_target, ramp_steps, endpoint=False)
    
    # Concatenate: Ramp + Trajectory
    full_sequence_np = np.vstack([ramp, target_path])
    currents_seq = torch.tensor(full_sequence_np, dtype=torch.float64)
    num_steps = len(currents_seq)
    
    print(f"  Trajectory loaded: {len(target_path)} steps")
    print(f"  Prepended Ramp: {ramp_steps} steps")
    print(f"  Total rollout: {num_steps} steps")
    
    # 3. Parameters
    ins = torch.tensor([94.3], dtype=torch.float64)
    dt = 0.02  # Use 20ms steps (internally sub-stepped to 1ms)
    physics.set_timestep(dt)
    
    # 4. Initialization (Physical Warm Start from FK)
    # This will now internally RETRY with 0A if the requested currents fail.
    print(f"  Initializing from FK...")
    init_state = physics.initialize_from_fk(currents_seq[0], ins)
    v, w, p, R, xf, mL, nL = [t.unsqueeze(0) for t in init_state]

    # 5. Rollout
    positions = []
    convergences = []
    start_time = time.time()
    
    print(f"  Executing rollout...")
    for i in range(num_steps):
        curr = currents_seq[i:i+1] # (1, 3)
        
        # Execute step with full state return
        res = physics.dyn_step(
            curr, ins, v, w, p, R, xf, mL, nL, 
            return_full_state=True
        )
        
        # Extract components
        next_state, next_v, next_w, next_p, next_R, next_xf, next_mL, next_nL, localmin = res
        
        # Only update state if converged (mimic stateful wrapper protection)
        if localmin.item() == 0:
            v, w, p, R, xf, mL, nL = next_v, next_w, next_p, next_R, next_xf, next_mL, next_nL
        
        # Log data
        positions.append(next_state[0, :3].detach().numpy())
        convergences.append(localmin.item())
        
        if i % 50 == 0:
            print(f"  Step {i}/{num_steps}... localmin={localmin.item()}")

    duration = time.time() - start_time
    positions = np.array(positions)
    convergences = np.array(convergences)
    
    print(f"Completed in {duration:.2f}s ({duration/num_steps*1000:.2f}ms/step)")
    print(f"Failures (localmin != 0): {np.sum(convergences != 0)}")
    
    return positions, convergences

def main():
    trajectories = [
        ("data/output/circle1_currents_y40_r10_dp_0p1mm_n200.csv", "Circle"),
        ("data/output/lemniscate1_currents_y40_a10_dp_0p1mm_n200.csv", "Lemniscate")
    ]
    
    os.makedirs("plots", exist_ok=True)
    
    fig = plt.figure(figsize=(15, 10))
    
    for i, (path, name) in enumerate(trajectories):
        if not os.path.exists(path):
            print(f"Skipping {name}: file not found")
            continue
            
        pos, conv = run_trajectory_stress_test(path, name)
        
        # Plot 3D
        ax = fig.add_subplot(1, 2, i+1, projection='3d')
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'b-', label='C++ Extension')
        
        # Highlight failures
        fail_mask = conv != 0
        if np.any(fail_mask):
            ax.scatter(pos[fail_mask, 0], pos[fail_mask, 1], pos[fail_mask, 2], c='r', marker='x', label='Failure')
            
        ax.set_title(f"Stress Test: {name}\n(Max Dev: {np.max(np.linalg.norm(pos, axis=1)):.1f}mm)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()

    plt.tight_layout()
    plt.savefig("plots/trajectory_stress_test_results.png")
    print("\nSaved summary plot to plots/trajectory_stress_test_results.png")

if __name__ == "__main__":
    main()
