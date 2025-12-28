import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add repository root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from crm_ml_rl.wrappers.torch_physics import TorchCRMPhysics

def visualize_test_inputs(csv_path, name):
    print(f"Loading {name} trajectory...")
    df = pd.read_csv(csv_path)
    currents = df[['c1', 'c2', 'c3']].values
    
    # 1. Plot Currents
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    time_steps = np.arange(len(currents))
    ax1.plot(time_steps, currents[:, 0], 'r-', label='C1')
    ax1.plot(time_steps, currents[:, 1], 'g-', label='C2')
    ax1.plot(time_steps, currents[:, 2], 'b-', label='C3')
    
    # Highlight the "Start" point (teleport from 0)
    ax1.scatter([0, 0, 0], [0, 0, 0], c='k', marker='o', s=50, label='Cold Start (0A)')
    ax1.scatter([0], [currents[0, 0]], c='r', marker='x', s=100)
    ax1.scatter([0], [currents[0, 1]], c='g', marker='x', s=100)
    ax1.scatter([0], [currents[0, 2]], c='b', marker='x', s=100)
    
    ax1.set_title(f"{name}: Currents Jump at Step 0")
    ax1.set_xlabel("CSV Row (Time Step)")
    ax1.set_ylabel("Current (Amps)")
    ax1.legend()
    ax1.grid(True)

    # 2. Run FK for the sequence to see the "Target" path
    param_file = "data/catheter_params/CatheterParameterSet_1_dyn.txt"
    config_file = "data/catheter_params/CatheterSpatialConfiguration_1.txt"
    physics = TorchCRMPhysics(param_file, config_file, use_cpp_extension=True)
    
    ins = torch.tensor([94.3], dtype=torch.float64)
    tip_pos = []
    
    print("Computing FK for reference path...")
    for i in range(len(currents)):
        # Use initialize_from_fk directly to get static results
        # (We skip the dynamics solver here just to see where the points ARE)
        c_tensor = torch.tensor(currents[i], dtype=torch.float64)
        # Note: initialize_from_fk was updated to retry with 0, 
        # so for this plot we use the internal binding directly to see the target.
        import crm_torch_ext
        res = crm_torch_ext.initialize_from_fk(c_tensor, ins)
        # res[4] is xf (tip state)
        tip_pos.append(res[4][:3].numpy())
        
    tip_pos = np.array(tip_pos)
    
    # Plot Tip Path (X-Y projection)
    ax2.plot(tip_pos[:, 0], tip_pos[:, 1], 'b-', label='Target Path (FK)')
    ax2.scatter(tip_pos[0, 0], tip_pos[0, 1], c='r', marker='*', s=200, label='Target Start Point')
    ax2.scatter(0, 0, c='k', marker='o', s=100, label='Straight Rod (0,0)')
    
    ax2.set_title(f"{name}: Tip Path (X-Y Projection)")
    ax2.set_xlabel("X (mm)")
    ax2.set_ylabel("Y (mm)")
    ax2.axis('equal')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f"plots/test_analysis_{name.lower()}.png")
    print(f"Saved plot to plots/test_analysis_{name.lower()}.png")

if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    visualize_test_inputs("data/output/circle1_currents_y40_r10_dp_0p1mm_n200.csv", "Circle")
    visualize_test_inputs("data/output/lemniscate1_currents_y40_a10_dp_0p1mm_n200.csv", "Lemniscate")
