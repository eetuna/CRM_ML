import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from crm_ml_rl.wrappers.crm_wrapper import CRMWrapper
import os
import sys
from pathlib import Path

def plot_trajectory_multiview(tip_positions, output_dir, filename_prefix):
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(f'Catheter Tip Trajectory - {filename_prefix}', fontsize=16)
    # ... (rest of function unchanged)

    # 3D View
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(tip_positions[:, 0], tip_positions[:, 1], tip_positions[:, 2], 'b-', linewidth=2)
    ax1.scatter(tip_positions[0, 0], tip_positions[0, 1], tip_positions[0, 2], c='g', s=100, label='Start')
    ax1.scatter(tip_positions[-1, 0], tip_positions[-1, 1], tip_positions[-1, 2], c='r', s=100, label='End')
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")
    ax1.set_zlabel("Z (mm)")
    ax1.set_title("3D View")
    ax1.legend()
    ax1.view_init(elev=30, azim=-60)

    # Top-down (X-Y) View
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(tip_positions[:, 0], tip_positions[:, 1], 'b-')
    ax2.set_xlabel("X (mm)")
    ax2.set_ylabel("Y (mm)")
    ax2.set_title("Top-Down (X-Y) View")
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True)

    # Front (X-Z) View
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(tip_positions[:, 0], tip_positions[:, 2], 'b-')
    ax3.set_xlabel("X (mm)")
    ax3.set_ylabel("Z (mm)")
    ax3.set_title("Front (X-Z) View")
    ax3.set_aspect('equal', adjustable='box')
    ax3.grid(True)

    # Side (Y-Z) View
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(tip_positions[:, 1], tip_positions[:, 2], 'b-')
    ax4.set_xlabel("Y (mm)")
    ax4.set_ylabel("Z (mm)")
    ax4.set_title("Side (Y-Z) View")
    ax4.set_aspect('equal', adjustable='box')
    ax4.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = os.path.join(output_dir, f"{filename_prefix}_multiview.png")
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/plot_trajectories_fk.py <path_to_currents_csv>")
        return
        
    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print(f"Error: File not found {csv_path}")
        return

    base_name = Path(csv_path).stem
    output_dir = os.path.join("data", "fk_trajectories")
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    # Support both comma and space/tab delimited, and handle potential header issues
    if 'c1' not in df.columns:
         df = pd.read_csv(csv_path, sep=None, engine='python')

    currents_seq = df[['c1', 'c2', 'c3']].values
    
    wrapper = CRMWrapper(use_cpp=True)
    insertion_length = 94.3
    
    tip_positions = []
    du0_guess = None
    
    print(f"Processing {base_name}: Running FK for {len(currents_seq)} points...")
    for i, currents in enumerate(currents_seq):
        res = wrapper.forward_kinematics(currents, insertion_length=insertion_length, deltau0_initialguess=du0_guess)
        if res['converged']:
            tip_positions.append(res['tip_position'])
            du0_guess = res['delta_u0']
        else:
            print(f"Warning: FK failed to converge at step {i} for currents {currents}")
            res_no_guess = wrapper.forward_kinematics(currents, insertion_length=insertion_length)
            if res_no_guess['converged']:
                tip_positions.append(res_no_guess['tip_position'])
                du0_guess = res_no_guess['delta_u0']
            else:
                tip_positions.append([np.nan, np.nan, np.nan])
                du0_guess = None

    tip_positions = np.array(tip_positions)
    np.save(os.path.join(output_dir, f"{base_name}_tip_positions.npy"), tip_positions)
    
    mask = ~np.isnan(tip_positions).any(axis=1)
    if not mask.any():
        print("Error: No converged tip positions to plot.")
        return
        
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    plot_trajectory_multiview(tip_positions[mask], plots_dir, base_name)
    plot_components(tip_positions[mask], plots_dir, base_name)

def plot_components(tip_positions, output_dir, filename_prefix):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    labels = ['X', 'Y', 'Z']
    for i, ax in enumerate(axes):
        ax.plot(tip_positions[:, i], 'b-')
        ax.set_ylabel(f"{labels[i]} (mm)")
        ax.grid(True)
    axes[-1].set_xlabel("Step")
    fig.suptitle(f"Tip Position Components - {filename_prefix}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = os.path.join(output_dir, f"{filename_prefix}_components.png")
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    main()
