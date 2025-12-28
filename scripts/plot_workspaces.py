import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def plot_workspace(file_path, title_suffix=""):
    data = np.load(file_path)
    P = data['P']
    conv = data['conv']
    
    # Filter only converged points
    tip_positions = P[conv == 1]
    
    if len(tip_positions) == 0:
        print(f"No converged points in {file_path}")
        return

    # Plot the results
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(f'Reachable Workspace - {os.path.basename(file_path)} {title_suffix}', fontsize=16)

    # 3D View
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    sc1 = ax1.scatter(tip_positions[:, 0], tip_positions[:, 1], tip_positions[:, 2], c=tip_positions[:, 2], cmap='viridis', marker='.', s=1)
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")
    ax1.set_zlabel("Z (mm)")
    ax1.set_title("3D View")
    ax1.view_init(elev=30, azim=-60)
    plt.colorbar(sc1, ax=ax1, label='Z (mm)')

    # Top-down (X-Y) View
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(tip_positions[:, 0], tip_positions[:, 1], c=tip_positions[:, 2], cmap='viridis', marker='.', s=1)
    ax2.set_xlabel("X (mm)")
    ax2.set_ylabel("Y (mm)")
    ax2.set_title("Top-Down (X-Y) View")
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True)

    # Front (X-Z) View
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(tip_positions[:, 0], tip_positions[:, 2], c=tip_positions[:, 2], cmap='viridis', marker='.', s=1)
    ax3.set_xlabel("X (mm)")
    ax3.set_ylabel("Z (mm)")
    ax3.set_title("Front (X-Z) View")
    ax3.set_aspect('equal', adjustable='box')
    ax3.grid(True)

    # Side (Y-Z) View
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(tip_positions[:, 1], tip_positions[:, 2], c=tip_positions[:, 2], cmap='viridis', marker='.', s=1)
    ax4.set_xlabel("Y (mm)")
    ax4.set_ylabel("Z (mm)")
    ax4.set_title("Side (Y-Z) View")
    ax4.set_aspect('equal', adjustable='box')
    ax4.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_filename = os.path.join("plots", os.path.basename(file_path).replace('.npz', '.png'))
    os.makedirs("plots", exist_ok=True)
    plt.savefig(output_filename)
    print(f"Saved plot to {output_filename}")
    
    # If a GUI is available, this would open separate windows
    # Since we are likely in a headless environment, we save to files.
    # plt.show()

if __name__ == "__main__":
    files = [
        'data/output/workspace_fk_ins94.3_b0.3_step0.01_int0.2.npz',
        'data/output/workspace_fk_ins94.3_b0.3_step0.01_int0.2_withDU0.npz'
    ]
    
    for f in files:
        if os.path.exists(f):
            plot_workspace(f)
        else:
            print(f"File not found: {f}")
