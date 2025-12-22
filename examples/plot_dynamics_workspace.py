"""
Perform a grid sweep of the C++ dynamics and plot the resulting workspace.

This script iterates through a grid of current values, runs one step of the
dynamics for each combination using the proper kinematics-based initialization,
and then generates a 3D scatter plot of the resulting tip positions.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from crm_ml_rl.wrappers.crm_wrapper import CRMWrapper, HAS_CPP_BINDINGS

def plot_dynamics_workspace():
    if not HAS_CPP_BINDINGS:
        print("C++ bindings not available, skipping.")
        return

    # Grid data/simulation_parameters
    current_min = -0.3
    current_max = 0.3
    current_step = 0.1  # Using a larger step for reasonable execution time
    
    current_vals = np.arange(current_min, current_max + current_step, current_step)
    
    insertion_length = 50.0
    dt = 0.05

    wrapper = CRMWrapper(
        param_file="data/catheter_params/CatheterParameterSet_1_dyn.txt",
        config_file="data/catheter_params/CatheterSpatialConfiguration_1.txt",
        use_cpp=True,
    )
    if not wrapper.is_using_cpp:
        raise RuntimeError("C++ bindings are not available")

    tip_positions = []
    
    total_combinations = len(current_vals) ** 3
    count = 0

    for c1 in current_vals:
        for c2 in current_vals:
            for c3 in current_vals:
                count += 1
                currents = np.array([c1, c2, c3])
                print(f"Testing currents {currents} ({count}/{total_combinations})")

                # Initialize the dynamics from the FK solution for the given currents
                init_success = wrapper.initialize_dynamics(currents, insertion_length=insertion_length)

                if not init_success:
                    print(f"  --> Failed to initialize from FK.")
                    continue

                result = wrapper.step_dynamics(currents, insertion_length=insertion_length, dt=dt)
                
                if result["converged"]:
                    tip_positions.append(result["tip_position"])
                else:
                    print(f"  --> Failed to converge.")

    if not tip_positions:
        print("No successful simulations to plot.")
        return

    tip_positions = np.array(tip_positions)

    # Plot the results
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle('Reachable Workspace of C++ Dynamics', fontsize=16)

    # 3D View
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.scatter(tip_positions[:, 0], tip_positions[:, 1], tip_positions[:, 2], c=tip_positions[:, 2], cmap='viridis', marker='.')
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")
    ax1.set_zlabel("Z (mm)")
    ax1.set_title("3D View")
    ax1.view_init(elev=30, azim=-60)

    # Top-down (X-Y) View
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(tip_positions[:, 0], tip_positions[:, 1], c=tip_positions[:, 2], cmap='viridis', marker='.')
    ax2.set_xlabel("X (mm)")
    ax2.set_ylabel("Y (mm)")
    ax2.set_title("Top-Down (X-Y) View")
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True)

    # Front (X-Z) View
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(tip_positions[:, 0], tip_positions[:, 2], c=tip_positions[:, 2], cmap='viridis', marker='.')
    ax3.set_xlabel("X (mm)")
    ax3.set_ylabel("Z (mm)")
    ax3.set_title("Front (X-Z) View")
    ax3.set_aspect('equal', adjustable='box')
    ax3.grid(True)

    # Side (Y-Z) View
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(tip_positions[:, 1], tip_positions[:, 2], c=tip_positions[:, 2], cmap='viridis', marker='.')
    ax4.set_xlabel("Y (mm)")
    ax4.set_ylabel("Z (mm)")
    ax4.set_title("Side (Y-Z) View")
    ax4.set_aspect('equal', adjustable='box')
    ax4.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("dynamics_workspace_multiview.png")
    print("\nSaved plot to dynamics_workspace_multiview.png")


if __name__ == "__main__":
    plot_dynamics_workspace()
