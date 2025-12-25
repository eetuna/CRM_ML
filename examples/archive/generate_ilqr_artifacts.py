#!/usr/bin/env python3
"""
Generate iLQR demo artifacts for CP-03 validation.
Outputs:
  - ilqr_trajectory.png: Trajectory visualization
  - convergence.json: Convergence data
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ilqr_catheter_demo import setup_dyn, iLQRController


def main():
    print("="*60)
    print("CP-03: Generate iLQR Proof Artifacts")
    print("="*60)

    # Setup
    insertion = 94.3
    dyn = setup_dyn(insertion=insertion)

    # Use short horizon for quick execution
    controller = iLQRController(dyn, insertion, horizon=5, max_iters=5, use_implicit=True)

    # Initial state
    tip_pos = np.array(dyn.get_tip_position(), dtype=np.float64)
    x0 = np.concatenate([tip_pos, np.zeros(3)])

    # Target: achievable nearby target
    target = np.array([0.0, 54.0, 71.0])

    print(f"\nConfiguration:")
    print(f"  Initial position: {x0[:3]} mm")
    print(f"  Target position: {target} mm")
    print(f"  Initial distance: {np.linalg.norm(x0[:3] - target):.2f} mm")
    print(f"  Horizon: {controller.horizon} steps")
    print(f"  Max iterations: {controller.max_iters}")
    print()

    # Solve
    print("Running iLQR optimization...")
    states, actions, info = controller.solve(x0, target)

    # Extract results
    final_pos = states[-1, :3]
    final_error = np.linalg.norm(final_pos - target)
    iterations = len(info['costs']) - 1

    print(f"\nResults:")
    print(f"  Final position: {final_pos} mm")
    print(f"  Final error: {final_error:.2f} mm")
    print(f"  Iterations: {iterations}")
    print(f"  Cost history: {[f'{c:.2f}' for c in info['costs'][:5]]}")

    # Generate artifacts
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # 1. Generate trajectory plot
    print("\nGenerating ilqr_trajectory.png...")
    generate_trajectory_plot(states, target, info['costs'], output_dir / "ilqr_trajectory.png")

    # 2. Generate convergence JSON
    print("Generating convergence.json...")
    generate_convergence_json(x0, target, states, actions, info, output_dir / "convergence.json")

    print(f"\n{'='*60}")
    print("CP-03: Artifacts Generated Successfully")
    print(f"{'='*60}")
    print(f"  ✅ {output_dir / 'ilqr_trajectory.png'}")
    print(f"  ✅ {output_dir / 'convergence.json'}")
    print()


def generate_trajectory_plot(states, target, costs, output_path):
    """Generate 3D trajectory visualization."""
    fig = plt.figure(figsize=(14, 10))

    # 3D trajectory plot
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')

    # Extract positions
    x = states[:, 0]
    y = states[:, 1]
    z = states[:, 2]

    # Plot trajectory
    ax1.plot(x, y, z, 'b-', linewidth=2, label='iLQR Trajectory')
    ax1.scatter(x[0], y[0], z[0], c='green', s=100, marker='o', label='Start', zorder=5)
    ax1.scatter(x[-1], y[-1], z[-1], c='red', s=100, marker='s', label='Final', zorder=5)
    ax1.scatter(target[0], target[1], target[2], c='gold', s=150, marker='*', label='Target', zorder=5)

    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title('3D Catheter Tip Trajectory')
    ax1.legend()
    ax1.grid(True)

    # XY projection
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(x, y, 'b-', linewidth=2)
    ax2.scatter(x[0], y[0], c='green', s=100, marker='o', label='Start')
    ax2.scatter(x[-1], y[-1], c='red', s=100, marker='s', label='Final')
    ax2.scatter(target[0], target[1], c='gold', s=150, marker='*', label='Target')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_title('XY Projection')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')

    # XZ projection
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(x, z, 'b-', linewidth=2)
    ax3.scatter(x[0], z[0], c='green', s=100, marker='o', label='Start')
    ax3.scatter(x[-1], z[-1], c='red', s=100, marker='s', label='Final')
    ax3.scatter(target[0], target[2], c='gold', s=150, marker='*', label='Target')
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Z (mm)')
    ax3.set_title('XZ Projection')
    ax3.legend()
    ax3.grid(True)
    ax3.axis('equal')

    # Cost convergence
    ax4 = fig.add_subplot(2, 2, 4)
    iterations = range(len(costs))
    ax4.plot(iterations, costs, 'o-', linewidth=2, markersize=6)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Cost (mm)')
    ax4.set_title('Cost Convergence')
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Plot saved: {output_path}")


def generate_convergence_json(x0, target, states, actions, info, output_path):
    """Generate convergence data JSON."""

    # Convert numpy arrays to lists for JSON serialization
    data = {
        "metadata": {
            "checkpoint": "CP-03",
            "date": "2025-12-25",
            "description": "iLQR convergence validation for catheter control"
        },
        "configuration": {
            "horizon": len(actions),
            "max_iterations": info.get('max_iters', len(info['costs']) - 1),
            "linearization_method": "implicit_AD",
            "initial_position_mm": x0[:3].tolist(),
            "target_position_mm": target.tolist(),
            "initial_distance_mm": float(np.linalg.norm(x0[:3] - target))
        },
        "results": {
            "iterations_run": len(info['costs']) - 1,
            "final_position_mm": states[-1, :3].tolist(),
            "final_error_mm": float(np.linalg.norm(states[-1, :3] - target)),
            "converged": bool(info['converged']),
            "cost_history": [float(c) for c in info['costs']],
            "initial_cost": float(info['costs'][0]),
            "final_cost": float(info['costs'][-1]),
            "cost_reduction": float(info['costs'][0] - info['costs'][-1]),
            "cost_reduction_percent": float((info['costs'][0] - info['costs'][-1]) / info['costs'][0] * 100)
        },
        "performance": {
            "avg_linearization_time_s": float(np.mean(info['times_linearize'])),
            "avg_backward_pass_time_s": float(np.mean(info['times_backward'])),
            "avg_forward_pass_time_s": float(np.mean(info['times_forward'])),
            "total_time_s": sum(info['times_linearize']) + sum(info['times_backward']) + sum(info['times_forward'])
        },
        "trajectory": {
            "num_timesteps": len(states),
            "positions_mm": states[:, :3].tolist(),
            "velocities_mm_s": states[:, 3:6].tolist(),
            "actions_normalized": actions.tolist()
        }
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"  JSON saved: {output_path}")


if __name__ == '__main__':
    main()
