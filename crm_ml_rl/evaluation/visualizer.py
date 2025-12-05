"""
Visualization tools for catheter trajectories and training metrics.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Visualization features unavailable.")


class TrajectoryVisualizer:
    """Visualizer for catheter trajectories."""

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer.

        Args:
            figsize: Default figure size
        """
        self.figsize = figsize

    def plot_trajectory_3d(
        self,
        trajectory: np.ndarray,
        reference: Optional[np.ndarray] = None,
        target: Optional[np.ndarray] = None,
        title: str = "Catheter Trajectory",
        save_path: Optional[str] = None
    ):
        """
        Plot 3D trajectory.

        Args:
            trajectory: Trajectory positions (T, 3)
            reference: Optional reference trajectory
            target: Optional target position
            title: Plot title
            save_path: Path to save figure
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib not available")
            return

        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Plot trajectory
        ax.plot(
            trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
            'b-', linewidth=2, label='Actual'
        )

        # Mark start and end
        ax.scatter(*trajectory[0], c='g', s=100, marker='o', label='Start')
        ax.scatter(*trajectory[-1], c='r', s=100, marker='x', label='End')

        # Plot reference if provided
        if reference is not None:
            ax.plot(
                reference[:, 0], reference[:, 1], reference[:, 2],
                'g--', linewidth=1.5, alpha=0.7, label='Reference'
            )

        # Plot target if provided
        if target is not None:
            ax.scatter(*target, c='m', s=200, marker='*', label='Target')

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(title)
        ax.legend()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_trajectory_2d(
        self,
        trajectory: np.ndarray,
        reference: Optional[np.ndarray] = None,
        plane: str = "xy",
        title: str = "Trajectory (Top View)",
        save_path: Optional[str] = None
    ):
        """
        Plot 2D projection of trajectory.

        Args:
            trajectory: Trajectory positions (T, 3)
            reference: Optional reference trajectory
            plane: Projection plane ("xy", "xz", "yz")
            title: Plot title
            save_path: Path to save figure
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib not available")
            return

        plane_indices = {
            "xy": (0, 1, "X", "Y"),
            "xz": (0, 2, "X", "Z"),
            "yz": (1, 2, "Y", "Z")
        }

        i, j, xlabel, ylabel = plane_indices[plane]

        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot trajectory
        ax.plot(
            trajectory[:, i], trajectory[:, j],
            'b-', linewidth=2, label='Actual'
        )

        # Mark start and end
        ax.plot(trajectory[0, i], trajectory[0, j], 'go', markersize=10, label='Start')
        ax.plot(trajectory[-1, i], trajectory[-1, j], 'rx', markersize=10, label='End')

        # Plot reference
        if reference is not None:
            ax.plot(
                reference[:, i], reference[:, j],
                'g--', linewidth=1.5, alpha=0.7, label='Reference'
            )

        ax.set_xlabel(f'{xlabel} (mm)')
        ax.set_ylabel(f'{ylabel} (mm)')
        ax.set_title(title)
        ax.legend()
        ax.axis('equal')
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_trajectory_components(
        self,
        times: np.ndarray,
        trajectory: np.ndarray,
        reference: Optional[np.ndarray] = None,
        title: str = "Trajectory Components",
        save_path: Optional[str] = None
    ):
        """
        Plot trajectory components over time.

        Args:
            times: Time array
            trajectory: Trajectory positions (T, 3)
            reference: Optional reference trajectory
            title: Plot title
            save_path: Path to save figure
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib not available")
            return

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

        labels = ['X', 'Y', 'Z']

        for i, (ax, label) in enumerate(zip(axes, labels)):
            ax.plot(times, trajectory[:, i], 'b-', linewidth=2, label='Actual')

            if reference is not None:
                ax.plot(times, reference[:, i], 'g--', linewidth=1.5,
                       alpha=0.7, label='Reference')

            ax.set_ylabel(f'{label} (mm)')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Time (s)')
        fig.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_tracking_error(
        self,
        times: np.ndarray,
        trajectory: np.ndarray,
        reference: np.ndarray,
        title: str = "Tracking Error",
        save_path: Optional[str] = None
    ):
        """
        Plot tracking error over time.

        Args:
            times: Time array
            trajectory: Actual trajectory (T, 3)
            reference: Reference trajectory (T, 3)
            title: Plot title
            save_path: Path to save figure
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib not available")
            return

        # Compute error
        error = np.linalg.norm(trajectory - reference, axis=1)

        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(times, error, 'b-', linewidth=2)
        ax.axhline(y=np.mean(error), color='r', linestyle='--',
                  label=f'Mean: {np.mean(error):.2f} mm')
        ax.fill_between(times, 0, error, alpha=0.3)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Tracking Error (mm)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_currents(
        self,
        times: np.ndarray,
        currents: np.ndarray,
        title: str = "Applied Currents",
        save_path: Optional[str] = None
    ):
        """
        Plot applied currents over time.

        Args:
            times: Time array
            currents: Current values (T, 3) or (3, T)
            title: Plot title
            save_path: Path to save figure
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib not available")
            return

        # Ensure correct shape
        if currents.shape[0] == 3 and currents.shape[1] != 3:
            currents = currents.T

        fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

        colors = ['r', 'g', 'b']
        labels = ['I_x', 'I_y', 'I_z']

        for i, (ax, color, label) in enumerate(zip(axes, colors, labels)):
            ax.plot(times, currents[:, i], color=color, linewidth=1.5)
            ax.set_ylabel(f'{label} (A)')
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Time (s)')
        fig.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def compare_trajectories(
        self,
        trajectories: Dict[str, np.ndarray],
        reference: Optional[np.ndarray] = None,
        title: str = "Trajectory Comparison",
        save_path: Optional[str] = None
    ):
        """
        Compare multiple trajectories.

        Args:
            trajectories: Dictionary of {name: trajectory}
            reference: Optional reference trajectory
            title: Plot title
            save_path: Path to save figure
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib not available")
            return

        fig = plt.figure(figsize=(14, 6))

        # 3D plot
        ax1 = fig.add_subplot(121, projection='3d')

        colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))

        for (name, traj), color in zip(trajectories.items(), colors):
            ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                    color=color, linewidth=2, label=name)

        if reference is not None:
            ax1.plot(reference[:, 0], reference[:, 1], reference[:, 2],
                    'k--', linewidth=1.5, alpha=0.5, label='Reference')

        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_zlabel('Z (mm)')
        ax1.legend()
        ax1.set_title('3D View')

        # Error comparison
        ax2 = fig.add_subplot(122)

        if reference is not None:
            for (name, traj), color in zip(trajectories.items(), colors):
                min_len = min(len(traj), len(reference))
                error = np.linalg.norm(traj[:min_len] - reference[:min_len], axis=1)
                ax2.plot(error, color=color, linewidth=2, label=f'{name}: μ={np.mean(error):.2f}')

            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Error (mm)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_title('Tracking Error')

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class TrainingVisualizer:
    """Visualizer for training metrics."""

    def __init__(self, figsize: Tuple[int, int] = (12, 6)):
        """
        Initialize visualizer.

        Args:
            figsize: Default figure size
        """
        self.figsize = figsize

    def plot_training_curves(
        self,
        history: Dict,
        title: str = "Training Progress",
        save_path: Optional[str] = None
    ):
        """
        Plot training and validation loss curves.

        Args:
            history: Dictionary with 'train_loss' and 'val_loss' lists
            title: Plot title
            save_path: Path to save figure
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib not available")
            return

        fig, ax = plt.subplots(figsize=self.figsize)

        epochs = range(1, len(history['train_loss']) + 1)

        ax.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss')

        if 'val_loss' in history:
            ax.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val Loss')

        if 'best_epoch' in history:
            ax.axvline(x=history['best_epoch'], color='g', linestyle='--',
                      label=f"Best: {history['best_val_loss']:.4f}")

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_rl_training(
        self,
        episode_rewards: List[float],
        episode_lengths: List[float] = None,
        window: int = 100,
        title: str = "RL Training Progress",
        save_path: Optional[str] = None
    ):
        """
        Plot RL training progress.

        Args:
            episode_rewards: Episode rewards
            episode_lengths: Episode lengths
            window: Smoothing window size
            title: Plot title
            save_path: Path to save figure
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib not available")
            return

        n_subplots = 2 if episode_lengths else 1
        fig, axes = plt.subplots(n_subplots, 1, figsize=(12, 4*n_subplots))

        if n_subplots == 1:
            axes = [axes]

        episodes = range(len(episode_rewards))

        # Smoothed rewards
        smoothed = np.convolve(
            episode_rewards,
            np.ones(window)/window,
            mode='valid'
        )

        axes[0].plot(episodes, episode_rewards, 'b-', alpha=0.3, linewidth=1)
        axes[0].plot(
            range(window-1, len(episode_rewards)),
            smoothed, 'b-', linewidth=2,
            label=f'Smoothed (window={window})'
        )
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward')
        axes[0].set_title('Episode Rewards')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        if episode_lengths:
            smoothed_len = np.convolve(
                episode_lengths,
                np.ones(window)/window,
                mode='valid'
            )

            axes[1].plot(episodes, episode_lengths, 'g-', alpha=0.3, linewidth=1)
            axes[1].plot(
                range(window-1, len(episode_lengths)),
                smoothed_len, 'g-', linewidth=2
            )
            axes[1].set_xlabel('Episode')
            axes[1].set_ylabel('Episode Length')
            axes[1].set_title('Episode Lengths')
            axes[1].grid(True, alpha=0.3)

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_comparison_bar(
        self,
        results: Dict[str, Dict],
        metrics: List[str] = None,
        title: str = "Model Comparison",
        save_path: Optional[str] = None
    ):
        """
        Plot bar chart comparing different models/algorithms.

        Args:
            results: Dictionary of {name: {metric: value}}
            metrics: Metrics to plot
            title: Plot title
            save_path: Path to save figure
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib not available")
            return

        if metrics is None:
            metrics = ['mean_reward', 'success_rate']

        n_metrics = len(metrics)
        n_models = len(results)

        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]

        x = np.arange(n_models)
        width = 0.6

        for ax, metric in zip(axes, metrics):
            values = []
            labels = []

            for name, res in results.items():
                labels.append(name)
                if isinstance(res, dict) and 'metrics' in res:
                    values.append(res['metrics'].get(metric, 0))
                elif isinstance(res, dict):
                    values.append(res.get(metric, 0))
                else:
                    values.append(0)

            bars = ax.bar(x, values, width, color=plt.cm.tab10(x))

            ax.set_ylabel(metric)
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right')

            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{val:.3f}', ha='center', va='bottom')

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_prediction_comparison(
        self,
        true_values: np.ndarray,
        predictions: Dict[str, np.ndarray],
        times: Optional[np.ndarray] = None,
        title: str = "Prediction Comparison",
        save_path: Optional[str] = None
    ):
        """
        Plot comparison of different model predictions.

        Args:
            true_values: True values (T, dim)
            predictions: Dictionary of {name: prediction}
            times: Optional time array
            title: Plot title
            save_path: Path to save figure
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib not available")
            return

        if times is None:
            times = np.arange(len(true_values))

        dim = true_values.shape[1] if len(true_values.shape) > 1 else 1

        fig, axes = plt.subplots(dim, 1, figsize=(12, 3*dim), sharex=True)
        if dim == 1:
            axes = [axes]

        colors = plt.cm.tab10(np.linspace(0, 1, len(predictions)))

        for i, ax in enumerate(axes):
            # True values
            y_true = true_values[:, i] if dim > 1 else true_values
            ax.plot(times, y_true, 'k-', linewidth=2, label='Ground Truth')

            # Predictions
            for (name, pred), color in zip(predictions.items(), colors):
                y_pred = pred[:, i] if dim > 1 else pred
                ax.plot(times, y_pred, color=color, linewidth=1.5,
                       linestyle='--', label=name)

            ax.set_ylabel(f'Dim {i}')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Time')
        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def create_training_report(
    output_dir: str,
    history: Dict,
    metrics: Dict,
    config: Dict
):
    """
    Create comprehensive training report with plots.

    Args:
        output_dir: Output directory
        history: Training history
        metrics: Evaluation metrics
        config: Training configuration
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available for report generation")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Training curves
    viz = TrainingVisualizer()
    viz.plot_training_curves(
        history,
        title="Training Progress",
        save_path=str(output_path / "training_curves.png")
    )

    # Save metrics as JSON
    with open(output_path / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save config
    with open(output_path / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    print("Testing visualizers...")

    if not HAS_MATPLOTLIB:
        print("matplotlib not installed, skipping tests")
    else:
        # Test TrajectoryVisualizer
        viz = TrajectoryVisualizer()

        # Generate test trajectory
        t = np.linspace(0, 10, 500)
        trajectory = np.column_stack([
            10 * np.cos(t),
            10 * np.sin(t),
            80 + 0.5 * t
        ])

        reference = np.column_stack([
            10 * np.cos(t),
            10 * np.sin(t),
            80 * np.ones_like(t)
        ])

        print("\nTesting 3D trajectory plot...")
        viz.plot_trajectory_3d(trajectory, reference=reference, title="Test Trajectory")

        print("Testing 2D trajectory plot...")
        viz.plot_trajectory_2d(trajectory, reference=reference, plane="xy")

        print("Testing tracking error plot...")
        viz.plot_tracking_error(t, trajectory, reference)

        # Test TrainingVisualizer
        train_viz = TrainingVisualizer()

        history = {
            'train_loss': np.exp(-np.linspace(0, 2, 100)).tolist(),
            'val_loss': (np.exp(-np.linspace(0, 2, 100)) * 1.1).tolist(),
            'best_epoch': 80,
            'best_val_loss': 0.15
        }

        print("\nTesting training curves plot...")
        train_viz.plot_training_curves(history)

        print("\nVisualization testing complete!")
