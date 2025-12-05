"""
Evaluation utilities for dynamics models and RL agents.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Type
from pathlib import Path
from dataclasses import dataclass
import json

from crm_ml_rl.envs import CatheterEnv, TrackingEnv, ReachingEnv, CatheterEnvConfig


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    # Position metrics
    mean_position_error: float = 0.0
    max_position_error: float = 0.0
    rmse_position: float = 0.0

    # Trajectory metrics
    tracking_accuracy: float = 0.0  # Percentage within threshold
    mean_distance: float = 0.0

    # Prediction metrics (for dynamics)
    single_step_error: float = 0.0
    multi_step_error: float = 0.0
    prediction_horizon: int = 0

    # RL metrics
    mean_reward: float = 0.0
    std_reward: float = 0.0
    success_rate: float = 0.0
    mean_episode_length: float = 0.0


class ModelEvaluator:
    """Evaluator for dynamics models."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "auto"
    ):
        """
        Initialize evaluator.

        Args:
            model: Dynamics model to evaluate
            device: Device for evaluation
        """
        self.model = model
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)
        self.model.eval()

    def evaluate_single_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        physics_predictions: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Evaluate single-step prediction accuracy.

        Args:
            states: Current states (N, state_dim)
            actions: Actions (N, action_dim)
            next_states: True next states (N, state_dim)
            physics_predictions: Optional physics model predictions

        Returns:
            Evaluation metrics dictionary
        """
        with torch.no_grad():
            states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
            actions_t = torch.tensor(actions, dtype=torch.float32, device=self.device)

            # Get predictions
            if physics_predictions is not None:
                physics_t = torch.tensor(physics_predictions, dtype=torch.float32, device=self.device)
                predicted = self.model(states_t, actions_t, physics_t)
            else:
                predicted = self.model(states_t, actions_t)

            predicted = predicted.cpu().numpy()

        # Compute errors
        position_error = np.linalg.norm(predicted[:, :3] - next_states[:, :3], axis=1)
        velocity_error = np.linalg.norm(predicted[:, 3:6] - next_states[:, 3:6], axis=1)
        full_error = np.linalg.norm(predicted - next_states, axis=1)

        metrics = {
            'mean_position_error': float(np.mean(position_error)),
            'max_position_error': float(np.max(position_error)),
            'std_position_error': float(np.std(position_error)),
            'rmse_position': float(np.sqrt(np.mean(position_error**2))),
            'mean_velocity_error': float(np.mean(velocity_error)),
            'mean_full_error': float(np.mean(full_error)),
            'rmse_full': float(np.sqrt(np.mean(full_error**2)))
        }

        return metrics

    def evaluate_multi_step(
        self,
        initial_states: np.ndarray,
        action_sequences: np.ndarray,
        true_trajectories: np.ndarray,
        horizons: List[int] = None
    ) -> Dict:
        """
        Evaluate multi-step rollout prediction.

        Args:
            initial_states: Initial states (N, state_dim)
            action_sequences: Action sequences (N, T, action_dim)
            true_trajectories: True state trajectories (N, T+1, state_dim)
            horizons: Horizons to evaluate at

        Returns:
            Evaluation metrics dictionary
        """
        if horizons is None:
            horizons = [1, 5, 10, 20, 50]

        n_samples, T, _ = action_sequences.shape

        # Predict trajectories
        with torch.no_grad():
            states_t = torch.tensor(initial_states, dtype=torch.float32, device=self.device)
            actions_t = torch.tensor(action_sequences, dtype=torch.float32, device=self.device)

            predicted_traj = [states_t.unsqueeze(1)]
            state = states_t

            for t in range(T):
                next_state = self.model(state, actions_t[:, t])
                predicted_traj.append(next_state.unsqueeze(1))
                state = next_state

            predicted_traj = torch.cat(predicted_traj, dim=1).cpu().numpy()

        # Compute errors at different horizons
        metrics = {}
        for h in horizons:
            if h > T:
                continue

            pos_error = np.linalg.norm(
                predicted_traj[:, h, :3] - true_trajectories[:, h, :3],
                axis=1
            )
            metrics[f'position_error_h{h}'] = float(np.mean(pos_error))
            metrics[f'rmse_h{h}'] = float(np.sqrt(np.mean(pos_error**2)))

        # Trajectory-level metrics
        traj_errors = np.linalg.norm(
            predicted_traj[:, :, :3] - true_trajectories[:, :, :3],
            axis=2
        )
        metrics['mean_trajectory_error'] = float(np.mean(traj_errors))
        metrics['max_trajectory_error'] = float(np.max(traj_errors))

        return metrics

    def compare_with_physics(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        physics_predictions: np.ndarray
    ) -> Dict:
        """
        Compare ML model with physics baseline.

        Args:
            states: Current states
            actions: Actions
            next_states: True next states
            physics_predictions: Physics model predictions

        Returns:
            Comparison metrics
        """
        # ML model predictions
        ml_metrics = self.evaluate_single_step(states, actions, next_states, physics_predictions)

        # Physics model errors
        physics_pos_error = np.linalg.norm(
            physics_predictions[:, :3] - next_states[:, :3],
            axis=1
        )

        physics_metrics = {
            'physics_mean_error': float(np.mean(physics_pos_error)),
            'physics_rmse': float(np.sqrt(np.mean(physics_pos_error**2)))
        }

        # Improvement
        improvement = {
            'error_reduction': (physics_metrics['physics_mean_error'] - ml_metrics['mean_position_error']) /
                             physics_metrics['physics_mean_error'] * 100,
            'rmse_reduction': (physics_metrics['physics_rmse'] - ml_metrics['rmse_position']) /
                            physics_metrics['physics_rmse'] * 100
        }

        return {
            'ml_model': ml_metrics,
            'physics_model': physics_metrics,
            'improvement': improvement
        }


class RLEvaluator:
    """Evaluator for RL agents."""

    def __init__(
        self,
        env: CatheterEnv,
        deterministic: bool = True
    ):
        """
        Initialize evaluator.

        Args:
            env: Environment for evaluation
            deterministic: Use deterministic actions
        """
        self.env = env
        self.deterministic = deterministic

    def evaluate_agent(
        self,
        agent,
        n_episodes: int = 100,
        render: bool = False,
        verbose: bool = True
    ) -> Dict:
        """
        Evaluate RL agent.

        Args:
            agent: RL agent with predict method
            n_episodes: Number of evaluation episodes
            render: Whether to render
            verbose: Print progress

        Returns:
            Evaluation metrics
        """
        episode_rewards = []
        episode_lengths = []
        successes = []
        all_distances = []
        all_tracking_errors = []

        for ep in range(n_episodes):
            obs, info = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            episode_distances = []

            while not done:
                action = agent.predict(obs, deterministic=self.deterministic)
                if isinstance(action, tuple):
                    action = action[0]

                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1

                # Track metrics
                if 'distance' in info:
                    episode_distances.append(info['distance'])

                if render:
                    self.env.render()

            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            successes.append(info.get('success', info.get('reached_target', False)))
            all_distances.extend(episode_distances)

            if verbose and (ep + 1) % 10 == 0:
                print(f"Episode {ep + 1}/{n_episodes}: "
                      f"Reward = {total_reward:.2f}, "
                      f"Length = {steps}")

        # Compute metrics
        metrics = EvaluationMetrics(
            mean_reward=float(np.mean(episode_rewards)),
            std_reward=float(np.std(episode_rewards)),
            success_rate=float(np.mean(successes)),
            mean_episode_length=float(np.mean(episode_lengths)),
            mean_distance=float(np.mean(all_distances)) if all_distances else 0.0,
            tracking_accuracy=float(np.mean([d < 3.0 for d in all_distances])) if all_distances else 0.0
        )

        return {
            'metrics': metrics.__dict__,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'successes': successes
        }

    def compare_agents(
        self,
        agents: Dict,
        n_episodes: int = 50
    ) -> Dict:
        """
        Compare multiple agents.

        Args:
            agents: Dictionary of {name: agent}
            n_episodes: Episodes per agent

        Returns:
            Comparison results
        """
        results = {}

        for name, agent in agents.items():
            print(f"\nEvaluating {name}...")
            results[name] = self.evaluate_agent(
                agent,
                n_episodes=n_episodes,
                verbose=False
            )

        # Summary
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"{'Agent':<15} {'Reward':>12} {'Success':>10} {'Length':>10}")
        print("-"*60)

        for name, result in results.items():
            m = result['metrics']
            print(f"{name:<15} {m['mean_reward']:>12.2f} "
                  f"{m['success_rate']*100:>9.1f}% "
                  f"{m['mean_episode_length']:>10.1f}")

        return results

    def evaluate_trajectory_tracking(
        self,
        agent,
        trajectory_types: List[str] = None,
        n_episodes_per_type: int = 10
    ) -> Dict:
        """
        Evaluate agent on different trajectory types.

        Args:
            agent: RL agent
            trajectory_types: Trajectory types to evaluate
            n_episodes_per_type: Episodes per trajectory type

        Returns:
            Evaluation results per trajectory type
        """
        if trajectory_types is None:
            trajectory_types = ["circle", "lemniscate", "spiral"]

        results = {}

        for traj_type in trajectory_types:
            print(f"\nEvaluating on {traj_type} trajectory...")

            # Create environment with specific trajectory
            if hasattr(self.env, 'trajectory_type'):
                self.env.trajectory_type = traj_type

            result = self.evaluate_agent(
                agent,
                n_episodes=n_episodes_per_type,
                verbose=False
            )
            results[traj_type] = result

        return results


def load_and_evaluate_model(
    model_path: str,
    test_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    model_class: Type = None
) -> Dict:
    """
    Load model and evaluate on test data.

    Args:
        model_path: Path to saved model
        test_data: Tuple of (states, actions, next_states)
        model_class: Model class for loading

    Returns:
        Evaluation results
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')

    # Create and load model
    if model_class is not None:
        config = checkpoint.get('config', {})
        model = model_class(
            state_dim=config.get('state_dim', 6),
            action_dim=config.get('action_dim', 3),
            hidden_dims=config.get('hidden_dims', [256, 256])
        )
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise ValueError("model_class required for loading")

    # Evaluate
    evaluator = ModelEvaluator(model)
    states, actions, next_states = test_data

    metrics = evaluator.evaluate_single_step(states, actions, next_states)

    return metrics


if __name__ == "__main__":
    print("Testing evaluators...")

    # Test ModelEvaluator with dummy model
    from crm_ml_rl.models.full_dynamics import FullDynamicsModel

    model = FullDynamicsModel(state_dim=6, action_dim=3)
    evaluator = ModelEvaluator(model)

    # Generate dummy data
    n_samples = 100
    states = np.random.randn(n_samples, 6).astype(np.float32)
    actions = np.random.randn(n_samples, 3).astype(np.float32)
    next_states = states + 0.1 * np.random.randn(n_samples, 6).astype(np.float32)

    metrics = evaluator.evaluate_single_step(states, actions, next_states)
    print(f"\nSingle-step metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Test RLEvaluator with dummy agent
    print("\nTesting RLEvaluator...")

    class DummyAgent:
        def predict(self, obs, deterministic=True):
            return np.zeros(3), None

    env = ReachingEnv()
    rl_evaluator = RLEvaluator(env)

    dummy_agent = DummyAgent()
    results = rl_evaluator.evaluate_agent(dummy_agent, n_episodes=5, verbose=True)

    print(f"\nRL evaluation results:")
    for k, v in results['metrics'].items():
        print(f"  {k}: {v:.4f}")

    print("\nEvaluator testing complete!")
