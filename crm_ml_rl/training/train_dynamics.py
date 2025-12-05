"""
Training script for dynamics models (both residual and full).

Trains neural network dynamics models on experimental data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import json
from datetime import datetime

from crm_ml_rl.models.residual_dynamics import ResidualDynamicsModel, EnsembleResidualDynamics
from crm_ml_rl.models.full_dynamics import FullDynamicsModel, EnsembleFullDynamics, ProbabilisticFullDynamics
from crm_ml_rl.data.data_loader import CRMDataLoader


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    model_type: str = "residual"  # "residual" or "full"
    state_dim: int = 6
    action_dim: int = 3
    hidden_dims: List[int] = None
    use_ensemble: bool = False
    num_ensemble: int = 5

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 100
    early_stopping_patience: int = 10

    # Data
    train_split: float = 0.8
    sequence_length: int = 1  # Single step prediction
    normalize: bool = True

    # Logging
    log_interval: int = 10
    save_interval: int = 10

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256, 128]


class DynamicsDataset(Dataset):
    """Dataset for dynamics model training."""

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray
    ):
        """
        Initialize dataset.

        Args:
            states: Current states (N, state_dim)
            actions: Actions (N, action_dim)
            next_states: Next states (N, state_dim)
        """
        self.states = torch.tensor(states, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.float32)
        self.next_states = torch.tensor(next_states, dtype=torch.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.actions[idx],
            self.next_states[idx]
        )


class DynamicsTrainer:
    """Trainer for dynamics models."""

    def __init__(
        self,
        config: TrainingConfig,
        physics_model=None,
        device: str = "auto"
    ):
        """
        Initialize trainer.

        Args:
            config: Training configuration
            physics_model: Physics model for residual learning
            device: Device for training
        """
        self.config = config
        self.physics_model = physics_model

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Create model
        self.model = self._create_model()
        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )

        # Loss function
        self.criterion = nn.MSELoss()

        # Normalization stats
        self.state_mean = None
        self.state_std = None
        self.action_mean = None
        self.action_std = None

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'best_val_loss': float('inf'),
            'best_epoch': 0
        }

    def _create_model(self):
        """Create dynamics model based on config."""
        if self.config.model_type == "residual":
            if self.config.use_ensemble:
                return EnsembleResidualDynamics(
                    state_dim=self.config.state_dim,
                    action_dim=self.config.action_dim,
                    hidden_dims=self.config.hidden_dims,
                    num_models=self.config.num_ensemble
                )
            else:
                return ResidualDynamicsModel(
                    state_dim=self.config.state_dim,
                    action_dim=self.config.action_dim,
                    hidden_dims=self.config.hidden_dims
                )
        elif self.config.model_type == "full":
            if self.config.use_ensemble:
                return EnsembleFullDynamics(
                    state_dim=self.config.state_dim,
                    action_dim=self.config.action_dim,
                    hidden_dims=self.config.hidden_dims,
                    num_models=self.config.num_ensemble
                )
            else:
                return FullDynamicsModel(
                    state_dim=self.config.state_dim,
                    action_dim=self.config.action_dim,
                    hidden_dims=self.config.hidden_dims
                )
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

    def compute_normalization(self, states: np.ndarray, actions: np.ndarray):
        """Compute normalization statistics."""
        self.state_mean = states.mean(axis=0)
        self.state_std = states.std(axis=0) + 1e-8
        self.action_mean = actions.mean(axis=0)
        self.action_std = actions.std(axis=0) + 1e-8

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state."""
        return (state - self.state_mean) / self.state_std

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Normalize action."""
        return (action - self.action_mean) / self.action_std

    def prepare_data(
        self,
        data_loader: CRMDataLoader,
        trajectory_types: List[str] = None,
        sampling_times: List[int] = None
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare training and validation data loaders.

        Args:
            data_loader: CRM data loader
            trajectory_types: Types of trajectories to use
            sampling_times: Sampling times to filter (in ms)

        Returns:
            Tuple of (train_loader, val_loader)
        """
        if trajectory_types is None:
            trajectory_types = ["circle", "lemniscate"]

        all_states = []
        all_actions = []
        all_next_states = []

        # Load trajectories
        for traj_type in trajectory_types:
            trajectories = data_loader.load_trajectories(traj_type)

            for traj in trajectories:
                # Filter by sampling time if specified
                if sampling_times is not None:
                    if traj.sampling_time_ms not in sampling_times:
                        continue

                # Get state transitions
                positions = traj.tip_positions  # (T, 3)
                currents = traj.currents  # (3, T)

                # Compute velocities (finite difference)
                dt = traj.sampling_time_ms / 1000.0
                velocities = np.diff(positions, axis=0) / dt
                velocities = np.vstack([velocities, velocities[-1:]])  # Pad

                # Build states: [position, velocity]
                states = np.hstack([positions, velocities])  # (T, 6)

                # Actions are currents
                actions = currents.T  # (T, 3)

                # Create transitions
                for t in range(len(states) - 1):
                    all_states.append(states[t])
                    all_actions.append(actions[t])
                    all_next_states.append(states[t + 1])

        all_states = np.array(all_states)
        all_actions = np.array(all_actions)
        all_next_states = np.array(all_next_states)

        print(f"Total transitions: {len(all_states)}")

        # Compute normalization
        if self.config.normalize:
            self.compute_normalization(all_states, all_actions)
            all_states = self.normalize_state(all_states)
            all_actions = self.normalize_action(all_actions)
            all_next_states = self.normalize_state(all_next_states)

        # Create dataset
        dataset = DynamicsDataset(all_states, all_actions, all_next_states)

        # Split
        train_size = int(len(dataset) * self.config.train_split)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )

        return train_loader, val_loader

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for states, actions, next_states in train_loader:
            states = states.to(self.device)
            actions = actions.to(self.device)
            next_states = next_states.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            if self.config.use_ensemble:
                pred_mean, pred_std = self.model(states, actions)
                predicted = pred_mean
            else:
                if self.config.model_type == "residual":
                    # For residual, we need physics prediction
                    if self.physics_model is not None:
                        with torch.no_grad():
                            physics_pred = self.physics_model(states, actions)
                        predicted = self.model(states, actions, physics_pred)
                    else:
                        # Simple physics: next = current + dt * action
                        physics_pred = states.clone()
                        physics_pred[:, :3] += 0.02 * states[:, 3:6]  # position update
                        physics_pred[:, 3:6] += 0.02 * actions * 100  # velocity update
                        predicted = self.model(states, actions, physics_pred)
                else:
                    predicted = self.model(states, actions)

            # Loss
            loss = self.criterion(predicted, next_states)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate(self, val_loader: DataLoader) -> float:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for states, actions, next_states in val_loader:
                states = states.to(self.device)
                actions = actions.to(self.device)
                next_states = next_states.to(self.device)

                if self.config.use_ensemble:
                    pred_mean, _ = self.model(states, actions)
                    predicted = pred_mean
                else:
                    if self.config.model_type == "residual":
                        physics_pred = states.clone()
                        physics_pred[:, :3] += 0.02 * states[:, 3:6]
                        physics_pred[:, 3:6] += 0.02 * actions * 100
                        predicted = self.model(states, actions, physics_pred)
                    else:
                        predicted = self.model(states, actions)

                loss = self.criterion(predicted, next_states)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_dir: Optional[str] = None
    ) -> Dict:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            save_dir: Directory to save checkpoints

        Returns:
            Training history
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        patience_counter = 0

        for epoch in range(self.config.epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)

            # Validate
            val_loss = self.validate(val_loader)
            self.history['val_loss'].append(val_loss)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Early stopping
            if val_loss < self.history['best_val_loss']:
                self.history['best_val_loss'] = val_loss
                self.history['best_epoch'] = epoch
                patience_counter = 0

                if save_dir:
                    self.save(save_dir / "best_model.pt")
            else:
                patience_counter += 1

            # Logging
            if epoch % self.config.log_interval == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, "
                      f"Val Loss = {val_loss:.6f}, "
                      f"LR = {self.optimizer.param_groups[0]['lr']:.2e}")

            # Save checkpoint
            if save_dir and epoch % self.config.save_interval == 0:
                self.save(save_dir / f"checkpoint_epoch_{epoch}.pt")

            # Early stopping check
            if patience_counter >= self.config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Save final model
        if save_dir:
            self.save(save_dir / "final_model.pt")

            # Save config and history
            with open(save_dir / "config.json", 'w') as f:
                json.dump(self.config.__dict__, f, indent=2)
            with open(save_dir / "history.json", 'w') as f:
                json.dump(self.history, f, indent=2)

        return self.history

    def save(self, path: str):
        """Save model and normalization stats."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'state_mean': self.state_mean,
            'state_std': self.state_std,
            'action_mean': self.action_mean,
            'action_std': self.action_std,
            'history': self.history
        }
        torch.save(checkpoint, path)

    def load(self, path: str):
        """Load model and normalization stats."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.state_mean = checkpoint.get('state_mean')
        self.state_std = checkpoint.get('state_std')
        self.action_mean = checkpoint.get('action_mean')
        self.action_std = checkpoint.get('action_std')
        self.history = checkpoint.get('history', self.history)


def train_dynamics_model(
    data_dir: str,
    output_dir: str,
    model_type: str = "residual",
    use_ensemble: bool = False,
    sampling_times: List[int] = None,
    epochs: int = 100,
    batch_size: int = 64
):
    """
    Main training function.

    Args:
        data_dir: Directory containing experimental data
        output_dir: Directory to save models
        model_type: "residual" or "full"
        use_ensemble: Whether to use ensemble
        sampling_times: Sampling times to include (in ms)
        epochs: Number of training epochs
        batch_size: Batch size
    """
    # Default to slower data (>=20ms)
    if sampling_times is None:
        sampling_times = [20, 25, 50, 100]

    print(f"Training {model_type} dynamics model")
    print(f"Using sampling times: {sampling_times} ms")

    # Create config
    config = TrainingConfig(
        model_type=model_type,
        use_ensemble=use_ensemble,
        epochs=epochs,
        batch_size=batch_size
    )

    # Create trainer
    trainer = DynamicsTrainer(config)

    # Load data
    data_loader = CRMDataLoader(data_dir)

    # Prepare data
    train_loader, val_loader = trainer.prepare_data(
        data_loader,
        trajectory_types=["circle", "lemniscate"],
        sampling_times=sampling_times
    )

    # Train
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(output_dir) / f"{model_type}_{timestamp}"

    history = trainer.train(train_loader, val_loader, save_dir=str(save_dir))

    print(f"\nTraining complete!")
    print(f"Best validation loss: {history['best_val_loss']:.6f} at epoch {history['best_epoch']}")
    print(f"Model saved to: {save_dir}")

    return trainer, history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train dynamics model")
    parser.add_argument("--data-dir", type=str, default="3D_dynamic_response_data_0124",
                        help="Data directory")
    parser.add_argument("--output-dir", type=str, default="crm_ml_rl/trained_models",
                        help="Output directory")
    parser.add_argument("--model-type", type=str, default="residual",
                        choices=["residual", "full"], help="Model type")
    parser.add_argument("--use-ensemble", action="store_true", help="Use ensemble")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--sampling-times", type=int, nargs="+", default=[20, 25, 50, 100],
                        help="Sampling times in ms")

    args = parser.parse_args()

    train_dynamics_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_type=args.model_type,
        use_ensemble=args.use_ensemble,
        sampling_times=args.sampling_times,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
