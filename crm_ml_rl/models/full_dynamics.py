"""
Full neural network model for dynamics (for comparison).

Learns complete dynamics mapping without physics:
    x_{t+1} = f_nn(x_t, u_t)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple

from .networks import MLP, DeepResidualMLP, LSTM_MLP


class FullDynamicsModel(nn.Module):
    """
    Full neural network dynamics model.

    Learns complete state transition without physics model.
    """

    def __init__(
        self,
        state_dim: int = 6,      # position (3) + velocity (3)
        action_dim: int = 3,     # currents
        hidden_dims: list = [512, 512, 256, 128],
        use_deep_residual: bool = True,
        predict_delta: bool = True,  # Predict state change vs absolute state
        dropout: float = 0.1
    ):
        """
        Initialize full dynamics model.

        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer dimensions
            use_deep_residual: Use deep residual architecture
            predict_delta: Predict delta state (helps with learning)
            dropout: Dropout probability
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.predict_delta = predict_delta

        input_dim = state_dim + action_dim

        if use_deep_residual:
            self.network = DeepResidualMLP(
                input_dim=input_dim,
                output_dim=state_dim,
                hidden_dim=hidden_dims[0],
                num_blocks=len(hidden_dims),
                dropout=dropout
            )
        else:
            self.network = MLP(
                input_dim=input_dim,
                output_dim=state_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
                layer_norm=True
            )

        # Normalization buffers
        self.register_buffer('state_mean', torch.zeros(state_dim))
        self.register_buffer('state_std', torch.ones(state_dim))
        self.register_buffer('action_mean', torch.zeros(action_dim))
        self.register_buffer('action_std', torch.ones(action_dim))

    def set_normalization(
        self,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        action_mean: np.ndarray,
        action_std: np.ndarray
    ):
        """Set normalization statistics."""
        self.state_mean = torch.tensor(state_mean, dtype=torch.float32)
        self.state_std = torch.tensor(state_std, dtype=torch.float32)
        self.action_mean = torch.tensor(action_mean, dtype=torch.float32)
        self.action_std = torch.tensor(action_std, dtype=torch.float32)

    def normalize_state(self, s: torch.Tensor) -> torch.Tensor:
        return (s - self.state_mean) / (self.state_std + 1e-8)

    def denormalize_state(self, s: torch.Tensor) -> torch.Tensor:
        return s * self.state_std + self.state_mean

    def normalize_action(self, a: torch.Tensor) -> torch.Tensor:
        return (a - self.action_mean) / (self.action_std + 1e-8)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        normalize: bool = False
    ) -> torch.Tensor:
        """
        Predict next state.

        Args:
            state: Current state (batch, state_dim)
            action: Action (batch, action_dim)
            normalize: Apply normalization

        Returns:
            Next state (batch, state_dim)
        """
        if normalize:
            state_norm = self.normalize_state(state)
            action_norm = self.normalize_action(action)
            x = torch.cat([state_norm, action_norm], dim=-1)
        else:
            x = torch.cat([state, action], dim=-1)

        output = self.network(x)

        if self.predict_delta:
            # Output is delta, add to current state
            if normalize:
                next_state = self.denormalize_state(
                    self.normalize_state(state) + output
                )
            else:
                next_state = state + output
        else:
            if normalize:
                next_state = self.denormalize_state(output)
            else:
                next_state = output

        return next_state

    def multi_step_prediction(
        self,
        initial_state: torch.Tensor,
        actions: torch.Tensor,
        normalize: bool = False
    ) -> torch.Tensor:
        """
        Multi-step rollout prediction.

        Args:
            initial_state: Initial state (batch, state_dim)
            actions: Action sequence (batch, horizon, action_dim)
            normalize: Apply normalization

        Returns:
            State trajectory (batch, horizon+1, state_dim)
        """
        batch_size, horizon, _ = actions.shape

        states = [initial_state]
        state = initial_state

        for t in range(horizon):
            state = self.forward(state, actions[:, t], normalize=normalize)
            states.append(state)

        return torch.stack(states, dim=1)


class FullDynamicsLSTM(nn.Module):
    """
    LSTM-based full dynamics model.
    """

    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int = 3,
        lstm_hidden_dim: int = 256,
        lstm_num_layers: int = 2,
        mlp_hidden_dims: list = [256, 128],
        dropout: float = 0.1
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.lstm_mlp = LSTM_MLP(
            input_dim=state_dim + action_dim,
            output_dim=state_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
            mlp_hidden_dims=mlp_hidden_dims,
            dropout=dropout
        )

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with sequence.

        Args:
            states: State sequence (batch, seq_len, state_dim)
            actions: Action sequence (batch, seq_len, action_dim)
            hidden: LSTM hidden state

        Returns:
            Tuple of (predicted_next_state, new_hidden)
        """
        x = torch.cat([states, actions], dim=-1)
        delta, hidden = self.lstm_mlp(x, hidden)

        # Predict delta, add to last state
        next_state = states[:, -1, :] + delta

        return next_state, hidden


class EnsembleFullDynamics(nn.Module):
    """
    Ensemble of full dynamics models for uncertainty estimation.
    """

    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int = 3,
        hidden_dims: list = [256, 256],
        num_models: int = 5,
        **kwargs
    ):
        super().__init__()

        self.num_models = num_models
        self.state_dim = state_dim

        self.models = nn.ModuleList([
            FullDynamicsModel(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=hidden_dims,
                **kwargs
            )
            for _ in range(num_models)
        ])

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through ensemble.

        Returns:
            Tuple of (mean_next_state, std_next_state)
        """
        predictions = []
        for model in self.models:
            pred = model(state, action)
            predictions.append(pred)

        predictions = torch.stack(predictions, dim=0)

        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)

        return mean, std

    def sample_prediction(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """Sample prediction from random ensemble member."""
        idx = np.random.randint(self.num_models)
        return self.models[idx](state, action)


class ProbabilisticFullDynamics(nn.Module):
    """
    Probabilistic dynamics model that outputs mean and variance.
    """

    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int = 3,
        hidden_dims: list = [256, 256, 128],
        min_var: float = 1e-4,
        max_var: float = 10.0
    ):
        super().__init__()

        self.state_dim = state_dim
        self.min_var = min_var
        self.max_var = max_var

        input_dim = state_dim + action_dim

        # Shared encoder
        self.encoder = MLP(
            input_dim=input_dim,
            output_dim=hidden_dims[-1],
            hidden_dims=hidden_dims[:-1],
            layer_norm=True
        )

        # Mean head (predicts delta)
        self.mean_head = nn.Linear(hidden_dims[-1], state_dim)

        # Log variance head
        self.logvar_head = nn.Linear(hidden_dims[-1], state_dim)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            Tuple of (next_state_mean, next_state_var)
        """
        x = torch.cat([state, action], dim=-1)
        features = self.encoder(x)

        delta_mean = self.mean_head(features)
        log_var = self.logvar_head(features)

        # Next state mean
        next_state_mean = state + delta_mean

        # Variance
        var = torch.exp(log_var)
        var = torch.clamp(var, self.min_var, self.max_var)

        return next_state_mean, var

    def sample(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        num_samples: int = 1
    ) -> torch.Tensor:
        """Sample next states from predicted distribution."""
        mean, var = self.forward(state, action)
        std = torch.sqrt(var)

        if num_samples == 1:
            return mean + std * torch.randn_like(mean)
        else:
            samples = []
            for _ in range(num_samples):
                sample = mean + std * torch.randn_like(mean)
                samples.append(sample)
            return torch.stack(samples, dim=0)

    def loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor
    ) -> torch.Tensor:
        """Negative log likelihood loss."""
        mean, var = self.forward(state, action)
        nll = 0.5 * (torch.log(var) + (next_state - mean)**2 / var)
        return nll.mean()


if __name__ == "__main__":
    print("Testing FullDynamicsModel...")

    model = FullDynamicsModel(
        state_dim=6,
        action_dim=3,
        hidden_dims=[256, 256],
        predict_delta=True
    )

    batch_size = 32
    state = torch.randn(batch_size, 6)
    action = torch.randn(batch_size, 3)

    next_state = model(state, action)
    print(f"Next state: {next_state.shape}")

    # Multi-step
    actions_seq = torch.randn(batch_size, 10, 3)
    trajectory = model.multi_step_prediction(state, actions_seq)
    print(f"Trajectory: {trajectory.shape}")

    print("\nTesting EnsembleFullDynamics...")
    ensemble = EnsembleFullDynamics(state_dim=6, action_dim=3, num_models=3)
    mean, std = ensemble(state, action)
    print(f"Mean: {mean.shape}, Std: {std.shape}")

    print("\nTesting ProbabilisticFullDynamics...")
    prob_model = ProbabilisticFullDynamics(state_dim=6, action_dim=3)
    mean, var = prob_model(state, action)
    print(f"Mean: {mean.shape}, Var: {var.shape}")

    samples = prob_model.sample(state, action, num_samples=5)
    print(f"Samples: {samples.shape}")
