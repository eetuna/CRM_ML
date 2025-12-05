"""
Residual model for dynamics prediction.

Learns corrections to physics-based dynamics model:
    x_{t+1} = f_physics(x_t, u_t) + f_residual(x_t, u_t, f_physics(x_t, u_t))
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple

from .networks import MLP, LSTM_MLP, DeepResidualMLP


class ResidualDynamicsModel(nn.Module):
    """
    Residual model for dynamics prediction.

    Learns to correct physics model dynamics predictions.
    """

    def __init__(
        self,
        state_dim: int = 6,      # position (3) + velocity (3)
        action_dim: int = 3,     # currents
        hidden_dims: list = [256, 256, 128],
        include_physics_in_input: bool = True,
        use_deep_residual: bool = False,
        dropout: float = 0.1,
        max_correction: float = 5.0,  # Maximum correction per timestep
    ):
        """
        Initialize residual dynamics model.

        Args:
            state_dim: State dimension
            action_dim: Action/control dimension
            hidden_dims: Hidden layer dimensions
            include_physics_in_input: Include physics prediction in input
            use_deep_residual: Use deep residual architecture
            dropout: Dropout probability
            max_correction: Maximum correction magnitude
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.include_physics_in_input = include_physics_in_input
        self.max_correction = max_correction

        # Input: state + action + (physics prediction)
        input_dim = state_dim + action_dim
        if include_physics_in_input:
            input_dim += state_dim  # Physics predicted next state

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

        # Learnable scaling
        self.output_scale = nn.Parameter(torch.ones(state_dim) * 0.1)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        physics_next_state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: Current state (batch, state_dim)
            action: Action/control input (batch, action_dim)
            physics_next_state: Physics model predicted next state

        Returns:
            State correction (batch, state_dim)
        """
        # Build input
        inputs = [state, action]
        if self.include_physics_in_input and physics_next_state is not None:
            inputs.append(physics_next_state)

        x = torch.cat(inputs, dim=-1)

        # Get residual
        residual = self.network(x) * self.output_scale

        # Clamp
        residual = torch.clamp(residual, -self.max_correction, self.max_correction)

        return residual

    def predict_next_state(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        physics_next_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict corrected next state.

        Args:
            state: Current state
            action: Action
            physics_next_state: Physics model prediction

        Returns:
            Corrected next state
        """
        residual = self.forward(state, action, physics_next_state)
        return physics_next_state + residual


class ResidualDynamicsLSTM(nn.Module):
    """
    LSTM-based residual dynamics model for sequential predictions.

    Uses history of states and actions to predict corrections.
    """

    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int = 3,
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 2,
        mlp_hidden_dims: list = [128, 64],
        max_correction: float = 5.0
    ):
        """
        Initialize LSTM residual dynamics model.

        Args:
            state_dim: State dimension
            action_dim: Action dimension
            lstm_hidden_dim: LSTM hidden dimension
            lstm_num_layers: Number of LSTM layers
            mlp_hidden_dims: MLP hidden dimensions
            max_correction: Maximum correction
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_correction = max_correction

        # Input: state + action + physics_next_state
        input_dim = state_dim + action_dim + state_dim

        self.lstm_mlp = LSTM_MLP(
            input_dim=input_dim,
            output_dim=state_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
            mlp_hidden_dims=mlp_hidden_dims
        )

        self.output_scale = nn.Parameter(torch.ones(state_dim) * 0.1)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        physics_next_states: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with sequence.

        Args:
            states: State sequence (batch, seq_len, state_dim)
            actions: Action sequence (batch, seq_len, action_dim)
            physics_next_states: Physics predictions (batch, seq_len, state_dim)
            hidden: LSTM hidden state

        Returns:
            Tuple of (residual, new_hidden)
        """
        # Concatenate inputs
        x = torch.cat([states, actions, physics_next_states], dim=-1)

        # LSTM forward
        residual, hidden = self.lstm_mlp(x, hidden)

        # Scale and clamp
        residual = residual * self.output_scale
        residual = torch.clamp(residual, -self.max_correction, self.max_correction)

        return residual, hidden


class EnsembleResidualDynamics(nn.Module):
    """
    Ensemble of residual dynamics models for uncertainty estimation.
    """

    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int = 3,
        hidden_dims: list = [256, 256],
        num_models: int = 5,
        **kwargs
    ):
        """
        Initialize ensemble.

        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dims: Hidden dimensions for each model
            num_models: Number of ensemble members
        """
        super().__init__()

        self.num_models = num_models
        self.state_dim = state_dim

        # Create ensemble
        self.models = nn.ModuleList([
            ResidualDynamicsModel(
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
        action: torch.Tensor,
        physics_next_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through ensemble.

        Returns:
            Tuple of (mean_residual, std_residual)
        """
        predictions = []
        for model in self.models:
            pred = model(state, action, physics_next_state)
            predictions.append(pred)

        predictions = torch.stack(predictions, dim=0)  # (num_models, batch, state_dim)

        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)

        return mean, std

    def predict_next_state(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        physics_next_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next state with uncertainty.

        Returns:
            Tuple of (mean_next_state, uncertainty)
        """
        mean_residual, std_residual = self.forward(state, action, physics_next_state)
        mean_next = physics_next_state + mean_residual
        return mean_next, std_residual

    def get_epistemic_uncertainty(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        physics_next_state: torch.Tensor
    ) -> torch.Tensor:
        """Get epistemic uncertainty (disagreement between models)."""
        _, std = self.forward(state, action, physics_next_state)
        return std.mean(dim=-1)


if __name__ == "__main__":
    print("Testing ResidualDynamicsModel...")

    model = ResidualDynamicsModel(
        state_dim=6,
        action_dim=3,
        hidden_dims=[128, 128],
        include_physics_in_input=True
    )

    batch_size = 32
    state = torch.randn(batch_size, 6)
    action = torch.randn(batch_size, 3)
    physics_next = torch.randn(batch_size, 6)

    residual = model(state, action, physics_next)
    print(f"Residual shape: {residual.shape}")

    next_state = model.predict_next_state(state, action, physics_next)
    print(f"Predicted next state: {next_state.shape}")

    print("\nTesting EnsembleResidualDynamics...")
    ensemble = EnsembleResidualDynamics(
        state_dim=6,
        action_dim=3,
        num_models=3
    )

    mean, std = ensemble.forward(state, action, physics_next)
    print(f"Mean: {mean.shape}, Std: {std.shape}")

    uncertainty = ensemble.get_epistemic_uncertainty(state, action, physics_next)
    print(f"Uncertainty: {uncertainty.shape}, range: [{uncertainty.min():.4f}, {uncertainty.max():.4f}]")
