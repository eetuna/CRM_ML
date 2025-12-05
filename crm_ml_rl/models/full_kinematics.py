"""
Full neural network model for forward kinematics (for comparison).

This model learns the complete FK mapping without using physics:
    y = f_nn(currents, state)

Used to compare against residual models.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple

from .networks import MLP, DeepResidualMLP, LSTM_MLP


class FullKinematicsModel(nn.Module):
    """
    Full neural network forward kinematics model.

    Learns complete mapping from inputs to outputs without physics.
    """

    def __init__(
        self,
        input_dim: int = 6,      # currents (3) + base_state (3)
        output_dim: int = 3,     # tip position
        hidden_dims: list = [512, 512, 256, 128],
        use_deep_residual: bool = True,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        """
        Initialize full kinematics model.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dims: Hidden layer dimensions
            use_deep_residual: Use deep residual architecture
            dropout: Dropout probability
            activation: Activation function
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        if use_deep_residual:
            self.network = DeepResidualMLP(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=hidden_dims[0],
                num_blocks=len(hidden_dims),
                dropout=dropout
            )
        else:
            self.network = MLP(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
                activation=activation,
                layer_norm=True
            )

        # Statistics for normalization
        self.register_buffer('input_mean', torch.zeros(input_dim))
        self.register_buffer('input_std', torch.ones(input_dim))
        self.register_buffer('output_mean', torch.zeros(output_dim))
        self.register_buffer('output_std', torch.ones(output_dim))

    def set_normalization(
        self,
        input_mean: np.ndarray,
        input_std: np.ndarray,
        output_mean: np.ndarray,
        output_std: np.ndarray
    ):
        """Set normalization statistics."""
        self.input_mean = torch.tensor(input_mean, dtype=torch.float32)
        self.input_std = torch.tensor(input_std, dtype=torch.float32)
        self.output_mean = torch.tensor(output_mean, dtype=torch.float32)
        self.output_std = torch.tensor(output_std, dtype=torch.float32)

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input."""
        return (x - self.input_mean) / (self.input_std + 1e-8)

    def denormalize_output(self, y: torch.Tensor) -> torch.Tensor:
        """Denormalize output."""
        return y * self.output_std + self.output_mean

    def forward(self, x: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, input_dim)
            normalize: Apply input/output normalization

        Returns:
            Predicted output (batch, output_dim)
        """
        if normalize:
            x = self.normalize_input(x)

        y = self.network(x)

        if normalize:
            y = self.denormalize_output(y)

        return y


class FullKinematicsLSTM(nn.Module):
    """
    LSTM-based full kinematics model for sequential predictions.
    """

    def __init__(
        self,
        input_dim: int = 6,
        output_dim: int = 3,
        lstm_hidden_dim: int = 256,
        lstm_num_layers: int = 2,
        mlp_hidden_dims: list = [256, 128],
        dropout: float = 0.1
    ):
        """
        Initialize LSTM kinematics model.

        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension
            lstm_hidden_dim: LSTM hidden dimension
            lstm_num_layers: Number of LSTM layers
            mlp_hidden_dims: MLP hidden dimensions
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.lstm_mlp = LSTM_MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
            mlp_hidden_dims=mlp_hidden_dims,
            dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input sequence (batch, seq_len, input_dim)
            hidden: LSTM hidden state

        Returns:
            Tuple of (output, new_hidden)
        """
        return self.lstm_mlp(x, hidden)


class FullKinematicsTransformer(nn.Module):
    """
    Transformer-based full kinematics model.
    """

    def __init__(
        self,
        input_dim: int = 6,
        output_dim: int = 3,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        max_seq_len: int = 100,
        dropout: float = 0.1
    ):
        """
        Initialize transformer kinematics model.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input sequence (batch, seq_len, input_dim)

        Returns:
            Output (batch, output_dim) - prediction for last timestep
        """
        batch_size, seq_len, _ = x.shape

        # Project input
        x = self.input_proj(x)

        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]

        # Transformer
        x = self.transformer(x)

        # Use last timestep
        x = x[:, -1, :]

        # Output projection
        return self.output_proj(x)


if __name__ == "__main__":
    print("Testing FullKinematicsModel...")

    model = FullKinematicsModel(
        input_dim=6,
        output_dim=3,
        hidden_dims=[256, 256, 128],
        use_deep_residual=True
    )

    batch_size = 32
    x = torch.randn(batch_size, 6)
    y = model(x)
    print(f"MLP input: {x.shape}, output: {y.shape}")

    print("\nTesting FullKinematicsLSTM...")
    model_lstm = FullKinematicsLSTM(
        input_dim=6,
        output_dim=3,
        lstm_hidden_dim=128
    )

    x_seq = torch.randn(batch_size, 10, 6)
    y, _ = model_lstm(x_seq)
    print(f"LSTM input: {x_seq.shape}, output: {y.shape}")

    print("\nTesting FullKinematicsTransformer...")
    model_trans = FullKinematicsTransformer(
        input_dim=6,
        output_dim=3,
        d_model=64,
        nhead=4,
        num_layers=2
    )

    y = model_trans(x_seq)
    print(f"Transformer input: {x_seq.shape}, output: {y.shape}")
