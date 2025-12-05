"""
Base neural network architectures.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class MLP(nn.Module):
    """
    Multi-Layer Perceptron with configurable architecture.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        output_activation: Optional[str] = None,
        dropout: float = 0.0,
        batch_norm: bool = False,
        layer_norm: bool = False
    ):
        """
        Initialize MLP.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ('relu', 'tanh', 'leaky_relu', 'elu', 'gelu')
            output_activation: Output activation (None for linear)
            dropout: Dropout probability
            batch_norm: Use batch normalization
            layer_norm: Use layer normalization
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Build activation function
        self.activation = self._get_activation(activation)
        self.output_activation = self._get_activation(output_activation) if output_activation else None

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            layers.append(self.activation)

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'sigmoid': nn.Sigmoid(),
            'softplus': nn.Softplus(),
        }
        return activations.get(name, nn.ReLU())

    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out = self.network(x)
        if self.output_activation is not None:
            out = self.output_activation(out)
        return out


class LSTM_MLP(nn.Module):
    """
    LSTM followed by MLP for sequential data.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 2,
        mlp_hidden_dims: List[int] = [256, 256],
        dropout: float = 0.1,
        bidirectional: bool = False
    ):
        """
        Initialize LSTM-MLP.

        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension
            lstm_hidden_dim: LSTM hidden dimension
            lstm_num_layers: Number of LSTM layers
            mlp_hidden_dims: MLP hidden dimensions
            dropout: Dropout probability
            bidirectional: Use bidirectional LSTM
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.bidirectional = bidirectional

        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # MLP
        lstm_output_dim = lstm_hidden_dim * (2 if bidirectional else 1)
        self.mlp = MLP(
            input_dim=lstm_output_dim,
            output_dim=output_dim,
            hidden_dims=mlp_hidden_dims,
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
            x: Input tensor of shape (batch, seq_len, input_dim)
            hidden: Optional initial hidden state

        Returns:
            Tuple of (output, (h_n, c_n))
        """
        # LSTM forward
        lstm_out, hidden = self.lstm(x, hidden)

        # Use last timestep
        if self.bidirectional:
            last_out = lstm_out[:, -1, :]
        else:
            last_out = lstm_out[:, -1, :]

        # MLP forward
        output = self.mlp(last_out)

        return output, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state."""
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(
            self.lstm_num_layers * num_directions,
            batch_size,
            self.lstm_hidden_dim,
            device=device
        )
        c0 = torch.zeros(
            self.lstm_num_layers * num_directions,
            batch_size,
            self.lstm_hidden_dim,
            device=device
        )
        return h0, c0


class ResidualBlock(nn.Module):
    """Residual block for deeper networks."""

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class DeepResidualMLP(nn.Module):
    """
    Deep MLP with residual connections.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout)
            for _ in range(num_blocks)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)


if __name__ == "__main__":
    # Test networks
    print("Testing MLP...")
    mlp = MLP(input_dim=10, output_dim=3, hidden_dims=[64, 64])
    x = torch.randn(32, 10)
    y = mlp(x)
    print(f"MLP input: {x.shape}, output: {y.shape}")

    print("\nTesting LSTM-MLP...")
    lstm_mlp = LSTM_MLP(input_dim=6, output_dim=3, lstm_hidden_dim=64)
    x_seq = torch.randn(32, 10, 6)  # (batch, seq, features)
    y, _ = lstm_mlp(x_seq)
    print(f"LSTM-MLP input: {x_seq.shape}, output: {y.shape}")

    print("\nTesting DeepResidualMLP...")
    deep_mlp = DeepResidualMLP(input_dim=10, output_dim=3)
    x = torch.randn(32, 10)
    y = deep_mlp(x)
    print(f"DeepResidualMLP input: {x.shape}, output: {y.shape}")
