"""
Residual model for forward kinematics.

The residual model learns to correct the physics model predictions:
    y_final = y_physics + f_residual(x, y_physics)

This preserves the physics knowledge while learning to correct errors.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple

from .networks import MLP, DeepResidualMLP


class ResidualKinematicsModel(nn.Module):
    """
    Residual model for kinematics prediction.

    Learns to predict corrections to physics model outputs.
    """

    def __init__(
        self,
        input_dim: int = 6,  # currents (3) + current_position (3)
        output_dim: int = 3,  # position correction
        hidden_dims: list = [256, 256, 128],
        physics_output_dim: int = 3,  # physics model output dimension
        include_physics_in_input: bool = True,
        use_deep_residual: bool = False,
        dropout: float = 0.1,
        residual_scale: float = 1.0,  # Scale factor for residual output
        max_correction: float = 10.0,  # Maximum correction magnitude (mm)
    ):
        """
        Initialize residual kinematics model.

        Args:
            input_dim: Input dimension (currents + state)
            output_dim: Output dimension (position correction)
            hidden_dims: Hidden layer dimensions
            physics_output_dim: Dimension of physics model output
            include_physics_in_input: Include physics prediction in network input
            use_deep_residual: Use deep residual network architecture
            dropout: Dropout probability
            residual_scale: Scale factor for residual output
            max_correction: Maximum allowed correction magnitude
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.physics_output_dim = physics_output_dim
        self.include_physics_in_input = include_physics_in_input
        self.residual_scale = residual_scale
        self.max_correction = max_correction

        # Adjust input dimension if including physics output
        network_input_dim = input_dim
        if include_physics_in_input:
            network_input_dim += physics_output_dim

        # Build network
        if use_deep_residual:
            self.network = DeepResidualMLP(
                input_dim=network_input_dim,
                output_dim=output_dim,
                hidden_dim=hidden_dims[0] if hidden_dims else 256,
                num_blocks=len(hidden_dims),
                dropout=dropout
            )
        else:
            self.network = MLP(
                input_dim=network_input_dim,
                output_dim=output_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
                layer_norm=True
            )

        # Output scaling layer
        self.output_scale = nn.Parameter(torch.ones(output_dim) * residual_scale)

    def forward(
        self,
        x: torch.Tensor,
        physics_output: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (currents, state) of shape (batch, input_dim)
            physics_output: Physics model prediction of shape (batch, physics_output_dim)

        Returns:
            Residual correction of shape (batch, output_dim)
        """
        # Concatenate physics output if provided and enabled
        if self.include_physics_in_input and physics_output is not None:
            network_input = torch.cat([x, physics_output], dim=-1)
        else:
            network_input = x

        # Get raw residual
        raw_residual = self.network(network_input)

        # Scale and clamp residual
        residual = raw_residual * self.output_scale

        # Clamp to maximum correction
        residual = torch.clamp(residual, -self.max_correction, self.max_correction)

        return residual

    def predict_with_physics(
        self,
        x: torch.Tensor,
        physics_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict final output by adding residual to physics output.

        Args:
            x: Input tensor
            physics_output: Physics model prediction

        Returns:
            Final prediction (physics + residual)
        """
        residual = self.forward(x, physics_output)
        return physics_output + residual

    def get_residual_stats(self) -> Dict[str, float]:
        """Get statistics about the residual predictions."""
        return {
            'output_scale': self.output_scale.detach().cpu().numpy().tolist(),
            'max_correction': self.max_correction,
        }


class ResidualKinematicsWithUncertainty(nn.Module):
    """
    Residual kinematics model with uncertainty estimation.

    Predicts both mean correction and variance for uncertainty quantification.
    """

    def __init__(
        self,
        input_dim: int = 6,
        output_dim: int = 3,
        hidden_dims: list = [256, 256, 128],
        min_var: float = 1e-6,
        max_var: float = 100.0
    ):
        """
        Initialize model with uncertainty.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dims: Hidden layer dimensions
            min_var: Minimum variance
            max_var: Maximum variance
        """
        super().__init__()

        self.output_dim = output_dim
        self.min_var = min_var
        self.max_var = max_var

        # Shared encoder
        self.encoder = MLP(
            input_dim=input_dim,
            output_dim=hidden_dims[-1],
            hidden_dims=hidden_dims[:-1],
            layer_norm=True
        )

        # Mean head
        self.mean_head = nn.Linear(hidden_dims[-1], output_dim)

        # Variance head (output log variance for numerical stability)
        self.var_head = nn.Linear(hidden_dims[-1], output_dim)

    def forward(
        self,
        x: torch.Tensor,
        physics_output: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor
            physics_output: Physics model prediction (concatenated if provided)

        Returns:
            Tuple of (mean, variance)
        """
        if physics_output is not None:
            x = torch.cat([x, physics_output], dim=-1)

        features = self.encoder(x)

        mean = self.mean_head(features)
        log_var = self.var_head(features)

        # Clamp variance to valid range
        var = torch.exp(log_var)
        var = torch.clamp(var, self.min_var, self.max_var)

        return mean, var

    def loss(
        self,
        x: torch.Tensor,
        physics_output: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Negative log likelihood loss.

        Args:
            x: Input tensor
            physics_output: Physics model prediction
            target: Target residual (target - physics)

        Returns:
            NLL loss
        """
        mean, var = self.forward(x, physics_output)

        # Target residual
        residual_target = target - physics_output

        # NLL loss
        nll = 0.5 * (torch.log(var) + (residual_target - mean)**2 / var)
        return nll.mean()


if __name__ == "__main__":
    # Test residual kinematics model
    print("Testing ResidualKinematicsModel...")

    model = ResidualKinematicsModel(
        input_dim=6,
        output_dim=3,
        hidden_dims=[128, 128],
        physics_output_dim=3,
        include_physics_in_input=True
    )

    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 6)
    physics_out = torch.randn(batch_size, 3)

    residual = model(x, physics_out)
    print(f"Input: {x.shape}, Physics: {physics_out.shape}, Residual: {residual.shape}")

    final_pred = model.predict_with_physics(x, physics_out)
    print(f"Final prediction: {final_pred.shape}")

    # Test with uncertainty
    print("\nTesting ResidualKinematicsWithUncertainty...")
    model_unc = ResidualKinematicsWithUncertainty(
        input_dim=9,  # 6 + 3 physics
        output_dim=3,
        hidden_dims=[128, 128, 64]
    )

    mean, var = model_unc(x, physics_out)
    print(f"Mean: {mean.shape}, Variance: {var.shape}")
    print(f"Variance range: [{var.min().item():.4f}, {var.max().item():.4f}]")
