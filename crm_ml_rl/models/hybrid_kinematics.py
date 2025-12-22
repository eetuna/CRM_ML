"""
Hybrid Kinematics Model - Integrates CRM Physics with Neural Network Residuals.

This model tightly couples the C++ CRM physics engine with a learned residual
correction network:

    y_final = CRM_FK(currents, insertion_length) + f_residual(x, CRM_FK(...))

The physics model provides the base prediction, and the neural network learns
to correct systematic errors in the physics model.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, Union
from pathlib import Path

from .residual_kinematics import ResidualKinematicsModel, ResidualKinematicsWithUncertainty
from .networks import MLP
from ..wrappers.crm_wrapper import CRMWrapper, HAS_CPP_BINDINGS


class HybridKinematicsModel(nn.Module):
    """
    Hybrid Forward Kinematics Model combining CRM physics with learned residuals.

    This model:
    1. Runs CRM forward kinematics (C++ or simplified) to get physics prediction
    2. Passes the input + physics prediction through a residual neural network
    3. Returns the corrected prediction: physics + residual

    The residual network learns to correct systematic errors in the physics model,
    such as unmodeled friction, material nonlinearities, or manufacturing variations.
    """

    def __init__(
        self,
        param_file: Optional[str] = None,
        config_file: Optional[str] = None,
        use_cpp: bool = True,
        hidden_dims: list = [256, 256, 128],
        dropout: float = 0.1,
        max_correction: float = 10.0,
        use_deep_residual: bool = False,
        include_physics_in_input: bool = True,
        learnable_physics_weight: bool = False,
        use_torch_physics: bool = False,
        device: str = "cpu"
    ):
        """
        Initialize hybrid kinematics model.

        Args:
            param_file: Path to CRM catheter parameter file
            config_file: Path to CRM catheter configuration file
            use_cpp: Whether to use C++ physics bindings
            hidden_dims: Hidden layer dimensions for residual network
            dropout: Dropout probability
            max_correction: Maximum allowed residual correction (mm)
            use_deep_residual: Use deep residual network architecture
            include_physics_in_input: Include physics prediction in residual network input
            learnable_physics_weight: Learn a weight for physics vs residual blend
            device: Device for neural network computations
        """
        super().__init__()

        self.device = torch.device(device)
        self.use_cpp = use_cpp
        self.include_physics_in_input = include_physics_in_input
        self.use_torch_physics = use_torch_physics and use_cpp

        self._torch_physics = None
        if self.use_torch_physics:
            try:
                from ..wrappers.torch_physics import TorchCRMPhysics
                if TorchCRMPhysics is not None:
                    self._torch_physics = TorchCRMPhysics(
                        param_file=param_file or "data/catheter_params/CatheterParameterSet_1_dyn.txt",
                        config_file=config_file or "data/catheter_params/CatheterSpatialConfiguration_1.txt",
                        device=str(self.device),
                    )
            except Exception:
                self._torch_physics = None
                self.use_torch_physics = False

        # Initialize CRM physics wrapper
        self.physics = CRMWrapper(
            param_file=param_file,
            config_file=config_file,
            use_cpp=use_cpp
        )

        # Input dimensions:
        # - currents: 3 (or num_act_set * 3)
        # - insertion_length: 1
        # - (optional) physics prediction: 3
        input_dim = 4  # currents + insertion_length
        physics_output_dim = 3  # tip position

        # Residual network
        self.residual_net = ResidualKinematicsModel(
            input_dim=input_dim,
            output_dim=3,  # position correction
            hidden_dims=hidden_dims,
            physics_output_dim=physics_output_dim,
            include_physics_in_input=include_physics_in_input,
            use_deep_residual=use_deep_residual,
            dropout=dropout,
            max_correction=max_correction
        )

        # Optional learnable blend weight between physics and residual
        self.learnable_physics_weight = learnable_physics_weight
        if learnable_physics_weight:
            # Sigmoid applied to get weight in [0, 1]
            self.physics_weight_logit = nn.Parameter(torch.tensor(2.0))  # ~0.88 after sigmoid

        # Move to device
        self.to(self.device)

        # Cache for physics predictions (avoid recomputing)
        self._physics_cache = {}
        self._cache_enabled = True

    def get_physics_prediction(
        self,
        currents: np.ndarray,
        insertion_length: float
    ) -> np.ndarray:
        """
        Get physics prediction from CRM model.

        Args:
            currents: Applied currents (3,) or (batch, 3)
            insertion_length: Insertion length in mm

        Returns:
            Tip position prediction (3,) or (batch, 3)
        """
        currents = np.asarray(currents, dtype=np.float64)

        if currents.ndim == 1:
            # Single prediction
            result = self.physics.forward_kinematics(currents, insertion_length)
            return result['tip_position']
        else:
            # Batch prediction (loop - could be parallelized)
            positions = []
            for i in range(len(currents)):
                result = self.physics.forward_kinematics(currents[i], insertion_length)
                positions.append(result['tip_position'])
            return np.array(positions)

    def forward(
        self,
        currents: torch.Tensor,
        insertion_length: Union[float, torch.Tensor],
        return_components: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass: compute hybrid prediction.

        Args:
            currents: Current inputs (batch, 3) as torch tensor
            insertion_length: Insertion length (scalar or (batch,))
            return_components: If True, return (final, physics, residual)

        Returns:
            Predicted tip position (batch, 3)
            Or tuple of (final, physics, residual) if return_components=True
        """
        batch_size = currents.shape[0]

        if self.use_torch_physics and self._torch_physics is not None:
            if isinstance(insertion_length, (int, float)):
                ins_tensor = torch.full((batch_size,), float(insertion_length), dtype=currents.dtype, device=currents.device)
            else:
                ins_tensor = insertion_length.to(currents.device).view(-1)
            physics_pred = self._torch_physics.fk(currents.to(currents.device), ins_tensor).to(self.device)
        else:
            # Convert to numpy for physics computation
            currents_np = currents.detach().cpu().numpy()
            if isinstance(insertion_length, torch.Tensor):
                insertion_np = insertion_length.detach().cpu().numpy()
            else:
                insertion_np = np.full(batch_size, insertion_length)

            # Get physics predictions
            physics_predictions = []
            for i in range(batch_size):
                pred = self.get_physics_prediction(
                    currents_np[i],
                    float(insertion_np[i] if np.ndim(insertion_np) > 0 else insertion_np),
                )
                physics_predictions.append(pred)

            physics_pred = torch.tensor(
                np.array(physics_predictions),
                dtype=torch.float32,
                device=self.device
            )

        # Prepare input for residual network
        # Input: [currents, insertion_length]
        if isinstance(insertion_length, (int, float)):
            ins_tensor = torch.full((batch_size, 1), insertion_length, dtype=torch.float32, device=self.device)
        else:
            ins_tensor = insertion_length.unsqueeze(-1).to(self.device)

        residual_input = torch.cat([currents.to(self.device), ins_tensor], dim=-1)

        # Get residual correction
        residual = self.residual_net(residual_input, physics_pred)

        # Combine physics and residual
        if self.learnable_physics_weight:
            weight = torch.sigmoid(self.physics_weight_logit)
            final_pred = weight * physics_pred + (1 - weight) * (physics_pred + residual)
        else:
            final_pred = physics_pred + residual

        if return_components:
            return final_pred, physics_pred, residual
        return final_pred

    def predict(
        self,
        currents: np.ndarray,
        insertion_length: float
    ) -> np.ndarray:
        """
        Convenience method for numpy input/output prediction.

        Args:
            currents: Applied currents (3,) or (batch, 3)
            insertion_length: Insertion length in mm

        Returns:
            Predicted tip position (3,) or (batch, 3)
        """
        currents = np.asarray(currents, dtype=np.float32)
        single_input = currents.ndim == 1

        if single_input:
            currents = currents[np.newaxis, :]

        currents_tensor = torch.tensor(currents, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            pred = self.forward(currents_tensor, insertion_length)

        result = pred.cpu().numpy()

        if single_input:
            return result[0]
        return result

    def compute_loss(
        self,
        currents: torch.Tensor,
        insertion_length: Union[float, torch.Tensor],
        target_positions: torch.Tensor,
        physics_weight: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.

        Args:
            currents: Input currents (batch, 3)
            insertion_length: Insertion length
            target_positions: Ground truth positions (batch, 3)
            physics_weight: Weight for physics prediction loss (regularization)

        Returns:
            Dictionary with loss components
        """
        final_pred, physics_pred, residual = self.forward(
            currents, insertion_length, return_components=True
        )

        target = target_positions.to(self.device)

        # Main loss: final prediction error
        final_loss = nn.functional.mse_loss(final_pred, target)

        # Physics loss: encourage residual to be small when physics is accurate
        physics_loss = nn.functional.mse_loss(physics_pred, target)

        # Residual regularization: penalize large corrections
        residual_reg = (residual ** 2).mean()

        total_loss = final_loss + physics_weight * physics_loss + 0.01 * residual_reg

        return {
            'total': total_loss,
            'final': final_loss,
            'physics': physics_loss,
            'residual_reg': residual_reg,
            'residual_magnitude': residual.abs().mean()
        }

    def get_physics_weight(self) -> float:
        """Get current physics-residual blend weight."""
        if self.learnable_physics_weight:
            return torch.sigmoid(self.physics_weight_logit).item()
        return 1.0  # Pure addition when not learnable

    @property
    def is_using_cpp(self) -> bool:
        """Check if using C++ physics backend."""
        return self.physics.is_using_cpp

    def enable_cache(self, enabled: bool = True):
        """Enable/disable physics prediction caching."""
        self._cache_enabled = enabled
        if not enabled:
            self._physics_cache.clear()

    def clear_cache(self):
        """Clear physics prediction cache."""
        self._physics_cache.clear()


class HybridKinematicsWithUncertainty(nn.Module):
    """
    Hybrid kinematics model with uncertainty estimation.

    Returns both mean prediction and uncertainty estimate, useful for:
    - Model-based RL with uncertainty-aware exploration
    - Safe control with confidence bounds
    - Active learning for data collection
    """

    def __init__(
        self,
        param_file: Optional[str] = None,
        config_file: Optional[str] = None,
        use_cpp: bool = True,
        hidden_dims: list = [256, 256, 128],
        dropout: float = 0.1,
        device: str = "cpu"
    ):
        """
        Initialize hybrid kinematics with uncertainty model.

        Args:
            param_file: Path to CRM catheter parameter file
            config_file: Path to CRM catheter configuration file
            use_cpp: Whether to use C++ physics bindings
            hidden_dims: Hidden layer dimensions
            dropout: Dropout probability
            device: Device for computations
        """
        super().__init__()

        self.device = torch.device(device)

        # Physics wrapper
        self.physics = CRMWrapper(
            param_file=param_file,
            config_file=config_file,
            use_cpp=use_cpp
        )

        # Residual network with uncertainty
        # Input: currents (3) + insertion (1) + physics (3) = 7
        self.residual_net = ResidualKinematicsWithUncertainty(
            input_dim=7,
            output_dim=3,
            hidden_dims=hidden_dims
        )

        self.to(self.device)

    def forward(
        self,
        currents: torch.Tensor,
        insertion_length: Union[float, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with uncertainty.

        Args:
            currents: Input currents (batch, 3)
            insertion_length: Insertion length

        Returns:
            Tuple of (mean_prediction, variance) both (batch, 3)
        """
        batch_size = currents.shape[0]

        # Get physics predictions
        currents_np = currents.detach().cpu().numpy()
        if isinstance(insertion_length, torch.Tensor):
            insertion_np = insertion_length.detach().cpu().numpy()
        else:
            insertion_np = insertion_length

        physics_predictions = []
        for i in range(batch_size):
            ins = float(insertion_np[i]) if np.ndim(insertion_np) > 0 else float(insertion_np)
            result = self.physics.forward_kinematics(currents_np[i], ins)
            physics_predictions.append(result['tip_position'])

        physics_pred = torch.tensor(
            np.array(physics_predictions),
            dtype=torch.float32,
            device=self.device
        )

        # Prepare input
        if isinstance(insertion_length, (int, float)):
            ins_tensor = torch.full((batch_size, 1), insertion_length, dtype=torch.float32, device=self.device)
        else:
            ins_tensor = insertion_length.unsqueeze(-1).to(self.device)

        residual_input = torch.cat([currents.to(self.device), ins_tensor], dim=-1)

        # Get residual mean and variance
        residual_mean, residual_var = self.residual_net(residual_input, physics_pred)

        # Final prediction
        mean = physics_pred + residual_mean

        return mean, residual_var

    def predict(
        self,
        currents: np.ndarray,
        insertion_length: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convenience method for numpy prediction with uncertainty.

        Returns:
            Tuple of (mean_position, variance) both (3,) or (batch, 3)
        """
        currents = np.asarray(currents, dtype=np.float32)
        single_input = currents.ndim == 1

        if single_input:
            currents = currents[np.newaxis, :]

        currents_tensor = torch.tensor(currents, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            mean, var = self.forward(currents_tensor, insertion_length)

        mean_np = mean.cpu().numpy()
        var_np = var.cpu().numpy()

        if single_input:
            return mean_np[0], var_np[0]
        return mean_np, var_np


if __name__ == "__main__":
    print("Testing HybridKinematicsModel...")

    # Test basic model
    model = HybridKinematicsModel(use_cpp=True)
    print(f"Using C++ physics: {model.is_using_cpp}")

    # Test single prediction
    currents = np.array([0.1, 0.0, 0.0])
    pred = model.predict(currents, insertion_length=94.3)
    print(f"Single prediction: {pred}")

    # Test batch prediction
    currents_batch = np.array([
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.0, 0.0, 0.1]
    ])
    preds = model.predict(currents_batch, insertion_length=94.3)
    print(f"Batch predictions:\n{preds}")

    # Test forward with components
    currents_tensor = torch.tensor(currents_batch, dtype=torch.float32)
    final, physics, residual = model.forward(currents_tensor, 50.0, return_components=True)
    print(f"\nPhysics predictions:\n{physics.detach().numpy()}")
    print(f"Residual corrections:\n{residual.detach().numpy()}")
    print(f"Final predictions:\n{final.detach().numpy()}")

    # Test loss computation
    target = torch.randn(3, 3)
    losses = model.compute_loss(currents_tensor, 50.0, target)
    print(f"\nLosses: {', '.join(f'{k}={v.item():.4f}' for k, v in losses.items())}")

    # Test with uncertainty
    print("\n\nTesting HybridKinematicsWithUncertainty...")
    model_unc = HybridKinematicsWithUncertainty(use_cpp=True)

    mean, var = model_unc.predict(currents, insertion_length=94.3)
    print(f"Mean: {mean}")
    print(f"Variance: {var}")
    print(f"Std: {np.sqrt(var)}")

    print("\nHybrid kinematics tests complete!")
