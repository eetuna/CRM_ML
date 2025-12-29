"""
Hybrid Dynamics Model - Integrates CRM Dynamics with Neural Network Residuals.

This model tightly couples the C++ CRM dynamics engine with a learned residual
correction network:

    x_{t+1} = CRM_DYN(x_t, u_t) + f_residual(x_t, u_t, CRM_DYN(...))

The physics model provides the base dynamics prediction, and the neural network
learns to correct systematic errors such as unmodeled damping, friction, or
other nonlinear effects.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, Union, List
from pathlib import Path
from dataclasses import dataclass

from .residual_dynamics import ResidualDynamicsModel, EnsembleResidualDynamics
from .networks import MLP, LSTM_MLP
from ..wrappers.crm_wrapper import CRMWrapper, CRMSimulator, CatheterState, HAS_CPP_BINDINGS


@dataclass
class HybridDynamicsConfig:
    """Configuration for hybrid dynamics model."""
    # Physics data/simulation_parameters
    param_file: Optional[str] = None
    config_file: Optional[str] = None
    use_cpp: bool = True
    dt: float = 0.02
    insertion_length: float = 50.0

    # Damping coefficients
    damping: np.ndarray = None

    # Neural network data/simulation_parameters
    hidden_dims: List[int] = None
    dropout: float = 0.1
    max_correction: float = 5.0
    use_deep_residual: bool = False

    # Model options
    include_physics_in_input: bool = True
    use_ensemble: bool = False
    num_ensemble: int = 5
    learnable_blend: bool = False
    use_torch_physics: bool = False
    use_option_c: bool = True  # Use Option C (recommended, validated) instead of torch_physics

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256, 128]
        if self.damping is None:
            self.damping = np.array([
                12.1761626666366, 12.1761626666366, 284.429938756989,
                0.0304776127617393, 0.0304776127617393, 0.00502712804532508
            ])


class HybridDynamicsModel(nn.Module):
    """
    Hybrid Dynamics Model combining CRM physics with learned residuals.

    This model:
    1. Maintains internal physics state via CRMSimulator
    2. Runs CRM dynamics to get physics prediction for next state
    3. Applies residual neural network correction
    4. Returns corrected state prediction

    State representation:
        - Position (3): tip position in mm
        - Velocity (3): tip velocity in mm/s
        Total state dim: 6
    """

    def __init__(
        self,
        config: Optional[HybridDynamicsConfig] = None,
        device: str = "cpu"
    ):
        """
        Initialize hybrid dynamics model.

        Args:
            config: Model configuration
            device: Device for neural network computations
        """
        super().__init__()

        self.config = config or HybridDynamicsConfig()
        self.device = torch.device(device)
        self.state_dim = 6  # position (3) + velocity (3)
        self.action_dim = 3  # currents

        # Initialize CRM physics simulator
        self.simulator = CRMSimulator(
            param_file=self.config.param_file,
            config_file=self.config.config_file,
            dt=self.config.dt,
            use_cpp=self.config.use_cpp
        )

        # Set damping if using C++
        if self.simulator.is_using_cpp:
            self.simulator.wrapper.set_damping(self.config.damping)

        # Optional: differentiable (torch) physics helpers for planning/MPC.
        self._torch_physics = None
        self._option_c = None
        self._cpp_dyn = None

        if self.simulator.is_using_cpp:
            try:
                # Private member, but stable within this repo.
                self._cpp_dyn = self.simulator.wrapper._cpp_dynamics
            except Exception:
                self._cpp_dyn = None

            # Option C (recommended): Validated differentiable physics via crm_torch
            if self.config.use_option_c:
                try:
                    from ..wrappers.option_c_physics import OptionCPhysics, OptionCConfig
                    option_c_config = OptionCConfig(
                        param_file=self.config.param_file or "data/catheter_params/CatheterParameterSet_1_dyn.txt",
                        config_file=self.config.config_file or "data/catheter_params/CatheterSpatialConfiguration_1.txt",
                        insertion_length=self.config.insertion_length,
                        dt=self.config.dt,
                        damping=self.config.damping
                    )
                    self._option_c = OptionCPhysics(option_c_config, device=str(self.device))
                except Exception as e:
                    print(f"Warning: Could not initialize Option C: {e}")
                    self._option_c = None

            # Legacy torch_physics (fallback if Option C not available)
            if not self.config.use_option_c or self._option_c is None:
                try:
                    from ..wrappers.torch_physics import TorchCRMPhysics
                    if TorchCRMPhysics is not None:
                        self._torch_physics = TorchCRMPhysics(
                            param_file=self.config.param_file or "data/catheter_params/CatheterParameterSet_1_dyn.txt",
                            config_file=self.config.config_file or "data/catheter_params/CatheterSpatialConfiguration_1.txt",
                            device=str(self.device),
                        )
                except Exception:
                    self._torch_physics = None

        # Use Option C if available, otherwise fall back to torch_physics
        self.use_option_c = (
            bool(self.config.use_option_c)
            and (self._cpp_dyn is not None)
            and (self._option_c is not None)
        )

        self.use_torch_physics = (
            bool(self.config.use_torch_physics)
            and (self._cpp_dyn is not None)
            and (self._torch_physics is not None)
            and not self.use_option_c  # Don't use both
        )

        # Create residual network
        if self.config.use_ensemble:
            self.residual_net = EnsembleResidualDynamics(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dims=self.config.hidden_dims,
                num_models=self.config.num_ensemble,
                include_physics_in_input=self.config.include_physics_in_input,
                max_correction=self.config.max_correction
            )
        else:
            self.residual_net = ResidualDynamicsModel(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dims=self.config.hidden_dims,
                include_physics_in_input=self.config.include_physics_in_input,
                use_deep_residual=self.config.use_deep_residual,
                dropout=self.config.dropout,
                max_correction=self.config.max_correction
            )

        # Learnable blend weight
        self.learnable_blend = self.config.learnable_blend
        if self.learnable_blend:
            self.blend_logit = nn.Parameter(torch.tensor(2.0))

        # Internal state tracking
        self._current_state = np.zeros(self.state_dim)
        self._initialized = False

        self.to(self.device)

    def reset(
        self,
        initial_currents: Optional[np.ndarray] = None,
        insertion_length: Optional[float] = None
    ) -> np.ndarray:
        """
        Reset the dynamics model to initial state.

        Args:
            initial_currents: Initial currents for FK initialization
            insertion_length: Insertion length in mm

        Returns:
            Initial state (position, velocity)
        """
        if initial_currents is None:
            initial_currents = np.zeros(3)
        if insertion_length is None:
            insertion_length = self.config.insertion_length

        # Reset physics simulator
        self.simulator.reset(
            initial_currents=initial_currents,
            insertion_length=insertion_length
        )

        # Get initial state
        position = self.simulator.state.position.copy()
        velocity = np.zeros(3)

        self._current_state = np.concatenate([position, velocity])
        self._initialized = True

        return self._current_state.copy()

    def get_physics_prediction(
        self,
        state: np.ndarray,
        action: np.ndarray,
        insertion_length: Optional[float] = None
    ) -> np.ndarray:
        """
        Get physics prediction for next state.

        Args:
            state: Current state (6,) - [position, velocity]
            action: Current action (3,) - currents
            insertion_length: Insertion length

        Returns:
            Predicted next state (6,)
        """
        if insertion_length is None:
            insertion_length = self.config.insertion_length

        # Set simulator state to current state
        self.simulator.state.position = state[:3].copy()
        self.simulator.state.velocity = state[3:].copy()

        # Reinitialize dynamics from current state
        self.simulator.wrapper.initialize_dynamics(
            action, insertion_length
        )

        # Step physics simulation
        result = self.simulator.wrapper.step_dynamics(action, insertion_length)

        # Extract next state
        next_position = result['tip_position']
        next_velocity = result.get('tip_velocity', np.zeros(3))

        return np.concatenate([next_position, next_velocity])

    def get_physics_action_jacobian(
        self,
        currents: np.ndarray,
        insertion_length: Optional[float] = None,
        eps: float = 1e-4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (next_state, B) where B = d(next_state)/d(currents) around the *current*
        internal C++ dynamics seed.

        This is useful for MPC/iLQR-style controllers that need local linearizations.
        """
        if insertion_length is None:
            insertion_length = self.config.insertion_length

        if self._cpp_dyn is None:
            raise RuntimeError("C++ dynamics not available for linearization.")

        seed = self._cpp_dyn.get_seed_state()
        out = self._cpp_dyn.linearize_action_from_seed(
            np.asarray(currents, dtype=np.float64).reshape(-1),
            float(insertion_length),
            seed["v"], seed["w"], seed["p"], seed["R"], seed["xf"],
            seed.get("mL"), seed.get("nL"),
            float(eps),
        )
        next_state = np.asarray(out["next_state"], dtype=np.float64).reshape(6)
        B = np.asarray(out["B"], dtype=np.float64).reshape(6, 3)
        return next_state, B

    def torch_physics_step(
        self,
        currents: torch.Tensor,
        insertion_length: Optional[float] = None,
        eps: float = 1e-4,
    ) -> torch.Tensor:
        """
        Differentiable physics step w.r.t. currents, linearized around the current
        internal C++ dynamics seed.

        Returns next_state (batch, 6). Gradients are provided for `currents` only.
        """
        if insertion_length is None:
            insertion_length = self.config.insertion_length

        # Use Option C if available (recommended, validated)
        if self.use_option_c and self._option_c is not None:
            return self._option_c.step_differentiable(currents)

        # Fallback to legacy torch_physics
        if self._cpp_dyn is None or self._torch_physics is None:
            raise RuntimeError("Torch physics requires C++ bindings and TorchCRMPhysics/OptionC.")

        seed = self._cpp_dyn.get_seed_state()
        seed_v = torch.tensor(seed["v"], dtype=currents.dtype, device=currents.device).unsqueeze(0)
        seed_w = torch.tensor(seed["w"], dtype=currents.dtype, device=currents.device).unsqueeze(0)
        seed_p = torch.tensor(seed["p"], dtype=currents.dtype, device=currents.device).unsqueeze(0)
        seed_R = torch.tensor(seed["R"], dtype=currents.dtype, device=currents.device).unsqueeze(0)
        seed_xf = torch.tensor(seed["xf"], dtype=currents.dtype, device=currents.device).unsqueeze(0)

        batch = currents.shape[0]
        ins = torch.full((batch,), float(insertion_length), dtype=currents.dtype, device=currents.device)
        return self._torch_physics.dyn_step(
            currents, ins, seed_v, seed_w, seed_p, seed_R, seed_xf, eps_u=float(eps)
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        insertion_length: Optional[float] = None,
        return_components: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass: predict next state.

        Args:
            state: Current state (batch, 6)
            action: Action/currents (batch, 3)
            insertion_length: Insertion length
            return_components: Return (final, physics, residual)

        Returns:
            Predicted next state (batch, 6)
        """
        if insertion_length is None:
            insertion_length = self.config.insertion_length

        batch_size = state.shape[0]
        state_np = state.detach().cpu().numpy()
        action_np = action.detach().cpu().numpy()

        if self.use_option_c or self.use_torch_physics:
            # Differentiable w.r.t. `action` only. Seeds are prepared via the C++ wrapper.
            action_device = action.to(self.device)
            next_states = []
            for i in range(batch_size):
                # Seed the internal C++ dynamics from the provided (state, action) values.
                self.simulator.state.position = state_np[i, :3].copy()
                self.simulator.state.velocity = state_np[i, 3:].copy()
                self.simulator.wrapper.initialize_dynamics(action_np[i], float(insertion_length))

                if self.use_option_c:
                    # Use Option C (recommended, validated)
                    nxt = self._option_c.step_differentiable(action_device[i : i + 1])
                else:
                    # Use legacy torch_physics
                    seed = self._cpp_dyn.get_seed_state()
                    seed_v = torch.tensor(seed["v"], dtype=action_device.dtype, device=self.device).unsqueeze(0)
                    seed_w = torch.tensor(seed["w"], dtype=action_device.dtype, device=self.device).unsqueeze(0)
                    seed_p = torch.tensor(seed["p"], dtype=action_device.dtype, device=self.device).unsqueeze(0)
                    seed_R = torch.tensor(seed["R"], dtype=action_device.dtype, device=self.device).unsqueeze(0)
                    seed_xf = torch.tensor(seed["xf"], dtype=action_device.dtype, device=self.device).unsqueeze(0)

                    ins = torch.full((1,), float(insertion_length), dtype=action_device.dtype, device=self.device)
                    nxt = self._torch_physics.dyn_step(
                        action_device[i : i + 1],
                        ins,
                        seed_v,
                        seed_w,
                        seed_p,
                        seed_R,
                        seed_xf,
                        eps_u=1e-4,
                    )
                next_states.append(nxt)

            physics_pred = torch.cat(next_states, dim=0).to(dtype=torch.float32, device=self.device)
        else:
            # Get physics predictions for each sample (nondifferentiable).
            # Note: This requires resetting simulator state for each sample.
            physics_predictions = []
            for i in range(batch_size):
                # Set simulator state to current state
                self.simulator.state.position = state_np[i, :3].copy()
                self.simulator.state.velocity = state_np[i, 3:].copy()

                # Reinitialize dynamics from current position
                self.simulator.wrapper.initialize_dynamics(action_np[i], insertion_length)

                # Step physics
                result = self.simulator.wrapper.step_dynamics(action_np[i], insertion_length)
                next_pos = result["tip_position"]
                next_vel = result.get("tip_velocity", np.zeros(3))
                physics_predictions.append(np.concatenate([next_pos, next_vel]))

            physics_pred = torch.tensor(np.array(physics_predictions), dtype=torch.float32, device=self.device)

        # Get residual correction
        state_device = state.to(self.device)
        action_device = action.to(self.device)

        if self.config.use_ensemble:
            residual_mean, residual_std = self.residual_net(
                state_device, action_device, physics_pred
            )
            residual = residual_mean
        else:
            residual = self.residual_net(state_device, action_device, physics_pred)

        # Combine predictions
        if self.learnable_blend:
            weight = torch.sigmoid(self.blend_logit)
            final = weight * physics_pred + (1 - weight) * (physics_pred + residual)
        else:
            final = physics_pred + residual

        if return_components:
            return final, physics_pred, residual
        return final

    def step(
        self,
        action: np.ndarray,
        insertion_length: Optional[float] = None
    ) -> np.ndarray:
        """
        Step the model forward by one timestep.

        This is the main interface for sequential prediction.

        Args:
            action: Current action (3,) - currents
            insertion_length: Insertion length

        Returns:
            Next state (6,) - [position, velocity]
        """
        if not self._initialized:
            raise RuntimeError("Model not initialized. Call reset() first.")

        if insertion_length is None:
            insertion_length = self.config.insertion_length

        # Convert to tensors
        state_tensor = torch.tensor(
            self._current_state[np.newaxis, :],
            dtype=torch.float32,
            device=self.device
        )
        action_tensor = torch.tensor(
            action[np.newaxis, :],
            dtype=torch.float32,
            device=self.device
        )

        # Forward pass
        with torch.no_grad():
            next_state = self.forward(state_tensor, action_tensor, insertion_length)

        # Update internal state
        self._current_state = next_state[0].cpu().numpy()

        return self._current_state.copy()

    def predict_trajectory(
        self,
        actions: np.ndarray,
        initial_state: Optional[np.ndarray] = None,
        initial_currents: Optional[np.ndarray] = None,
        insertion_length: Optional[float] = None
    ) -> np.ndarray:
        """
        Predict a full trajectory given action sequence.

        Args:
            actions: Action sequence (T, 3)
            initial_state: Initial state (6,) - if None, reset is called
            initial_currents: Initial currents for reset
            insertion_length: Insertion length

        Returns:
            State trajectory (T+1, 6)
        """
        if insertion_length is None:
            insertion_length = self.config.insertion_length

        if initial_state is None:
            self.reset(initial_currents, insertion_length)
        else:
            self._current_state = initial_state.copy()
            self._initialized = True

        T = len(actions)
        trajectory = np.zeros((T + 1, self.state_dim))
        trajectory[0] = self._current_state.copy()

        for t in range(T):
            next_state = self.step(actions[t], insertion_length)
            trajectory[t + 1] = next_state

        return trajectory

    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        insertion_length: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.

        Args:
            states: Current states (batch, 6)
            actions: Actions (batch, 3)
            next_states: Ground truth next states (batch, 6)
            insertion_length: Insertion length

        Returns:
            Dictionary with loss components
        """
        final_pred, physics_pred, residual = self.forward(
            states, actions, insertion_length, return_components=True
        )

        target = next_states.to(self.device)

        # Main loss
        final_loss = nn.functional.mse_loss(final_pred, target)

        # Physics baseline loss
        physics_loss = nn.functional.mse_loss(physics_pred, target)

        # Residual regularization
        residual_reg = (residual ** 2).mean()

        # Position and velocity losses separately
        pos_loss = nn.functional.mse_loss(final_pred[:, :3], target[:, :3])
        vel_loss = nn.functional.mse_loss(final_pred[:, 3:], target[:, 3:])

        total_loss = final_loss + 0.01 * residual_reg

        return {
            'total': total_loss,
            'final': final_loss,
            'physics': physics_loss,
            'position': pos_loss,
            'velocity': vel_loss,
            'residual_reg': residual_reg,
            'residual_magnitude': residual.abs().mean(),
            'improvement': physics_loss - final_loss  # Positive = hybrid is better
        }

    def get_uncertainty(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        insertion_length: Optional[float] = None
    ) -> torch.Tensor:
        """
        Get prediction uncertainty (only for ensemble model).

        Args:
            state: Current state (batch, 6)
            action: Action (batch, 3)
            insertion_length: Insertion length

        Returns:
            Uncertainty estimate (batch, 6) or scalar if not ensemble
        """
        if not self.config.use_ensemble:
            return torch.zeros(state.shape[0], self.state_dim, device=self.device)

        # Get ensemble predictions
        _, physics_pred, _ = self.forward(
            state, action, insertion_length, return_components=True
        )

        # For ensemble, residual_net returns (mean, std)
        _, std = self.residual_net(
            state.to(self.device),
            action.to(self.device),
            physics_pred
        )

        return std

    @property
    def is_using_cpp(self) -> bool:
        """Check if using C++ physics backend."""
        return self.simulator.is_using_cpp

    @property
    def current_state(self) -> np.ndarray:
        """Get current internal state."""
        return self._current_state.copy()


class HybridDynamicsLSTM(nn.Module):
    """
    Hybrid dynamics with LSTM for history-dependent corrections.

    Uses sequence of past states/actions to predict residual, capturing
    temporal patterns that might be missed by single-step residual.
    """

    def __init__(
        self,
        config: Optional[HybridDynamicsConfig] = None,
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 2,
        history_length: int = 10,
        device: str = "cpu"
    ):
        """
        Initialize LSTM-based hybrid dynamics.

        Args:
            config: Model configuration
            lstm_hidden_dim: LSTM hidden dimension
            lstm_num_layers: Number of LSTM layers
            history_length: Number of past steps to use
            device: Device for computations
        """
        super().__init__()

        self.config = config or HybridDynamicsConfig()
        self.device = torch.device(device)
        self.state_dim = 6
        self.action_dim = 3
        self.history_length = history_length

        # Physics simulator
        self.simulator = CRMSimulator(
            param_file=self.config.param_file,
            config_file=self.config.config_file,
            dt=self.config.dt,
            use_cpp=self.config.use_cpp
        )

        if self.simulator.is_using_cpp:
            self.simulator.wrapper.set_damping(self.config.damping)

        # LSTM residual network
        # Input: state + action + physics_next_state
        input_dim = self.state_dim + self.action_dim + self.state_dim
        self.lstm = LSTM_MLP(
            input_dim=input_dim,
            output_dim=self.state_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
            mlp_hidden_dims=[128, 64]
        )

        # Output scaling
        self.output_scale = nn.Parameter(torch.ones(self.state_dim) * 0.1)

        # History buffer
        self._history = []
        self._lstm_hidden = None

        self.to(self.device)

    def reset(
        self,
        initial_currents: Optional[np.ndarray] = None,
        insertion_length: Optional[float] = None
    ) -> np.ndarray:
        """Reset model and clear history."""
        if initial_currents is None:
            initial_currents = np.zeros(3)
        if insertion_length is None:
            insertion_length = self.config.insertion_length

        self.simulator.reset(initial_currents, insertion_length)
        self._history = []
        self._lstm_hidden = None

        return np.concatenate([
            self.simulator.state.position,
            np.zeros(3)
        ])

    def step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        insertion_length: Optional[float] = None
    ) -> np.ndarray:
        """
        Step with history tracking.

        Args:
            state: Current state (6,)
            action: Action (3,)
            insertion_length: Insertion length

        Returns:
            Next state (6,)
        """
        if insertion_length is None:
            insertion_length = self.config.insertion_length

        # Get physics prediction using current simulator state
        result = self.simulator.wrapper.step_dynamics(action, insertion_length)
        physics_next = np.concatenate([
            result['tip_position'],
            result.get('tip_velocity', np.zeros(3))
        ])

        # Add to history
        self._history.append({
            'state': state.copy(),
            'action': action.copy(),
            'physics_next': physics_next.copy()
        })

        # Keep only recent history
        if len(self._history) > self.history_length:
            self._history.pop(0)

        # Build sequence for LSTM
        seq_len = len(self._history)
        sequence = np.zeros((seq_len, self.state_dim + self.action_dim + self.state_dim))
        for i, h in enumerate(self._history):
            sequence[i] = np.concatenate([h['state'], h['action'], h['physics_next']])

        # Get residual from LSTM
        seq_tensor = torch.tensor(
            sequence[np.newaxis, :, :],
            dtype=torch.float32,
            device=self.device
        )

        with torch.no_grad():
            residual, self._lstm_hidden = self.lstm(seq_tensor, self._lstm_hidden)
            residual = residual[0] * self.output_scale
            residual = torch.clamp(residual, -self.config.max_correction, self.config.max_correction)

        # Combine
        next_state = physics_next + residual.cpu().numpy()

        return next_state


if __name__ == "__main__":
    print("Testing HybridDynamicsModel...")

    # Create config
    config = HybridDynamicsConfig(
        use_cpp=True,
        dt=0.02
    )

    # Create model
    model = HybridDynamicsModel(config)
    print(f"Using C++ physics: {model.is_using_cpp}")

    # Test reset
    initial_state = model.reset()
    print(f"Initial state: {initial_state}")

    # Test step
    action = np.array([0.1, 0.0, 0.0])
    next_state = model.step(action)
    print(f"After step: {next_state}")

    # Test trajectory prediction
    T = 50
    t = np.linspace(0, 1, T)
    actions = np.column_stack([
        0.1 * np.sin(2 * np.pi * t),
        0.1 * np.cos(2 * np.pi * t),
        np.zeros(T)
    ])

    trajectory = model.predict_trajectory(actions)
    print(f"Trajectory shape: {trajectory.shape}")
    print(f"Final position: {trajectory[-1, :3]}")

    # Test batch forward
    batch_states = torch.randn(8, 6)
    batch_actions = torch.randn(8, 3) * 0.1
    batch_next = model.forward(batch_states, batch_actions)
    print(f"Batch output shape: {batch_next.shape}")

    # Test loss
    target_next = torch.randn(8, 6)
    losses = model.compute_loss(batch_states, batch_actions, target_next)
    print(f"Losses: {', '.join(f'{k}={v.item():.4f}' for k, v in losses.items())}")

    # Test ensemble model
    print("\n\nTesting Ensemble HybridDynamicsModel...")
    config_ensemble = HybridDynamicsConfig(use_ensemble=True, num_ensemble=3)
    model_ensemble = HybridDynamicsModel(config_ensemble)

    uncertainty = model_ensemble.get_uncertainty(batch_states, batch_actions)
    print(f"Uncertainty shape: {uncertainty.shape}")
    print(f"Mean uncertainty: {uncertainty.mean().item():.4f}")

    # Test LSTM model
    print("\n\nTesting HybridDynamicsLSTM...")
    model_lstm = HybridDynamicsLSTM(config, history_length=5)

    state = model_lstm.reset()
    for i in range(10):
        state = model_lstm.step(state, action)
    print(f"After 10 LSTM steps: {state}")

    print("\nHybrid dynamics tests complete!")
