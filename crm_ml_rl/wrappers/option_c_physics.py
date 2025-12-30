"""
Option C Physics Wrapper for crm_ml_rl.

Provides a unified interface for differentiable catheter dynamics using
the validated Option C implementation (crm_torch.CRMDynamicsStep).
"""

import torch
import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

import crm_torch
from . import crm_python


@dataclass
class OptionCConfig:
    """Configuration for Option C physics."""
    param_file: str = "data/catheter_params/CatheterParameterSet_1_dyn.txt"
    config_file: str = "data/catheter_params/CatheterSpatialConfiguration_1.txt"
    insertion_length: float = 94.3
    dt: float = 0.05
    eps_seed: float = 1e-4
    damping: np.ndarray = None  # Uses default from CRMDynamics

    def __post_init__(self):
        if self.damping is None:
            self.damping = np.array([
                12.1761626666366, 12.1761626666366, 284.429938756989,
                0.0304776127617393, 0.0304776127617393, 0.00502712804532508
            ])


class OptionCPhysics:
    """
    Differentiable physics wrapper using Option C (crm_torch).

    Provides:
    - Forward pass: Compute next state from currents
    - Backward pass: Gradients w.r.t. currents (validated)
    - Phase 3B: Gradients w.r.t. seed state (enables multi-step optimization)
    - State management: Maintains seed state for multi-step simulation

    Usage (Phase 3A - gradients w.r.t. currents only):
        physics = OptionCPhysics()
        physics.reset(initial_currents)

        # Single-step differentiable
        currents = torch.tensor([[0.1, 0.0, 0.0]], requires_grad=True)
        next_state = physics.step_differentiable(currents)
        loss = next_state[:, :3].sum()  # Loss on position
        loss.backward()  # Gradients w.r.t. currents
        print(currents.grad)

    Usage (Phase 3B - gradients through multi-step trajectory):
        physics = OptionCPhysics()
        physics.reset()

        actions = torch.randn(5, 3, requires_grad=True)  # 5 steps
        seed = None
        loss = 0
        for t in range(5):
            output, seed = physics.step_with_seed_gradients(actions[t], seed)
            loss += torch.norm(output[:3] - target)**2

        loss.backward()  # Gradients flow through ALL steps!
        print(actions.grad)  # All actions have gradients
    """

    def __init__(self, config: Optional[OptionCConfig] = None, device: str = "cpu"):
        self.config = config or OptionCConfig()
        self.device = torch.device(device)

        # Verify Option C is available
        if not crm_torch.is_available():
            raise RuntimeError(f"Option C not available: {crm_torch.get_import_error()}")

        # Option A dynamics for state management
        self._dyn = crm_python.CRMDynamics()
        if not self._dyn.load_parameters(self.config.param_file, self.config.config_file):
            raise RuntimeError("Failed to load CRM parameters")
        self._dyn.dt = self.config.dt
        self._dyn.set_damping(self.config.damping)

        self._initialized = False
        self._current_position = np.zeros(3)
        self._current_velocity = np.zeros(3)

    def reset(self, initial_currents: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Reset physics to initial state.

        Args:
            initial_currents: Initial currents for FK initialization (default: [0,0,0.01])

        Returns:
            Initial state [position(3), velocity(3)]
        """
        if initial_currents is None:
            initial_currents = np.array([0.0, 0.0, 0.01])

        initial_currents = np.asarray(initial_currents, dtype=np.float64)

        # Initialize Option A dynamics (sets internal seed state)
        ok = self._dyn.initialize_from_kinematics(initial_currents, self.config.insertion_length)
        if not ok:
            raise RuntimeError("Failed to initialize dynamics")

        # Get initial position
        self._current_position = self._dyn.get_tip_position()
        self._current_velocity = np.zeros(3)
        self._initialized = True

        return np.concatenate([self._current_position, self._current_velocity])

    def get_seed_tensors(self) -> Dict[str, torch.Tensor]:
        """Get current seed state as torch tensors (non-differentiable)."""
        if not self._initialized:
            raise RuntimeError("Physics not initialized. Call reset() first.")

        seed = self._dyn.get_seed_state()

        return {
            'v': torch.from_numpy(seed['v']).unsqueeze(0).double().to(self.device),
            'w': torch.from_numpy(seed['w']).unsqueeze(0).double().to(self.device),
            'p': torch.from_numpy(seed['p']).unsqueeze(0).double().to(self.device),
            'R': torch.from_numpy(seed['R']).unsqueeze(0).double().to(self.device),
            'xf': torch.from_numpy(seed['xf']).unsqueeze(0).double().to(self.device),
            'mL': torch.from_numpy(seed['mL']).unsqueeze(0).double().to(self.device),
            'nL': torch.from_numpy(seed['nL']).unsqueeze(0).double().to(self.device),
        }

    def get_seed_tensors_differentiable(self) -> Dict[str, torch.Tensor]:
        """
        Get seed state as differentiable tensors (Phase 3B).

        Returns tensors with requires_grad=True to enable gradient flow
        through seed state updates in multi-step trajectories.

        Returns:
            Dictionary of seed tensors with gradients enabled:
            - v: velocity (1, num_sets, 3)
            - w: angular velocity (1, num_sets, 3)
            - p: position (1, num_sets, 3)
            - R: rotation matrix (1, num_sets, 9)
            - xf: frame positions (1, 15)
            - mL: magnetic field L (1, num_sets, 3)
            - nL: normal field L (1, num_sets, 3)
        """
        if not self._initialized:
            raise RuntimeError("Physics not initialized. Call reset() first.")

        seed = self._dyn.get_seed_state()

        return {
            'v': torch.from_numpy(seed['v']).unsqueeze(0).double()
                 .to(self.device).requires_grad_(True),
            'w': torch.from_numpy(seed['w']).unsqueeze(0).double()
                 .to(self.device).requires_grad_(True),
            'p': torch.from_numpy(seed['p']).unsqueeze(0).double()
                 .to(self.device).requires_grad_(True),
            'R': torch.from_numpy(seed['R']).unsqueeze(0).double()
                 .to(self.device).requires_grad_(True),
            'xf': torch.from_numpy(seed['xf']).unsqueeze(0).double()
                  .to(self.device).requires_grad_(True),
            'mL': torch.from_numpy(seed['mL']).unsqueeze(0).double()
                  .to(self.device).requires_grad_(True),
            'nL': torch.from_numpy(seed['nL']).unsqueeze(0).double()
                  .to(self.device).requires_grad_(True),
        }

    def step_differentiable(self, currents: torch.Tensor) -> torch.Tensor:
        """
        Differentiable physics step.

        Args:
            currents: (batch, 3) control currents with requires_grad=True

        Returns:
            next_state: (batch, 6) [tip_position(3), tip_velocity(3)]

        Note:
            Gradients are computed w.r.t. currents only (Option C Phase 3A).
            Seed gradients are zero (can be enabled in Phase 3B if needed).
        """
        if not self._initialized:
            raise RuntimeError("Physics not initialized. Call reset() first.")

        # Ensure correct shape and dtype
        if currents.dim() == 1:
            currents = currents.unsqueeze(0)
        currents = currents.double()

        batch_size = currents.shape[0]
        insertion = torch.tensor([self.config.insertion_length], dtype=torch.float64, device=self.device)

        # Get seed state tensors (replicated for batch)
        seed = self.get_seed_tensors()
        seed_v = seed['v'].expand(batch_size, -1, -1)
        seed_w = seed['w'].expand(batch_size, -1, -1)
        seed_p = seed['p'].expand(batch_size, -1, -1)
        seed_R = seed['R'].expand(batch_size, -1, -1)
        seed_xf = seed['xf'].expand(batch_size, -1)
        seed_mL = seed['mL'].expand(batch_size, -1, -1)
        seed_nL = seed['nL'].expand(batch_size, -1, -1)

        # Forward pass through Option C
        # Phase 3B: Now returns tuple (output, next_v, next_w, ...)
        result = crm_torch.CRMDynamicsStep.apply(
            currents, insertion,
            seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL,
            self.config.param_file, self.config.config_file, self.config.eps_seed
        )

        # Unpack and return only the output (seed updates not used in this interface)
        output = result[0]
        return output.float()  # (batch, 6): [pos(3), vel(3)]

    def step(self, currents: np.ndarray, update_state: bool = True) -> np.ndarray:
        """
        Non-differentiable physics step (faster, for simulation).

        Args:
            currents: (3,) control currents
            update_state: Whether to update internal state for next step

        Returns:
            next_state: (6,) [tip_position(3), tip_velocity(3)]
        """
        if not self._initialized:
            raise RuntimeError("Physics not initialized. Call reset() first.")

        currents = np.asarray(currents, dtype=np.float64)

        # Step Option A dynamics (updates internal seed)
        result = self._dyn.step(currents, self.config.insertion_length)

        next_position = np.array(result['tip_position'], dtype=np.float64)
        next_velocity = (next_position - self._current_position) / self.config.dt

        if update_state:
            self._current_position = next_position
            self._current_velocity = next_velocity

        return np.concatenate([next_position, next_velocity])

    def step_and_update_differentiable(self, currents: torch.Tensor) -> torch.Tensor:
        """
        Differentiable step that also updates internal state.

        Use this for multi-step differentiable trajectories.

        Args:
            currents: (3,) or (1, 3) control currents with requires_grad=True

        Returns:
            next_state: (6,) [tip_position(3), tip_velocity(3)]

        Note:
            This method updates the internal seed state by calling Option A's step()
            in a detached manner. This breaks the gradient graph intentionally, as
            gradients w.r.t. seed state are not yet supported (Option C Phase 3A).

            For continuous gradient flow through multi-step trajectories, use this
            method carefully or manually manage seed state updates.
        """
        # Get differentiable output (this has gradients w.r.t currents)
        output = self.step_differentiable(currents)

        # Update internal state via Option A for next step's seed
        # Note: This detaches from computation graph. Seed gradients not yet supported.
        with torch.no_grad():
            currents_np = currents.detach().cpu().numpy().reshape(-1)
            self._dyn.step(currents_np, self.config.insertion_length)

            # Update tracked state
            self._current_position = output[0, :3].detach().cpu().numpy()
            self._current_velocity = output[0, 3:6].detach().cpu().numpy()

        return output[0]  # Return unbatched

    def step_with_seed_gradients(
        self,
        currents: torch.Tensor,
        seed_state: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Differentiable step with seed gradient flow (Phase 3B).

        This method enables gradients to flow through both currents AND seed state,
        allowing true multi-step trajectory optimization.

        Args:
            currents: (3,) or (1, 3) control currents with requires_grad=True
            seed_state: Optional seed state dict. If None, uses current state with gradients enabled.

        Returns:
            Tuple of:
            - output: (6,) [tip_position(3), tip_velocity(3)]
            - updated_seed: Dict of updated seed tensors (for next step)

        Example:
            # Multi-step optimization with seed gradients
            physics = OptionCPhysics()
            physics.reset()

            seed = None
            loss = 0
            for t in range(horizon):
                output, seed = physics.step_with_seed_gradients(actions[t], seed)
                loss += torch.norm(output[:3] - target)**2

            loss.backward()  # Gradients flow through ALL steps!
        """
        if not self._initialized:
            raise RuntimeError("Physics not initialized. Call reset() first.")

        # Ensure correct shape and dtype
        if currents.dim() == 1:
            currents = currents.unsqueeze(0)
        currents = currents.double()

        # Get seed state
        if seed_state is None:
            seed_state = self.get_seed_tensors_differentiable()

        # Expand seeds for batch if needed
        batch_size = currents.shape[0]
        seed_v = seed_state['v']
        seed_w = seed_state['w']
        seed_p = seed_state['p']
        seed_R = seed_state['R']
        seed_xf = seed_state['xf']
        seed_mL = seed_state['mL']
        seed_nL = seed_state['nL']

        if seed_v.shape[0] != batch_size:
            seed_v = seed_v.expand(batch_size, -1, -1)
            seed_w = seed_w.expand(batch_size, -1, -1)
            seed_p = seed_p.expand(batch_size, -1, -1)
            seed_R = seed_R.expand(batch_size, -1, -1)
            seed_xf = seed_xf.expand(batch_size, -1)
            seed_mL = seed_mL.expand(batch_size, -1, -1)
            seed_nL = seed_nL.expand(batch_size, -1, -1)

        insertion = torch.tensor([self.config.insertion_length], dtype=torch.float64, device=self.device)

        # Forward pass through Option C (with seed gradients!)
        # Phase 3B FIX: CRMDynamicsStep now returns updated seeds!
        result = crm_torch.CRMDynamicsStep.apply(
            currents, insertion,
            seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL,
            self.config.param_file, self.config.config_file, self.config.eps_seed
        )

        # Unpack result: (next_state, next_v, next_w, next_p, next_R, next_xf, next_mL, next_nL)
        output, next_v, next_w, next_p, next_R, next_xf, next_mL, next_nL = result

        # Phase 3B FIX: Return the UPDATED seeds from the forward pass (not the input seeds!)
        # This connects seed_t+1 to seed_t in the gradient graph, enabling multi-step optimization
        updated_seed = {
            'v': next_v,
            'w': next_w,
            'p': next_p,
            'R': next_R,
            'xf': next_xf,
            'mL': next_mL,
            'nL': next_nL,
        }

        # Also update internal state for non-differentiable tracking
        with torch.no_grad():
            currents_np = currents[0].detach().cpu().numpy()
            self._dyn.step(currents_np, self.config.insertion_length)

        return output[0].float(), updated_seed

    @property
    def current_state(self) -> np.ndarray:
        """Get current state [position(3), velocity(3)]."""
        return np.concatenate([self._current_position, self._current_velocity])

    @property
    def is_initialized(self) -> bool:
        return self._initialized
