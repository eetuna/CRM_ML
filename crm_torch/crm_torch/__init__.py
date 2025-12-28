"""
crm_torch - PyTorch C++ Extension for CRM Physics Simulation

This package provides a high-performance, differentiable wrapper around the
CRM catheter physics engine.

Phase 2A: Forward pass via Python bindings (sequential processing)
Phase 3: Backward pass (autograd) implementation
Phase 4: Validation and integration
"""

import torch

__version__ = "0.1.0"

# Try to import the C++ extension
try:
    import _crm_torch_ext
    _extension_available = True
    _import_error = None
except ImportError as e:
    _extension_available = False
    _import_error = str(e)
    _crm_torch_ext = None

def is_available():
    """Check if the C++ extension is successfully loaded."""
    return _extension_available

def get_import_error():
    """Get the import error message if extension failed to load."""
    return _import_error if not _extension_available else None


class CRMDynamicsStep(torch.autograd.Function):
    """
    PyTorch autograd Function for CRM dynamics stepping.

    Phase 2A: Forward pass implemented via C++ extension calling Python bindings.
    Phase 3: Backward pass will use implicit differentiation.
    """

    @staticmethod
    def forward(ctx, currents, insertion_length, seed_v, seed_w, seed_p, seed_R,
                seed_xf, seed_mL, seed_nL, param_file, config_file, eps_seed=1e-4):
        """
        Forward pass: compute next_state from inputs.

        Args:
            currents: (B, 3) - Control currents
            insertion_length: (B,) or scalar - Insertion length
            seed_v, seed_w, seed_p: (B, num_sets, 3) - Velocity/position seeds
            seed_R: (B, num_sets, 9) - Rotation matrix seeds
            seed_xf: (B, 15) - Tip state seed
            seed_mL, seed_nL: (B, num_sets, 3) - Load seeds
            param_file, config_file: CRM configuration files
            eps_seed: Epsilon for implicit differentiation (backward pass)

        Returns:
            next_state: (B, output_dim) where output_dim = 3 + 3*num_sets
        """
        if not _extension_available:
            raise RuntimeError("C++ extension not available. " + str(_import_error))

        # Convert all inputs to CPU float64 (required by C++ extension)
        currents = currents.detach().cpu().double().contiguous()
        insertion_length = insertion_length.detach().cpu().double().contiguous()
        seed_v = seed_v.detach().cpu().double().contiguous()
        seed_w = seed_w.detach().cpu().double().contiguous()
        seed_p = seed_p.detach().cpu().double().contiguous()
        seed_R = seed_R.detach().cpu().double().contiguous()
        seed_xf = seed_xf.detach().cpu().double().contiguous()
        seed_mL = seed_mL.detach().cpu().double().contiguous()
        seed_nL = seed_nL.detach().cpu().double().contiguous()

        # Call C++ forward function
        next_state = _crm_torch_ext.dynamics_forward(
            currents, insertion_length,
            seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL,
            param_file, config_file
        )

        # Save for backward
        ctx.save_for_backward(
            currents, insertion_length,
            seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL
        )
        ctx.param_file = param_file
        ctx.config_file = config_file
        ctx.eps_seed = eps_seed

        return next_state

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: compute gradients via implicit differentiation.

        Phase 3A (Implemented): Computes current gradients via Option A's implicit linearization.
        Uses linearize_full_seed_action_from_seed_implicit to get B Jacobian (∂y/∂currents).
        Computes grad_currents = B^T @ grad_output.

        Seed gradients: Currently zero (Phase 3B - can be added if needed).
        """
        if not _extension_available:
            raise RuntimeError("C++ extension not available")

        # Retrieve saved tensors
        saved = ctx.saved_tensors
        currents, insertion_length = saved[0], saved[1]
        seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL = saved[2:]

        # Call C++ backward function (Phase 3 - currently returns zeros)
        grads = _crm_torch_ext.dynamics_backward(
            grad_output.detach().cpu().double().contiguous(),
            currents, insertion_length,
            seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL,
            ctx.param_file, ctx.config_file, ctx.eps_seed
        )

        # Return gradients for all inputs (+ None for non-tensor args)
        return grads + (None, None, None)  # None for param_file, config_file, eps_seed


# Export C++ functions and autograd wrapper
if _extension_available:
    dynamics_forward = _crm_torch_ext.dynamics_forward
    dynamics_backward = _crm_torch_ext.dynamics_backward

__all__ = [
    'is_available',
    'get_import_error',
    'CRMDynamicsStep',
    'dynamics_forward',
    'dynamics_backward',
]
