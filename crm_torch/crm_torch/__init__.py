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
            Tuple of (next_state, next_v, next_w, next_p, next_R, next_xf, next_mL, next_nL)
            - next_state: (B, output_dim) where output_dim = 3 + 3*num_sets
            - next_v, next_w, next_p, next_R, next_xf, next_mL, next_nL: Updated seed tensors
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

        # Call C++ forward function (Phase 3B: now returns updated seeds!)
        result = _crm_torch_ext.dynamics_forward(
            currents, insertion_length,
            seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL,
            param_file, config_file
        )

        # Unpack result tuple
        next_state, next_v, next_w, next_p, next_R, next_xf, next_mL, next_nL = result

        # Save for backward
        ctx.save_for_backward(
            currents, insertion_length,
            seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL
        )
        ctx.param_file = param_file
        ctx.config_file = config_file
        ctx.eps_seed = eps_seed

        return next_state, next_v, next_w, next_p, next_R, next_xf, next_mL, next_nL

    @staticmethod
    def backward(ctx, grad_output, grad_next_v, grad_next_w, grad_next_p,
                 grad_next_R, grad_next_xf, grad_next_mL, grad_next_nL):
        """
        Backward pass: compute gradients via implicit differentiation.

        Phase 3B: Handles gradients from both next_state and updated seeds.
        This enables multi-step gradient flow.

        Args:
            grad_output: Gradient w.r.t. next_state
            grad_next_v, grad_next_w, etc.: Gradients w.r.t. updated seeds (from next step)
        """
        # DEBUG: Print when backward is called
        import os
        if os.getenv("CRM_DEBUG_BACKWARD") == "1":
            print(f"\n[CRMDynamicsStep.backward] Called!")
            print(f"  grad_output is None: {grad_output is None}")
            print(f"  grad_next_v is None: {grad_next_v is None}")
            if grad_output is not None:
                print(f"  grad_output norm: {torch.norm(grad_output).item():.6e}")
            if grad_next_v is not None:
                print(f"  grad_next_v norm: {torch.norm(grad_next_v).item():.6e}")

        if not _extension_available:
            raise RuntimeError("C++ extension not available")

        # Retrieve saved tensors
        saved = ctx.saved_tensors
        currents, insertion_length = saved[0], saved[1]
        seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL = saved[2:]

        # Phase 3B FIX: Pass grad_next_* to C++ for proper multi-step gradient flow
        # If None, use zeros (no gradient from that output)
        if grad_next_v is None:
            grad_next_v = torch.zeros_like(seed_v)
        if grad_next_w is None:
            grad_next_w = torch.zeros_like(seed_w)
        if grad_next_p is None:
            grad_next_p = torch.zeros_like(seed_p)
        if grad_next_R is None:
            grad_next_R = torch.zeros_like(seed_R)
        if grad_next_xf is None:
            grad_next_xf = torch.zeros_like(seed_xf)
        if grad_next_mL is None:
            grad_next_mL = torch.zeros_like(seed_mL)
        if grad_next_nL is None:
            grad_next_nL = torch.zeros_like(seed_nL)

        # Call C++ backward function with grad_next_* for multi-step gradient flow
        # C++ now computes: grad_currents = B^T @ grad_output + B_seeds^T @ grad_next_seeds
        grads = _crm_torch_ext.dynamics_backward(
            grad_output.detach().cpu().double().contiguous(),
            grad_next_v.detach().cpu().double().contiguous(),
            grad_next_w.detach().cpu().double().contiguous(),
            grad_next_p.detach().cpu().double().contiguous(),
            grad_next_R.detach().cpu().double().contiguous(),
            grad_next_xf.detach().cpu().double().contiguous(),
            grad_next_mL.detach().cpu().double().contiguous(),
            grad_next_nL.detach().cpu().double().contiguous(),
            currents, insertion_length,
            seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL,
            ctx.param_file, ctx.config_file, ctx.eps_seed
        )

        # grads = (grad_currents, grad_insertion, grad_seed_v, grad_seed_w,
        #          grad_seed_p, grad_seed_R, grad_seed_xf, grad_seed_mL, grad_seed_nL)
        #
        # Note: C++ now handles the multi-step contribution to grad_currents!
        # The seed gradients still need the pass-through term added here
        # (for seeds that are essentially copied forward like v, w, p, R)
        grad_currents = grads[0]
        grad_insertion = grads[1]
        # Add pass-through gradients for non-solved seeds (v, w, p, R, xf)
        # These are approximately identity-transformed, so grad flows directly
        grad_seed_v = grads[2] + grad_next_v.cpu().double().contiguous()
        grad_seed_w = grads[3] + grad_next_w.cpu().double().contiguous()
        grad_seed_p = grads[4] + grad_next_p.cpu().double().contiguous()
        grad_seed_R = grads[5] + grad_next_R.cpu().double().contiguous()
        grad_seed_xf = grads[6] + grad_next_xf.cpu().double().contiguous()
        # For mL/nL: the C++ already included their contribution to grad_currents
        # but we still add them here for the seed input gradient
        grad_seed_mL = grads[7] + grad_next_mL.cpu().double().contiguous()
        grad_seed_nL = grads[8] + grad_next_nL.cpu().double().contiguous()

        # Return gradients for all inputs (+ None for non-tensor args)
        return (grad_currents, grad_insertion, grad_seed_v, grad_seed_w,
                grad_seed_p, grad_seed_R, grad_seed_xf, grad_seed_mL, grad_seed_nL,
                None, None, None)  # None for param_file, config_file, eps_seed


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
