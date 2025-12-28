"""
crm_torch - PyTorch C++ Extension for CRM Physics Simulation

This package provides a high-performance, differentiable wrapper around the
CRM catheter physics engine with OpenMP batch parallelization.

Phase 1 (Current): Build infrastructure and basic imports
Phase 2: Forward pass implementation
Phase 3: Backward pass (autograd) implementation
Phase 4: Validation and integration
"""

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

# Phase 1: Export placeholder for testing
if _extension_available:
    dynamics_forward_placeholder = _crm_torch_ext.dynamics_forward_placeholder

__all__ = [
    'is_available',
    'get_import_error',
    'dynamics_forward_placeholder',
]
