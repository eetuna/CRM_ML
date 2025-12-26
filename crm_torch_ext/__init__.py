# CRM Torch C++ Extension
# End-to-End Differentiable Catheter Dynamics Simulator
# Option C: Native PyTorch C++ Extension

__version__ = "0.1.0"

try:
    # Ensure torch libraries are available
    import torch
    from . import _crm_torch_ext

    # Expose the main operator
    crm_step = _crm_torch_ext.crm_step

    __all__ = ['crm_step']

except ImportError as e:
    import warnings
    warnings.warn(
        f"Failed to import crm_torch_ext C++ extension: {e}\n"
        "The extension may not be built yet. Run: pip install -e ./crm_torch_ext\n"
        "If built, ensure PyTorch is installed: pip install torch",
        ImportWarning
    )
    __all__ = []
