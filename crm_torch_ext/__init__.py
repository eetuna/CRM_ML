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

    # Expose parameter management functions
    initialize_params = _crm_torch_ext.initialize_params
    initialize_from_fk = _crm_torch_ext.initialize_from_fk
    set_timestep = _crm_torch_ext.set_timestep
    set_integrator = _crm_torch_ext.set_integrator
    set_integration_step_size = _crm_torch_ext.set_integration_step_size
    set_damping = _crm_torch_ext.set_damping

    __all__ = [
        'crm_step',
        'initialize_params',
        'initialize_from_fk',
        'set_timestep',
        'set_integrator',
        'set_integration_step_size',
        'set_damping',
    ]

except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import crm_torch_ext C++ extension: {e}")
    import warnings
    warnings.warn(
        f"Failed to import crm_torch_ext C++ extension: {e}\n"
        "The extension may not be built yet. Run: pip install -e ./crm_torch_ext\n"
        "If built, ensure PyTorch is installed: pip install torch",
        ImportWarning
    )
    __all__ = []
