# CRM Torch C++ Extension

A high-performance PyTorch C++ extension for the CRM Catheter Dynamics physics engine.

## Overview

This extension packages the CRM physics engine (C++ core) as a native custom operator for PyTorch. It allows for:
- **Faster Forward Pass:** ~85x speedup compared to the pure Python wrapper.
- **Differentiable Physics:** Full autograd support via implicit differentiation.
- **Drop-in Replacement:** Compatible with existing `TorchCRMPhysics` wrapper.

## Installation

### Prerequisites
- PyTorch >= 1.10
- CMake >= 3.10
- C++17 compliant compiler (GCC/Clang)
- `crm_dynamics` shared library (built via root CMake)

### Build & Install

Run this from the project root:

```bash
# Build locally (recommended for development)
# MAX_JOBS=1 prevents memory issues during compilation
cd crm_torch_ext
MAX_JOBS=1 python setup.py build_ext --inplace
```

Ensure the root directory is in your `PYTHONPATH` to import it.

## Usage

### Low-Level API

```python
import torch
import crm_torch_ext

# Initialize (must be done once)
crm_torch_ext.initialize_params("path/to/params.txt", "path/to/config.txt")
crm_torch_ext.set_integrator("rk4")

# Tensors (must be double precision/float64)
currents = torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64)
insertion = torch.tensor([0.0], dtype=torch.float64)
# ... define seed tensors (v, w, p, R, xf, mL, nL) ...

# Forward Step
next_state = crm_torch_ext.crm_step(
    currents, insertion,
    seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL
)

# Backward
loss = next_state.sum()
loss.backward()
```

### High-Level API (Recommended)

Use the existing `TorchCRMPhysics` wrapper with the `use_cpp_extension` flag:

```python
from crm_ml_rl.wrappers.torch_physics import TorchCRMPhysics

physics = TorchCRMPhysics(
    "data/catheter_params/CatheterParameterSet_1_dyn.txt",
    "data/catheter_params/CatheterSpatialConfiguration_1.txt",
    use_cpp_extension=True  # <--- Enable extension
)

# Use exactly as before
out = physics.dyn_step(currents, insertion, ...)
```

## Performance

Benchmark results (100 iterations):
- **Forward Pass:** ~0.1 ms (vs 8.0 ms for Python wrapper) -> **~80x Speedup**
- **Backward Pass:** ~1.1 ms (Calculates Jacobians on demand)
- **Total Iteration:** ~1.2 ms (vs 8.2 ms for Python wrapper) -> **~6.8x Speedup**

## Troubleshooting

- **ImportError:** Ensure you ran `build_ext --inplace` and your `PYTHONPATH` is correct.
- **RuntimeError (Shape Mismatch):** The extension is strict about shapes. `insertion_length` must be `[1]`, `currents` must be `[3]`.
- **Divergence:** If the BVP solver fails, the extension returns zero gradients and logs a warning. Try reducing timestep or improving the initial guess (seed).