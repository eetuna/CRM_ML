# CRM Torch C++ Extension

PyTorch C++ extension for high-performance end-to-end differentiable catheter dynamics simulation.

## Overview

This extension packages the CRM_ML Option A infrastructure (C++ dynamics + AD Jacobians) into a native PyTorch custom operator, eliminating Python-level overhead.

## Installation

### Prerequisites

1. Build the main CRM_ML library first:
   ```bash
   cd /workspaces/catheter/CRM_ML
   mkdir -p build && cd build
   cmake ..
   make -j$(nproc)
   cd ..
   ```

2. Install the extension:
   ```bash
   pip install -e ./crm_torch_ext
   ```

## Usage

```python
import torch
from crm_torch_ext import crm_step

# Prepare inputs (example shapes for NUM_ACT_SET=1)
currents = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
insertion_length = torch.tensor([94.3], requires_grad=True)

# Seed state tensors
seed_v = torch.zeros(1, 3, requires_grad=True)   # [num_sets, 3]
seed_w = torch.zeros(1, 3, requires_grad=True)
seed_p = torch.zeros(1, 3, requires_grad=True)
seed_R = torch.eye(3).reshape(1, 9, requires_grad=True)  # [num_sets, 9]
seed_xf = torch.zeros(15, requires_grad=True)
seed_mL = torch.zeros(1, 3, requires_grad=True)
seed_nL = torch.zeros(1, 3, requires_grad=True)

# Forward pass
next_state = crm_step(
    currents, insertion_length,
    seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL
)

# Backward pass (automatic via PyTorch autograd)
loss = next_state.sum()
loss.backward()

print("Gradient w.r.t. currents:", currents.grad)
```

## Status

**Implementation Status:** In Progress

- [x] CP-C01: Package skeleton created
- [ ] CP-C02: Build system functional
- [ ] CP-C03: Forward pass implementation
- [ ] CP-C04: Backward pass implementation
- [ ] CP-C05: Autograd registration

See `docs/architecture/OPTION_C_IMPLEMENTATION_PLAN.md` for full roadmap.

## Performance

Target performance vs Python wrapper:
- Forward overhead: <0.1ms per call (vs ~1ms for pybind)
- Total iLQR iteration (horizon=30): 10-20s (vs 15-30s)

## License

[Specify license]
