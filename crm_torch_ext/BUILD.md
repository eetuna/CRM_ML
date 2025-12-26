# Build Instructions

## Prerequisites

1. **Main CRM_ML library must be built first:**
   ```bash
   cd /workspaces/catheter/CRM_ML
   mkdir -p build && cd build
   cmake ..
   make -j$(nproc)
   cd ..
   ```

2. **Required system packages:**
   - PyTorch (tested with 2.9.1+cpu)
   - Eigen3 (system package)
   - Python 3.8+

## Building the Extension

### Method 1: In-place build (for development)

```bash
cd /workspaces/catheter/CRM_ML/crm_torch_ext
python3 setup.py build_ext --inplace
```

This creates `_crm_torch_ext.cpython-*.so` directly in the `crm_torch_ext/` directory.

### Method 2: pip install (recommended)

```bash
cd /workspaces/catheter/CRM_ML
pip install -e ./crm_torch_ext
```

This installs the package in development mode with automatic build.

## Verification

Run the build system test:

```bash
python3 crm_torch_ext/test/test_build_system.py
```

Expected output:
```
✓ CP-C02 ACCEPTANCE CRITERIA MET
  - Build system finds all required headers
  - Link step finds CRM library
  - Extension is importable from Python
```

## Usage

```python
import torch
from crm_torch_ext import crm_step

# Create inputs
currents = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
insertion_length = torch.tensor([94.3], requires_grad=True)
# ... other seed tensors ...

# Forward pass
result = crm_step(currents, insertion_length, ...)
```

## Troubleshooting

### "cannot open shared object file: libc10.so"

Ensure PyTorch libraries are in your library path. The extension automatically loads torch, which handles this.

### "cannot find -lCRMCPPLib"

The main CRM library hasn't been built. Run:
```bash
cd /workspaces/catheter/CRM_ML
mkdir -p build && cd build
cmake .. && make -j$(nproc)
```

### Headers not found

Check that these directories exist:
- `/workspaces/catheter/CRM_ML/src/`
- `/workspaces/catheter/CRM_ML/third_party/autodiff/`
- `/usr/include/eigen3/`

## Build Configuration

The extension is configured to match the main project:
- **C++ Standard:** C++17
- **Optimization:** -O3
- **Defines:** NUM_ACT_SET=1, USE_AUTODIFF
- **Linked Libraries:** CRMCPPLib (static), torch, c10

See `setup.py` for full configuration.
