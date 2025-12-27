## Features

- **Fast Forward Dynamics**: C++ implementation with Python bindings
- **PyTorch C++ Extension**: Native custom operator (~85x faster forward pass)
- **Automatic Differentiation**: Full AD support via autodiff library (Option A architecture)
- **Implicit Differentiation**: Efficient gradient computation through BVP residuals
- **Multi-Actuator Support**: Handles single or multiple actuator sets
- **iLQR Controller**: Iterative Linear Quadratic Regulator for trajectory optimization
- **PyTorch Integration**: Seamless integration for deep learning workflows

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd CRM_ML

# Build C++ library and Python bindings
mkdir -p build
cd build
cmake ..
make -j$(nproc)
cd ..

# Install Python package
pip install -e .

# Build High-Performance C++ Extension (Optional but Recommended)
# Ensure root is in PYTHONPATH
cd crm_torch_ext
MAX_JOBS=1 python setup.py build_ext --inplace
cd ..
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### Running the iLQR Demo

The iLQR (Iterative Linear Quadratic Regulator) demo demonstrates trajectory optimization for catheter control:

```bash
# Basic reaching task (default: implicit AD linearization)
python3 examples/ilqr_catheter_demo.py

# With verbose output (show iteration details)
python3 examples/ilqr_catheter_demo.py --verbose

# Compare implicit AD vs finite differences
python3 examples/ilqr_catheter_demo.py --mode compare

# Trajectory tracking demo
python3 examples/ilqr_catheter_demo.py --mode tracking

# Use finite differences for linearization (slower)
python3 examples/ilqr_catheter_demo.py --method fd
```

**Expected Output:**
```
============================================================
REACHING DEMO (Implicit AD)
============================================================
Initial position: [X Y Z] mm
Target position: [X Y Z] mm
Initial distance: N mm

Final position: [X Y Z]
Final error: N mm
Total time: N s
Converged: True
```

**Performance Notes:**
- Implicit AD linearization: ~0.5-1.0s per step
- For horizon=30 steps: ~15-30s per iLQR iteration
- Convergence typically achieved in 3-10 iterations

### Using the Dynamics API

```python
from crm_ml_rl.wrappers import crm_python
import numpy as np

# Initialize dynamics
dyn = crm_python.CRMDynamics()
dyn.load_parameters(
    "data/catheter_params/CatheterParameterSet_1_dyn.txt",
    "data/catheter_params/CatheterSpatialConfiguration_1.txt"
)

# Set stable damping values
BASE_DAMPING = np.array([12.18, 12.18, 284.43, 0.0305, 0.0305, 0.00503])
dyn.set_damping(BASE_DAMPING)
dyn.dt = 0.02  # 20ms timestep
dyn.integration_step_size = 0.1

# Initialize from kinematics
currents = np.array([0.0, 0.0, 0.0])
insertion_length = 94.3  # mm
dyn.initialize_from_kinematics(currents, insertion_length)

# Step forward dynamics
result = dyn.step(currents, insertion_length)
tip_position = result['tip_position']
coil_velocities = result['coil_velocities']  # Array of shape (num_actuators, 3)

# Get linearization (for control/RL)
seed = dyn.get_seed_state()
lin_result = dyn.linearize_full_seed_action_from_seed_implicit(
    currents, insertion_length,
    seed['v'], seed['w'], seed['p'], seed['R'], seed['xf'],
    seed['mL'], seed['nL']
)
A_matrix = lin_result['A']  # State Jacobian
B_matrix = lin_result['B']  # Control Jacobian (includes gradients for currents)
grad_ins = lin_result['grad_insertion']  # Gradient w.r.t. insertion length
```

## Architecture: Option A & C (Hybrid)

This implementation uses a hybrid architecture for end-to-end differentiation:

- **Option A (Core)**: C++ forward dynamics + AD-based implicit linearization.
- **Option C (Acceleration)**: Native PyTorch C++ Extension (`crm_torch_ext`) wrapping Option A.
    - **Speedup**: ~85x faster forward pass, ~6x faster overall iteration.
    - **Integration**: Transparently used by `TorchCRMPhysics` when built.

See `docs/architecture/PLAN_end_to_end_differentiable_simulator_options_A_B_C.md` for architectural details.

## Examples

### Validation Scripts

```bash
# Verify AD correctness with fine epsilon
python3 examples/verify_ad_with_fine_epsilon.py

# Verify output Jacobian implementation
python3 examples/verify_output_jacobian_gth.py

# Verify multi-actuator output support
python3 examples/verify_multi_actuator_output.py

# Run full validation suite
python3 examples/validate_full_suite.py
```

### Performance Benchmarks

```bash
# Benchmark linearization methods
python3 examples/benchmark_linearization.py

# Benchmark C++ Extension vs Python Wrapper
python3 crm_torch_ext/benchmark/benchmark_overhead.py
```

## Development Status

### Completed Checkpoints (Stabilization & Remediation)

- ✅ **CP-01 through CP-08**: Stabilization plan executed (BVP workaround, AD verification, Demo cleanup).
- ✅ **Remediation Phase 1**: Physical AD Chain complete. **Non-zero control gradients** and **Insertion Length gradients** enabled.
- ✅ **Remediation Phase 2**: Forward Stability Synchronization. **Adaptive RK4** backported to forward simulation.
- ✅ **Remediation Phase 3**: Solver Homotopy. `step_from_seed` made robust for consecutive stepping via **velocity continuation**.
- ✅ **Remediation Phase 4**: Multi-Actuator Generalization. Dynamic sizing and recursive chaining infrastructure implemented.
- ✅ **Option C Phase 1-5**: Native PyTorch C++ Extension implemented and integrated.

### Key Achievements

1. **End-to-End Differentiable**: Control inputs (currents, insertion) now produce physically accurate, non-zero gradients.
2. **Robust BVP Solver**: Resolved `localmin=3` errors in moving frames via homotopy continuation.
3. **Unified Stability**: Forward simulation now uses the same adaptive timestep logic as the AD pass.
4. **Scalable Architecture**: Support for `NUM_ACT_SET > 1` through dynamic sizing and `coil_velocities` API.
5. **High Performance**: Native C++ operator eliminates Python overhead for critical loops.
