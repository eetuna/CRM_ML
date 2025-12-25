# Option C Implementation Plan: Torch C++ Extension

**Date:** 2025-12-25
**Goal:** Create a high-performance PyTorch C++ extension (`crm_torch`) that directly exposes the stabilized CRM physics engine as a custom autograd function, eliminating Python overhead and enabling massive batch parallelization.

**Prerequisites:** Option A Stabilization (Complete).

---

## Phase 1: Infrastructure & Build System
**Objective:** Set up the build environment to compile `src/` code directly into a Python extension module using `torch.utils.cpp_extension`.

### Task 1.1: Extension Structure Setup
*   **File:** `crm_torch/` (New Directory)
*   **Subtask 1.1.1:** Create directory structure:
    ```
    crm_torch/
    ├── src/
    │   ├── crm_torch_binding.cpp  (Main entry point)
    │   ├── torch_utils.hpp        (Type conversion helpers)
    │   └── dynamics_op.cpp        (Autograd function impl)
    ├── setup.py                   (Build script)
    └── __init__.py                (Package init)
    ```
*   **Subtask 1.1.2:** Create `setup.py` configured to:
    *   Find Eigen3 and AutoDiff headers.
    *   Link against `CRMCPPLib` (or compile source directly).
    *   Enable OpenMP for parallel batch processing.

### Task 1.2: Type Conversion & Header Strategy
*   **File:** `crm_torch/src/torch_utils.hpp`
*   **Subtask 1.2.1:** Implement efficient zero-copy mapping between `torch::Tensor` and `Eigen::Map`.
*   **Subtask 1.2.2:** Create helpers to extract `DYNNLEqnParams` from Python dictionaries.
*   **Subtask 1.2.3 (Refinement):** Create `crm_config.hpp` shim to standardize include paths for `Eigen` and `autodiff` (handling local `third_party` vs system paths), ensuring robust compilation across environments.

### Checkpoint 1: Build Verification
*   **Action:** Run `pip install -e .` inside `crm_torch/`.
*   **Success Criteria:** `import crm_torch` succeeds in Python.

---

## Phase 2: Forward Pass Implementation
**Objective:** Implement the batched forward dynamics stepping in C++ with OpenMP parallelization.

### Task 2.1: The Forward Operator
*   **File:** `crm_torch/src/dynamics_op.cpp`
*   **Subtask 2.1.1:** Implement `DynamicsStepForward`:
    *   Inputs: `currents` (B,3), `insertion` (B,1), `seeds` (B, seed_dim).
    *   Output: `next_state` (B, output_dim).
*   **Subtask 2.1.2:** Implement OpenMP loop:
    ```cpp
    #pragma omp parallel for
    for (int i = 0; i < batch_size; ++i) {
        // 1. Unpack seed[i]
        // 2. Initialize DynamicsContext
        // 3. Call step() (using Adaptive RK4 from Option A)
        // 4. Pack result into next_state[i]
    }
    ```

### Task 2.2: Python Binding Exposure
*   **File:** `crm_torch/src/crm_torch_binding.cpp`
*   **Subtask 2.2.1:** Bind the forward operator using `m.def("dynamics_forward", ...)`

### Checkpoint 2: Performance Benchmark
*   **Action:** Run `examples/benchmark_linearization.py` (adapted for Option C).
*   **Success Criteria:** Option C forward pass is >10x faster than Option A (Python loop) for batch size > 1000.

---

## Phase 3: Backward Pass (Autograd) Implementation
**Objective:** Implement the backward pass using the Implicit Differentiation logic from Option A, vectorized in C++.

### Task 3.1: The Backward Operator
*   **File:** `crm_torch/src/dynamics_op.cpp`
*   **Subtask 3.1.1:** Implement `DynamicsStepBackward`:
    *   Inputs: `grad_output` (B, output_dim), `saved_tensors` (inputs from forward).
    *   Output: `grad_currents`, `grad_insertion`, `grad_seeds`.
*   **Subtask 3.1.2:** Integrate AD Logic:
    *   Inside the OpenMP loop, instantiate `DynamicsContextAD`.
    *   Call `DYNNLEquationOutputJacobianEigenAD` (Option A logic).
    *   Compute Implicit Gradients: $dx/du = -J_{xx}^{-1} J_{xu}$.
    *   Compute Final Gradients: $dy/du = g_u + g_x (dx/du)$.
    *   Perform Vector-Jacobian Product (VJP) with `grad_output`.

### Task 3.2: Custom Autograd Function
*   **File:** `crm_torch/dynamics.py`
*   **Subtask 3.2.1:** Create `class CRMDynamicsStep(torch.autograd.Function)` that calls the C++ forward and backward ops.

### Checkpoint 3: Gradient Correctness
*   **Action:** Run `torch.autograd.gradcheck`.
*   **Success Criteria:** Gradients match Finite Difference baselines within $10^{-4}$.

---

## Phase 4: Integration & Optimization
**Objective:** Polish the extension for ease of use and maximum speed.

### Task 4.1: Memory Optimization
*   **Subtask 4.1.1:** Minimize memory allocations inside the hot loop. Reuse `DynamicsContext` where possible.
*   **Subtask 4.1.2:** Ensure thread safety of all static/global variables in the AD stack (Option A audit confirmed most are safe, but double-check `static` buffers).

### Task 4.2: RL Wrapper Update
*   **File:** `crm_ml_rl/wrappers/torch_physics.py`
*   **Subtask 4.2.1:** Add logic to detect if `crm_torch` is installed.
*   **Subtask 4.2.2:** If installed, swap the Python-loop implementation for `crm_torch.CRMDynamicsStep.apply`.

---

## Final Deliverable
*   A standalone `crm_torch` package.
*   Zero-config acceleration for all existing RL scripts.
