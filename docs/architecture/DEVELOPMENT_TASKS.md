# Development Tasks & Status Report

**Date:** December 21, 2025
**Status:** Repository Stabilized & Verified

## 1. Repository Health
*   **Structure:** Organized (`data/`, `docs/`, `src/`, `tests/`).
*   **Verification:** All tests passed (`pytest`, `make`, validation scripts).
*   **Fixes:** Resolved broken paths (`load_parameters`, `data/simulation_parameters`) and repaired Python bindings.
*   **Integrity:** Documentation is consistent with the code.

## 2. Differentiable Simulator Plan (Option A) Status
We are currently at **Phase 1 (Proof of Concept)**.
*   **Done:**
    *   `autodiff_eigen` branch merged.
    *   `DYNNLEquationJacobianEigenAD` computes $\partial F/\partial x$ (solver state Jacobian).
*   **Missing (Next Steps):**
    *   Gradient w.r.t parameters ($\,\partial F/\partial \theta$).
    *   Gradient w.r.t control inputs ($\,\partial F/\partial u$).
    *   Multi-actuator support (`NUM_ACT_SET > 1`).

## 3. Recommended Tasks (Next Steps)

### 🚀 Priority 1: Implement Parameter Gradients (Task A1)
**Goal:** Enable "End-to-End" training by computing gradients of the dynamics residual w.r.t physical parameters.
- [ ] **Refactor `DYNNLEqnParams`:** Create a templated version of this struct that can hold `autodiff::real` types for learnable parameters (damping, stiffness, etc.).
- [ ] **Update Residual:** Modify `DYNNLEquationResidualEigenAD` to use these templated parameters.
- [ ] **Expose Jacobian:** Implement `DYNNLEquationParameterJacobianEigenAD` to return $\,\partial F/\partial \theta$.
- [ ] **Python Binding:** Expose this new Jacobian to Python via `crm_bindings.cpp`.

### 🛡️ Priority 2: System Identification Validation
**Goal:** Prove that the differentiable simulator can actually learn.
- [ ] **Synthetic Test:** Generate a trajectory with known parameters. Initialize the model with wrong parameters. Use gradient descent (using the new Jacobian) to recover the true parameters.
- [ ] **Sim-to-Real Test:** Load `data/experimental` data and fine-tune simulation parameters to minimize tracking error.

### ⚡ Optimization (Optional / Low Priority)
*   **Batching:** Move the `for batch_idx in range(N)` loop from Python (`torch_physics.py`) into C++ to reduce overhead.
*   **Memory Safety (C++ Core):** *Suggestion Only*: Replace raw array allocations in `CRM_MatrixOperations.hpp` with `Eigen` types for better safety and vectorization. **(Do not touch core C++ unless critical).**
*   **MATLAB Cleanup:** Mark MATLAB scripts as "Legacy" to prevent confusion with the active C++/Python pipeline.
