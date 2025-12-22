# Development Tasks & Status Report

**Date:** December 22, 2025
**Status:** Task A1 (Parameter Gradients) Implemented and Validated
**Parent Strategy:** [End-to-End Differentiable Simulator Plan (Options A/B/C)](PLAN_end_to_end_differentiable_simulator_options_A_B_C.md)

> **Context:** This document outlines the specific engineering tasks required to execute **Option A (C++ AD + Implicit Diff)** from the parent strategy. It translates the high-level goals of Option A into concrete, checkable coding tasks.

## 1. Repository Health
*   **Structure:** Organized (`data/`, `docs/`, `src/`, `tests/`).
*   **Verification:** All tests passed (`pytest`, `make`, validation scripts).
*   **Fixes:** Resolved broken paths (`load_parameters`, `data/simulation_parameters`) and repaired Python bindings.
*   **Integrity:** Documentation is consistent with the code.

## 2. Differentiable Simulator Plan (Option A) Status
We are currently at **Phase 1 (Proof of Concept)** - Task A1 Complete.
*   **Done:**
    *   **Task 1.1:** Evaluate Differentiable Frameworks (Chosen: `autodiff` + Eigen).
    *   **Task 1.2:** Prototype Core Physics Component (Implemented dynamics residual Jacobian $\partial F/\partial x$).
    *   `autodiff_eigen` branch merged.
    *   `DYNNLEquationJacobianEigenAD` computes $\partial F/\partial x$ (solver state Jacobian).
    *   **NEW:** `DYNNLEquationParameterJacobianEigenAD` computes $\partial F/\partial \theta$ (parameter Jacobian).
    *   **NEW:** Python bindings for `compute_parameter_jacobian()` and `compute_residual_at_state()`.
    *   **FIXED:** Parameter Reporting Bug ("Ghost Value" issue). See Section 3 for details.
*   **Missing (Next Steps):**
    *   ~~Gradient w.r.t parameters ($\partial F/\partial \theta$).~~ ✅ Done
    *   **Task 1.3:** Verify Gradient Accuracy (Finite-Difference validation complete; tests now run by default).
    *   ~~Task 1.4: Benchmark Performance (Profile AutoDiff vs. FD vs. Analytic).~~ ✅ Done (see `scripts/benchmark_task1_4.py`, results in `data/output/benchmark_task1_4.json`; benchmark now skips unconverged runs and records failure counts)
    *   ~~Task 1.5: Implement gradient w.r.t control inputs ($\partial F/\partial u$).~~ ✅ Done
    *   **Task A1.6:** Dynamics Stabilization (Mitigated; convergence stable with validated damping values).
    *   **Task A2:** Multi-actuator support (`NUM_ACT_SET > 1`).
    *   **Known Issue (RESOLVED):** `packLearnableParams` returned unexpected theta values due to memory aliasing in the legacy core. This is now handled via a manual synchronization layer. See Section 3.

## 3. Technical Notes & Known Issues

### ⚠️ Dynamics Instability ("Coil integration Unbounded")
The solver frequently crashes with "Coil integration Unbounded!!" during training or validaton.
*   **Root Cause:** The `CoilDynamics` solver uses an **Explicit** ABM4 integrator with a fixed timestep (`t_step = 0.001`). This is unstable for stiff systems (high stiffness/damping, low mass).
*   **Plan (Task A1.6):**
    1.  **Adaptive Stepping:** Implement a simple adaptive scheme or reduce `t_step` dynamically when accelerations are high.
    2.  **Force Clamping:** Clamp maximum forces/moments passed to the integrator to prevent non-physical explosions.
    3.  **Soft Failure:** Update `crm_bindings.cpp` to catch `isnan` and return a large penalty instead of terminating the process.

### 🐞 The "Legacy Prep Overlap" Bug
During implementation of Task A1, a critical memory aliasing issue was identified in the legacy C++ core (`src/CoilDynamics_Defs.cpp` and `src/CRM_BVPIVP_APIDeclarations.hpp`):
*   **Root Cause:** The classes `CRMShootingMethodParams` and `CRMIVPCoreParams` mix heap-allocated pointers (e.g., `double (*MagMoment)[3]`) with fixed-size arrays (e.g., `double damping[NUM_ACT_SET][6]`). 
*   **The Error:** When the legacy `Prep` or `Construct` functions run, the calculated **Magnetic Moment** (e.g., `0.018448` for 10mA) is written to a memory location that overlaps with the `damping` array. This is likely due to a compiler padding mismatch or a `NUM_ACT_SET` synchronization error between headers.
*   **Symptoms:** Python would report all learnable parameters as `0.018448`, regardless of their actual values.
*   **The Fix:** A **Manual Synchronization** layer was added to `crm_bindings.cpp`. After the legacy C++ initialization runs, the bindings manually re-inject the correct physical values (damping, stiffness, etc.) into the AutoDiff parameter vector. This ensures gradients are calculated on the **true** values while keeping the legacy core untouched.

#### 🛠️ Proper Fix Implementation Plan (Long-term)
To permanently resolve this without relying on the bindings sync, the following core refactor is required:

**Phase 1: Type Modernization**
1.  **Refactor `CRMIVPCoreParams`:**
    *   Replace `double (*K)[9]` with `std::vector<Eigen::Matrix3d>`.
    *   Replace `double damping[NUM_ACT_SET][6]` with `std::vector<Eigen::Matrix<double, 6, 1>>`.
    *   This eliminates manual pointer arithmetic and prevents the compiler from generating unsafe fixed-offset code that assumes a specific `NUM_ACT_SET`.

**Phase 2: Structural Decoupling**
1.  **Create `DynamicsContext`:** Move dynamics-specific parameters (damping, mass, inertia) out of the general `CRMIVPCoreParams` and into a dedicated `DynamicsContext` struct.
2.  **Update `CoilDynamics_Defs.cpp`:** Rewrite the `Prep` function to populate this new struct explicitly, using `Eigen::Map` for safe data transfer.

**Phase 3: Full Templatization**
1.  **Templatize Solvers:** Templatize `CRMSolverIVP` and `CRMShootingMethodBVP` on a `Scalar` type.
2.  **Unified Params:** Define `CRMParams<Scalar>` that can hold either `double` or `autodiff::real`.
3.  **Remove Shadow Structs:** Once the core solvers accept `CRMParams<autodiff::real>`, the `DYNNLEqnParamsAD` shadow struct and the manual sync layer can be deleted.

## 4. Primary Task: Core C++ Refactor (Task A1.7)
**Goal:** Replace the brittle legacy core with a robust, modern C++ foundation that supports AutoDiff natively and uses stable integration. This supersedes the temporary "Manual Sync" and "Acceleration Clamp" patches.

### Phase 1: Memory & Type Modernization
*   **Target:** `CRMIVPCoreParams` and `CRMShootingMethodParams`.
*   **Action:** Replace all raw heap pointers (e.g., `double (*K)[9]`) and fixed-size arrays with modern C++ containers:
    *   `std::vector<Eigen::Matrix3d>` for stiffness/rotation matrices.
    *   `std::vector<Eigen::Vector3d>` for moments/positions.
    *   `std::vector<Eigen::Matrix<double, 6, 1>>` for damping.
*   **Benefit:** Eliminates memory aliasing/corruption (the "Ghost Value" bug) and simplifies initialization.

### Phase 2: Solver Templatization
*   **Target:** `CoilDynamics`, `CRMIntegrand`, and the `BVP/IVP` solvers.
*   **Action:** Fully templatize these functions on `<typename Scalar>` to support `autodiff::real` natively.
*   **Benefit:** Removes the need for "Shadow Structs" and ensures the same verified math flows through both simulation and gradient calculation.

### Phase 3: Integrator Stabilization
*   **Target:** `CoilDynamics_Defs.cpp`.
*   **Action:** Replace the brittle, history-dependent **ABM4** integrator with a memoryless **Runge-Kutta 4th Order (RK4)** scheme.
*   **Benefit:** Prevents numerical explosions ("Coil integration Unbounded") during AutoDiff parameter perturbations.

## 5. Recommended Tasks (Next Steps)

### ✅ Priority 1: Implement Parameter Gradients (Task A1) - COMPLETE
**Goal:** Enable "End-to-End" training by computing gradients of the dynamics residual w.r.t physical parameters.
- [x] **Refactor `DYNNLEqnParams`:** Created `DYNNLEqnParamsAD<Scalar>` shadow struct with templated learnable parameters.
- [x] **Update Residual:** Implemented `DYNNLEquationResidualWithParamsAD<Scalar>` to use templated parameters.
- [x] **Expose Jacobian:** Implemented `DYNNLEquationParameterJacobianEigenAD` returning $\partial F/\partial \theta$ (16 params).
- [x] **Python Binding:** Exposed `compute_parameter_jacobian()` via `crm_bindings.cpp`.
- [x] **Tests:** Added `tests/test_parameter_jacobian_autodiff.py` with shape/finite validation.

**Learnable Parameters (16 total):**
| Parameter | Dim | Description |
|-----------|-----|-------------|
| `damping` | 6 | Linear (3) + angular (3) damping coefficients |
| `K_diag` | 3 | Diagonal of stiffness matrix |
| `ustar` | 3 | Rest curvature |
| `actMass` | 1 | Actuator mass |
| `MagMoment` | 3 | Magnetic moment vector |

**Known Limitation:** None for Task A1 validation; full AD vs FD tests run by default.

### 🛡️ Priority 2: System Identification Validation
**Goal:** Prove that the differentiable simulator can actually learn.
- [ ] **Synthetic Test:** Generate a trajectory with known parameters. Initialize the model with wrong parameters. Use gradient descent (using the new Jacobian) to recover the true parameters.
- [ ] **Sim-to-Real Test:** Load `data/experimental` data and fine-tune simulation parameters to minimize tracking error.

### ⚡ Optimization (Optional / Low Priority)
*   **Batching:** Move the `for batch_idx in range(N)` loop from Python (`torch_physics.py`) into C++ to reduce overhead.
*   **Memory Safety (C++ Core):** *Suggestion Only*: Replace raw array allocations in `CRM_MatrixOperations.hpp` with `Eigen` types for better safety and vectorization. **(Do not touch core C++ unless critical).**
*   **MATLAB Cleanup:** Mark MATLAB scripts as "Legacy" to prevent confusion with the active C++/Python pipeline.
