# Code Audit Phase 1: Core C++ (`src/`)

**Date:** 2025-12-25
**Auditor:** Gemini Agent
**Status:** Completed

## 1. Executive Summary
The core C++ codebase has been modernized significantly (Task A1.7). The critical memory aliasing ("Ghost Value") bug appears resolved by the migration to `std::vector` and `Eigen` containers. However, a discrepancy exists between the AD-enabled dynamics (used for gradients) and the legacy double-precision dynamics (likely used for forward simulation), specifically regarding adaptive stepping. Additionally, multi-actuator output support is structurally present but logic-incomplete.

## 2. Detailed Findings

### 2.1 "Ghost Value" Bug Regression Check
*   **Status:** ✅ **RESOLVED**
*   **Evidence:**
    *   `CRMShootingMethodParams` and `CRMIVPCoreParams` in `src/CRM_BVPIVP_APIDeclarations.hpp` now use `std::vector<Eigen::Matrix3d>`, `std::vector<double>`, etc.
    *   The `DynamicsContext` struct is properly integrated.
    *   No raw pointer arrays (e.g., `double (*K)[9]`) were found in the core parameter structs.
    *   Legacy C-style APIs (`CRMSolverIVP`) still accept raw arrays, but these are safely copied into the vector-based storage.

### 2.2 Integrator Stability & Adaptive Stepping
*   **Status:** ⚠️ **INCONSISTENT**
*   **Issue:** Adaptive stepping (RK4 subdivision on acceleration spikes) is **implemented only in the AD path**, not the legacy path.
*   **Evidence:**
    *   **AD Path** (`src/CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp`): Contains `rk4_step_adaptive` with `kAngularAccelThreshold = 1000.0` and recursion logic.
    *   **Legacy Path** (`src/CoilDynamics_Defs.cpp`): Contains `RK4_coildyn` and `CoilDynamicsRK4`, but `CoilDynamicsRK4` only checks for divergence *after* the step loop or step completion. It does **not** appear to have the `rk4_step_adaptive` subdivision logic.
*   **Impact:** The forward simulation (if it uses `CoilDynamics_Defs.cpp`) may be less stable than the gradient computation path. If `step()` uses the legacy path, it might diverge on stiff inputs where the AD path (used for linearization) would succeed (or fail gracefully).

### 2.3 AutoDiff Architecture (Option A)
*   **Status:** ✅ **MOSTLY CORRECT**
*   **Evidence:**
    *   `DYNNLEquationResidualWithParamsAD` is fully templated on `Scalar`.
    *   Output Jacobian `g_θ` logic is wired in `DYNNLEquationOutputJacobianEigenAD`.
    *   Scaling constants (`IVALUE_SCALE_*`) are applied consistently.

### 2.4 Multi-Actuator Scalability
*   **Status:** ⚠️ **INCOMPLETE LOGIC**
*   **Issue:** `eval_output_AD` calculates dynamic output size but fails to populate data for actuators > 0.
*   **Evidence:**
    *   In `src/CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp`:
        ```cpp
        // Task 4.9: Build output with dynamic sizing
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> y(output_dim);
        y(0..2) = ... // Tip
        y(3..5) = vw_out(0..2); // Actuator 0
        // TODO: For NUM_ACT_SET > 1, compute remaining actuator velocities
        ```
    *   The code structurally supports `NUM_ACT_SET > 1` (loops exist), but this specific output function truncates data for secondary actuators.

### 2.5 Build System
*   **Status:** ✅ **CORRECT**
*   **Evidence:** `CMakeLists.txt` correctly includes `src/numerical` and sets up include paths.

## 3. Recommendations

1.  **Backport Adaptive Stepping:** Port `rk4_step_adaptive` from `autodiff_eigen.hpp` to `CoilDynamics_Defs.cpp` to ensure the forward simulation is as robust as the AD path.
2.  **Complete Multi-Actuator Output:** Implement the missing loop in `eval_output_AD` to populate `y(6+)` for additional actuators.
3.  **Verify Forward Path:** Confirm which integrator `step()` actually uses. (To be checked in Phase 2).
