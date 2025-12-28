# Code Audit Phase 2: Python Bindings (`crm_bindings.cpp`)

**Date:** 2025-12-25
**Auditor:** Gemini Agent
**Status:** Completed

## 1. Executive Summary
The Python bindings provide a comprehensive interface to the CRM physics engine. The migration to `DynamicsContext` has successfully decoupled the bindings from brittle legacy memory structures. However, while the "Option A" AD-based implicit linearization is structurally sound and supports dynamic sizing, the legacy FD-based paths and some helper functions still contain hardcoded 6D state assumptions (Tip + 1 Actuator), which will break for `NUM_ACT_SET > 1`.

## 2. Detailed Findings

### 2.1 Interface Exposure
*   **Status:** ✅ **EXCELLENT**
*   **Evidence:**
    - `step()` and `step_from_seed()` are clearly distinguished.
    - `step_from_seed` includes a loud warning about the BVP solver's consecutive stepping limitation.
    - `initialize_from_kinematics` is well-implemented, ensuring a valid physical starting point for dynamics.

### 2.2 Memory Safety & Data Types
*   **Status:** ✅ **SAFE**
*   **Evidence:**
    - Extensive use of `py::array_t<double>` with buffer checks.
    - Proper use of `mutable_unchecked` for array population.
    - Inputs are safely copied into local fixed-size buffers (`double[NUM_ACT_SET][3]`) before entering C++ core, respecting the compile-time limits.

### 2.3 Dynamic Sizing Logic (Multi-Actuator Support)
*   **Status:** ⚠️ **INCONSISTENT**
*   **Issues:**
    1.  **Hardcoded FD State:** The `get_state6` lambda used in `linearize_full_seed_action_from_seed` (FD baseline) is hardcoded to return a 6D vector (line 1815). It will ignore additional actuators.
    2.  **Hardcoded dictionary output:** `step_from_seed` (line 1765) returns `tip_position` (3D) and `tip_velocity` (3D). For multi-actuator systems, users would expect velocities for all actuators.
    3.  **Leftover Lambda:** The `eval_output` lambda inside `linearize_full_seed_action_from_seed_implicit` (line 2255) is also hardcoded to 6D. Although the AD-path (`DYNNLEquationOutputJacobianEigenAD`) uses dynamic sizing, any fallback to FD for the output map will truncate data.
    4.  **Wait-State Mapping Bug:** In `DYNSolverIVP` wrapper (line 1515), `xf` is initialized from `in_u0` using `xf[12] = in_u0[i-12]`. This is correct for the 15D state vector layout (`p[3], R[9], u[3]`), but should be verified against the `CRM_StateVector_Definitions.hpp`.

## 3. Recommendations

1.  **Generalize `get_state6`:** Rename to `get_output_state` and use `3 + 3 * num_sets` sizing to support multi-actuator FD baselines.
2.  **Standardize Step Output:** Update `step_from_seed` and `step` to return a `coil_velocities` array of shape `(num_sets, 3)` instead of just a 3D `tip_velocity`.
3.  **Clean up `eval_output`:** Either remove the hardcoded lambda in the implicit solver or update it to match the dynamic sizing of the AD path.
