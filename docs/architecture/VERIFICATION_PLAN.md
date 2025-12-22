# Verification Plan: Core C++ Refactor & AutoDiff

**Date:** December 22, 2025
**Branch:** `refactor/core-cpp-stabilization`

## 1. Goal
Definitively verify that the modernized C++ core (using Eigen/std::vector) correctly implements the Cosserat Rod dynamics and that the AutoDiff engine accurately computes parameter gradients.

## 2. Diagnosis of Current Failures
The recent test failures (`AssertionError: AutoDiff angular damping gradients are all zero`) are caused by a logical flaw in the test setup, not necessarily the C++ code:
*   **Static Initialization:** The test initializes the catheter from kinematics, resulting in $v=0, w=0$.
*   **Physics Consequence:** Damping force $F_d = -d \cdot v$. If $v=0$, then $\frac{\partial F}{\partial d} = 0$.
*   **False Negative:** The AutoDiff engine correctly reports 0 gradient, but the test assumes this is an error.

## 3. Execution Steps

### Step 1: Fix Test Logic (The "Wind Tunnel")
*   **Action:** Modify `tests/test_parameter_jacobian_autodiff.py`.
*   **Detail:** In `test_parameter_jacobian_vs_finite_difference`, explicitly inject non-zero velocity ($v \neq 0, w \neq 0$) into the state vector before calling both `compute_parameter_jacobian` (AutoDiff) and `compute_residual_at_state` (Finite Difference).
*   **Expected Result:** Damping gradients should become non-zero.

### Step 2: Verify Residual Consistency
*   **Action:** Run the test and compare the "base residual" returned by C++ vs the residual expected by the Finite Difference loop.
*   **Check:** If `residual_ad != residual_fd_base`, then the AutoDiff residual equation (`DYNNLEquationResidualEigenAD`) does not match the core physics equation (`CRMIntegrand`).

### Step 3: Gradient Validation
*   **Action:** Compare Analytical Jacobian ($J_{AD}$) vs Finite Difference Jacobian ($J_{FD}$).
*   **Criteria:** Relative error $< 1e-4$.

### Step 4: Final Cleanup
*   **Action:** Remove `[DEBUG CPP]` prints from headers.
*   **Action:** Commit changes.

## 4. Fallback Strategy
If Step 3 fails (gradients mismatch):
1.  Isolate the mismatch to specific parameters (e.g., is it just damping? or stiffness too?).
2.  Audit `src/CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp` line-by-line against `src/CoilDynamics_Defs.cpp`.
