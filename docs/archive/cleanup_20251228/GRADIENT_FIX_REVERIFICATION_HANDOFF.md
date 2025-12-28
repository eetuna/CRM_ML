# Gradient Fix Reverification Handoff

**Date:** 2025-12-27
**Status:** Phase 2/3 - **NEEDS RE-VERIFICATION**
**Next Session Goal:** Double-check Phase 2 and Phase 3 findings before proceeding to Phase 4.

---

## Session Summary

In this session, we attempted to fix the gradient computation in the C++ extension (`crm_torch_ext`).

### Key Actions Taken
1.  **Build Fix:** Enabled `-fPIC` in `CMakeLists.txt` to allow linking `CRMCPPLib` into the Python extension.
2.  **Scaling Fix:** Identified a 100x scaling mismatch in `crm_step_op.cpp`. Changed hardcoded `1e-2`/`1e-1` to `IVALUE_SCALE_M` (10000.0) / `IVALUE_SCALE_N` (10000.0).
    *   *Result:* BVP solver now converges in the backward pass (`localmin: 0`).
    *   *Result:* Gradient norm changed from ~1e-3 (essentially zero) to ~188 (correct magnitude).
3.  **Missing Term Identification:** Identified that the C++ implementation calculates $B = g_x \]cdot \frac{dx}{du}$ but misses the direct sensitivity term $g_\theta = \frac{\partial y}{\partial u}$.
    *   *Symptom:* The first 3 columns of $B$ (gradients w.r.t. currents) differ significantly from Python, while column 4 (w.r.t. insertion length) matches ($0$).

### Current State
- **Forward Pass:** Correct.
- **Backward Pass:** BVP converges.
- **Gradients:** Finite (no NaN), correct magnitude, but numerically incorrect values.
- **Code:** Modified `crm_step_op.cpp` (scaling fix applied, FD epsilon reduced).

---

## Plan to Move Forward (Strict Protocol)

**Objective:** Rigorously re-verify Phase 2 and Phase 3 of `GRADIENT_FIX_COMPREHENSIVE_PLAN.md` to ensure the findings are correct and complete. **Do not assume the current state is correct.**

### Step 1: Re-Execute Phase 2 (Code Audit)
*Refer to `docs/GRADIENT_FIX_COMPREHENSIVE_PLAN.md`*

1.  **Audit DYNNLEParams (Task 2.1):**
    *   Systematically compare `crm_step_op.cpp` params setup vs `crm_bindings.cpp`.
    *   *Verification Question:* Are we absolutely sure `DYNNLEParams` are identical?
2.  **Audit Autodiff Call Sites (Task 2.2):**
    *   Verify input vector scaling/unscaling logic.
    *   *Verification Question:* Is the `x_star_scaled` passed to autodiff exactly `mL / 10000`?
3.  **Audit FD Fallback (Task 2.3):**
    *   Check if `eval_residual` in C++ correctly rebuilds the full parameter set for every perturbation.

### Step 2: Re-Execute Phase 3 (Root Cause Identification)
1.  **Verify Scaling Fix:**
    *   Prove that the 100x scaling change was necessary and correct.
2.  **Verify Missing Term ($g_\theta$):**
    *   Confirm mathematically and via code inspection that the Python wrapper includes `gth` (theta Jacobian) and the C++ extension does not.
    *   This is the primary suspect for the remaining value mismatch.

### Step 3: STOP
*   **Do not proceed to Phase 4 (Implementation)** until Phase 2 and 3 are re-verified and documented.
*   Update the status document.

---

## Critical Files
- `crm_torch_ext/csrc/crm_step_op.cpp`: Current implementation (contains scaling fix).
- `crm_ml_rl/wrappers/crm_bindings.cpp`: Reference implementation (Golden Standard).
- `docs/GRADIENT_FIX_COMPREHENSIVE_PLAN.md`: The master plan.
