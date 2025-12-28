# Gradient Fix Phase 3 Verification Report

**Date:** 2025-12-27
**Status:** Verification Complete
**Outcome:** Root Cause Identified (New Finding)

---

## 1. Scaling Fix Verification
**Status:** ✅ **Verified Correct**
- `crm_step_op.cpp` uses `IVALUE_SCALE_M` / `IVALUE_SCALE_N` (10000.0).
- `crm_bindings.cpp` (Python wrapper) uses the same constants.
- The previous issue of hardcoded `1e-2` is resolved.

## 2. Missing Term ($g_theta$) Verification
**Status:** ❌ **Hypothesis Incorrect (Term is Present)**
- **Finding:** The C++ implementation in `crm_step_op.cpp` **does** include the direct sensitivity term `gth` (theta Jacobian).
- **Code:** `B.leftCols(3) = gth.leftCols(3) + gx * dx_du.leftCols(3);`
- **Verification:** `examples/verify_output_jacobian_gth.py` confirms `gth` is computed, non-zero, and has correct shape/magnitude via AD.
- **Conclusion:** The discrepancy is NOT due to a missing `gth` term.

## 3. New Root Cause Identification
**Status:** 🔍 **Identified: Seed State Mismatch in IVP_Prep**

**The Issue:**
There is a subtle difference in how `DYNNLEParams` is initialized for the Jacobian computation between C++ and Python.

*   **Python Wrapper (`crm_bindings.cpp`):**
    *   Uses the **raw** `mL_seed` (from `seed` state) to call `IVP_Prep` inside `linearize_full_seed_action_from_seed_implicit`.
    *   Does *not* apply damping compensation to this seed before linearization.

*   **C++ Extension (`crm_step_op.cpp`):**
    *   In `crm_step_backward`, the `mL_guess` array is populated from `seed_mL` input.
    *   **Damping compensation is applied in-place** to `mL_guess` (to help the BVP solver).
    *   This **modified (compensated)** `mL_guess` is then passed to `compute_implicit_jacobians`.
    *   `compute_implicit_jacobians` uses this modified seed for `IVP_Prep`.

**Impact:**
- `IVP_Prep` sets up the `DYNNLEParams` (system structure).
- Different inputs to `IVP_Prep` -> Different `DYNNLEParams`.
- Different `DYNNLEParams` -> Different `J_xx` (Jacobian of residual).
- **Observed Difference:** `J_xx` norm differs by ~2% between C++ and Python.
- **Result:** $B = g_theta + g_x \cdot (-J_{xx}^{-1} J_{xu})$.
- Since $g_theta$ (large, ~867) and $g_x \frac{dx}{du}$ (large, opposite sign) cancel out significantly, the 2% error in $J_{xx}$ leads to a massive error in the final $B$ matrix.

**Secondary Finding:**
- C++ uses FD epsilon `1e-8` (hardcoded).
- Python uses FD epsilon `1e-5` (default).
- While AD is working now (rendering FD epsilon less critical for `J_xx`), consistency is preferred.

---

## 4. Next Steps (Phase 4 Implementation)

1.  **Fix `crm_step_op.cpp`:**
    *   In `crm_step_backward`, make a copy of `mL_guess` *before* applying damping compensation.
    *   Pass the *original* (uncompensated) seed to `compute_implicit_jacobians`.
    *   Use the *compensated* seed for `DynamicsBVP`.
2.  **Align Epsilon:**
    *   Update FD epsilon in `crm_step_op.cpp` to `1e-5` to match Python wrapper.
3.  **Verify:**
    *   Run `test_backward_debug.py` and verify `B` matrix matches Python.
