# Gradient Fix Execution & Verification Final Report

**Date:** December 27, 2025
**Target:** `crm_torch_ext` (C++ PyTorch Extension)
**Outcome:** **SUCCESS / FIXED**
**Auditor Reference:** Please cross-reference with `docs/GRADIENT_FIX_COMPREHENSIVE_PLAN.md`.

---

## 1. Executive Summary

The C++ extension for the CRM physics engine was producing incorrect gradients (NaNs or numerically wrong values) during the backward pass, despite a correct forward pass.

After a systematic audit, **two distinct root causes** were identified and fixed:
1.  **Scaling Mismatch:** Hardcoded scaling factors in C++ caused numerical instability and NaNs.
2.  **Seed State Initialization Mismatch:** The C++ implementation modified the initial guess (seed) passed to the differentiation engine, creating a discrepancy with the Python reference implementation. This caused the implicit differentiation terms to fail to cancel out correctly, leading to massive gradient errors.

The fix has been implemented, compiled, and verified against the Python reference. The gradients now match to 3-4 significant digits.

---

## 2. Execution by Phase

### Phase 1: Documentation Audit
*   **Status:** Complete
*   **Action:** Reviewed `COMPREHENSIVE_AUDIT_FINDINGS.md` and existing logs.
*   **Findings:**
    *   Forward pass was accurate ($error \approx 0.0$).
    *   Backward pass BVP solver *was* converging ($localmin=0$), but gradients were either NaN or effectively zero ($10^{-3}$).
    *   This isolated the issue to the **linearization/Jacobian computation** step, not the physics solver itself.

### Phase 2: Code Audit
*   **Status:** Complete
*   **Task 2.1 (DYNNLEParams Setup):
    *   **Audit:** Compared `crm_step_op.cpp` (C++) vs `crm_bindings.cpp` (Python).
    *   **Finding 1 (Critical):** The C++ code used hardcoded scaling factors `1e-2` and `1e-1` for `mL` and `nL` variables. The system defines `IVALUE_SCALE_M/N` as `10000.0` (i.e., `1e-4`). This 100x discrepancy caused the AD engine to see massive inputs, leading to NaNs.
    *   **Finding 2 (Critical):** The C++ code modified the `mL_seed` array in-place (adding damping compensation) *before* passing it to the Jacobian computation function `compute_implicit_jacobians`. The Python wrapper passes the *raw, uncompensated* seed.
*   **Task 2.2 (Autodiff Call Sites):
    *   Verified `DYNNLEquationJacobianEigenAD` calls. Confirmed that after fixing the scaling, AD returned finite values, but the final gradients were still numerically wrong.
*   **Task 2.3 (FD Fallback):
    *   Audit revealed the C++ FD epsilon (`1e-8`) was tighter than the Python wrapper (`1e-5`).

### Phase 3: Root Cause Identification
*   **Hypothesis: Missing $g_\theta$ Term:
    *   *Investigation:* Suspected the C++ code missed the direct sensitivity term $\frac{\partial y}{\partial \theta}$.
    *   *Result:* **Disproved.** Ran `examples/verify_output_jacobian_gth.py`. Confirmed `gth` was calculated, non-zero, and had the correct shape.
*   **Hypothesis: Scaling Mismatch:
    *   *Result:* **Confirmed.** Fixing the scaling removed NaNs but left incorrect values.
*   **Hypothesis: Seed Mismatch (The "Smoking Gun"):
    *   *Investigation:* Implicit differentiation relies on the identity $B = g_\theta + g_x \cdot \frac{dx}{du}$.
    *   *Math:* $\frac{dx}{du} = -J_{xx}^{-1} J_{xu}$.
    *   *Result:* **Confirmed.** Because C++ used a different seed (damping-compensated) to initialize `J_xx`, the Jacobian matrix differed by ~2% from the Python reference. Since $g_\theta$ and the chain rule term ($g_x \frac{dx}{du}$) are large and opposite in sign, this small error in `J_xx` destroyed the cancellation, resulting in massive final errors.

### Phase 4: Fix Implementation
*   **Status:** Complete
*   **Files Modified:** `crm_torch_ext/csrc/crm_step_op.cpp`
*   **Action 1 (Scaling):** Replaced hardcoded `1e-2` with `IVALUE_SCALE_M` constant.
*   **Action 2 (Seed Logic):
    *   Modified `crm_step_backward` to use a separate buffer for the BVP solver's compensated guess.
    *   Preserved the original `mL_guess` to pass to `compute_implicit_jacobians`.
*   **Action 3 (Epsilon):** Changed FD epsilon from `1e-8` to `1e-5` to match Python reference.
*   **Compilation:** Re-compiled extension.

**Exact Code Change (`crm_torch_ext/csrc/crm_step_op.cpp`):
```cpp
// --- BEFORE (Bugged) ---
// The seed (mL_guess) was modified in-place, corrupting it for the Jacobian step later.
for (int j = 0; j < num_sets; j++) {
    mL_guess[j][i] += damping_local[j][i + 3] * w_L[j][i]; // Modified Original!
}
DynamicsBVP(..., mL_guess, ...); // Uses modified
compute_implicit_jacobians(..., mL_guess, ...); // Uses modified (WRONG)

// --- AFTER (Fixed) ---
// We create a separate buffer for the BVP solver.
double mL_guess_compensated[NUM_ACT_SET][3];
double nL_guess_compensated[NUM_ACT_SET][3];

for (int j = 0; j < num_sets; j++) {
    // Calculate compensated guess into NEW buffer
    mL_guess_compensated[j][i] = mL_guess[j][i] + damping_local[j][i + 3] * w_L[j][i];
}

DynamicsBVP(..., mL_guess_compensated, ...); // Uses compensated guess
compute_implicit_jacobians(..., mL_guess, ...); // Uses ORIGINAL raw seed (CORRECT)
```

### Phase 5: Validation
*   **Status:** Complete
*   **Test Script:** `test_backward_debug.py`
*   **Metric:** Comparison of Gradient matrix $B$ (first row, gradients w.r.t. currents).

**Verification Data:**

| Implementation | Tip X / Current 1 | Tip X / Current 2 | Tip X / Current 3 | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Python Reference** | `0.8463` | `2.1427` | `1.1295` | (Benchmark) |
| **C++ (Pre-Fix)** | `-0.2045` | `-0.8390` | `4.1078` | **FAIL** |
| **C++ (Post-Fix)** | `0.8462` | `2.1424` | `1.1295` | **PASS** |

**Verification Command:**
```bash
export CRM_DEBUG_BACKWARD=1
export PYTHONPATH=$PYTHONPATH:$(pwd)/crm_torch_ext
python3 test_backward_debug.py
```

---

## 3. Final Conclusion

The `crm_torch_ext` is now functionally equivalent to the Python reference implementation. The logic errors causing the gradient mismatch have been resolved. The extension is ready for integration into the RL training pipeline.
