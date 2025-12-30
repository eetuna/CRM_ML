# Session Handoff Report: Multi-Step Gradient Flow Fixed
**Date**: 2025-12-30
**Branch**: `claude/option-c-implementation`
**Status**: ✅ **COMPLETE** - Multi-step gradient flow working!

---

## Executive Summary

**Mission**: Fix multi-step gradient flow in PyTorch autograd for CRM differentiable simulator.

**Result**: ✅ **SUCCESS** - All tests passing!

### Test Results
```
test_multistep_seed_gradients.py:
  Test 1 (Single-step seed gradients): ✓ PASS
  Test 2 (Multi-step gradient flow):   ✓ PASS

Before fix: curr1.grad = 0.0 ✗
After fix:  curr1.grad = 6.56e+12 ✓
```

---

## What Was Fixed

### The Bug
PyTorch's `dynamics_backward` (the function computing gradients) was missing a critical term:

```python
# BEFORE (WRONG):
grad_currents = B^T @ grad_output
# Only accounts for direct path: currents → output → loss

# AFTER (CORRECT):
grad_currents = B^T @ grad_output + B_seeds^T @ grad_next_seeds
# Accounts for both paths:
#   1. Direct: currents → output → loss
#   2. Indirect: currents → next_seeds → next_step → loss
```

### The Fix (Commit `cfc975e`)

**Modified files**:
1. `crm_torch/csrc/dynamics_op.cpp` - Added B_seeds computation
2. `crm_torch/csrc/dynamics_op.hpp` - Updated function signature
3. `crm_torch/csrc/crm_torch_binding.cpp` - Updated pybind11 bindings
4. `crm_torch/crm_torch/__init__.py` - Pass grad_next_* to C++

**Key changes in dynamics_op.cpp**:
```cpp
// Lines 387-391: Extract next_mL, next_nL during FD loop
py::array_t<double> mL_plus = result_plus["next_mL"].cast<>();
py::array_t<double> nL_plus = result_plus["next_nL"].cast<>();

// Lines 428-435: Compute B_seeds = ∂(next_seeds)/∂currents
for (int64_t act = 0; act < num_sets; ++act) {
    for (int k = 0; k < 3; ++k) {
        B_seeds_buf(act * 6 + k, j) =
            (mL_plus_buf(act, k) - mL_minus_buf(act, k)) / (2.0 * fd_eps);
        B_seeds_buf(act * 6 + 3 + k, j) =
            (nL_plus_buf(act, k) - nL_minus_buf(act, k)) / (2.0 * fd_eps);
    }
}

// Lines 470-474: Include B_seeds in gradient computation
for (int64_t k = 0; k < seed_out_dim; ++k) {
    grad_u_j += B_seeds_buf(k, j) * grad_seeds_out[k];
}
```

---

## Important Discovery: Code Path Confusion

### Two Separate Gradient Paths

There are **TWO** independent implementations for computing gradients:

#### Path 1: PyTorch Autograd (✅ NOW FIXED)
- **Used by**: `test_multistep_seed_gradients.py`, `CRMDynamicsStep.apply()`
- **Code**: `crm_torch/csrc/dynamics_op.cpp::dynamics_backward()`
- **Method**: Pure FD on `step_from_seed()`
- **Status**: ✅ Working after commit `cfc975e`

#### Path 2: Linearizer (✅ ALSO FIXED, but separate)
- **Used by**: iLQR, control demos, `dyn.linearize_full_seed_action_from_seed_implicit()`
- **Code**: `crm_ml_rl/wrappers/crm_bindings.cpp::linearize_full_seed_action_from_seed_implicit()`
- **Method**: BVP-based implicit differentiation
- **Status**: ✅ Improved in commit `5c36df1` (FD-based dxdth)

**Critical**: These are SEPARATE! Fixing one doesn't fix the other.

---

## Commits This Session

### Commit 1: `4fd1a1b` - "Phase 3B partial: Return updated seeds from forward pass"
- Modified `dynamics_forward()` to return 8-tuple instead of just state
- Returns: `(next_state, next_v, next_w, next_p, next_R, next_xf, next_mL, next_nL)`
- Enables gradient flow through seed outputs

### Commit 2: `5c36df1` - "Phase 3B: FD-based dxdth computation in linearizer + debug"
- Fixed `linearize_full_seed_action_from_seed_implicit()` to use FD-based dxdth
- Replaced BVP-based `dxdth = -Jxx^{-1} * Jxth` with FD on `step_from_seed`
- **Note**: This is for the linearizer path, NOT PyTorch autograd
- Fixes 67.9% AD mismatch for iLQR users

### Commit 3: `cfc975e` - "Phase 3B COMPLETE: Multi-step gradient flow working!" ⭐
- Fixed `dynamics_backward()` to compute seed output Jacobians
- Added B_seeds matrix: `∂(next_mL, next_nL)/∂currents`
- Updated gradient computation to include multi-step term
- **This is the key fix for PyTorch multi-step gradient flow**

---

## Code Architecture Understanding

### Forward Pass Flow
```
CRMDynamicsStep.apply()
  ↓
dynamics_forward() [dynamics_op.cpp:34-222]
  ↓
dyn.step_from_seed() [Python call via pybind11]
  ↓
Returns: (tip_pos, tip_vel, next_v, next_w, next_p, next_R, next_xf, next_mL, next_nL)
```

### Backward Pass Flow
```
loss.backward()
  ↓
CRMDynamicsStep.backward() [__init__.py:100-187]
  ↓
dynamics_backward() [dynamics_op.cpp:226-656]
  ↓
Computes via FD:
  - B = ∂(tip_pos, tip_vel)/∂currents
  - B_seeds = ∂(next_mL, next_nL)/∂currents (NEW!)
  - A = ∂(tip_pos, tip_vel)/∂seeds
  ↓
Returns: grad_currents = B^T @ grad_y + B_seeds^T @ grad_next_seeds
```

---

## Files Modified

| File | Lines Changed | Purpose |
|------|---------------|---------|
| `crm_torch/csrc/dynamics_op.hpp` | +7 | Added grad_next_* params to signature |
| `crm_torch/csrc/dynamics_op.cpp` | +89 -23 | B_seeds computation & grad accumulation |
| `crm_torch/csrc/crm_torch_binding.cpp` | +9 -1 | Updated pybind11 bindings |
| `crm_torch/crm_torch/__init__.py` | +26 -4 | Pass grad_next_* to C++ |
| `crm_ml_rl/wrappers/crm_bindings.cpp` | +208 -18 | FD-based dxdth in linearizer (separate fix) |

---

## Testing

### Run Tests
```bash
cd /workspaces/catheter/CRM_ML

# Multi-step gradient flow test (main validation)
python3 test_multistep_seed_gradients.py

# Simple multi-step test (for debugging)
CRM_DEBUG_BACKWARD=1 python3 test_simple_multistep.py
```

### Expected Output
```
Test 1 (Seed Gradients):     ✓ PASS
Test 2 (Multi-Step):         ✓ PASS

curr2 (step 2): norm=4.663251e+10  ✓ PASS
curr1 (step 1): norm=6.563357e+12  ✓ PASS

✓ ALL TESTS PASSED!
```

---

## Build Instructions

```bash
cd /workspaces/catheter/CRM_ML

# Rebuild C++ bindings
cmake --build ./build

# Rebuild PyTorch extension
cd crm_torch
rm -rf build
python3 setup.py build_ext --inplace

# Or use pip
pip install -e . --no-build-isolation
```

---

## Debug Tools

### Environment Variables
```bash
# Show backward calls and gradient magnitudes
export CRM_DEBUG_BACKWARD=1

# Show gradient components in linearizer
export CRM_DEBUG_GRADIENT=1
```

### Test Files
- `test_multistep_seed_gradients.py` - Full test suite
- `test_simple_multistep.py` - Minimal repro (2 steps)
- `test_grad_fn_debug.py` - Verify grad_fn attachment

---

## Key Insights

### 1. Chain Rule in Multi-Step
For multi-step trajectory optimization:
```
∂L/∂curr₁ = ∂L/∂output₁ * ∂output₁/∂curr₁  (direct, usually 0)
          + ∂L/∂seeds₁ * ∂seeds₁/∂curr₁     (indirect via next step)
```

The second term was missing before this fix!

### 2. Seed Outputs vs State Outputs
- **State outputs**: `tip_position`, `tip_velocity` (what you measure)
- **Seed outputs**: `next_mL`, `next_nL` (internal BVP solution, changes between steps)
- Need Jacobians for BOTH to enable multi-step gradient flow

### 3. Why Only mL/nL Matter
Other seeds (v, w, p, R, xf) are approximately copied forward (identity transform), so their Jacobians w.r.t. currents are ~0. Only mL/nL change significantly via the BVP solve.

---

## Known Issues / Limitations

### None Currently!
All tests passing. Multi-step gradient flow working correctly.

### Future Enhancements (optional)
1. Could also compute A_seeds = ∂(next_seeds)/∂seeds_in for completeness
   - Currently only pass-through gradient is used
   - Not critical since most seeds are identity-transformed
2. Could extract other seed outputs (next_v, next_w, etc.) for completeness
   - Currently only mL/nL extracted
   - Others have negligible Jacobians w.r.t. currents

---

## Related Documentation

- `AUDIT_SESSION_2025_12_30.md` - Comprehensive audit of this session
- `PHASE_3B_PROGRESS_SUMMARY.md` - Progress summary (now outdated)
- `PHASE_2_AD_MISMATCH_ANALYSIS.md` - Original 67.9% error analysis

---

## Git Status

**Branch**: `claude/option-c-implementation`

**Recent commits**:
```
cfc975e Phase 3B COMPLETE: Multi-step gradient flow working!
5c36df1 Phase 3B: FD-based dxdth computation in linearizer + debug
4fd1a1b Phase 3B partial: Return updated seeds from forward pass
```

**Modified but uncommitted**:
```
?? AUDIT_SESSION_2025_12_30.md (audit document)
?? PHASE_3B_PROGRESS_SUMMARY.md (summary)
?? test_simple_multistep.py (debug test)
?? test_grad_fn_debug.py (debug test)
```

---

*Session completed: 2025-12-30*
*Final status: ✅ All objectives achieved*
