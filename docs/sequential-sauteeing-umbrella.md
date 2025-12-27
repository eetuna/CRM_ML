# Comprehensive Gradient Fix Plan: Systematic Audit and Resolution

**Date:** 2025-12-27
**Objective:** Fix NaN/infinite/incorrect gradients in Option C C++ extension through systematic audit

---

## Problem Summary

The Option C C++ extension has persistent gradient computation failures:
- **Autodiff returns NaN** for J_xx and J_xu Jacobians
- **Finite difference fallback returns zeros** (Fp == Fm, both ~2.4e6)
- **Python wrapper works correctly** (have_ad_jxx = True, B norm ~187)
- Both use same `CRMCPPLib.a` library

---

## Root Cause Hypotheses

1. **DYNNLEParams initialization differs** between C++ extension and Python wrapper
2. **Residual evaluated at wrong point** - huge residual (~2.4e6) instead of near-zero at solution
3. **Scaling mismatch** - Python uses 1e-2/1e-1, header defines 10000.0
4. **IVP_Prep not called correctly** for each FD perturbation

---

## Phase 1: Documentation Audit (Complete)

Key findings from docs:
- Forward pass works perfectly (0.00e+00 error)
- BVP re-solve in backward converges (localmin=0, mL_star/nL_star non-zero)
- Autodiff fails silently (returns NaN, no exception)
- FD fallback: Fp and Fm are identical despite different x values

---

## Phase 2: Code Audit

### Task 2.1: Compare Python vs C++ DYNNLEParams Setup
**Files:**
- C++: `crm_torch_ext/csrc/crm_step_op.cpp:497-558`
- Python: `crm_ml_rl/wrappers/crm_bindings.cpp:2458-2558`

**Check:**
- `CRMDYNSolverIVP_Prep()` call parameters
- `DYNNLEParams` field initialization
- `xf` array population

### Task 2.2: Compare Autodiff Call Sites
**Files:**
- C++: `crm_step_op.cpp:583-619` (J_xx), `crm_step_op.cpp:647-695` (J_xu)
- Python: `crm_bindings.cpp:2700-2730` (J_xx), `crm_bindings.cpp:2816-2830` (J_xu)

**Check:**
- Input vector format and scaling
- DYNNLEParams state at call time
- Error handling

### Task 2.3: Compare FD Fallback Implementation
**Files:**
- C++: `crm_step_op.cpp:622-679`
- Python: `crm_bindings.cpp:2712-2727`

**Check:**
- How `eval_residual` is implemented
- Whether DYNNLEParams is rebuilt for each perturbation
- Epsilon values used

### Task 2.4: Verify Residual Function Behavior
**Test:** Call `DYNNLEquation` directly at x_star and verify residual is near-zero

---

## Phase 3: Root Cause Identification

### Task 3.1: Instrument Python Wrapper
Add debug output to capture:
- `x_star_scaled` values
- `DYNNLEParams` key fields
- Residual at x_star (should be ~0)
- J_xx first row values

### Task 3.2: Create Minimal C++ Test
Test autodiff in isolation:
```cpp
// Call DYNNLEquationJacobianEigenAD with known-good params from Python
// Verify it returns finite values
```

### Task 3.3: Test Residual Sensitivity
```cpp
// Verify F(x+eps) != F(x-eps) for small eps
// If equal, the residual is not responding to x changes
```

---

## Phase 4: Fix Implementation

### Option A: Fix Autodiff (Preferred)
1. Match DYNNLEParams initialization exactly to Python
2. Ensure IVP_Prep is called with correct parameters
3. Verify scaling is consistent

### Option B: Implement Correct FD Fallback
If autodiff cannot be fixed:
1. Copy Python's `eval_residual` logic exactly
2. Rebuild BVPParams for each perturbation (not just DYNNLEParams)
3. Use same epsilon values as Python (1e-5)

### Option C: Hybrid Approach
1. Use Python wrapper's linearization function directly via pybind
2. Call existing working code instead of reimplementing

---

## Phase 5: Validation

### Task 5.1: Gradient Parity Test
```bash
python3 crm_torch_ext/test/test_gradient_parity.py
```
**Target:** `||B_cpp - B_python|| < 1e-6`

### Task 5.2: Finite Difference Cross-Check
**Target:** Relative error < 1% vs numerical gradient

### Task 5.3: Regression Test
**Target:** Forward pass still 0.00e+00 error

---

## Phase 6: Documentation

- Update `OPTION_C_CORRECTED_STATUS.md`
- Update `GRADIENT_DEBUG_SESSION_HANDOFF.md`
- Create completion report with root cause and fix details

---

## Critical Files to Modify

1. `crm_torch_ext/csrc/crm_step_op.cpp` - Main fix location
2. `crm_torch_ext/csrc/bindings.cpp` - May need debug output removal

## Reference Files (Read-Only)

1. `crm_ml_rl/wrappers/crm_bindings.cpp` - Working Python implementation
2. `src/CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp` - Autodiff functions

---

## Success Criteria

| Criterion | Target |
|-----------|--------|
| Gradient accuracy vs Python | < 1e-6 |
| FD cross-check | < 1% relative error |
| No NaN/Inf | 0 occurrences |
| Forward pass unchanged | 0.00e+00 error |

---

## Immediate Next Steps

1. Run Python wrapper with debug to get known-good intermediate values
2. Create minimal C++ test calling autodiff directly
3. Compare DYNNLEParams field-by-field
4. Identify first point of divergence
