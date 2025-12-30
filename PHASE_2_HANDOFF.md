# Phase 2: True AD Implementation - Handoff Document

**Date:** 2025-12-29
**Session Status:** Investigation Complete - Ready for Implementation
**Context Used:** 135k/200k tokens (68%)

---

## 🎯 Quick Start for Next Session

```
Continue Phase 2: Implement true automatic differentiation for the implicit linearization method.

CURRENT STATUS:
- Phase 1 COMPLETE: FD-based gradients work with 0% error (both Options A and C)
- Phase 2 IN PROGRESS: True AD has 67.9% gradient error - root cause identified

ROOT CAUSE CONFIRMED:
The linearize_full_seed_action_from_seed_implicit() function computes gradients using:
  dy/dθ = gth + gx * dxdth

Where:
- gth = ∂y/∂θ|ₓ (partial derivative, x fixed) ✓ CORRECT
- gx = ∂y/∂x ✓ CORRECT
- dxdth = dx/dθ from BVP residual ✗ WRONG

The bug: dxdth is computed from BVP equilibrium (backward integration), but the actual
forward pass uses IVP (forward integration). These are DIFFERENT operations.

VERIFIED FACTS:
1. gth is a PARTIAL derivative (∂y/∂θ|ₓ), NOT total derivative
   - Proof: Line 1538 of autodiff header captures x_ad as constant
2. Chain rule IS mathematically necessary
3. Forward pass uses DYNSolverIVP (IVP solver), NOT BVP solver
   - Confirmed by grep on crm_bindings.cpp
4. Current dxdth from line 2862 solves: Jxx · dx/dθ = -Jxθ where J is BVP residual Jacobian

SOLUTION: Option 2C (Hybrid FD+AD)
- Compute dx/dθ by finite-differencing the actual forward pass (IVP)
- Use that in chain rule with existing gth and gx (which are correct)
- Minimal code changes (~100 lines in crm_bindings.cpp)

FILES TO MODIFY:
- crm_ml_rl/wrappers/crm_bindings.cpp lines 2700-2862

TEST:
- python3 test_ad_forward_backward_mismatch.py
- Should show <1% error (currently shows 67.9% error)

See PHASE_2_IMPLEMENTATION_PLAN.md for detailed implementation steps.
```

---

## 📊 Investigation Summary

### What Was Attempted

1. **Initial hypothesis**: Remove chain rule term (just use gth)
   - **Result**: FAILED - Error increased from 67.9% → 387.1%
   - **Learning**: Chain rule is mathematically necessary

2. **Rigorous analysis**: Traced through autodiff code execution
   - **Finding**: gth is ∂y/∂θ|ₓ (partial), not total derivative
   - **Proof**: Line 1538 captures x_ad by reference (constant)
   - **Conclusion**: Chain rule dy/dθ = ∂y/∂θ|ₓ + ∂y/∂x · dx/dθ is required

3. **Forward pass analysis**: Checked what step_from_seed actually does
   - **Finding**: Calls DYNSolverIVP (Initial Value Problem solver)
   - **Not**: BVP solver
   - **Implication**: dx/dθ from BVP residual is WRONG for this forward pass

### Critical Code Locations

**Forward pass (IVP):**
```cpp
// crm_bindings.cpp:2292
py::dict base = step_from_seed(currents, insertion_length, v_in, w_in, ...);
  ↓
// Calls DYNSolverIVP (forward integration)
// Returns: tip_position, tip_velocity, next_mL, next_nL
```

**Current (buggy) backward pass:**
```cpp
// crm_bindings.cpp:2702
Eigen::MatrixXd Jxx = DYNNLEquationJacobianEigenAD(...);  // BVP residual Jacobian

// crm_bindings.cpp:2862
Eigen::MatrixXd dxdth = qr.solve(-Jxth);  // dx/dθ from BVP

// crm_bindings.cpp:2950
DYNNLEquationOutputJacobianEigenAD(..., gx, gth);  // ∂y/∂x and ∂y/∂θ|ₓ

// crm_bindings.cpp:2964
const Eigen::MatrixXd dydth = gth + gx * dxdth;  // WRONG dxdth!
```

**Why it's wrong:**
- `dxdth` assumes: ∂F/∂x · dx/dθ + ∂F/∂θ = 0 where F is BVP residual
- Forward pass doesn't satisfy F(x,θ) = 0
- Forward pass computes x via explicit IVP integration
- Different x(θ) relationship → wrong dx/dθ → wrong gradients

---

## ✅ Correct Solution: Option 2C

### High-Level Strategy

1. **Keep** the chain rule: dy/dθ = gth + gx · dx/dθ
2. **Keep** gth and gx from output Jacobian (correct)
3. **Replace** dxdth computation:
   - OLD: From BVP residual (lines 2700-2862)
   - NEW: From finite differences on forward pass

### Implementation Pseudocode

```cpp
// After line 2292: base = step_from_seed(...)
// Extract x_star from forward pass
auto next_mL = base["next_mL"].cast<py::array_t<double>>();
auto next_nL = base["next_nL"].cast<py::array_t<double>>();
// ... unpack to x_star_scaled

// NEW: Compute dx/dθ via FD on forward pass
Eigen::MatrixXd dxdth_correct(x_dim, theta_dim);
const double eps = 1e-5;

for (int j = 0; j < theta_dim; j++) {
    // Perturb theta[j]
    Eigen::VectorXd theta_plus = theta0;
    Eigen::VectorXd theta_minus = theta0;
    theta_plus(j) += eps;
    theta_minus(j) -= eps;

    // Run forward pass with perturbed theta
    py::dict base_plus = step_from_seed(theta_plus_currents, ..., theta_plus_seed, ...);
    py::dict base_minus = step_from_seed(theta_minus_currents, ..., theta_minus_seed, ...);

    // Extract x from both
    Eigen::VectorXd x_plus = extract_x(base_plus);
    Eigen::VectorXd x_minus = extract_x(base_minus);

    // Central difference
    dxdth_correct.col(j) = (x_plus - x_minus) / (2.0 * eps);
}

// Keep existing code for gth and gx (lines 2864-2952)
// ...

// Use corrected dxdth in chain rule
const Eigen::MatrixXd dydth = gth + gx * dxdth_correct;  // NOW CORRECT!
```

### Expected Outcome

- Test `test_ad_forward_backward_mismatch.py` should show **<1% error**
- Currently shows 67.9% error, should drop to <1%

---

## 📁 Key Files

### Test Files
- **test_ad_forward_backward_mismatch.py** - Diagnostic test (67.9% error currently)
- **validate_option_a_fix.py** - Phase 1 validation (passes with 0% error)

### Documentation
- **PHASE_2_AD_MISMATCH_ANALYSIS.md** - Problem analysis (CORRECTED)
- **PHASE_2_IMPLEMENTATION_PLAN.md** - Detailed implementation guide
- **RK4_ABM4_INVESTIGATION_REPORT.md** - RK4 investigation results

### Code Files
- **crm_ml_rl/wrappers/crm_bindings.cpp** - Main file to modify
  - Line 2197: `linearize_full_seed_action_from_seed_implicit()` function
  - Lines 2700-2862: dx/dθ computation (needs replacement)
  - Line 2964: Chain rule (correct once dxdth is fixed)

- **src/CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp** - AD infrastructure
  - Line 1318: `eval_output_AD_with_params()` - Output function
  - Line 1491: `DYNNLEquationOutputJacobianEigenAD()` - Computes gth, gx
  - Line 1538: Proves gth is partial derivative (x_ad captured as constant)

---

## ⚠️ Lessons Learned

### Oversight #1: "gth contains full gradients"
**Claim**: gth already has everything we need, remove chain rule
**Reality**: gth = ∂y/∂θ|ₓ (partial derivative only)
**Proof**: Line 1538 captures x_ad by reference (constant during differentiation)
**Result**: Removing chain rule made error WORSE (67.9% → 387.1%)

### Verification Method
When analyzing code:
1. **Trace actual execution** - Read the code line by line
2. **Check captured variables** - Lambda captures reveal what's constant vs variable
3. **Test hypothesis** - Implement and measure before concluding
4. **Verify forward pass** - Check actual solver used (grep for "Solver")

---

## 📋 Implementation Checklist

### Phase 2C: Hybrid FD+AD (Recommended)

- [ ] **Step 1**: Extract x_star from forward pass result (line ~2295)
  - [ ] Get next_mL and next_nL from `base` dict
  - [ ] Unpack to x_star_scaled vector
  - [ ] Verify matches existing x_star_scaled

- [ ] **Step 2**: Replace dx/dθ computation (lines 2700-2862)
  - [ ] Remove BVP residual Jacobian code
  - [ ] Implement FD loop over theta dimensions
  - [ ] Handle currents (first 3 dims) and seed (remaining dims) separately
  - [ ] Use eps = 1e-5 for FD
  - [ ] Store result in `dxdth_forward`

- [ ] **Step 3**: Update chain rule to use new dxdth (line ~2964)
  - [ ] Change variable name for clarity: `dxdth_forward`
  - [ ] Verify dimensions match (x_dim × theta_dim)

- [ ] **Step 4**: Test
  - [ ] Rebuild: `cmake --build build -j8`
  - [ ] Run: `python3 test_ad_forward_backward_mismatch.py`
  - [ ] Verify: Error < 1% (target: ~0.1%)

- [ ] **Step 5**: Validate existing tests still pass
  - [ ] Run: `python3 validate_option_a_fix.py`
  - [ ] Run: `pytest tests/test_option_c_integration.py`

---

## 🔗 Context Preservation

**Current state:**
- Phase 1: ✅ Complete (FD works perfectly)
- Phase 2: 🔄 Root cause identified, implementation ready
- Phase 3: ⏸️ Pending

**No breaking changes** - Phase 1 FD-based system works. Phase 2 is enhancement.

**Estimated effort:** 2-3 hours for Option 2C implementation + testing

---

## 🚀 Next Session Prompt

See top of document for copy-paste prompt.

---

**Document Status:** CORRECTED - Previous claims about gth verified and corrected
