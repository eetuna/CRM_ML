# Phase 2: AD Implementation Plan - Updated

**Date:** 2025-12-29
**Status:** Investigation Complete - Implementation Strategy Revised

---

## What We Learned

### Initial Hypothesis (WRONG)
- Thought we could just remove the `gx * dxdth` term
- **Result:** Made error WORSE (67.9% → 387.1%)

### Root Cause (CORRECT)
The problem is **architectural mismatch**:

```
Forward Pass (step_from_seed):
  Input: (currents, seed_state)
  ↓
  Explicit Integration (IVP)
  ↓
  Output: tip_position, tip_velocity

Backward Pass (linearize_implicit):
  Input: (currents, seed_state)
  ↓
  BVP Solve → Get x* = [m_L*, n_L*]
  ↓
  eval_output_AD_with_params(x*, currents, seed)
  ↓
  Chain rule: dy/dθ = ∂y/∂θ|x + ∂y/∂x * dx/dθ
  ↓
  Output: gradients

The BVP solve for x* is a DIFFERENT operation than explicit IVP!
```

### Why `eval_output_AD_with_params` Uses x

Looking at the function (line 1318 of autodiff header):
1. It unpacks `x_scaled` to get `m_L_all`, `n_L_all` (line 1334-1340)
2. Uses them to compute `net_mL = m_L_all[0] - tau` (line 1442)
3. Passes `net_mL` to `CoilDynamicsDispatch` (line 1451)

So **x (m_L, n_L) is required** - we can't just ignore it.

### The Chain Rule IS Needed

`dy/dθ = ∂y/∂θ|x + ∂y/∂x * dx/dθ` is mathematically correct.

The problem is `dx/dθ` is computed from:
```cpp
// Line 2702: Compute Jxx = ∂F/∂x where F is BVP residual
Jxx = DYNNLEquationJacobianEigenAD(x_star_scaled, ...)

// Line 2862: Solve for dx/dθ
dx/dθ = -Jxx^{-1} * Jxθ
```

This gives dx/dθ for the BVP equilibrium, NOT for the explicit IVP!

---

## The Correct Solution

We need to **bypass the BVP entirely** and compute gradients through the actual forward pass.

### Option 2A-Revised: Direct Forward-Pass AD

**Don't use `linearize_full_seed_action_from_seed_implicit` at all!**

Instead, create a new function that:
1. Takes (currents, seed) as AD variables
2. Does explicit forward integration (same as `step_from_seed`)
3. Returns tip position/velocity
4. Let autodiff compute all gradients automatically

**Implementation:**
```cpp
// New function in crm_bindings.cpp
py::dict linearize_full_seed_action_from_seed_FORWARD_AD(
    currents, insertion_length, v, w, p, R, xf, mL, nL
) {
    // 1. Forward pass (same as step_from_seed)
    py::dict base = step_from_seed(...);

    // 2. Use autodiff to compute gradients through forward pass
    //    (This requires making step_from_seed AD-compatible)

    // 3. Return B and A matrices
}
```

**Challenge:** `step_from_seed` calls the legacy BVP solver which is NOT AD-compatible.

### Option 2B: Make Forward Pass Use Same BVP

Make `step_from_seed` also solve the BVP instead of explicit integration.

**Problem:** This changes what the forward pass computes! Users expect IVP, not BVP.

### Option 2C: Compute dx/dθ Differently

Keep the current structure but compute `dx/dθ` by differentiating through the explicit forward pass, not the BVP residual.

**How:**
1. Forward pass: `x* = explicit_solve(currents, seed)` (current step_from_seed)
2. Compute `dx/dθ` via finite differences on the forward pass
3. Use that in the chain rule with `gth` and `gx`

**Pseudo-code:**
```cpp
// Compute dx/dθ via FD on forward pass
for each parameter θ_i:
    x_plus = extract_x_from(step_from_seed(θ + ε))
    x_minus = extract_x_from(step_from_seed(θ - ε))
    dx/dθ_i = (x_plus - x_minus) / (2ε)

// Then use chain rule
dy/dθ = gth + gx * (dx/dθ)  // Now dx/dθ is from forward pass!
```

---

## Recommended Next Steps

### Immediate: Option 2C (Hybrid Approach)

**Rationale:**
- Minimal code changes
- Keeps existing AD infrastructure
- Just fixes the `dx/dθ` computation
- Compatible with current forward pass

**Implementation Plan:**

1. **Extract x from forward pass**
   ```cpp
   // After line 2292: base = step_from_seed(...)
   // Extract mL*, nL* from base dict
   x_star_from_forward = extract_x(base);
   ```

2. **Compute dx/dθ via FD on forward pass**
   ```cpp
   // Instead of line 2702-2862 (BVP-based dx/dθ)
   // Compute via FD on step_from_seed:
   for (int j = 0; j < theta_dim; j++) {
       θ_plus = θ + ε
       θ_minus = θ - ε
       base_plus = step_from_seed(θ_plus, ...)
       base_minus = step_from_seed(θ_minus, ...)
       x_plus = extract_x(base_plus)
       x_minus = extract_x(base_minus)
       dxdth.col(j) = (x_plus - x_minus) / (2ε)
   }
   ```

3. **Use in chain rule (line 2964 unchanged)**
   ```cpp
   dydth = gth + gx * dxdth;  // Now correct!
   ```

**Files to modify:**
- `crm_ml_rl/wrappers/crm_bindings.cpp` lines 2700-2862

**Expected result:** <1% error vs FD

---

### Long-term: Option 2A (Clean Solution)

Make the entire forward pass AD-compatible:

1. Refactor forward kinematics to use autodiff types
2. Replace BVP solver with AD-compatible version
3. Direct forward-mode AD through entire forward pass

**Effort:** Large (weeks)
**Benefit:** True forward-mode AD, no FD overhead

---

## Current State

**Files modified:**
- `crm_bindings.cpp:2954-2964` - Documented the issue, reverted to original (buggy) code

**Test results:**
- Original code: 67.9% error
- Attempted fix (remove chain rule): 387.1% error (worse!)
- Current: Back to 67.9% error

**Next session should:**
1. Implement Option 2C (FD-based dx/dθ from forward pass)
2. Test and validate
3. Document results

---

## Key Files Reference

- **Test:** `test_ad_forward_backward_mismatch.py`
- **Implementation:** `crm_ml_rl/wrappers/crm_bindings.cpp:2197-3000`
- **AD functions:** `src/CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp`
  - Line 1318: `eval_output_AD_with_params`
  - Line 1491: `DYNNLEquationOutputJacobianEigenAD`
  - Line 2080: `DYNNLEquationJacobianEigenAD` (BVP residual - WRONG for our purpose)

---

**Status:** Ready for Option 2C implementation
