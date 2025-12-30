# Phase 2: AD Forward/Backward Mismatch Analysis

**Date:** 2025-12-29
**Status:** Investigation Complete - Ready for Implementation

---

## Executive Summary

**Confirmed:** The `linearize_full_seed_action_from_seed_implicit()` method has a **67.9% gradient error** due to a fundamental mathematical mismatch between forward and backward passes.

**Root Cause:**
- **Forward pass:** Computes explicit time-stepping integration (Initial Value Problem)
- **Backward pass:** Differentiates implicit BVP equilibrium residual (Boundary Value Problem)
- These are **mathematically different operations**

---

## Quantitative Evidence

Test results from `test_ad_forward_backward_mismatch.py`:

```
B matrix (AD):                    B matrix (FD reference):
[[ 7.63  7.21 -6.46]              [[  8.91  10.84 -14.09]
 [-0.84 -1.43 35.89]               [ -2.63  -3.99  49.56]
 [ 0.45  0.49 -5.61]]              [  0.59   0.76  -6.64]]

Relative error:
[[14.3%  33.4%  54.2%]
 [67.9%  64.2%  27.6%]
 [23.9%  36.2%  15.5%]]

MAX ERROR: 67.9%
```

This matches the plan's prediction of "50-200% error".

---

## The Mathematical Mismatch Explained

### Forward Pass: Initial Value Problem (IVP)

File: `crm_bindings.cpp:2292`
```cpp
py::dict base = step_from_seed(currents, insertion_length, v_in, w_in, p_in, R_in, xf_in, mL_use, nL_use);
```

**What it does:**
1. Start with **known** actuator state (v, w, p, R) at root
2. Apply currents → compute forces/torques
3. **Forward integrate** through time
4. Return final tip position/velocity

**Mathematical operation:**
`y(t+Δt) = Φ(y(t), u, Δt)`  where Φ is the forward integrator

### Backward Pass: Boundary Value Problem (BVP)

Files:
- `crm_bindings.cpp:2702` calls `DYNNLEquationJacobianEigenAD`
- `CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp:1557` implements residual

**What it does:**
1. Start with **known** tip state (p_t, R_t)
2. **Backward integrate** through flexible segments
3. Compute actuator dynamics
4. Check if result matches **desired** root configuration
5. Residual F(m_L, n_L) = (p_integrated - p_desired)

**Mathematical operation:**
`F(x*) = 0` where x* = [m_L, n_L] are the unknowns

The AD computes:
`∂F/∂θ` where θ = [currents, seed]

### Why They Don't Match

| Aspect | Forward (IVP) | Backward (BVP) |
|--------|--------------|----------------|
| **Problem type** | Initial value | Boundary value |
| **Direction** | Root → Tip | Tip → Root |
| **Known** | Initial actuator state | Final tip state + root target |
| **Unknown** | Final tip position | Internal forces m_L, n_L |
| **Integration** | Explicit stepping | Solve equilibrium |
| **Gradient of** | `∂(tip_position)/∂(currents)` | `∂F/∂θ` where F is residual |

**The gradients are fundamentally different:**
- IVP gradient: How does changing input affect final output?
- BVP gradient: How does changing parameter affect equilibrium condition?

---

## Additional Issue Discovered

The forward pass in the linearization returns suspicious values:
```
Tip position: [-0.547, -0.547, -0.547]
Tip velocity: [-0.547, -0.547, -0.547]
```

All components are identical! This suggests the `linearize_..._implicit` function is also not correctly extracting the forward pass result. The actual forward pass from `step_from_seed` gives:
```
Tip position: [-0.980, 4.137, 94.140]  ← Correct
```

---

## Two Approaches to Fix

### Option 2A: Make Backward Match Forward ✅ RECOMMENDED

**Goal:** Differentiate the actual forward integration, not the BVP residual

**Approach:**
1. Replace residual-based AD with forward integration AD
2. Use `eval_output_AD_with_params()` which already exists!
3. Differentiate through explicit RK4/ABM4 integrator
4. This gives true gradients of the IVP

**Advantages:**
- ✅ Mathematically correct for IVP
- ✅ Code already exists (`eval_output_AD_with_params`)
- ✅ Matches what the forward pass actually computes
- ✅ No need to change forward pass

**Disadvantages:**
- More expensive (AD through full integration)
- Requires integrator to be AD-compatible (already is!)

**Implementation:**
- File: `crm_bindings.cpp:2700-2800`
- Replace `DYNNLEquationJacobianEigenAD` call
- Use `DYNNLEquationOutputJacobianEigenAD` instead (line 2950)
- This already computes output Jacobians via AD!

### Option 2B: Make Forward Match Backward

**Goal:** Use implicit BVP solve in forward pass too

**Approach:**
1. Change forward pass to solve BVP instead of explicit integration
2. Then implicit differentiation is mathematically correct
3. Forward and backward both solve `F(x*) = 0`

**Advantages:**
- Implicit differentiation is elegant
- Potentially more accurate for steady-state

**Disadvantages:**
- ❌ Changes the forward pass completely
- ❌ Breaks compatibility with existing code
- ❌ BVP solve is more expensive
- ❌ Not what users expect (IVP → BVP change)

---

## Recommended Path: Option 2A

**CORRECTION: Previous analysis was WRONG**

The `gth` matrix is **NOT** the full gradient. Rigorous analysis shows:

Looking at line 1538 of `CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp`:
```cpp
auto output_fn_theta = [&ctx_base, &x_ad](const VectorXreal& th_) -> VectorXreal {
    // x_ad is captured by reference (constant during differentiation)
    return dynnl_ad_eigen::eval_output_AD_with_params(x_ad, curr_ad, seed_ad, ctx_ad);
};
```

This means `gth` = **∂output/∂θ|ₓ** (partial derivative with x held FIXED).

**Chain rule IS needed:**
```
dy/dθ = ∂y/∂θ|ₓ + ∂y/∂x · dx/dθ
         ↑         ↑       ↑
        gth        gx     dxdth
```

**Current code structure:**
1. Line 2292: Forward pass via `step_from_seed()` → uses DYNSolverIVP (forward integration)
2. Line 2702: Compute `dxdth` from BVP residual → **WRONG** (BVP ≠ IVP)
3. Line 2950: Compute `gth` and `gx` via output AD → Correct but incomplete
4. Line 2964: Chain rule `dydth = gth + gx * dxdth` → Wrong because dxdth is wrong

**The actual bug:**
- `dxdth` is computed from BVP equilibrium condition F(x,θ)=0
- But forward pass uses IVP (forward integration), not BVP
- These give DIFFERENT dx/dθ values → 67% error

---

## Correct Implementation Plan (Option 2C)

### Step 1: Extract x from Forward Pass

After line 2292, extract the actual mL, nL values that `step_from_seed` computes:
```cpp
py::dict base = step_from_seed(...);
// Extract mL*, nL* from base["next_mL"], base["next_nL"]
x_star_from_forward = extract_internal_state(base);
```

### Step 2: Compute dx/dθ via Finite Differences on Forward Pass

Replace lines 2700-2862 (BVP-based computation) with:
```cpp
// Compute dx/dθ by finite-differencing the forward pass
Eigen::MatrixXd dxdth_forward(x_dim, theta_dim);
double eps = 1e-5;

for (int j = 0; j < theta_dim; j++) {
    // Perturb theta
    theta_plus = theta + eps * e_j
    theta_minus = theta - eps * e_j

    // Run forward pass
    base_plus = step_from_seed(theta_plus, ...)
    base_minus = step_from_seed(theta_minus, ...)

    // Extract x
    x_plus = extract_internal_state(base_plus)
    x_minus = extract_internal_state(base_minus)

    // Central difference
    dxdth_forward.col(j) = (x_plus - x_minus) / (2 * eps)
}
```

### Step 3: Use Correct dxdth in Chain Rule

Keep line 2964 but use the corrected dxdth:
```cpp
const Eigen::MatrixXd dydth = gth + gx * dxdth_forward;  // Now CORRECT!
```

### Step 4: Validate

Run tests:
- `test_ad_forward_backward_mismatch.py` should show <1% error
- All existing tests should still pass

---

## Files to Modify

1. **crm_ml_rl/wrappers/crm_bindings.cpp**
   - Function: `linearize_full_seed_action_from_seed_implicit` (line 2197)
   - Change: Use output Jacobian instead of residual Jacobian

2. **src/CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp**
   - Function: `DYNNLEquationOutputJacobianEigenAD` (line 1491)
   - Verify: Already computes correct gradients
   - No changes needed (maybe)

---

## Success Criteria

✅ AD gradients match FD with <1% error
✅ Forward pass returns correct tip position
✅ All existing tests pass
✅ Performance acceptable (AD through integration is slower than residual AD)

---

## Next Steps

1. ✅ Understanding complete - documented here
2. ⏭️ Implement Option 2A fix in `crm_bindings.cpp`
3. ⏭️ Test and validate
4. ⏭️ Update documentation

---

**Ready to proceed with implementation!**
