# Phase 3B Progress Summary
**Date**: 2025-12-30
**Status**: FD-based dxdth implemented, multi-step flow debugging in progress

## Completed Work

### 1. Forward Pass Seed Return (✅ DONE)
**Commit**: `4fd1a1b` - "Phase 3B partial: Return updated seeds from forward pass"

Modified files to return updated seeds from forward pass:
- `crm_torch/csrc/dynamics_op.hpp`: Changed return type to 8-tuple
- `crm_torch/csrc/dynamics_op.cpp`: Extract and return updated seeds
- `crm_torch/crm_torch/__init__.py`: Handle 8 outputs, pass to backward
- `crm_ml_rl/wrappers/option_c_physics.py`: Return updated seeds to Python
- `test_multistep_seed_gradients.py`: Updated API usage

Result: Single-step seed gradients work! (A matrix gradients are non-zero)

### 2. FD-Based dxdth Computation (✅ DONE)
**Commit**: `5c36df1` - "Phase 3B: FD-based dxdth computation in linearizer + debug"

**Root Cause Identified**:
```
Forward pass:  Uses IVP (explicit integration via step_from_seed)
Backward pass: Uses BVP (equilibrium condition differentiation)
→ These are DIFFERENT operations → 67.9% gradient mismatch
```

**Fix Implemented** (crm_bindings.cpp:2928-3025):
- Replace: `dxdth = -Jxx^{-1} * Jxth` (BVP-based, WRONG)
- With: `dxdth` via FD on `step_from_seed` (IVP-based, CORRECT)
- Compute dx/dθ by perturbing inputs and calling actual forward function
- Keep old BVP approach for comparison/debugging

**Debug Infrastructure**:
- `CRM_DEBUG_GRADIENT=1`: Shows dx/dθ comparison (FD vs BVP)
- `CRM_DEBUG_BACKWARD=1`: Traces backward calls and gradient flow

## Current Issue: Multi-Step Gradient Flow

### Problem
Multi-step test FAILS:
```
curr2 (step 2): norm=3.8e+14  ✓ PASS
curr1 (step 1): norm=0.0      ✗ FAIL
```

### Diagnostic Results
Using `CRM_DEBUG_BACKWARD=1` on `test_simple_multistep.py`:

**Step 2 backward** (direct loss):
```
grad_output norm: 7.978531e+04  ← From loss
grad_next_v norm: 0.000000e+00  ← No future steps
```

**Step 1 backward** (chained from Step 2):
```
grad_output norm: 0.000000e+00  ← output1 not in loss
grad_next_v norm: 1.839241e+14  ← HUGE gradient from Step 2!
→ But curr1.grad = 0.0 ✗
```

### Root Cause Analysis

The gradient chain:
```
curr1 → [Step1] → next1_v → [Step2] → output2 → loss
```

Gradient flow:
```
∂loss/∂curr1 = ∂loss/∂next1_v * ∂next1_v/∂curr1
               = 1.8e14        * ???
```

**The Problem**: `dynamics_backward` only computes:
- ✓ `∂(tip_pos, tip_vel)/∂currents` (B matrix)
- ✓ `∂(tip_pos, tip_vel)/∂seeds_in` (A matrix)
- ✗ `∂(next_mL, next_nL)/∂currents` ← **MISSING!**
- ✗ `∂(next_mL, next_nL)/∂seeds_in` ← **MISSING!**

**Evidence** (dynamics_op.cpp:362-395):
```cpp
// Extract ONLY tip position/velocity for Jacobian computation
py::array_t<double> pos_plus = result_plus["tip_position"].cast<>();
py::array_t<double> vel_plus = result_plus["tip_velocity"].cast<>();
// ❌ Does NOT extract next_mL, next_nL!

// Compute B matrix using only tip_pos/tip_vel
B_buf(k, j) = (pos_plus_buf(k) - pos_minus_buf(k)) / (2.0 * fd_eps);
```

So when Step 1's backward receives `grad_next_v = 1.8e14`:
1. It computes `grad_currents = B^T @ grad_output = B^T @ 0 = 0` ← WRONG!
2. It should also compute `grad_currents += (∂next_seeds/∂curr)^T @ grad_next_seeds`
3. But `∂next_seeds/∂curr` is never computed!

## Solution: Extend dynamics_backward

Need to compute full Jacobians for ALL outputs (state + seeds) w.r.t. ALL inputs.

### Current Jacobians (6 rows = tip_pos + tip_vel):
```
B matrix (6, 3):     ∂output/∂currents
A matrix (6, K):     ∂output/∂seeds_in  (K = seed_dim ≈ 39)
```

### Required Jacobians (add seed output rows):
```
B_seeds (6*num_sets, 3):  ∂(next_v,w,p,R,xf,mL,nL)/∂currents
A_seeds (6*num_sets, K):  ∂(next_v,w,p,R,xf,mL,nL)/∂seeds_in
```

### Implementation Plan

**Option A**: Extend FD in dynamics_backward
```cpp
// For each current perturbation:
for (int j = 0; j < 3; ++j) {
    // ... existing code ...

    // ✓ Already extract tip_pos/vel
    pos_plus = result_plus["tip_position"];
    vel_plus = result_plus["tip_velocity"];

    // ➕ ALSO extract updated seeds
    next_v_plus = result_plus["next_v"];
    next_w_plus = result_plus["next_w"];
    // ... etc for p, R, xf, mL, nL

    // Compute Jacobians for ALL outputs
    B_state[..., j] = Δtip / Δcurr[j];
    B_seeds[..., j] = Δseeds / Δcurr[j];  // NEW!
}

// In backward, compute FULL gradient:
grad_currents = B_state^T @ grad_output
              + B_seeds^T @ grad_next_seeds;  // NEW!
```

**Option B**: Use the linearizer
- The linearizer ALREADY computes `dxdth` (which I just fixed!)
- `dxdth[:, :3]` = `∂(mL*, nL*)/∂currents`
- Could call linearizer from dynamics_backward
- But linearizer is more complex and slower

**Recommendation**: Option A (extend FD) for consistency

### Files to Modify
1. `crm_torch/csrc/dynamics_op.cpp`:
   - Extract all 8 outputs from `step_from_seed` in FD loops
   - Compute extended Jacobians (state + seeds)
   - Accumulate gradients from both paths

2. `crm_torch/crm_torch/__init__.py`:
   - Already correct! Just needs C++ to compute right Jacobians

## Test Plan

After implementing Option A:

1. **Unit test**: `test_simple_multistep.py`
   - Should show `curr1.grad ≠ 0`
   - Gradients flow curr1 → next1_v → curr2

2. **Full test**: `test_multistep_seed_gradients.py`
   - Multi-step gradient flow test should PASS

3. **Validation**: Compare against pure FD end-to-end
   - Perturb curr1, measure change in final loss
   - Should match computed curr1.grad

## Performance Notes

**Current**: dynamics_backward does ~3*3 + 39 = 48 FD evaluations per backward call
- 3 currents × 3 perturbations (±, but optimized)
- 39 seed components × 2 perturbations

**After fix**: ~51 FD evaluations
- Same as current (we're already perturbing, just need to extract more outputs)
- No significant slowdown!

## References

- Original audit: `/home/vscode/.claude/plans/cuddly-tinkering-twilight.md`
- AD mismatch analysis: `/workspaces/catheter/CRM_ML/PHASE_2_AD_MISMATCH_ANALYSIS.md`
- Test files:
  - `test_multistep_seed_gradients.py` (full test)
  - `test_simple_multistep.py` (minimal repro)
  - `test_grad_fn_debug.py` (grad_fn verification)

## Next Steps

1. ✅ Commit current progress (FD-based dxdth + diagnostics)
2. ⏳ Implement Option A (extend dynamics_backward)
3. ⏳ Test multi-step gradient flow
4. ⏳ Validate against pure FD
5. ⏳ Final commit and PR

---
**Status**: Ready to implement seed output Jacobians in dynamics_backward
