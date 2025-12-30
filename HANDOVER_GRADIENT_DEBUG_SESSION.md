# Handover: Gradient Debugging Session Summary

**Date**: 2025-12-29
**Task**: Debug 67.9% gradient error in implicit linearization (Phase 2)
**Status**: ❌ INCOMPLETE - Fundamental blocking issue identified

---

## Critical Discovery: The Linearization Function Doesn't Work

**Most Important Finding**: `linearize_full_seed_action_from_seed_implicit()` **HANGS** and never completes.

### Verified Facts

1. ✅ **Forward pass works**: `step_from_seed()` completes successfully
   ```python
   result = dyn.step_from_seed(currents, insertion, v, w, p, R, xf, mL, nL)
   # Returns tip position, velocity in ~1 second
   ```

2. ❌ **Linearization hangs**: `linearize_full_seed_action_from_seed_implicit()` times out after 30+ seconds
   ```python
   result = dyn.linearize_full_seed_action_from_seed_implicit(
       currents, insertion, v, w, p, R, xf, mL, nL, ...)
   # HANGS - never returns
   ```

3. ✅ **This was ALREADY broken**: Not caused by any changes during this session
   - Tested original code (git HEAD) - also hangs
   - Tested modified code - also hangs
   - Both produce identical behavior

### Test Script to Reproduce

```python
# File: test_minimal.py
import numpy as np
import sys
sys.path.insert(0, '/workspaces/catheter/CRM_ML')
from crm_ml_rl.wrappers import crm_python

dyn = crm_python.CRMDynamics()
dyn.load_parameters(
    "data/catheter_params/CatheterParameterSet_1_dyn.txt",
    "data/catheter_params/CatheterSpatialConfiguration_1.txt")
dyn.dt = 0.05
dyn.integration_step_size = 0.001
dyn.initialize_from_kinematics([0.0, 0.0, 0.01], 94.3)
seed = dyn.get_seed_state()

v = np.asarray(seed["v"], dtype=np.float64)
w = np.asarray(seed["w"], dtype=np.float64)
p = np.asarray(seed["p"], dtype=np.float64)
R = np.asarray(seed["R"], dtype=np.float64)
xf = np.asarray(seed["xf"], dtype=np.float64)
mL = np.asarray(seed["mL"], dtype=np.float64)
nL = np.asarray(seed["nL"], dtype=np.float64)
currents = np.array([0.1, 0.05, 0.02], dtype=np.float64)

# THIS WORKS
result = dyn.step_from_seed(currents, 94.3, v, w, p, R, xf, mL, nL)
print(f"✅ Forward: {result['tip_position']}")

# THIS HANGS (timeout after 30s+)
result = dyn.linearize_full_seed_action_from_seed_implicit(
    currents, 94.3, v, w, p, R, xf, mL, nL)  # <-- NEVER RETURNS
```

---

## What Was Attempted (But Cannot Be Verified)

### Diagnostic Work Done

1. **Added debug output** to C++ gradient computation
   - File: `crm_bindings.cpp:2857-3020`
   - Added `CRM_DEBUG_GRADIENT` environment variable
   - Prints BVP residuals, Jacobian norms, gradient components
   - **Cannot test** - linearization hangs before reaching these prints

2. **Created diagnostic scripts**
   - `debug_ad_gradient_components.py` - Component-wise gradient verification
   - `test_gx_verification.py` - Attempt to verify gx via finite differences
   - **Cannot run** - all depend on linearization completing

3. **Identified potential issue** with gx computation
   - Debug output showed `gx norm: 4.50e+07` (extremely large)
   - Finite difference estimate showed `gx_fd norm: 5.26e+04` (~1000x smaller)
   - **But this output was only seen once and cannot be reproduced** (tests hang)

### Hypothesis (Unverified)

There MAY be an issue where `eval_output_AD()` recomputes `tau` from backward integration, creating spurious derivatives. The actual forward pass uses `tau` as a fixed input copied from the BVP solve.

**Evidence supporting this**:
- Line 1548: `crm_bindings.cpp` - `DYNSolverIVP` receives tau as input, copied at line 1343 of `CoilDynamics_Defs.cpp`
- Lines 1213-1224: `CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp` - `eval_output_AD` recomputes tau via backward integration
- When autodiff differentiates through this, it computes ∂tau/∂(mL,nL) which shouldn't exist

**But cannot verify** because tests don't run.

### Attempted Fix (Also Hangs)

Modified `eval_output_AD` to accept tau as a fixed parameter:
```cpp
// src/CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp:1157
template <typename Scalar>
inline Eigen::Matrix<Scalar, Eigen::Dynamic, 1> eval_output_AD(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x_scaled,
    const Vec3<Scalar>& tau_fixed,  // FIX: tau is FIXED, not recomputed
    const DynamicsContextAD<Scalar>& ctx)
```

**Status**: Compiles successfully, but linearization still hangs. Reverted via `git checkout`.

---

## Architecture Understanding (Verified)

### Legacy CRM_Dynamics Uses BVP

**File**: `/workspaces/catheter/CRM_Dynamics/src/CRM_BVPSolver.cpp`

```
CRMShootingMethodBVP (line 13)
    ↓
Trust Region Dogleg (line 83)
    ↓ (iterative loop)
CRM_NLEquation (line 173)
    ↓
CRMSolverIVP_Core (line 194) - forward integration
    ↓
Returns residuals (lines 199-205)
    ↑
Iterate until F(x) ≈ 0
```

### CRM_ML Preserves BVP Architecture

**File**: `src/CoilDynamics_Defs.cpp:1457` and `crm_bindings.cpp:1211`

```
step_from_seed()
    ↓
DynamicsBVP() - solves F(mL,nL) = 0 via Trust Region
    Output: mL*, nL*, tau*, u0*
    ↓
DYNSolverIVP() - forward integrate with solved values
    Input: mL*, nL*, tau* (as fixed parameters)
    Output: tip_position, tip_velocity
```

**Key**: tau is computed by BVP solver (line 900 of CoilDynamics_Defs.cpp), then **passed as input** to DYNSolverIVP (line 1343, copied; line 1392, used in `net_mL = m_L - tau`; lines 1419-1420, used in `u_0 = ustar + Kinv * tau`).

---

## Analysis of Phase 2 Handoff Claim

### Handoff States (PHASE_2_HANDOFF.md)

> "The bug: dxdth is computed from BVP equilibrium (backward integration), but the actual forward pass uses IVP (forward integration)."

### This is Misleading

The forward pass uses **BOTH**:
1. BVP solver (DynamicsBVP) to find equilibrium forces (mL*, nL*, tau*)
2. IVP solver (DYNSolverIVP) to integrate forward with those forces

So the implicit differentiation approach (computing dx/dθ from BVP residual Jacobian) is **architecturally correct**.

### Option 2C Analysis

**Handoff proposes**: "Compute dx/dθ by finite-differencing the actual forward pass (IVP)"

**Problem**: This doesn't make sense because:
- x = [mL, nL] are **outputs of the BVP solver**, not the IVP solver
- If you finite-difference the full forward pass, you get dy/dθ directly (the full gradient you want)
- You wouldn't need the chain rule at all - that would be Option A (pure FD), not Option 2C (hybrid)

**Conclusion**: Option 2C as described is based on architectural misunderstanding. If you want FD gradients, just use Option A (pure FD of the forward pass).

---

## Current State of Code

### Files Modified (with debug output)
- `crm_bindings.cpp:2857-3020` - Added CRM_DEBUG_GRADIENT prints
- All changes committed to current branch

### Files Created
- `debug_ad_gradient_components.py` - Diagnostic script (cannot run)
- `test_gx_verification.py` - gx verification (cannot run)
- `test_quick_gradient.py` - Minimal test showing hang
- `test_minimal.py` - Demonstrates forward pass works, linearization hangs
- `GRADIENT_BUG_ANALYSIS.md` - Unverified analysis
- `HANDOVER_GRADIENT_DEBUG_SESSION.md` - This document

### Git Status
```bash
git status
# Modified files with debug prints
# Untracked analysis/test files
```

---

## Recommended Next Steps

### Priority 1: Fix Why Linearization Hangs (BLOCKING)

**Without this, you cannot test ANY gradient hypothesis.**

Possible causes to investigate:
1. Infinite loop in Jacobian computation (FD or AD)
2. Numerical issue in `Jxx` inversion (line 2858-2862 of crm_bindings.cpp)
3. Issue in `DYNNLEquationJacobianEigenAD` or `DYNNLEquationOutputJacobianEigenAD`
4. Problem with epsilon values (eps_residual_x, eps_residual_theta, etc.)

**Debugging approach**:
- Add print statements progressively through `linearize_full_seed_action_from_seed_implicit`
- Binary search to find exactly where it hangs
- Check if it's hanging on first Jacobian computation or later

### Priority 2: Once Linearization Works, Verify Gradient Accuracy

If you get linearization working and there IS a gradient error:

1. **Run diagnostics**:
   ```bash
   CRM_DEBUG_GRADIENT=1 python3 debug_ad_gradient_components.py
   ```

2. **Check component magnitudes**:
   - Is `gx` unreasonably large? (Expected ~1e4-1e5, saw 4e7)
   - Is `gth` reasonable? (Expected ~1e2-1e3, saw 178)
   - Is `Jxx` well-conditioned?
   - Is BVP residual actually zero? (`||F(x*, θ)|| < 1e-6`)

3. **If gx is wrong**: Investigate the tau computation issue I identified
4. **If gth is wrong**: Check if x is truly held constant during differentiation
5. **If Jxx/Jxθ wrong**: Verify BVP residual Jacobians

### Priority 3: Alternative Approaches

If debugging proves too difficult:

**Option A (Pure FD)**: Just use finite differences on the forward pass
```python
def compute_gradient_fd(dyn, currents, insertion, seed, eps=1e-6):
    # Perturb currents, run forward pass, compute dy/dθ
    # Simple, reliable, works
```

**Option 3 (Full AD)**: Differentiate through the entire BVP solve
- Very complex - need to make Trust Region solver differentiable
- Likely not worth it

---

## Questions for Next Person

1. **Has `linearize_full_seed_action_from_seed_implicit` EVER worked?**
   - Check git history
   - Look for any successful test runs

2. **Is there a simpler linearization function that works?**
   - Maybe `linearize_from_seed` (without "implicit")?
   - Check Python API for alternatives

3. **What is the epsilon policy?**
   - Default values for eps_residual_x, eps_residual_theta, eps_g_x, eps_g_theta?
   - Could wrong epsilon cause hang?

4. **Do you actually need implicit linearization?**
   - If FD works (Option A), use that
   - "Pure AD" is a nice-to-have, not a must-have

---

## Files and Locations Reference

### Key C++ Files
- `crm_ml_rl/wrappers/crm_bindings.cpp:2197-3000` - Linearization implementation
- `src/CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp:1157-1310` - eval_output_AD
- `src/CoilDynamics_Defs.cpp:1457-1606` - DynamicsBVP
- `src/CoilDynamics_Defs.cpp:1286-1455` - CRMIVP_DYN

### Key Python Tests
- `test_ad_forward_backward_mismatch.py` - Original test showing 67.9% error (HANGS)
- `test_minimal.py` - Demonstrates forward works, linearization hangs
- `debug_ad_gradient_components.py` - Would diagnose components (CANNOT RUN)

### Documentation
- `PHASE_2_HANDOFF.md` - Original problem statement (contains misdiagnosis)
- `GRADIENT_BUG_ANALYSIS.md` - Unverified analysis from this session
- `HANDOVER_GRADIENT_DEBUG_SESSION.md` - This document

---

## Summary

**What we know**: The linearization function hangs. It was already broken before this debugging session started.

**What we suspect (but cannot verify)**: There may be an issue with tau being recomputed in eval_output_AD, creating spurious derivatives. The gx Jacobian may be ~1000x too large.

**What we tried**: Added debug output, created diagnostic scripts, attempted a fix. Nothing worked because the fundamental issue (linearization hanging) blocks all testing.

**What to do next**: Fix why linearization hangs. Only then can you test gradient accuracy.

**Bottom line**: You cannot debug gradients when the gradient computation doesn't complete. Fix the hang first, THEN debug the gradients.
