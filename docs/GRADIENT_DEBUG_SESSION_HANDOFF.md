# Gradient Debugging Session Handoff

**Date:** 2025-12-26
**Session Goal:** Fix gradient validation failures in Option C backward pass
**Current Status:** CP-C07 (Gradient Validation) - IN PROGRESS - Critical bug found and partial fix applied

---

## Problem Summary

The C++ extension backward pass (implicit differentiation) was implemented but **gradients are 6+ orders of magnitude too small**:
- **Python wrapper gradients:** ~1-100 (correct, from `linearize_full_seed_action_from_seed_implicit`)
- **C++ extension gradients:** ~1e-8 to 1e-6 (WRONG - essentially zero)
- **Finite difference gradients:** ~1-100 (confirms Python is correct)

**Test results:**
```
Python wrapper: B[:3, 0] = [3.19, -0.81, 0.01]  (norm: 1.88e+02)
C++ extension:  B[:3, 0] = [5.0e-08, -3.7e-07, 1.3e-08]  (norm: 1.87e-03)
Difference: 1.85e+02 (FAILED - tolerance was 1e-07)
```

---

## Root Cause Identified

**The backward pass was using zero/guess values for mL and nL instead of the converged BVP solution!**

### The Issue:
1. Forward pass solves BVP and gets `mL_star`, `nL_star` (the solution)
2. Autograd saves the *input* `seed_mL`, `seed_nL` (the guess, often zeros)
3. Backward pass was using these guess values (zeros) to compute Jacobians
4. With `mL=0, nL=0`, the residual Jacobian is near-identity, giving tiny gradients

### The Fix Applied:
Modified `crm_step_backward()` in `crm_torch_ext/csrc/crm_step_op.cpp` (lines 752-826):
```cpp
// CRITICAL FIX: Re-solve BVP to get converged mL_star and nL_star
// The input seed_mL and seed_nL are just guesses (often zeros)
// We need the actual converged solution from the BVP solver
// This matches the Python bindings approach (line 2292 in crm_bindings.cpp)

// ... setup BVPParams ...

// Solve BVP to get converged mL_star and nL_star
DynamicsBVP(BVPParams, xf_local, mL_guess, nL_guess, ftip_guess,
            out_u0, mL_star, nL_star, out_tau, ftip_calc, localmin);

// Now use mL_star, nL_star for Jacobian computation
auto [B, A] = compute_implicit_jacobians(..., mL_star, nL_star, ...);
```

This matches what the Python wrapper does - it calls `step_from_seed` in the backward pass to get the converged solution before computing gradients.

---

## Current Status

**Code changes made:**
1. ✅ Added BVP re-solve in backward pass (lines 752-826 of `crm_step_op.cpp`)
2. ✅ Added damping compensation to BVP guess (lines 776-782)
3. ✅ Added convergence check and error handling (lines 794-818)
4. ✅ Added debug output (lines 794-801) - can enable with `CRM_DEBUG_BACKWARD=1`
5. ✅ Code compiles successfully

**Testing status:**
- ❌ NOT YET TESTED with the fix
- Need to run: `CRM_DEBUG_BACKWARD=1 python3 /tmp/test_backward_debug.py`
- Need to run: `python3 crm_torch_ext/test/test_gradient_parity.py`

---

## Next Steps (Priority Order)

###  1. **Verify the Fix Works** (IMMEDIATE)
```bash
# Quick test to see if gradients are now correct
CRM_DEBUG_BACKWARD=1 python3 /tmp/test_backward_debug.py

# Full validation suite
python3 crm_torch_ext/test/test_gradient_parity.py
```

**Expected outcome:**
- BVP should converge (localmin=0)
- mL_star, nL_star should be non-zero
- Gradients should match Python wrapper within 1e-7
- Finite difference check should pass (<1% error)

### 2. **If Still Failing - Debug BVP Solve**
Check if BVP is converging properly in backward pass:
- Look at debug output: `mL_star` and `nL_star` values
- Check `localmin` status
- Compare with forward pass BVP solve

Possible issues:
- BVP parameters not matching forward pass
- Damping compensation formula incorrect
- Initial guess not good enough

### 3. **If BVP Converges But Gradients Still Wrong**
The issue would be in `compute_implicit_jacobians()` function (lines 421-659):
- Verify autodiff calls: `DYNNLEquationJacobianEigenAD`, `DYNNLEquationControlJacobianEigenAD`
- Check IFT solver: `qr.solve(-J_xu)`
- Verify output Jacobian `g_x` computation (finite differences)
- Check chain rule: `B = g_x * dx_du`

### 4. **Complete CP-C07**
Once gradients match:
- Run full validation suite
- Update status documents
- Mark CP-C07 as COMPLETE
- Proceed to CP-C08 (Operator Parity Test)

---

## Key Files

**Implementation:**
- `crm_torch_ext/csrc/crm_step_op.cpp` - Main backward pass implementation
  - Lines 421-659: `compute_implicit_jacobians()` - IFT math
  - Lines 661-896: `crm_step_backward()` - Main backward function (NOW WITH BVP RE-SOLVE)
- `crm_torch_ext/csrc/bindings.cpp` - Autograd Function wrapper

**Tests:**
- `crm_torch_ext/test/test_gradient_parity.py` - **PRIMARY TEST** - Compares C++ vs Python
- `crm_torch_ext/test/test_backward_simple.py` - Simple non-zero gradient check
- `/tmp/test_backward_debug.py` - Quick debug script

**Reference:**
- `crm_ml_rl/wrappers/crm_bindings.cpp` - Python wrapper implementation
  - Lines 2265-3028: `linearize_full_seed_action_from_seed_implicit()` - THE REFERENCE
  - Line 2292: Where Python re-solves BVP in backward pass

**Documentation:**
- `docs/OPTION_C_TASK_2.2_COMPLETION_REPORT.md` - Original backward pass completion
- `docs/OPTION_C_CORRECTED_STATUS.md` - Current overall status
- `docs/architecture/OPTION_C_IMPLEMENTATION_PLAN.md` - Full plan

---

## Mathematical Background

**Implicit Differentiation Chain:**
```
Given: F(x, u) = 0  (BVP residual)
       y = g(x, u)  (output from IVP)

Compute: dy/du

Steps:
1. J_xx = ∂F/∂x  (via autodiff on residual)
2. J_xu = ∂F/∂u  (via autodiff on residual)
3. dx/du = -J_xx^{-1} @ J_xu  (Implicit Function Theorem)
4. g_x = ∂g/∂x  (via finite differences on IVP)
5. dy/du = g_x @ dx/du  (chain rule)
```

**Why BVP solution matters:**
- All Jacobians (J_xx, J_xu, g_x) are evaluated at the converged point x*
- If x* is wrong (e.g., zeros), Jacobians are completely incorrect
- The BVP solution x* = (mL_star, nL_star) must come from solving F(x,u)=0

---

## Quick Reference Commands

```bash
# Rebuild after changes
python3 crm_torch_ext/setup.py build_ext --inplace

# Quick gradient check with debug
CRM_DEBUG_BACKWARD=1 python3 /tmp/test_backward_debug.py 2>&1 | grep -v DEBUG

# Full validation test
python3 crm_torch_ext/test/test_gradient_parity.py

# Check what Python wrapper does
python3 /tmp/test_bvp_solve.py 2>&1 | grep -v DEBUG
```

---

## Debug Test Script

If `/tmp/test_backward_debug.py` is missing, recreate it:

```python
import sys, os
sys.path.insert(0, '/workspaces/catheter/CRM_ML')
os.chdir('/workspaces/catheter/CRM_ML')
import torch, numpy as np, crm_torch_ext
from crm_ml_rl.wrappers import crm_python

crm_torch_ext.initialize_params(
    "data/catheter_params/CatheterParameterSet_1_dyn.txt",
    "data/catheter_params/CatheterSpatialConfiguration_1.txt")
crm_torch_ext.set_timestep(0.02)
crm_torch_ext.set_integrator("abm4")
crm_torch_ext.set_integration_step_size(0.1)
crm_torch_ext.set_damping([12.18, 12.18, 284.43, 0.0305, 0.0305, 0.00503])

dyn = crm_python.CRMDynamics()
dyn.load_parameters(
    "data/catheter_params/CatheterParameterSet_1_dyn.txt",
    "data/catheter_params/CatheterSpatialConfiguration_1.txt")
dyn.set_damping(np.array([12.18, 12.18, 284.43, 0.0305, 0.0305, 0.00503]))
dyn.dt = 0.02
dyn.integration_step_size = 0.1
dyn.set_integrator("abm4")
dyn.initialize_from_kinematics(np.array([0.1, 0.0, 0.0]), 94.3)
seed = dyn.get_seed_state()

currents = torch.tensor([0.1, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
insertion = torch.tensor([94.3], dtype=torch.float64)
seed_v = torch.from_numpy(seed['v'])
seed_w = torch.from_numpy(seed['w'])
seed_p = torch.from_numpy(seed['p'])
seed_R = torch.from_numpy(seed['R'])
seed_xf = torch.from_numpy(seed['xf'])
seed_mL = torch.from_numpy(seed['mL'])
seed_nL = torch.from_numpy(seed['nL'])

print("Forward...")
output = crm_torch_ext.crm_step(currents, insertion, seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL)
print(f"Output: {output}")

print("\nBackward...")
output[0].backward()
print(f"Currents grad: {currents.grad}")
print(f"Grad norm: {currents.grad.norm().item()}")

# Compare with Python
result = dyn.linearize_full_seed_action_from_seed_implicit(
    currents=np.array([0.1, 0.0, 0.0]), insertion_length=94.3,
    v_in=seed['v'], w_in=seed['w'], p_in=seed['p'], R_in=seed['R'],
    xf_in=seed['xf'], mL_in=seed['mL'], nL_in=seed['nL'])
print(f"\nPython B[:, 0]: {result['B'][:, 0]}")
print(f"Expected grad[0]: {result['B'][0, 0]:.6e}")
print(f"Actual grad[0]:   {currents.grad[0].item():.6e}")
```

---

## Summary

**What was done:**
- Identified root cause: backward pass using mL=0, nL=0 instead of BVP solution
- Implemented fix: re-solve BVP in backward pass (like Python wrapper does)
- Code compiles successfully

**What needs to be done:**
- Test if gradients are now correct
- Debug if still failing
- Complete CP-C07 validation

**Expected fix time:** 10-30 minutes if BVP converges properly, longer if additional debugging needed.

The fix should work because it exactly matches the Python wrapper's approach. The key insight is that implicit differentiation requires evaluating Jacobians at the converged solution, not at the initial guess.
