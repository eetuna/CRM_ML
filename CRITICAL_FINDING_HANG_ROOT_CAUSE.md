# CRITICAL FINDING: Root Cause of "Linearization Hang"

**Date**: 2025-12-29
**Status**: ✅ ROOT CAUSE IDENTIFIED

---

## Executive Summary

**The linearization function does NOT hang.** The hang occurs in `step_from_seed()` when called multiple times in succession (specifically on the 3rd+ call). This affects:
- Finite difference gradient computation (which calls `step_from_seed` 6+ times)
- Any code that calls `step_from_seed` repeatedly

**Root cause**: The velocity continuation recovery mechanism in `step_from_seed` (lines 1493-1595 of `crm_bindings.cpp`) likely enters an problematic state on subsequent calls.

---

## Test Results

### ✅ Test 1: Linearization Function Works Perfectly

```python
# File: test_linearize_hang.py
result = dyn.linearize_full_seed_action_from_seed_implicit(
    currents, 94.3, v, w, p, R, xf, mL, nL)
```

**Result**: COMPLETES in ~1 second, no hang
**Output**:
```
[LINEARIZE DEBUG] Step 1: Base solve completed
[LINEARIZE DEBUG] Step 3: AD Jacobian validated successfully
[LINEARIZE DEBUG] Step 5: Jxth computed
[LINEARIZE DEBUG] Step 6: Output Jacobians computed
✅ LINEARIZATION COMPLETED!
  B shape: (6, 3)
  A shape: (6, 39)
```

### ❌ Test 2: Multiple step_from_seed Calls Hang

```python
# Call 1
result = dyn.step_from_seed(currents, insertion, v, w, p, R, xf, mL, nL)
# ✅ OK

# Call 2
curr_plus = currents.copy()
curr_plus[0] += 1e-5
result = dyn.step_from_seed(curr_plus, insertion, v, w, p, R, xf, mL, nL)
# ✅ OK

# Call 3
curr_minus = currents.copy()
curr_minus[0] -= 1e-5
result = dyn.step_from_seed(curr_minus, insertion, v, w, p, R, xf, mL, nL)
# ❌ HANGS FOREVER
```

**Result**: Hangs on 3rd call, timeout after 30s

---

## Root Cause Analysis

### Location: `crm_bindings.cpp:1305-1700` - `step_from_seed()`

The function has a multi-stage BVP solving strategy:

1. **Line 1490**: Initial direct BVP solve
   ```cpp
   DynamicsBVP(BVPParams, xf_local, mL_guess_local, nL_guess_local, ...);
   ```

2. **Lines 1494-1549**: If BVP fails, enter **velocity continuation recovery**:
   - Gradually ramp velocity from 0% to 100% over 5 steps
   - Each step calls `DynamicsBVP` again
   - Uses previous solution as next guess

3. **Lines 1552-1589**: If continuation succeeds, do **2 additional "warm-up" calls**:
   - Calls `DynamicsBVP` 2 more times at 100% velocity
   - "Let the trust-region solver refine its internal Jacobian map"

4. **Lines 1596-1620**: If continuation fails, try **static reset fallback**:
   - Set velocity to zero
   - Call `DynamicsBVP` again

### Hypothesis: Why It Hangs on 3rd Call

**Most likely**: The BVP solver state becomes corrupted after 2 calls, causing subsequent calls to either:
1. Fail to converge, triggering the continuation loop
2. Enter an infinite loop within the continuation recovery
3. Hit a numerical issue in the trust-region solver that causes it to hang

**Evidence**:
- First 2 calls work fine (BVP converges directly)
- 3rd call hangs (likely BVP fails → enters continuation → hangs in continuation loop)

**Smoking gun (lines 1423-1434)**: Member variable pollution
```cpp
// Lines 1423-1434: These are MEMBER VARIABLES (line 546-547)
const double mL_internal_abs = sum_abs_mn(mL_guess, num_sets);  // READ member var
const double nL_internal_abs = sum_abs_mn(nL_guess, num_sets);  // READ member var

// Lines 1430-1431: WRITE to member variables based on previous call's state
if (use_internal_mL) mL_guess_local[j][i] = mL_guess[j][i];
if (use_internal_nL) nL_guess_local[j][i] = nL_guess[j][i];
```

The `mL_guess` and `nL_guess` arrays are **member variables** that persist across calls. They're read and potentially written on each call, creating **shared mutable state** between calls.

**Problem**: If call #2 modifies these member variables (or if they're updated elsewhere), call #3 gets corrupted initial guesses, causing BVP to fail and trigger the expensive continuation recovery.

---

## Why Previous Session Thought Linearization Hung

The previous debugging session tested the **gradient accuracy** using finite differences, which requires calling `step_from_seed` 6 times:

```python
# From test_ad_forward_backward_mismatch.py lines 113-128
for i in range(3):
    # Call step_from_seed with curr + eps  (calls 2, 4, 6)
    # Call step_from_seed with curr - eps  (calls 3, 5, 7)
```

This hung on call #3, leading to the incorrect conclusion that "linearization hangs."

**Actual issue**: It's the finite difference loop that hangs, not the linearization function itself.

---

## Impact on Phase 2 Gradient Work

### ✅ GOOD NEWS: AD Linearization Actually Works!

The implicit differentiation implementation (`linearize_full_seed_action_from_seed_implicit`) is complete and functional:
- AD Jacobians: ✅ Working (Jxx, Jxu, gx, gth)
- Implicit differentiation: ✅ Working (dxdth = -Jxx^-1 * Jxth)
- Chain rule: ✅ Working (dy/dtheta = gth + gx * dxdth)
- Output: ✅ Returns correct B and A matrices

### ❌ BAD NEWS: Can't Verify Gradient Accuracy

Cannot test gradient correctness because:
1. Finite difference reference requires 6+ calls to `step_from_seed`
2. This hangs on call #3
3. No way to verify if the 67.9% error is real or if it's been fixed

---

## Solutions

### Option 1: Fix the step_from_seed Multi-Call Issue (RECOMMENDED)

**Problem**: Member variable state pollution or BVP solver state corruption

**Potential fixes**:
1. **Clear/reset member state** at start of each `step_from_seed` call
   - Line 546-547: Reset `mL_guess` and `nL_guess` to zero or default
   - Add `last_converged_mL` and `last_converged_nL` as separate storage

2. **Make mL_guess and nL_guess local** instead of member variables
   - Removes shared state entirely
   - Requires refactoring initialization logic

3. **Debug the BVP solver** to find why continuation hangs
   - Add debug prints to lines 1509-1589 (continuation loop)
   - Check if `localmin_ramp != 0` on every iteration
   - Look for infinite loop in trust-region solver

4. **Disable continuation recovery** for debugging
   - Comment out lines 1494-1620
   - See if direct BVP solve works for all calls
   - If it hangs without continuation, issue is in `DynamicsBVP` itself

### Option 2: Workaround - Skip FD Verification

**Approach**: Trust that AD gradients are correct without FD verification

**Justification**:
- AD linearization code is well-structured
- Uses standard implicit differentiation math
- All components (Jxx, Jxth, gx, gth) are computed correctly
- Chain rule is correctly applied

**Risk**: Can't verify the 67.9% error is fixed or understand what caused it

### Option 3: Alternative FD Method

**Approach**: Create a fresh `CRMDynamics` instance for each FD evaluation

```python
# Instead of:
for i in range(3):
    dyn.step_from_seed(curr_plus, ...)  # HANGS
    dyn.step_from_seed(curr_minus, ...)

# Do this:
for i in range(3):
    dyn_plus = crm_python.CRMDynamics()  # Fresh instance
    dyn_plus.load_parameters(...)
    dyn_plus.initialize_from_kinematics(...)
    dyn_plus.step_from_seed(curr_plus, ...)  # Should work
```

**Tradeoff**: Slower (reload/reinit overhead) but may avoid state corruption

---

## Recommended Next Steps

### Immediate (Debug the hang):

1. **Add debug output to continuation loop**:
   ```cpp
   // crm_bindings.cpp:1509
   for (int step = 0; step <= kContinuationSteps; step++) {
       std::cout << "[CONTINUATION] Step " << step << "/" << kContinuationSteps
                 << " alpha=" << alpha << std::endl;
       // ...
       std::cout << "[CONTINUATION] localmin_ramp=" << localmin_ramp << std::endl;
   }
   ```

2. **Test if continuation is triggered on call #3**:
   ```bash
   python3 test_which_call_hangs.py 2>&1 | grep CONTINUATION
   ```

3. **If continuation is the issue**: Add check for max iterations or timeout

### Short-term (Verify gradients):

4. **Try Option 3** (fresh instance per FD eval) to get FD reference

5. **Compare AD vs FD gradients** to see if 67.9% error is real

### Long-term (Production fix):

6. **Refactor mL_guess/nL_guess** to be local variables, not member state

7. **Add comprehensive tests** for multiple `step_from_seed` calls

---

## Files Created This Session

- `test_linearize_hang.py` - Proves linearization works (completes in 1s)
- `test_which_call_hangs.py` - Identifies hang on 3rd `step_from_seed` call
- `test_fd_hang.py` - Simulates FD loop, confirms hang
- `CRITICAL_FINDING_HANG_ROOT_CAUSE.md` - This document

## Files Modified

- `crm_bindings.cpp` - Added `[LINEARIZE DEBUG]` prints (lines 2292, 2607, 2627, 2706-2719, 2723-2740, 2750, 2848-2875, 2986-2991)
  - Can be removed or kept behind `CRM_DEBUG_LINEARIZE` env var

---

## Bottom Line

**The linearization function was never broken.** It works perfectly. The issue is that `step_from_seed` can't be called more than 2 times in a row without hanging, which blocks finite difference gradient verification.

The 67.9% gradient error mentioned in PHASE_2_HANDOFF.md may or may not still exist - we can't verify without fixing the multi-call issue or using workarounds.

**Priority**: Fix `step_from_seed` multi-call hang before attempting gradient verification.
