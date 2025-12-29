# Phase 0 Verification Experiments - Comprehensive Comparison

**Date:** 2025-12-29
**Purpose:** Test gradient errors across different initialization strategies
**Status:** EXPERIMENTS COMPLETE

---

## Executive Summary

**All three experiments confirm catastrophic gradient errors (33-99% Frobenius norm).**

The gradient error is a **fundamental issue**, not dependent on half-plane choice. However, **sign-aware initialization helps** - matching initialization to target half-plane reduces error from 99.54% to 94.36% (positive) or 33.12% (negative).

---

## Experimental Setup

| Experiment | Initialization | Test Currents | Sign Match? | Purpose |
|------------|---------------|---------------|-------------|---------|
| **A** | `[0, 0, 0.01]` | `[0.1, 0.05, 0.02]` | ✅ Yes | Baseline: positive half-plane |
| **B** | `[0, 0, -0.01]` | `[0.1, 0.05, -0.02]` | ✅ Yes | Test: negative half-plane |
| **C** | `[0, 0, 0.01]` | `[0.1, 0.05, -0.02]` | ❌ **No** | Test: cross half-plane mismatch |

All experiments:
- Insertion depth: 94.3 mm
- FD epsilon: 1e-5 (central differences)
- Parameters: `CatheterParameterSet_1_dyn.txt`

---

## Results Summary

### Error Metrics Comparison

| Experiment | Frobenius Norm Error | Mean Relative Error | Max Relative Error |
|------------|---------------------|--------------------|--------------------|
| **A (Positive match)** | 94.36% | 106.92% | 200.52% |
| **B (Negative match)** | 33.12% | 1444.86% | 25374.80% |
| **C (Mismatch)** | **99.54%** | 100.19% | 117.05% |

### Key Observations

1. **All experiments show catastrophic errors** (>30% Frobenius norm)
   - Even best case (B): 33.12% error
   - Original requirement: <1% error
   - Actual performance: 33-100× worse than requirement

2. **Sign mismatch worsens error** (Experiment C)
   - Positive init + negative currents: 99.54% error
   - Worse than matched cases A (94%) and B (33%)
   - **Conclusion:** Sign-aware initialization helps

3. **Negative half-plane is better** (Experiment B)
   - Negative match (B): 33.12% error
   - Positive match (A): 94.36% error
   - **3× improvement in Frobenius norm**
   - But mean/max errors are worse (possibly due to magnitude scaling)

---

## Detailed Results

### Experiment A: Positive Initialization, Positive Currents ✅

**Configuration:**
- Init: `[0, 0, 0.01]` → tip at `[0.026, 4.337, 94.251]` (y > 0)
- Currents: `[0.1, 0.05, 0.02]` (c3 > 0)

**Results:**
```
B from FD (ground truth):
[[   9.548,    0.839,  -10.897],
 [  -2.373,   15.473,   40.995],
 [  -6.831,   -0.254,    9.369],
 [ 234.621, 1434.981,  668.066],
 [ 196.985,  504.375,  185.216],
 [-416.210,   48.935,  756.985]]

B from implicit (wrong):
[[ -1.086,  -0.844, -17.277],
 [ -0.379,   0.848,  21.563],
 [  0.148,   0.124,  -1.781],
 [ -5.216, 161.368, 140.098],
 [-22.898, -79.838,-141.668],
 [  5.205,   9.605, -20.987]]
```

**Errors:**
- Frobenius norm: **94.36%**
- Mean relative: 106.92%
- Max relative: 200.52%

---

### Experiment B: Negative Initialization, Negative Currents ✅

**Configuration:**
- Init: `[0, 0, -0.01]` → tip at `[0.296, -8.884, 93.740]` (y < 0)
- Currents: `[0.1, 0.05, -0.02]` (c3 < 0)

**Results:**
```
B from FD (ground truth):
[[   5.644,    3.976,  -11.487],
 [   5.103,   -3.358,   31.245],
 [   0.631,   -0.842,    6.402],
 [  -0.049,  -18.658,  145.974],
 [ -25.453,  -95.341,  117.592],
 [   2.902,  -16.236,   99.092]]

B from implicit (wrong):
[[  5.954,   6.442,  -8.252],
 [  4.563,  -1.863,  21.750],
 [  0.515,  -0.834,   5.083],
 [-12.541, -51.206, 155.032],
 [ -3.438, -30.236, 111.446],
 [  1.814, -21.166,  96.068]]
```

**Errors:**
- Frobenius norm: **33.12%** ← BEST
- Mean relative: 1444.86%
- Max relative: 25374.80%

**Note:** Best Frobenius error but worst mean/max errors. Likely due to smaller absolute gradient magnitudes creating larger relative errors.

---

### Experiment C: Positive Initialization, Negative Currents ❌

**Configuration:**
- Init: `[0, 0, 0.01]` → tip at `[-0.793, 2.203, 94.265]` (y > 0)
- Currents: `[0.1, 0.05, -0.02]` (c3 < 0)
- **SIGN MISMATCH:** Initialized for positive y, but currents drive negative

**Results:**
```
B from FD (ground truth):
[[   16.089,     4.269,   764.089],
 [  225.596,   -81.883,  -131.662],
 [  -11.695,     5.951,    35.340],
 [-3746.082,  1308.642, -3190.310],
 [-1307.441,   746.118, 12534.107],
 [ -522.388,   287.529,  1586.144]]

B from implicit (wrong):
[[  1.067,   0.491,  -8.463],
 [  0.193,   1.595,  22.442],
 [ -0.008,  -0.015,  -0.201],
 [-19.353,  -5.586, 156.794],
 [ 47.911,  47.790, 107.071],
 [ -0.522,   0.184,   5.712]]
```

**Errors:**
- Frobenius norm: **99.54%** ← WORST
- Mean relative: 100.19%
- Max relative: 117.05%

**Conclusion:** Cross half-plane initialization causes near-total gradient failure.

---

## Common Findings Across All Experiments

### 1. Forward/Backward Mismatch

**ALL experiments show the same issue:**

| Experiment | Forward tip_position | Implicit next_state[:3] | Match? |
|------------|---------------------|------------------------|--------|
| A | `[0.026, 4.337, 94.251]` | `[6.138, 6.138, 6.138]` | ❌ |
| B | `[0.296, -8.884, 93.740]` | `[-1.583, -1.583, -1.583]` | ❌ |
| C | `[-0.793, 2.203, 94.265]` | `[0.208, 0.208, 0.208]` | ❌ |

The implicit linearization returns **corrupted state** - all components equal to the last velocity component from forward pass. This confirms they compute outputs from different mathematical operations.

### 2. Explicit Method Returns Zeros

**ALL experiments:**
```
Explicit B matrix: [[0, 0, 0], [0, 0, 0], [0, 0, 0], ...]
All zeros: True
```

Confirms the all-or-nothing failure mode documented in the plan. The explicit FD method exists but is unusable due to BVP convergence failures.

### 3. Element-wise Error Patterns

**Sample from Experiment A (relative errors):**
```
[[111.37%, 200.52%,  58.55%],
 [ 84.02%,  94.52%,  47.40%],
 [102.17%, 148.60%, 119.01%],
 [102.22%,  88.75%,  79.03%],
 [111.62%, 115.83%, 176.49%],
 [101.25%,  80.37%, 102.77%]]
```

- 16/18 elements: >80% error
- 10/18 elements: >100% error (wrong sign AND magnitude!)
- Even "best" elements: 47-58% error

---

## Interpretation

### What We Learned

1. **Error is fundamental, not half-plane specific**
   - All scenarios show >30% error
   - Root cause: forward uses explicit integration, backward differentiates implicit equilibrium
   - These are mathematically different operations

2. **Sign-aware initialization helps**
   - Matched init (A, B): 94% and 33% error
   - Mismatched init (C): 99% error
   - **Improvement: 6-66 percentage points**

3. **Negative half-plane is better**
   - Negative match (B): 33% Frobenius error
   - Positive match (A): 94% Frobenius error
   - Reason unclear - may be parameter-dependent

4. **Explicit method is broken**
   - Zero gradients in all cases
   - All-or-nothing BVP convergence failure
   - Cannot be used as-is

### Implications for Fix Strategy

**Phase A (Pure FD) is MANDATORY:**
- Errors are 33-99% (100× worse than requirement)
- No half-plane avoids the problem
- Sign-aware initialization alone insufficient
- Must implement correct backward pass

**Phase B (Fix explicit method) is STILL VALUABLE:**
- FD will be 6× slower (6 forward passes per backward)
- If explicit method can be fixed, it's more efficient
- But get correctness first (Phase A), optimize later (Phase B)

**Sign-aware initialization should be implemented:**
- Reduces error by 6-66 percentage points
- Cheap to implement (just sign check)
- Helps even though it doesn't solve the fundamental issue

---

## Files Generated

1. `/workspaces/catheter/CRM_ML/verify_gradient_issue.py` - Experiment A
2. `/workspaces/catheter/CRM_ML/verify_gradient_experiment_B.py` - Experiment B
3. `/workspaces/catheter/CRM_ML/verify_gradient_experiment_C.py` - Experiment C
4. This document: `/workspaces/catheter/CRM_ML/docs/VERIFICATION_EXPERIMENTS_COMPARISON_2025_12_29.md`

---

## Next Steps

**STOP for user approval.**

Based on these results, we should:

1. ✅ **Proceed with Phase A (Pure FD fix)** - Errors are catastrophic and universal
2. ✅ **Implement sign-aware initialization** - Helps reduce error
3. ✅ **Consider Phase B (fix explicit method)** - After Phase A, if performance matters
4. ❌ **Do NOT rely on half-plane choice alone** - All half-planes have major errors

**Ready to proceed to Phase A.2: Implement pure FD backward pass in C++ extension?**
