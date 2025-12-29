# Session Handoff: Seed Gradients Implementation (Phase 3B)

**Date:** 2025-12-29
**Status:** Task 0 Complete - Ready for Task B.1
**Current Model:** Switching from Opus (planning) to Sonnet (implementation)

---

## CRITICAL CONTEXT

### What Happened Before This Session

1. **Audit Conducted:** Full audit of Option C implementation revealed test weakening and missing seed gradients
2. **Plan Created:** Comprehensive TIER 1+2+3 plan at `/home/vscode/.claude/plans/expressive-percolating-snowglobe.md`
3. **Task 0 Completed:** Verified A matrix exists and has correct structure (6, 39)

### Current State

**Branch:** `claude/option-c-implementation`

**Files Modified (uncommitted):**
```
M crm_ml_rl/models/hybrid_dynamics.py
M crm_ml_rl/training/model_based_rl.py
M docs/NEXT_SESSION_HANDOFF_PROMPT.md
?? crm_ml_rl/envs/differentiable_catheter_env.py
?? crm_ml_rl/wrappers/option_c_physics.py
?? docs/SEED_GRADIENTS_IMPACT_ANALYSIS.md
?? docs/SESSION_HANDOFF_PHASE_3_2.md
?? tests/test_differentiable_env.py
?? tests/test_option_c_integration.py
?? tests/test_option_c_physics.py
```

**Tests Status:** 51/51 passing (but some tests weakened)

---

## NEXT TASK: Task B.1 - Implement Seed Gradients in C++

### What You Need to Do

**File to modify:** `/workspaces/catheter/CRM_ML/crm_torch/csrc/dynamics_op.cpp`

**Location:** Lines 321-322 (currently just a TODO comment)

**Current code:**
```cpp
// Phase 3B TODO: Compute seed gradients from A Jacobian
// For now, seed gradients remain zero (initialized above)
```

**What to add:** ~60 lines of C++ code to extract A matrix and compute seed gradients

**Key findings from Task 0:**
- ✅ A matrix exists in linearization result (verified)
- ✅ Shape: (6, 39) where 39 = total seed components
- ✅ A matrix already computed - just need to use it!

### Seed Component Mapping (39 total)

| Component | Count | Offset | Description |
|-----------|-------|--------|-------------|
| v | 3 | 0-2 | Velocity at sets |
| w | 3 | 3-5 | Angular velocity |
| p | 3 | 6-8 | Position |
| R | 9 | 9-17 | Rotation matrix |
| xf | 15 | 18-32 | Frame positions |
| mL | 3 | 33-35 | Magnetic field L |
| nL | 3 | 36-38 | Normal field L |

### Code Template to Implement

```cpp
// After line 320 (after computing grad_currents)

// Extract A Jacobian: ∂y/∂seed_state
py::array_t<double> A_np = result["A"].cast<py::array_t<double>>();
auto A_buf = A_np.unchecked<2>();

// Verify A shape (should be 6 x 39)
if (A_buf.shape(0) != 6) {
    throw std::runtime_error("Expected A to have 6 rows (output dim)");
}

int64_t num_seed_components = A_buf.shape(1);

// Compute seed gradients: grad_seed = A^T @ grad_y
// grad_y already extracted above (6,)

int seed_offset = 0;

// 1. grad_seed_v (num_sets * 3 components)
for (int64_t j = 0; j < num_sets; ++j) {
    for (int64_t k = 0; k < 3; ++k) {
        double grad_v_jk = 0.0;
        for (int l = 0; l < 6; ++l) {
            grad_v_jk += A_buf(l, seed_offset) * grad_y[l];
        }
        grad_seed_v.data_ptr<double>()[i * num_sets * 3 + j * 3 + k] = grad_v_jk;
        seed_offset++;
    }
}

// 2. grad_seed_w (num_sets * 3 components)
for (int64_t j = 0; j < num_sets; ++j) {
    for (int64_t k = 0; k < 3; ++k) {
        double grad_w_jk = 0.0;
        for (int l = 0; l < 6; ++l) {
            grad_w_jk += A_buf(l, seed_offset) * grad_y[l];
        }
        grad_seed_w.data_ptr<double>()[i * num_sets * 3 + j * 3 + k] = grad_w_jk;
        seed_offset++;
    }
}

// 3. grad_seed_p (num_sets * 3 components)
for (int64_t j = 0; j < num_sets; ++j) {
    for (int64_t k = 0; k < 3; ++k) {
        double grad_p_jk = 0.0;
        for (int l = 0; l < 6; ++l) {
            grad_p_jk += A_buf(l, seed_offset) * grad_y[l];
        }
        grad_seed_p.data_ptr<double>()[i * num_sets * 3 + j * 3 + k] = grad_p_jk;
        seed_offset++;
    }
}

// 4. grad_seed_R (num_sets * 9 components)
for (int64_t j = 0; j < num_sets; ++j) {
    for (int64_t k = 0; k < 9; ++k) {
        double grad_R_jk = 0.0;
        for (int l = 0; l < 6; ++l) {
            grad_R_jk += A_buf(l, seed_offset) * grad_y[l];
        }
        grad_seed_R.data_ptr<double>()[i * num_sets * 9 + j * 9 + k] = grad_R_jk;
        seed_offset++;
    }
}

// 5. grad_seed_xf (15 components)
for (int64_t k = 0; k < 15; ++k) {
    double grad_xf_k = 0.0;
    for (int l = 0; l < 6; ++l) {
        grad_xf_k += A_buf(l, seed_offset) * grad_y[l];
    }
    grad_seed_xf.data_ptr<double>()[i * 15 + k] = grad_xf_k;
    seed_offset++;
}

// 6. grad_seed_mL (num_sets * 3 components)
for (int64_t j = 0; j < num_sets; ++j) {
    for (int64_t k = 0; k < 3; ++k) {
        double grad_mL_jk = 0.0;
        for (int l = 0; l < 6; ++l) {
            grad_mL_jk += A_buf(l, seed_offset) * grad_y[l];
        }
        grad_seed_mL.data_ptr<double>()[i * num_sets * 3 + j * 3 + k] = grad_mL_jk;
        seed_offset++;
    }
}

// 7. grad_seed_nL (num_sets * 3 components)
for (int64_t j = 0; j < num_sets; ++j) {
    for (int64_t k = 0; k < 3; ++k) {
        double grad_nL_jk = 0.0;
        for (int l = 0; l < 6; ++l) {
            grad_nL_jk += A_buf(l, seed_offset) * grad_y[l];
        }
        grad_seed_nL.data_ptr<double>()[i * num_sets * 3 + j * 3 + k] = grad_nL_jk;
        seed_offset++;
    }
}

// Verification: seed_offset should equal num_seed_components
if (seed_offset != num_seed_components) {
    throw std::runtime_error(
        "Seed component count mismatch: expected " +
        std::to_string(num_seed_components) +
        ", got " + std::to_string(seed_offset)
    );
}
```

### After Adding Code

1. **Rebuild C++ extension:**
```bash
cd /workspaces/catheter/CRM_ML/crm_torch
cmake --build build
pip install -e .
```

2. **Test that it compiles:**
```bash
python -c "import crm_torch; print('Import successful')"
```

3. **Stop and ask for approval before testing**

---

## TODO LIST (15 Tasks Total)

**Current:** Task 0 ✅ (Complete)
**Next:** Task B.1 ← YOU ARE HERE

### Remaining Tasks
1. ✅ Task 0: Verify A matrix (DONE)
2. Task B.1: Implement seed gradients in C++ ← **NEXT**
3. Task B.2: Update OptionCPhysics wrapper
4. Task A.1: Add FD gradient validation tests
5. Task B.3: Add seed gradient tests
6. Task B.4: Validate seed gradients against FD
7. Task A.2: Fix multi-step gradient tests
8. Task A.3: Fix trajectory optimization tests
9. Task A.4: Validate numerical thresholds
10. Task C.1: Update DifferentiableMPCAgent
11. Task C.2: Add gradient stability features
12. Task D.1: Update __init__.py exports
13. Task D.2: Create demo script
14. Task D.3: Create documentation
15. Task E.1-E.3: Integration testing

**CRITICAL:** Stop after EACH task for user approval!

---

## IMPORTANT PROTOCOLS

### Protocol 1: Plan Generation
- **When using Opus for planning:** STOP after generating the plan
- **DO NOT proceed to implementation** with Opus
- User will switch to Sonnet for implementation
- Wait for explicit "proceed" command

### Protocol 2: Task-by-Task Approval
- Complete ONE task at a time
- Update todo list after each task
- Stop and wait for user approval before next task
- NEVER batch multiple tasks without approval

### Protocol 3: Test Honesty
- NEVER weaken test assertions to make them pass
- If a test fails, fix the root cause or document the limitation
- Tests must accurately reflect actual capabilities

---

## KEY FILES & LOCATIONS

### Plan File
`/home/vscode/.claude/plans/expressive-percolating-snowglobe.md`

### Current Implementation
- OptionCPhysics: `/workspaces/catheter/CRM_ML/crm_ml_rl/wrappers/option_c_physics.py`
- DifferentiableCatheterEnv: `/workspaces/catheter/CRM_ML/crm_ml_rl/envs/differentiable_catheter_env.py`
- C++ backend: `/workspaces/catheter/CRM_ML/crm_torch/csrc/dynamics_op.cpp`

### Tests
- Physics tests: `/workspaces/catheter/CRM_ML/tests/test_option_c_physics.py`
- Env tests: `/workspaces/catheter/CRM_ML/tests/test_differentiable_env.py`
- Integration tests: `/workspaces/catheter/CRM_ML/tests/test_option_c_integration.py`

### Documentation
- Seed gradients analysis: `/workspaces/catheter/CRM_ML/docs/SEED_GRADIENTS_IMPACT_ANALYSIS.md`
- Model-based RL requirements: `/workspaces/catheter/CRM_ML/docs/MODEL_BASED_RL_REQUIREMENTS.md`

---

## AUDIT FINDINGS (Why We're Doing This)

### Issues Found
1. ❌ Tests were weakened to hide Phase 3A limitations
2. ❌ Multi-step trajectory optimization diverges (no seed gradients)
3. ❌ No finite difference gradient validation
4. ❌ Missing exports in __init__.py files
5. ❌ No demo script
6. ❌ No user documentation

### Root Cause
Phase 3A (current) only computes ∂y/∂currents, NOT ∂y/∂seed.
This breaks multi-step gradient flow because seed state updates are detached.

### Solution
Implement Phase 3B seed gradients - the A matrix is already computed, just not used!

---

## VERIFICATION RESULTS (Task 0)

```
A matrix shape: (6, 39) ✅
Min value: -73.648534
Max value: 3943.519546
Nonzero count: 132 / 234
```

**Conclusion:** A matrix is ready to use - just need to extract it in C++ backward pass!

---

## SUCCESS CRITERIA

### For Task B.1
1. ✅ Code compiles without errors
2. ✅ C++ extension loads successfully
3. ✅ No segfaults when running forward/backward
4. ⏸️ Actual gradient values (test in later tasks)

### For Full Implementation
1. All tests pass with correct assertions (not weakened)
2. FD gradient validation passes (<10% error)
3. Multi-step optimization converges
4. Demo script runs successfully
5. Documentation complete

---

**STATUS: Ready for Task B.1 Implementation**

**Handoff Complete - Please continue from Task B.1**
