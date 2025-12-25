# Comprehensive Code Audit Plan (v2)

**Date:** 2025-12-25
**Scope:** Full Repository (`src`, `crm_ml_rl`, `examples`, `scripts`, `tests`)
**Objective:** Identify bugs, divergence, stability issues, regression risks, obsolete code, and documentation mismatches.
**Constraint:** Non-invasive audit. No code modification allowed during this phase.

---

## Phase 1: Core C++ Audit (`src/`, `src/numerical/`)

**Goal:** Verify numerical stability, memory safety, and architectural compliance (Option A).

### 1.1 "Ghost Value" Bug Regression Check
*   **Target:** `src/CRM_BVPIVP_APIDeclarations.hpp`, `src/CoilDynamics_Defs.cpp`
*   **Audit Task:**
    *   Confirm `CRMIVPCoreParams` and `CRMShootingMethodParams` use `std::vector<Eigen::...>` containers.
    *   **CRITICAL:** Ensure NO raw pointer arrays (e.g., `double (*K)[9]`, `double damping[NUM_ACT_SET][6]`) remain.
    *   Verify `Prep` functions populate these vectors correctly from inputs.
*   **Checkpoint:** Confirm strict type safety is maintained.

### 1.2 Integrator Stability & Adaptive Stepping
*   **Target:** `src/CoilDynamics_Defs.cpp`, `src/CRMDYN_Numerical_Integration.hpp`
*   **Audit Task:**
    *   Verify `RK4` implementation is correctly exposed and wired as an option.
    *   Inspect adaptive stepping logic:
        *   Check acceleration threshold logic (`1000 rad/s^2`).
        *   Verify subdivision depth limit (max 4 levels).
        *   Ensure `diverged` flag is correctly propagated up the stack on failure.
    *   Check for hardcoded damping values vs. parameter-driven damping.

### 1.3 AutoDiff Architecture (Option A)
*   **Target:** `src/CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp`, `src/CRM_DynamicsContext_AD_impl.hpp`
*   **Audit Task:**
    *   Verify `DYNNLEquationResidualWithParamsAD` is fully templated on `Scalar`.
    *   Check `eval_output_AD` implementation:
        *   Does it support `NUM_ACT_SET > 1` (dynamic sizing)?
        *   Are `out_gth` (parameter gradients) correctly computed (non-zero)?
    *   Verify `DYNNLEquationOutputJacobianEigenAD` uses the correct scaling constants (`IVALUE_SCALE_M`, etc.).

### 1.4 Multi-Actuator Scalability
*   **Target:** All `src/*.cpp` files.
*   **Audit Task:**
    *   Search for hardcoded loops `for(int i=0; i<1; ...)` that should be `NUM_ACT_SET`.
    *   Verify `CRM_StateVector_Definitions.hpp` defines state offsets dynamically or using `NUM_ACT_SET` macro correctly.
    *   Check if `CRM_CatheterClass.cpp` correctly handles multiple flexible segments/coils in its `step` logic.

### 1.5 Build System & Paths
*   **Target:** `CMakeLists.txt`
*   **Audit Task:**
    *   Verify include paths point to the new `src/numerical` location.
    *   Check if `tests/cpp` is correctly added as a test directory.
    *   Ensure all new `.hpp` files (AD impls) are included in the build target headers.

---

## Phase 2: Python Bindings Audit (`crm_ml_rl/wrappers/`)

**Goal:** Verify safety and correctness of the C++/Python boundary.

### 2.1 Interface Exposure
*   **Target:** `crm_ml_rl/wrappers/crm_bindings.cpp`
*   **Audit Task:**
    *   Verify `step()` vs `step_from_seed()` exposure.
        *   Are docstrings clear about the "consecutive stepping" limitation of `step_from_seed`?
    *   Check `initialize_from_kinematics`:
        *   Does it properly reset internal state?
        *   Does it handle `NUM_ACT_SET > 1` inputs?
    *   Verify `linearize_full_seed_action_from_seed_implicit` exposure.

### 2.2 Memory Safety & Data Types
*   **Target:** `crm_ml_rl/wrappers/crm_bindings.cpp`
*   **Audit Task:**
    *   Check `py::array_t` handling. Are strides respected? (Eigen mapping usually handles this, but explicit casts need care).
    *   Verify `return_value_policy`. Are we returning references to stack variables? (Should be `copy` or `move`).
    *   Check `std::vector` to `numpy` conversions.

### 2.3 Dynamic Sizing Logic
*   **Target:** `crm_ml_rl/wrappers/crm_bindings.cpp` (Linearization functions)
*   **Audit Task:**
    *   Inspect `output_dim` calculation. Is it `3 + 3*num_sets`?
    *   Verify `gx`, `gth`, `A`, `B` matrix resizing logic matches the C++ return types.

---

## Phase 3: ML/RL Integration Audit (`crm_ml_rl/`)

**Goal:** Ensure downstream components use the stabilized API correctly.

### 3.1 Environment Correctness
*   **Target:** `crm_ml_rl/envs/catheter_env.py`
*   **Audit Task:**
    *   **CRITICAL:** Verify the `step()` method uses the `dyn.step()` API for forward simulation, NOT `step_from_seed()`.
    *   Check reset logic: Does it call `initialize_from_kinematics`?
    *   Verify action scaling: Are actions (currents) clamped to safe ranges (e.g., ±20mA)?

### 3.2 Gradient Flow (Torch)
*   **Target:** `crm_ml_rl/wrappers/torch_physics.py`
*   **Audit Task:**
    *   Check `CRMDynamicsStepFunction.backward`.
    *   Are the correct Jacobians (`A`, `B`) used?
    *   Is `ctx.save_for_backward` saving the necessary state?
    *   Verify handling of `None` gradients (e.g., insertion length).

---

## Phase 4: Examples & Scripts Audit

**Goal:** Identify obsolete code and verify "Gold Standard" examples.

### 4.1 "Gold Standard" verification
*   **Target:** `examples/ilqr_catheter_demo.py`
*   **Audit Task:**
    *   Verify usage of `step()` for rollouts.
    *   Verify usage of `linearize_*` only for Jacobians.
    *   Check for any leftover debug prints or commented-out hacks.

### 4.2 Obsolete Code Identification
*   **Target:** `examples/`, `scripts/`
*   **Audit Task:**
    *   Flag scripts that rely on:
        *   Old directory structures (`numerical/` at root).
        *   Old parameter file formats.
        *   `step_from_seed` for loops.
    *   List files to be moved to `archive/` or updated.

---

## Phase 5: Documentation Consistency

**Goal:** Ensure docs match reality.

### 5.1 README & Guides
*   **Target:** `README.md`, `docs/guides/`
*   **Audit Task:**
    *   Verify "Quick Start" commands actually work (dry run mentally).
    *   Check API examples in docs against actual function signatures in `crm_bindings.cpp`.

---

## Execution Plan & Checkpoints

1.  **Run Phase 1 Audit:**
    *   **Action:** Manual code review of `src/` files.
    *   **Deliverable:** `docs/architecture/CODE_AUDIT_PHASE1_CPP.md` (Findings list).
    *   **Test:** Generate a C++ unit test `tests/cpp/test_ghost_value_regression.cpp` if strictly needed.

2.  **Run Phase 2 Audit:**
    *   **Action:** Manual code review of `crm_bindings.cpp`.
    *   **Deliverable:** `docs/architecture/CODE_AUDIT_PHASE2_BINDINGS.md`.

3.  **Run Phase 3 & 4 Audit:**
    *   **Action:** Review Python files.
    *   **Deliverable:** `docs/architecture/CODE_AUDIT_PHASE3_PYTHON.md`.

4.  **Final Report:**
    *   Compile findings into `COMPREHENSIVE_AUDIT_FINDINGS.md`.
    *   Create `REMEDIATION_PLAN.md` for any critical issues found.
