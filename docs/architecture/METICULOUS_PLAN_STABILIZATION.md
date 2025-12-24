# Meticulous Plan: Stabilization & Validation (Option A)

**Status:** Feature-Complete (Multi-actuator AD + Full Output AD).
**Goal:** Resolve numerical instability preventing valid iLQR rollouts and fix gradient precision.

**References:**
- Original Plan: `docs/architecture/PLAN_end_to_end_differentiable_simulator_options_A_B_C.md`
- Claude's Plans: `docs/architecture/CLAUDE_OPTION_A_IMPLEMENTATION_PLAN.md`
- Option A Review: `docs/architecture/GEMINI_OPTION_A_REVIEW_REPORT.md`

**Current Blockers (Leftovers from Claude's Implementation):**
1.  **Numerical Instability:** Forward dynamics diverge on the second consecutive step in control loops (Failed in Claude's Phase 3).
2.  **Gradient Mismatch:** `gradcheck` tests fail due to tolerance/scaling issues (Leftover from Claude's Phase 4).

---

## Step 1: Stability Root Cause Analysis (Immediate Priority)
**Corresponds to:** **Task A1.6 (Stabilization)** & **Task A5 (Experiments - Prerequisites)**
**Source:** Claude's Phase 3 (Unfinished - "Numerical Instability").
**Objective:** Fix the "Second Step Divergence" to enable valid trajectory rollouts for iLQR.

### Task 1.1: Create Reproduction Script
**File:** `examples/debug_consecutive_stepping.py`
**Goal:** Deterministically reproduce `nan` or `diverged=True` on Step 2.

*   **Action:** Implement `run_debug_stepping(integrator="abm4")` that:
    1.  Initializes dynamics at `insertion=50.0mm`.
    2.  Applies constant currents `[0.01, 0.0, 0.0]`.
    3.  Calls `step_from_seed` (Step 1).
    4.  Logs full output state: `v`, `w`, `mL`, `nL`, `converged`, `diverged`.
    5.  Calls `step_from_seed` (Step 2) using Step 1 outputs.
    6.  Logs full output state.
*   **Success Criteria:** Script prints "Step 2: converged=False" or "diverged=True".

### Task 1.2: Test Integrator Stability (ABM4 vs RK4)
**Goal:** Verify if RK4 fixes the divergence.

*   **Action:**
    1.  Run `python examples/debug_consecutive_stepping.py` (Default: ABM4).
    2.  Modify script or add flag to run with `dyn.set_integrator("rk4")`.
    3.  Run again.
*   **Hypothesis:** ABM4 (multistep) requires history (`x_nm1`, `x_nm2`...) which might be initialized improperly on the second step. RK4 (single-step) is history-free and should be robust.

### Task 1.3: Analyze State Physicality
**Goal:** Determine *why* it diverged.

*   **Action:** Inspect logs from Task 1.1.
    *   **Check:** Are Step 1 output velocities physical? ($|v| < 10$ m/s, $|w| < 100$ rad/s).
    *   **Check:** Are `mL`/`nL` boundary variables within typical ranges ($|n| < 10$ N, $|m| < 100$ mNm)?
    *   **Check:** Does `diverged=True` trigger even if `converged=True`? (Indicates solver found a root, but IVP exploded).

---

## Step 2: Gradient Precision Tuning
**Corresponds to:** **Task A3 (Torch Wrapper - Validation)** & **Task A4 (Tests)**
**Source:** Claude's Phase 4 (Unfinished - `gradcheck` XFAILs).
**Objective:** Verify that Analytical AD gradients match Finite Differences (FD).

### Task 2.1: Analyze Gradient Mismatch
**File:** `tests/test_torch_gradcheck.py`
**Goal:** Quantify the error.

*   **Action:** Run `pytest tests/test_torch_gradcheck.py -v -rP`.
*   **Check:** Look at the failure message for `test_dynamics_gradcheck_currents`.
    *   Is relative error $\approx 10^{-7}$? -> Float precision issue.
    *   Is relative error $> 10^{-2}$? -> Scaling/Logic bug.

### Task 2.2: Fix Scaling Constants
**File:** `crm_ml_rl/wrappers/crm_bindings.cpp` & `src/CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp`
**Goal:** Ensure AD and FD use identical scaling.

*   **Action:** Verify `IVALUE_SCALE_M` and `IVALUE_SCALE_N` are applied:
    1.  In `DYNNLEquationResidualEigenAD` (AD path).
    2.  In `linearize_full_seed_action_from_seed` (FD path).
*   **Action:** Adjust `atol` in `test_torch_gradcheck.py` to `1e-5` (realistic for 64-bit physics) if error is small.

---

## Step 3: The "Money Plot" (Closed-Loop Control)
**Corresponds to:** **Task A5 (Experiments - Controller Demo)**
**Source:** Claude's Phase 3 (Unfinished - "Demo Functional but Unstable").
**Objective:** Deliver the final proof of the paper's claims (iLQR Demo).

### Task 3.1: Tune iLQR Parameters
**File:** `examples/ilqr_catheter_demo.py`
**Goal:** Achieve stable convergence once simulator is fixed.

*   **Action:**
    1.  Set `R` (control cost) matrix diagonal to `1e-2` or higher (dampen updates).
    2.  Set `Q` (state cost) matrix to focus on Tip Position (`1.0`) and ignore velocity (`0.0`).
    3.  Set `max_iterations = 10` for fast debug cycles.

### Task 3.2: Generate Validation Artifacts
**Goal:** Visual proof of control.

*   **Action:**
    1.  Run `python examples/ilqr_catheter_demo.py --method implicit`.
    2.  Output: `ilqr_trajectory.png` showing tip trace vs target.
    3.  Output: `convergence.json` showing Cost vs Iteration.

---

## Step 4: Documentation & Handoff
**Corresponds to:** **Task A5 (Experiments - Reporting)**
**Objective:** Clean up for future users.

*   **Task 4.1:** Update `README.md` with "How to run the stable iLQR demo".
*   **Task 4.2:** Document `CRM_DYN_LINEARIZATION_METHOD=implicit` variable.
*   **Task 4.3:** Mark Option A as "Recommended" in architecture docs.