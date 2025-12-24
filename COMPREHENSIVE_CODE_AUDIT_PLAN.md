# Comprehensive Code Audit Plan

**Objective:** Meticulously verify the implementations of Phase 1, Phase 2, Phase 3, and Phase 4 against their original architectural plans. This audit aims to confirm adherence to requirements, identify any deviations, and document the current state of the codebase, particularly regarding the reported numerical instability.

**Reference Documents:**
*   **Phase 1 Plan:** `/home/vscode/.claude/plans/glowing-wandering-river.md`
*   **Phases 2-3-4 Plan:** `/home/vscode/.claude/plans/golden-inventing-tower.md`

---

## Part 1: Phase 1 Audit - Multi-Actuator Generalization

**Goal:** Verify the removal of single-actuator limitations and correct implementation of multi-actuator logic.

### 1.1 Source Code Verification
**File:** `src/CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp`

*   [ ] **Static Assertions Removal:**
    *   Verify `DYNNLEquationResidualEigenAD` (approx. line 886): Ensure `static_assert(NUM_ACT_SET == 1)` is removed.
    *   Verify `DYNNLEquationResidualWithParamsAD` (approx. line 1057): Ensure `static_assert` is removed.
    *   Verify `DYNNLEquationResidualWithControlsAD` (approx. line 1213): Ensure `static_assert` is removed.

*   [ ] **Actuator Loop Generalization:**
    *   Inspect residual functions for `for (int actno = 0; actno < NUM_ACT_SET; ++actno)` loops replacing hardcoded `act0`.
    *   Verify `m_L` and `n_L` unpacking uses `NUM_ACT_SET * 6` sizing (not fixed 6).
    *   Check pattern: `m_L_all[actno]` and `n_L_all[actno]` indexing.

*   [ ] **Control Jacobian Sizing:**
    *   Verify `DYNNLEquationControlJacobianEigenAD` signature accepts `Eigen::VectorXd` (or dynamic size) for currents, not `Vector3d`.
    *   Verify `u_ad` matches `NUM_ACT_SET * 3` dimensions.

*   [ ] **Force Propagation:**
    *   Verify `net_mL` / `net_nL` calculation propagates forces correctly between actuator stages: `net_nL = n_L_all[actno] - n_L_all[actno+1]` (or equivalent logic for final stage).

### 1.2 Configuration Verification
**File:** `src/CRM.hpp`

*   [ ] Verify `NUM_ACT_SET` definition exists and defaults to `1` (ensuring backward compatibility).

---

## Part 2: Phase 2 Audit - Full AD for Output Mapping

**Goal:** Confirm the replacement of Finite Difference (FD) gradients with Automatic Differentiation (AD) for the output mapping `y = g(x, θ)`.

### 2.1 Source Code Verification
**File:** `src/CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp`

*   [ ] **Templated Forward IVP:**
    *   Verify existence of `CRMFlexForward_passAD` (or similar name) and that it is templated on `<typename Scalar>`.
    *   Verify `CRMIVP_DYN_AD` is templated and supports AD types.

*   [ ] **Output Jacobian Helper:**
    *   Verify existence of `eval_output_AD<Scalar>`.
    *   Verify existence of `DYNNLEquationOutputJacobianEigenAD` wrapper.
    *   Check that `autodiff::jacobian` is used inside the wrapper.

**File:** `crm_ml_rl/wrappers/crm_bindings.cpp`

*   [ ] **Bindings Integration:**
    *   Locate `linearize_full_seed_action_from_seed_implicit`.
    *   **Crucial:** Verify that the manual FD loops (calculating `gx` and `gθ` via perturbation) have been *removed* or *commented out*.
    *   Verify they are replaced by a call to `DYNNLEquationOutputJacobianEigenAD`.

---

## Part 3: Phase 3 Audit - iLQR Controller & Instability

**Goal:** Verify iLQR algorithm structure and investigate the reported numerical instability in consecutive steps.

### 3.1 Algorithm Verification
**File:** `examples/ilqr_catheter_demo.py`

*   [ ] **Controller Class (`iLQRController`):**
    *   Verify `backward_pass`: Checks for computation of `Q_xx`, `Q_uu`, `Q_ux` and solving `K = -Q_uu^-1 Q_ux`.
    *   Verify `forward_pass`: Checks for line search loop and state update `u = u_nom + alpha*k + K(x - x_nom)`.
    *   Verify `cost_function`: Checks for quadratic tracking cost + control regularization.

*   [ ] **Demo Functions:**
    *   Verify `reaching_demo` setup (targets, horizon).
    *   Verify `tracking_demo` trajectory generation.

### 3.2 Instability Investigation Plan (Current Priority)
**File:** `examples/debug_consecutive_stepping.py` (Created during session)

*   [ ] **Divergence Reproduction:**
    *   Confirm Step 1 succeeds (`converged=True`).
    *   Confirm Step 2 fails (`converged=False` or `diverged=True`).
    *   Audit inputs passed to Step 2: Are `next_v`, `next_w`, etc., from Step 1 valid (non-NaN, reasonable magnitudes)?

*   [ ] **Root Cause Analysis (Code Level):**
    *   Audit `src/CRM_DynamicsContext.hpp` or relevant integrator state management: Is internal state reset correctly between `step_from_seed` calls?
    *   Check `step_from_seed` implementation in bindings: Does it re-initialize the IVP solver correctly for the second step?

---

## Part 4: Phase 4 Audit - Python Wrapper Polish

**Goal:** Verify documentation clarity and test coverage for PyTorch integration.

### 4.1 Documentation Verification
**File:** `crm_ml_rl/wrappers/torch_physics.py`

*   [ ] **Gradient Documentation:**
    *   Check `CRMDynamicsStepFunction.backward`.
    *   Verify docstring explicitly states `insertion_length` is *not* differentiated.
    *   Verify code returns `None` (or equivalent) for the insertion length gradient.
    *   Verify shape correctness for other gradients (e.g., `grad_inputs[:, 3:4]` fix mentioned in summary).

### 4.2 Test Verification
**File:** `tests/test_torch_gradcheck.py`

*   [ ] **Gradcheck Tests:**
    *   Verify `test_fk_gradcheck` exists and passes.
    *   Verify `test_dynamics_gradcheck` exists.
    *   Check for `xfail` markers on tests that require precision tuning (as noted in implementation summary).
    *   Verify `test_insertion_length_gradient_is_none` exists.

---

## Execution Strategy

1.  **Static Analysis:** Manually inspect the file content for the checklist items above.
2.  **Dynamic Analysis (Phase 3 Focus):**
    *   Run `examples/debug_consecutive_stepping.py` (Already done - confirmed failure).
    *   Run `tests/test_torch_gradcheck.py` to confirm Phase 4 status.
    *   Run `pytest tests/test_multi_actuator_ad.py` (if it exists) or check build config for Phase 1.

This plan provides a structured approach to validate the codebase state against the architectural intent.
