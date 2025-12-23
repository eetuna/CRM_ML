# Option A Implementation Plan: "Paper-Ready" Differentiable Simulator

**Status:** Ready for Implementation
**Goal:** Finalize Option A (C++ AD + Implicit Differentiation) to be robust, multi-actuator capable, and experimentally validated.
**Input:** Based on `GEMINI_OPTION_A_REVIEW_REPORT.md` and `PLAN_end_to_end_differentiable_simulator_options_A_B_C.md`.

---

## Phase 1: Multi-Actuator Generalization (Task A2)
**Objective:** Remove the single-actuator limitation to support complex catheter designs.
**Files:** `src/CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp`

1.  **Remove Static Assertions**
    *   [ ] Locate and remove `static_assert(NUM_ACT_SET == 1, ...)` in `DYNNLEquationResidualEigenAD` and `DYNNLEquationResidualWithControlsAD`.
    *   [ ] Ensure `DynamicsContextAD` is being used correctly to access actuator parameters dynamically (via `ctx.actuators[i]`) rather than assuming index 0.

2.  **Generalize Residual Loop**
    *   [ ] In `DYNNLEquationResidualEigenAD`, refactor the loop that iterates `segi = NUM_SEGMENTS - 1` down to 0.
    *   [ ] Ensure the logic handling `actno` (actuator index) correctly maps to `ctx.actuators[actno]` for any `NUM_ACT_SET`.
    *   [ ] Verify `net_mL` and `net_nL` propagation logic handles the multi-actuator chain correctly (passing forces/moments between stacked actuators).

3.  **Update Jacobian Helpers**
    *   [ ] In `DYNNLEquationJacobianEigenAD`, ensure the input vector `x` sizing logic accounts for `NUM_ACT_SET * 6` (mL + nL per actuator).
    *   [ ] In `DYNNLEquationControlJacobianEigenAD`, update input sizing to `NUM_ACT_SET * 3` (currents).

4.  **Verification**
    *   [ ] Create a unit test `tests/cpp/test_multi_actuator_ad.cpp` (or Python equivalent) that instantiates a 2-segment/2-actuator system and verifies the residual runs without error.

---

## Phase 2: Full AD for Output Mapping (Task A1)
**Objective:** Replace Finite Difference (FD) gradients for the output map $y = g(x^*, \theta)$ with AD gradients. Currently, `crm_bindings.cpp` uses FD for `gx` and `gθ` because the forward IVP is not templated for AD.
**Files:** `src/CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp`, `crm_ml_rl/wrappers/crm_bindings.cpp`

1.  **Template Forward Integration (IVP)**
    *   [ ] Port/Template `CRMFlexForward_pass` to `CRMFlexForward_passAD` in `src/CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp`.
    *   [ ] Port/Template `CRMIVP_DYN` to `CRMIVP_DYN_AD`.
    *   [ ] Implement `DYNSolverIVP_AD` that utilizes the above and the existing `CoilDynamicsRK4` (which is already templated!).
    *   *Note:* This allows computing the tip state $y$ from the solved boundary variables $x^*$ using autodiff types.

2.  **Expose Output Jacobian**
    *   [ ] Create a helper `DYNNLEquationOutputJacobianEigenAD` that:
        1.  Takes `x_star` (boundary vars), `u` (currents), `theta` (params).
        2.  Runs `DYNSolverIVP_AD`.
        3.  Returns Jacobians $\partial y / \partial x^*$ and $\partial y / \partial \theta$.

3.  **Integrate into Bindings**
    *   [ ] Update `linearize_full_seed_action_from_seed_implicit` in `crm_bindings.cpp`.
    *   [ ] Replace the FD loops for `gx` (output w.r.t boundary) and `gθ` (output w.r.t params) with calls to the new AD helper.
    *   [ ] *Result:* The implicit differentiation formula $dy/d\theta = \partial g/\partial \theta + (\partial g/\partial x^*) (-J_{xx}^{-1} J_{x\theta})$ will now be fully AD-based.

---

## Phase 3: Control Experiment Validation (Task A5)
**Objective:** Prove the utility of the differentiable simulator with a closed-loop control demo.
**Files:** `examples/ilqr_catheter_demo.py`

1.  **Implement iLQR**
    *   [ ] Create a script that defines a target tip position $p_{target}$.
    *   [ ] Implement the iLQR backward pass using `CRMDynamics.linearize_full_seed_action_from_seed_implicit` to get $A, B$ matrices.
    *   [ ] Implement the forward rollout using `CRMDynamics.step_from_seed`.

2.  **Comparison Experiment**
    *   [ ] Run optimization using **Implicit AD** gradients.
    *   [ ] Run optimization using **Full FD** gradients (`linearize_full_seed_action_from_seed`).
    *   [ ] **Metrics:**
        *   Wall-clock time per iteration.
        *   Convergence rate (error vs iteration).
        *   Stability (did it diverge?).

---

## Phase 4: Python Wrapper Polish (Task A3)
**Objective:** Clean up the Python/Torch interface for end-users.
**Files:** `crm_ml_rl/wrappers/torch_physics.py`

1.  **Insertion Length Gradients**
    *   [ ] In `CRMDynamicsStepFunction.backward`, handle `insertion_length`. Even if we don't compute the gradient (return `None` or 0.0), explicitly document/structure it rather than ignoring it.

2.  **Gradcheck**
    *   [ ] Add `tests/test_torch_gradcheck.py`.
    *   [ ] Use `torch.autograd.gradcheck` to verify the `CRMDynamicsStepFunction` against numerical perturbation on a stable, single-step example.

---

## Summary of Work Order
1.  **Phase 1** first to ensure the architecture supports the real hardware configuration (multi-actuator).
2.  **Phase 3** (Control Demo) second. Even with FD parts in the output map, the AD residual (Task 1.5) provides the heavy lifting. We can prove value *now* before optimizing the output map.
3.  **Phase 2** (Full AD) third. This is an optimization/accuracy boost to remove the final FDs.
4.  **Phase 4** (Polish) last.
