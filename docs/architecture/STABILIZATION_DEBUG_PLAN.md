# Stabilization & Validation Plan: Resolving iLQR Divergence

**Objective:** Isolate and resolve the numerical instability in consecutive dynamics stepping and verify the precision of Phase 2 AD gradients to enable successful iLQR trajectory optimization.

---

## Phase 1: Deterministic Reproduction & Isolation
**Goal:** Create a minimal reproduction of the "Second Step Divergence" identified in Phase 3.

*   **Task 1.1: Implement `debug_consecutive_stepping.py`**
    *   Create `/workspaces/catheter/CRM_ML/examples/debug_consecutive_stepping.py`.
    *   Initialize `CRMDynamics` with `CatheterParameterSet_1_dyn.txt`.
    *   Perform "Step 1" using `step_from_seed` with a small current (0.01A).
    *   Extract the `next_` state (v, w, p, R, xf, mL, nL) and feed it into "Step 2".
    *   Log `converged` and `diverged` flags for both steps.
*   **Task 1.2: Execute Baseline Test (ABM4)**
    *   Run the script using the default `abm4` integrator.
    *   Confirm if Step 1 succeeds while Step 2 returns NaNs or fails to converge.

---

## Phase 2: Integrator Sensitivity Analysis
**Goal:** Determine if the multi-step nature of the legacy integrator is the root cause.

*   **Task 2.1: Execute Comparison Test (RK4)**
    *   Modify the debug script to call `dyn.set_integrator("rk4")`.
    *   **Analysis:** If RK4 succeeds where ABM4 fails, the issue is likely the lack of derivative history initialization in the `step_from_seed` API. RK4 is "history-free" and preferred for control rollouts.

---

## Phase 3: Numerical State & Orthonormality Audit
**Goal:** Inspect the physical validity of the state variables between steps.

*   **Task 3.1: Orthonormality Check**
    *   Verify that the rotation matrix `R` returned after Step 1 satisfies $R^T R \approx I$.
*   **Task 3.2: Magnitude Audit**
    *   Check if angular velocity `w` or boundary forces `nL` spike to non-physical values ($|w| > 1000$ rad/s) before the divergence.
*   **Task 3.3: Jacobian Condition Number Audit**
    *   Calculate and log the condition number of the residual Jacobian $J_{xx}$ during the debug rollout.
    *   **Analysis:** If the condition number exceeds $10^{12}$, the implicit update is numerically garbage, indicating the system is too stiff for the current integration parameters.

---

## Phase 4: Multi-Actuator & AD Gradient Verification
**Goal:** Verify that the Phase 1 (Multi-Actuator) and Phase 2 (Full AD) implementations are numerically precise.

*   **Task 4.1: Multi-Actuator Sanity**
    *   Run the debug script with `NUM_ACT_SET=2` (if hardware config allows) to ensure the generalized indexing logic from Phase 1 is stable.
*   **Task 4.2: Run `gradcheck` for Phase 2 AD**
    *   Execute `pytest tests/test_torch_gradcheck.py`.
    *   **Audit:** If relative error is $> 10^{-2}$, verify that `IVALUE_SCALE_M` and `IVALUE_SCALE_N` are applied identically in both the forward C++ path and the `DYNNLEquationOutputJacobianEigenAD` implementation.
*   **Task 4.3: Performance Benchmark (Phase 2 Verification)**
    *   Compare the time taken for a 30-step linearization using the new Full AD vs. the previous FD-based output mapping. 
    *   **Success Criteria:** Linearization for a 30-step horizon should drop from ~2 minutes to under 10 seconds.
*   **Task 4.4: Torch Wrapper Gradient Audit (Phase 4 Verification)**
    *   Verify that `CRMDynamicsStepFunction.backward` correctly returns `None` for the `insertion_length` gradient.
    *   Confirm that the `grad_inputs[:, 3:4]` indexing fix (mentioned in the Phase 4 summary) correctly handles the state vector slicing during the backward pass.
*   **Task 4.5: Granular Jacobian Component Audit**
    *   Modify `debug_consecutive_stepping.py` to call `linearize_full_seed_action_from_seed_implicit(..., return_debug=True)`.
    *   Compare the returned `Jxx` (AD) and `Jxx_fd` (Finite Difference) matrices.
    *   **Success Criteria:** The relative error between AD and FD components of the residual Jacobian should be $< 10^{-7}$ for double precision.
*   **Task 4.6: FD Epsilon Sensitivity Check**
    *   Verify that the FD baselines used in `gradcheck` are stable by varying `eps` (e.g., testing `1e-4`, `1e-5`, and `1e-6`). 
    *   If the gradients change significantly with `eps`, the system may be too stiff for standard FD, further justifying the need for the Phase 2 AD implementation.
*   **Task 4.7: Implement Direct Output Jacobian ($g_\theta$) AD Logic**
    *   **Context:** `DYNNLEquationOutputJacobianEigenAD` currently sets `out_gth` to zero.
    *   **Action:** Extend the AD implementation to accept `currents` and `seed` as AD variables and compute the direct partial derivatives $\partial y / \partial \theta$.
*   **Task 4.8: Generalize Parameter Jacobian Bindings**
    *   **Context:** `compute_parameter_jacobian` in `crm_bindings.cpp` is restricted to `num_sets == 1`.
    *   **Action:** Remove this restriction and ensure the multi-actuator logic from Phase 1 is fully exposed for system identification.
*   **Task 4.9: Generalize Output Vector for Multi-Actuator Support**
    *   **Context:** `eval_output_AD` currently returns a fixed 6D vector (tip + 1st actuator).
    *   **Action:** Scale the output vector $y$ and Jacobians $g_x, g_\theta$ to include velocities for all `NUM_ACT_SET` actuators.
*   **Task 4.10: Validate Damping Propagation in AD Path**
    *   **Context:** Damping is a critical stability parameter.
    *   **Action:** Verify that values set via `dyn.set_damping()` in Python correctly propagate into the `DynamicsContextAD` and are used by the templated AD integrators.
*   **Task 4.11: Add Compile-time vs. Runtime Actuator Count Guard**
    *   **Action:** Implement checks in `crm_bindings.cpp` to ensure that input currents and seed state shapes match the compile-time `NUM_ACT_SET`, preventing silent errors in multi-actuator scenarios.

---

## Phase 5: Controller Stabilization & Tuning
**Goal:** Finalize the iLQR demo once the underlying dynamics are stable.

*   **Task 5.1: Regularization Tuning**
    *   In `examples/ilqr_catheter_demo.py`, increase the `R` matrix (control cost) to penalize aggressive current changes.
*   **Task 5.2: Implement "Settling" Steps**
    *   Before starting iLQR, run the dynamics for 5-10 steps with zero currents to allow the catheter to reach physical equilibrium from its kinematic initialization.
*   **Task 5.3: iLQR Robustness (Backtracking on Divergence)**
    *   Modify the `iLQRController._forward_pass` to check the `converged` and `diverged` flags from the dynamics result.
    *   **Action:** If a step returns `diverged=True`, treat the cost as infinite and force the line search to backtrack to a smaller $\alpha$.
*   **Task 5.4: Control Input Scaling**
    *   Implement a normalization layer for currents (e.g., mapping $[-1, 1]$ to the physical range $[-0.5, 0.5]$ Amps).
    *   **Goal:** Improve the conditioning of the $B$ matrix ($\partial y / \partial u$) to prevent the optimizer from taking "wild" steps.
*   **Task 5.5: Fix iLQR State Jacobian Mapping ($A_t$)**
    *   **Context:** `iLQRController` currently uses `A_t = B_t` as a placeholder.
    *   **Action:** Map the high-dimensional `seed_state` Jacobian (the `A` matrix from `linearize_full_seed_action_from_seed_implicit`) to the 6D tip state to accurately model error propagation.

---

## Phase 6: Documentation & Final Artifacts
**Goal:** Clean up the codebase and provide visual proof of stabilization.

*   **Task 6.1: Generate Proof of Control**
    *   Run the finalized iLQR demo.
    *   **Output:** `ilqr_trajectory.png` showing tip trace vs. target.
    *   **Output:** `convergence.json` documenting Cost vs. Iteration.
*   **Task 6.2: Update README.md**
    *   Add a section: "How to run the stable iLQR demo."
    *   Document the `CRM_DYN_LINEARIZATION_METHOD=implicit` environment variable.
*   **Task 6.3: Mark Option A as Recommended**
    *   Update `docs/architecture/PLAN_end_to_end_differentiable_simulator_options_A_B_C.md` to reflect that Option A is now the stable, verified standard.

---

## Success Criteria
1.  **Stability:** `debug_consecutive_stepping.py` completes 10 consecutive steps without NaNs using RK4.
2.  **Precision:** `gradcheck` passes with a relative error $< 10^{-5}$ for the dynamics step.
3.  **Performance:** The iLQR "Point Reaching" demo converges to within 2mm of the target.