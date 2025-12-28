# Meticulous Remediation Implementation Plan

**Date:** 2025-12-25
**Parent Doc:** `COMPREHENSIVE_AUDIT_FINDINGS.md`
**Project:** CRM_ML Option A Stabilization

---

## Phase 1: Completing the Physical AD Chain
**Goal:** Enable non-zero gradients for Currents and Insertion Length.

### Task 1.1: Control Input AD Instrumenting
*   **File:** `src/CRM_DynamicsContext_AD.hpp`
    *   **Action:** Add `std::vector<Mat3<Scalar>> catam` to `LearnableParamsAD` to store the Coil-Alignment-Turn-Area Matrices.
*   **File:** `src/CRM_DynamicsContext_AD_impl.hpp`
    *   **Action:** Update `DynamicsContextAD` constructor to populate `catam` from `Params.CoilAlignmentTurnAreaMatrix`.
*   **File:** `src/CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp`
    *   **Action (eval_output_AD_with_params):** 
        1. Locate `// NOTE: currents_ad is not directly used here`.
        2. Replace the usage of the constant `muhat` with a dynamic calculation:
           ```cpp
           Vec3<Scalar> mu = ctx.learnable.catam[0] * currents_ad;
           Mat3<Scalar> muhat_ad = wHat(mu);
           ```
        3. Pass `muhat_ad` to `CoilDynamicsDispatch`.

### Task 1.2: Differentiable Insertion Length
*   **File:** `src/CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp`
    *   **Action (interpolate_fcum):** Change signature to `interpolate_fcum(..., const Scalar& Li_ad, const double s)`. Use `Li_ad` to calculate the integration `lambda`.
    *   **Action (DYNNLEquationControlJacobianEigenAD):** 
        1. Increase `u_ad` dimension from 3 to 4. 
        2. Set `u_ad(3) = insertion_length`.
        3. Pass `u_ad(3)` into the residual and output functions.
*   **File:** `crm_ml_rl/wrappers/crm_bindings.cpp`
    *   **Action:** Update `linearize_full_seed_action_from_seed_implicit` to extract the 4th column of the control Jacobian as `grad_insertion`.

---

## Phase 2: Forward Stability Synchronization
**Goal:** Ensure the `step()` API is as robust as the AD engine.

### Task 2.1: Porting Adaptive RK4
*   **File:** `src/CoilDynamics_Defs.cpp`
    *   **Action:** Implement `rk4_step_adaptive_double` (non-templated version of the recursive subdivision logic).
    *   **Action:** Implement `is_acceleration_safe_double` to detect torque spikes.

### Task 2.2: Updating Legacy Dispatcher
*   **File:** `src/CoilDynamics_Defs.cpp`
    *   **Action:** Modify the `CoilDynamicsRK4` loop. Replace the basic `RK4_coildyn` call with the new `rk4_step_adaptive_double`.
    *   **Action:** Ensure the `out_diverged` flag is correctly set and returned to Python if max subdivisions are reached.

---

## Phase 3: Solver Homotopy (Robust `step_from_seed`)
**Goal:** Eliminate `localmin=3` divergence when jumping into moving seed states.

### Task 3.1: Damping-Compensated Initial Guess
*   **File:** `crm_ml_rl/wrappers/crm_bindings.cpp`
    *   **Action (step_from_seed):** Before calling `DynamicsBVP`, compute a "Physical Heuristic Guess":
        ```cpp
        nL_guess_local[j] += damping[j].linear * v_seed[j];
        mL_guess_local[j] += damping[j].angular * w_seed[j];
        ```

### Task 3.2: Internal Velocity Continuation Loop & Warm-Up
*   **File:** `crm_ml_rl/wrappers/crm_bindings.cpp`
    *   **Action:** If `DynamicsBVP` returns `localmin != 0`, enter recovery:
        1. **Continuation Ramp:** Gradually ramp velocity $v$ and $w$ from 0 to 100% over 5 internal iterations.
        2. **Converged-Guess Chain:** Carry over the solved $m_L, n_L$ from each ramp step to the next as the `initial_guess`.
        3. **Multi-Pass Refinement (Warm-Up):** At 100% velocity, perform **two additional calls** to `DynamicsBVP` to allow the trust-region solver to refine its internal Jacobian map.
        4. **Failure Recovery (Fallback):** If homotopy fails, perform an internal static reset (velocities=0), solve, and then retry the ramp once.

### Task 3.3: Multi-Actuator Initialization Validation
*   **File:** `crm_ml_rl/wrappers/crm_bindings.cpp`
    *   **Action (initialize_from_kinematics):** Update the `det(R)` sanity check loop through **all** actuators, not just the first.
    *   **Action:** Apply Gram-Schmidt orthonormalization to any $R$ matrix that drifts beyond $10^{-3}$ tolerance.

---

## Phase 4: Multi-Actuator Generalization
**Goal:** Remove 6D hardcoding and support scalable actuator counts.

### Task 4.1: Dynamic Binding Resizing
*   **File:** `crm_ml_rl/wrappers/crm_bindings.cpp`
    *   **Action:** Search and replace hardcoded `Eigen::Matrix<double, 6, 1>` with `Eigen::VectorXd(3 + 3 * num_sets)`.
    *   **Action:** In `step_from_seed`, update return dict: rename `tip_velocity` to `coil_velocities` (NumPy array of shape `(num_sets, 3)`).

### Task 4.2: Recursive State Chaining (AD)
*   **File:** `src/CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp`
    *   **Action (eval_output_AD):** Replace the `// TODO` with a **Recursive Chaining Loop**. 
        *   Iterate from `segi = NUM_SEGMENTS-1` down to `0`.
        *   Propagate the state through all segments and store velocities for ALL actuators in the output vector `y`.

### Task 4.3: Torch Wrapper Alignment
*   **File:** `crm_ml_rl/wrappers/torch_physics.py`
    *   **Action:** Update `CRMDynamicsStepFunction` forward/backward to handle variable-sized output and correctly index the new `grad_insertion` column.

---

## Phase 5: Documentation & Guide Synchronization
**Goal:** Align all repository documentation with the new stabilized and generalized API.

### Task 5.1: API Documentation Alignment
*   **File:** `README.md`, `docs/guides/USAGE_GUIDE.md`
    *   **Action:** Update all Python code snippets to use `coil_velocities`.
    *   **Action:** Update the "Differentiable Simulator" section to confirm that **Currents** and **Insertion Length** gradients are active.

---

## Final Verification Checkpoints
1.  **Gradients:** `pytest tests/test_torch_physics_gradients.py` (Must now return non-zero `currents.grad`).
2.  **Stability:** Re-run `step()` on historical failing cases (Must pass with zero NaNs).
3.  **Continuity:** `debug_consecutive_stepping.py` must succeed on Step 2.
4.  **Multi-Actuator:** `verify_multi_actuator_output.py` must return correctly sized Jacobians.