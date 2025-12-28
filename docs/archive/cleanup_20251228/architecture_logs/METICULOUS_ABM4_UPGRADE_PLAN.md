# Meticulous Implementation Plan: Upgrading Option C to ABM4

**Goal:** Enable the stateless C++ Extension (`crm_torch_ext`) to use the high-accuracy **ABM4 Integrator** by explicitly managing the required history state tensors in Python. This will close the convergence gap with the Python Wrapper.

---

## Phase 1: Verification & Feasibility (The "Can we do it?" Check)

**Objective:** Confirm that the Core C++ solver (`DYNSolverIVP`) exposes the internal history variables needed for ABM4.

*   **Task 1.1:** Audit `src/CRM_BVPIVP_APIDeclarations.hpp` and `src/CRMDYN.hpp`.
    *   *Subtask:* Identify the exact data structure used for history (`xdot_nm1`, `xdot_nm2`, `xdot_nm3`).
    *   *Subtask:* Determine the size of the history tensor (likely `3 * StateSize`).
*   **Task 1.2:** Audit `src/CRM_IVP_NumericalIntegrationTemplates.hpp`.
    *   *Subtask:* Verify that `ABM4` template function accepts history as an argument (or can be modified to do so).
    *   *Constraint Check:* If the Core solver *hides* history inside a local variable, we might need to modify the Core `src/` to expose it. (This is a major checkpoint).

---

## Phase 2: C++ Implementation (The "Heavy Lifting")

**Objective:** Update the Extension to accept, use, and return history tensors.

*   **Task 2.1:** Define Tensor Layout.
    *   *Plan:* Create a single `history` tensor of shape `[3, NUM_STATES]`.
    *   *Row 0:* $t_{-1}$ derivative.
    *   *Row 1:* $t_{-2}$ derivative.
    *   *Row 2:* $t_{-3}$ derivative.
*   **Task 2.2:** Update `crm_step_forward` signature.
    *   *Input:* Add `torch::Tensor history`.
    *   *Logic:*
        *   If `history` is empty/zero -> Use RK4 (Initialization phase).
        *   If `history` is populated -> Use ABM4.
    *   *Output:* Return `next_history` tensor alongside state.
*   **Task 2.3:** Update `crm_step_backward`.
    *   *Math Check:* Gradients for ABM4 are more complex (depend on history).
    *   *Simplification:* For the first iteration, we might treat history as constant (gradient stops) to avoid exploding complexity, relying on Implicit Differentiation of the final state.

---

## Phase 3: Python Integration (The "Manager")

**Objective:** Update `TorchCRMPhysics` to hold the history.

*   **Task 3.1:** Update `TorchCRMPhysics.__init__`.
    *   Initialize `self.history = None` or zero tensor.
*   **Task 3.2:** Update `TorchCRMPhysics.step`.
    *   Pass `self.history` to C++.
    *   Receive `next_history`.
    *   Update `self.history = next_history.detach()` (to manage graph size, though we need to be careful with BPTT).
*   **Task 3.3:** Reset Logic.
    *   Ensure `reset()` clears the history tensor.

---

## Phase 4: Validation (The "Parity Check")

**Objective:** Prove that the Extension now matches the Wrapper's convergence.

*   **Task 4.1:** Re-run `stress_test_trajectories.py`.
    *   *Expectation:* The "Circle" and "Lemniscate" trajectories should now converge (`localmin=0`) even for the 20ms steps, because ABM4 will provide the accurate prediction.
*   **Task 4.2:** Performance Benchmark.
    *   Measure overhead of passing the extra tensor. (Expected: Negligible).

---

## Checkpoints

1.  **[ ] Core Exposure:** Can we actually feed history into `DYNSolverIVP` without rewriting the Core library? (Crucial Go/No-Go).
2.  **[ ] Build Success:** Does the modified extension compile?
3.  **[ ] Convergence:** Does `localmin` finally stay 0?
