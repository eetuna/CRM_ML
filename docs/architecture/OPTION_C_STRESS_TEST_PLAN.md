# Option C Stress Test & Robustness Validation Plan

**Status:** DRAFT
**Target:** `crm_torch_ext` (C++ PyTorch Extension)
**Goal:** Guarantee numerical stability and gradient correctness across the entire operational workspace before RL integration.

---

## 1. Executive Summary

Historically, the CRM physics engine has suffered from "Divergence Instability," "NaN/Inf values," and "Wrong Gradients" when pushed to edge cases. Now that the gradient logic is fixed and the operator is performant, we must rigorously prove it is **robust**.

This plan defines a systematic "Stress Test" campaign to actively hunt for failures. We will not wait for RL training to crash; we will break the simulator now, in a controlled environment, to identify and guard against unsafe regions.

### Key Objectives
1.  **Workspace Coverage:** Verify stability at maximum deflection and insertion.
2.  **Dynamic Stability:** Ensure rapid changes in control do not cause BVP divergence.
3.  **Gradient Continuity:** Verify gradients remain finite and smooth across state transitions.
4.  **Failure Handling:** Confirm that when the physics *does* fail (it's a physical reality), the extension returns handled errors (zeros/flags) rather than crashing the process.

---

## Phase 1: Static Workspace Mapping (The "Safe Zone")

**Goal:** Identify the static boundaries where the BVP solver reliably converges.

### Task 1.1: Grid Search Sweep
*   **Action:** Create `stress_test_static_grid.py`.
*   **Input Space:**
    *   Currents $I \in [0, 20]$ mA (Step: 1.0 mA)
    *   Insertion $L \in [0, 100]$ mm (Step: 10 mm)
    *   All 3 actuators active simultaneously.
*   **Metric:** BVP Convergence (`localmin == 0`), Tip Position validity (not NaN).
*   **Output:** A 2D/3D heatmap of "Safe Convergence Zones".
*   **Checkpoint:** ST-01 - Stability Map Generated.

### Task 1.2: Extreme Boundary Probe
*   **Action:** Test specifically at physical limits and "corner cases".
    *   Max Current ($I_{max} = 25$ mA)
    *   Zero Current ($I = 0$ - Singularity check)
    *   Max Insertion + Max Bending (Buckling risk)
*   **Acceptance:** Solver must either converge OR return a graceful error code. **NO SEGFAULTS.**
*   **Checkpoint:** ST-02 - Boundaries Probed.

---

## Phase 2: Dynamic Stability (The "Motion Test")

**Goal:** Ensure the simulator can handle state transitions (time-stepping) without exploding.

### Task 2.1: Random Walk Trajectories
*   **Action:** Create `stress_test_random_walk.py`.
*   **Method:**
    *   Initialize at rest.
    *   Apply random control actions $\Delta u \sim N(0, \sigma)$ for 1000 steps.
    *   Run 50 parallel episodes.
*   **Checks:**
    *   Does state drift to Infinity?
    *   Does `next_state` contain NaNs?
    *   Does BVP fail to converge using the *previous* state as a seed?
*   **Checkpoint:** ST-03 - 50,000 Steps Verified.

### Task 2.2: High-Frequency Control Jumps
*   **Action:** Feed "bang-bang" or high-frequency noise inputs (simulating a bad RL policy).
*   **Hypothesis:** Rapid changes often break the BVP solver's initial guess logic.
*   **Success Criteria:** The continuation/homotopy logic in C++ must recover convergence, or report failure cleanly.
*   **Checkpoint:** ST-04 - High-Frequency Robustness Verified.

---

## Phase 3: Gradient Health & Continuity

**Goal:** Verify gradients don't just exist, but are *useful* (smooth and finite).

### Task 3.1: Jacobian Singularity Map
*   **Action:** Re-run the grid search from Task 1.1.
*   **Calculation:** At each point, compute the condition number $\kappa(B)$ of the control Jacobian.
*   **Fail Condition:** $\kappa(B) > 10^6$ implies singularity (loss of control authority).
*   **Output:** Map of "Uncontrollable Regions" (singularities).

### Task 3.2: Finite Difference Consistency Check
*   **Action:** Sample 100 random valid points.
*   **Test:** Compare Analytical Gradients vs. Finite Differences (FD).
*   **Strict Tolerance:** Relative Error < 1%.
*   **Fail Action:** Any point violating this must be logged to `failures.json` for manual inspection.
*   **Checkpoint:** ST-05 - Gradient Integrity Confirmed.

---

## Phase 4: Long-Horizon & Memory

**Goal:** Detect memory leaks and accumulated numerical error.

### Task 4.1: The "Endurance Run" (Memory Leak Check)
*   **Action:** Run a loop of 100,000 steps in a single process.
*   **Monitoring:** Track RAM usage (`psutil`).
*   **Fail Condition:** Memory usage grows linearly with steps.
*   **Checkpoint:** ST-06 - Memory Stability Verified.

### Task 4.2: Reversibility Test
*   **Action:** Move forward 100 steps, then apply negative reversed controls.
*   **Check:** Does the catheter return to (approximately) the start? (Checks physical consistency).
*   **Checkpoint:** ST-07 - Physics Consistency Verified.

---

## Failure Analysis & Reporting Protocol

If a failure (Crash, NaN, Divergence) is found:
1.  **Capture:** Save the exact `(currents, insertion, seed_state)` to `test/failure_cases/case_ID.json`.
2.  **Reproduce:** Create a minimal script `reproduce_failure.py` that loads that JSON and hits the C++ op.
3.  **Fix:** Adjust C++ BVP logic (e.g., improve damping, update continuation steps) to handle it.

---

## Schedule & Priorities

1.  **Phase 1 (Static):** Essential. Do this first to define the "playable area" for RL.
2.  **Phase 2 (Dynamic):** Critical for RL. An RL agent *will* try random walks.
3.  **Phase 3 (Gradients):** High Priority. Bad gradients kill training.
4.  **Phase 4 (Endurance):** Medium Priority.

**Recommendation:** Proceed immediately with **Task 1.1 and 2.1**.

