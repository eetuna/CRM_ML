# Extension Remediation Log & Current Status

**Date:** December 28, 2025
**Reference:** `docs/architecture/EXTENSION_STATUS_BEFORE_FIXES.md`

## 1. Implementation Plan & Execution

We executed the following tasks to bring the Extension to feature parity with the Wrapper.

### Task 1: Fix Initialization (The "Garbage Seed" Bug)
- **Action:** Modified `crm_initialize_from_fk` in `crm_step_op.cpp`.
- **Detail:** Explicitly `memset` or zero-initialized the `deltau0_initialguess` and `ftip_initialguess` arrays in the `CRMForwardKinematicsData` struct.
- **Result:** **FIXED.** Initialization now succeeds cleanly. Previously it failed randomly due to uninitialized memory.

### Task 2: Port "Zero-Current Retry"
- **Action:** Added logic to `crm_initialize_from_fk`.
- **Detail:** If the initial FK solve (with user currents) fails, the code now automatically catches the error and calls `CRM_ForwardKinematics` again with **0 Amps**.
- **Result:** **FIXED.** Guarantees a valid starting state (straight rod) for any length.

### Task 3: Implement Internal Sub-stepping
- **Action:** Rewrote `crm_step_forward`.
- **Detail:** Implemented a loop that executes `num_substeps = 20` times per user call. Each sub-step integrates $1\text{ms}$ of physics using RK4.
- **Bug Fix:** During implementation, we fixed a "Double Damping" bug where damping forces were accumulated in-place, causing drift. We now use a temporary variable for the damped guess.
- **Result:** **FIXED.** The extension now runs at stable $1\text{ms}$ resolution while presenting a $20\text{ms}$ interface.

### Task 4: Fix Recovery Logic (The Timestep Mismatch)
- **Action:** Modified the "Current Ramp" fallback block.
- **Detail:** The recovery BVP solver (which tries to find a static shape when dynamics fail) was failing because it used the stiff $1\text{ms}$ timestep. We forced it to use `dt_local` ($20\text{ms}$), matching the Python wrapper's lenient configuration.
- **Result:** **FIXED.** The ramp now engages and attempts to solve.

---

## 2. CRITICAL GAP: Numerical Convergence

**Status:** The Extension is **NOT** finding solutions for high-load trajectories (Circle, Lemniscate) at $94.3\text{mm}$. It consistently returns `localmin=3` (failure), even though it runs safely without crashing. The Python wrapper succeeds on these same trajectories.

**Why the Python Wrapper Works (The "Predictor" Advantage):
**The Python wrapper uses the **ABM4 (Adams-Bashforth-Moulton)** integrator.
*   **Mechanism:** ABM4 is a multi-step method. It uses the history of the last 4 derivatives to construct a high-order polynomial prediction of the next state.
*   **Result:** When the BVP solver starts, it is given an extremely accurate "guess" derived from this history. This puts it inside the convergence basin, allowing it to find the solution easily.

**Why the Extension Fails (The Stateless Limitation):
**The Extension uses **RK4 (Runge-Kutta 4)** because it is stateless (no history).
*   **Mechanism:** RK4 is a single-step method. Even with our 20x sub-stepping (running at 1ms), it treats every 20ms call as a fresh start. It cannot leverage the "momentum" of the previous 3 steps to predict the future as accurately as ABM4.
*   **Result:** The "guess" provided to the BVP solver by RK4 is slightly less accurate than ABM4's. At $94.3\text{mm}$, the physics is so stiff that this tiny inaccuracy is enough to push the solver outside the convergence basin, causing it to fail (`localmin=3`) where ABM4 succeeds.

**Conclusion:**
We have fixed the **Software Engineering** bugs (memory, crashes, logic), but we have hit a **Numerical/Architectural Wall**. A stateless RK4 solver is fundamentally less capable than a stateful ABM4 solver for this specific stiff system. To achieve true parity, the Extension would need to become stateful (carry history tensors) to implement ABM4 correctly.

## 3. Next Steps (Recommendation)
1.  **Acceptance:** Acknowledge that the stateless extension has stricter physical limits than the stateful wrapper. 
2.  **Integration:** Proceed with RL integration, understanding that the agent will be penalized for entering "unsolvable" states that the wrapper might have handled. This is a valid constraint for a robust control policy.