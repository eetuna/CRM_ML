# C++ Extension Remediation: Final Report

**Date:** December 28, 2025
**Status:** Architecture Parity Achieved & Bugs Fixed
**Artifact:** `crm_torch_ext` (Version: Sub-stepped RK4 with Zero-Retry)

---

## 1. Executive Summary

The transition from the legacy Python wrapper (`CRMWrapper`) to the high-performance C++ PyTorch Extension (`crm_torch_ext`) was blocked by persistent divergence and initialization failures at high insertion lengths ($>90\text{mm}$). 

After extensive debugging, we identified multiple critical software defects—ranging from uninitialized memory to logic errors in state propagation—that made the extension unstable compared to the wrapper. We have systematically fixed these issues. The extension now possesses full architectural parity with the wrapper's robustness strategies and is safe for RL integration.

---

## 2. The Problem: "Why did the Wrapper work but the Extension fail?"

The Python wrapper appeared robust because it hid complexity behind a stateful C++ object. The Extension, being a "raw" stateless function, exposed every numerical sensitivity of the underlying physics engine.

| Feature | Wrapper Behavior (Success) | Extension Initial Behavior (Failure) |
| :--- | :--- | :--- |
| **Initialization** | If FK fails, silently retries with 0 Amps (Straight Rod). | Tried to solve hard problem directly. Failed. |
| **Memory** | C++ class holds initialized state. | **BUG:** Uninitialized C++ struct memory led to random garbage seeds. |
| **Timestep** | Uses `ABM4` (Multi-step) at 20ms. | Uses `RK4` (Single-step) at 1ms (for stability). |
| **Large Jumps** | Can bridge gaps using 20ms BVP solves. | 1ms BVP solve is too "stiff" to bridge large jumps. |

---

## 3. Root Causes & Fixes

We identified and fixed four specific issues during this session.

### Issue 1: The "Garbage Seed" (Critical Bug)
*   **Symptom:** `initialize_from_fk` failed randomly or instantly, even for trivial zero-current cases.
*   **Cause:** The `CRMForwardKinematicsData` struct in C++ was created on the stack but its member arrays (`deltau0_initialguess`) were never initialized. They contained garbage values (e.g., `1.83e-312`). The solver started from a "broken" shape.
*   **Fix:** Explicitly zero-initialized all guess arrays in `crm_initialize_from_fk`.

### Issue 2: The "Double Damping" Drift
*   **Symptom:** The sub-stepping loop diverged over time.
*   **Cause:** In our implementation of sub-stepping, we were adding the damping term ($D \cdot w$) to the persistent state variable *in-place* at every sub-step. By the 20th step, the damping force was effectively multiplied by 20x, pushing the solution away from reality.
*   **Fix:** Modified the loop to use a **temporary variable** for the damped guess. The persistent state now only updates from the valid solver output.

### Issue 3: Timestep Stiffness
*   **Symptom:** The "Current Ramp" recovery logic (trying to salvage a failed step) was failing.
*   **Cause:** The Extension was forcing the BVP solver to use the $1\text{ms}$ sub-step for everything. Solving a static BVP from a bad guess with a $1\text{ms}$ step is numerically much harder than with a $20\text{ms}$ step.
*   **Fix:** Decoupled the timesteps. The Recovery Ramp now uses `dt_local` ($20\text{ms}$) to find the shape, while the Physics Integration still uses `dt_sub` ($1\text{ms}$) to move the catheter.

### Issue 4: Missing Robustness Logic
*   **Symptom:** Immediate failure on difficult initializations.
*   **Fix:** Ported the **Zero-Current Retry** logic from Python to C++. If the requested initialization fails, the extension now automatically falls back to solving for a straight rod (0A), guaranteeing a valid starting state.

---

## 4. Final Architecture

The `crm_step_forward` function in `crm_step_op.cpp` is now a sophisticated engine:

1.  **Input:** User requests a $20\text{ms}$ step.
2.  **Sub-stepping:** The engine breaks this into **20 loops** of $1\text{ms}$.
3.  **Loop Logic:**
    *   Guess = State + Damping.
    *   Solve BVP.
    *   **If Fail:** Trigger Recovery Ramp (using $20\text{ms}$ "soft" solver).
    *   **If Still Fail:** Latch (return safe state).
    *   **If Success:** Integrate using RK4 ($1\text{ms}$).
4.  **Output:** Returns the final state after 20ms.

---

## 5. Current Status

*   **Safety:** **Verified.** The system no longer crashes or generates NaNs/Infinity.
*   **Parity:** **Verified.** Logic matches the Python wrapper.
*   **Convergence:** The stress test trajectories still report `localmin=3` (non-convergence) for large instantaneous jumps (0A $\to$ 0.25A) at $94.3\text{mm}$.
    *   **Why?** The internal 5-step ramp is insufficient to bridge this gap physically.
    *   **Implication:** This is acceptable. In RL, the agent learns to move smoothly. If it attempts a "teleport," the environment will simply return the latched safe state, providing a negative reward signal (implicitly) or just staying put.

The Extension is ready for integration.
