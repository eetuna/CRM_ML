# Option A Analysis: The Stateful/Stateless Duality

**Date:** December 28, 2025
**Subject:** Understanding why Option A (Wrapper) succeeds where Option C (Extension) fails.

---

## 1. The Discovery

Option A is not a single algorithm. It employs two distinct strategies depending on the context.

### Mode 1: Simulation (`step()`)
*   **Context:** Used when running the environment loop (`env.step()`).
*   **Mechanism:** Calls `CRMDynamicsWrapper::step()`.
*   **State:** **Stateful.** It accesses private class members `xdot_nm1`, `xdot_nm2`, `xdot_nm3`.
*   **Integrator:** **ABM4 (Adams-Bashforth-Moulton).**
*   **Behavior:** Uses 4 steps of history to predict the next state.
*   **Result:** **High Robustness.** Can track 94.3mm trajectories because the predictor works.

### Mode 2: Linearization (`step_from_seed()` / `linearize_...`)
*   **Context:** Used when computing gradients (`backward()`).
*   **Mechanism:** Calls `CRMDynamicsWrapper::step_from_seed()`.
*   **State:** **Stateless.** It takes `v, w, p, R, xf` as arguments. It **IGNORES** the class members.
*   **Integrator:** **RK4** (or Cold-Start ABM4).
*   **Behavior:** Starts integration from scratch (no history).
*   **Result:** **Lower Robustness.** Explicitly documented in code as "NOT RELIABLE for consecutive stepping".

---

## 2. The Option C Trap

The C++ Extension (Option C) was implemented by porting the logic of **Mode 2 (Linearization)**.
*   It exposes `crm_step` which mirrors `step_from_seed`.
*   It accepts `v, w, p...` but **no history**.
*   **Consequence:** We are trying to run a long-horizon simulation using the "Linearization" engine.
*   **Outcome:** It fails exactly as the comment in `step_from_seed` predicted: "Consecutive stepping... is NOT RELIABLE".

## 3. The Required Fix

To make Option C usable for simulation, we must upgrade it to support **Mode 1**.
Since Option C is a functional operator (not a class), we cannot store `xdot_nm1` internally.
**Solution:** We must pass `xdot_nm1...` as **Tensor Arguments**.

**New Signature:**
`NextState, NextHistory = crm_step(State, Control, History)`

This bridges the gap, giving the stateless function the "memory" it needs to behave like the stateful class.
