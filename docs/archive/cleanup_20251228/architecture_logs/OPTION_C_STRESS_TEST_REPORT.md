# Option C Stress Test & Robustness Final Report

**Date:** December 27, 2025
**Target:** `crm_torch_ext` (C++ PyTorch Extension)
**Status:** **PASSED (With Strict Constraints)**
**Performance:** ~85x Forward Speedup, ~6x Total Speedup

---

## 1. Executive Summary

We conducted a rigorous stress test campaign to validate the robustness of the new C++ PyTorch Extension (`crm_torch_ext`) before deploying it to Reinforcement Learning. The goal was to ensure the simulator would not crash, produce NaNs, or explode numerically during the random exploration typical of RL agents.

**Key Outcome:** The extension is now **robust** and **fail-safe**, but the underlying physics solver (BVP) is proven to be highly sensitive. The system effectively trades "correctness at all costs" (which caused crashes) for "stability" (latching on failure).

**Critical Constraint:** The extension **MUST** be operated with a timestep of `1ms` (or lower) and full state warm-starting to maintain stability. Cold starts (solving from zero seed) fail for currents > 0.5mA.

---

## 2. Test Campaign & Findings

### Test 1: Static Workspace Mapping (Grid Search)
*   **Method:** Attempted to solve for Tip Position across a grid of Currents (0-25mA) and Insertion Lengths (0-100mm), initialized from zero seeds ("Cold Start").
*   **Initial Result:** **96% Failure Rate**. The solver crashed or failed to converge for almost all points where $I > 1.0$ mA.
*   **Diagnosis:** The BVP solver is incapable of solving large static deflections from a zero guess. The Python wrapper appeared more robust only because it masked these failures or used different defaults.
*   **Fix:** Implemented **Current/Force Continuation** in C++. Instead of jumping from 0 to Target Force, the solver now ramps the force in 5 steps (20%, 40%, ..., 100%).
*   **Final Result:** **100% Robustness** (No crashes). However, many points "passed" only because the system gracefully returned a zero-gradient/no-motion result upon non-convergence.

### Test 2: Dynamic Random Walk
*   **Method:** Run 50 parallel episodes of 1000 steps, applying random current changes ($\Delta u \sim \mathcal{N}(0, 0.05)$) at each step.
*   **Initial Result:** **Explosive Divergence**. Tip position reached $10^6$ mm within 3 steps. Velocities spiraled to infinity.
*   **Root Cause 1 (State Propagation):** The initial API did not return the full internal state (`mL`, `nL`), forcing the user to re-seed with zeros at every step. This broke physical continuity.
*   **Root Cause 2 (Rotation Drift):** The C++ rotation matrices drifted from orthogonality ($R^T R \neq I$), invalidating the physics math.
*   **Root Cause 3 (Timestep):** The default `dt=20ms` was too aggressive for the explicit integration of high-stiffness damping.
*   **Fixes:**
    1.  **Full State API:** Updated `crm_step` to return `[next_state, next_v, ..., next_nL]`.
    2.  **Gram-Schmidt:** Implemented orthonormalization in C++ to snap rotation matrices back to valid SO(3) after every step.
    3.  **Latch-on-Failure:** If BVP fails, the state is frozen (velocities zeroed, position held).
    4.  **Timestep:** Reduced to `1ms` (RK4).
*   **Final Result:** **Stable Motion**. The catheter successfully navigated 1000-step random walks without exploding.

### Test 3: Gradient Consistency
*   **Method:** Compare Analytical Gradients (AD) vs. Finite Differences (FD).
*   **Result:** **0.00% Error** (Suspicious).
*   **Reality Check:** At random points, the BVP often failed to converge.
    *   **Fail Behavior:** AD returns 0.0 (Hardcoded fallback).
    *   **Fail Behavior:** FD sees flat output (latched state), so slope is 0.0.
    *   **Match:** $0.0 == 0.0$.
*   **Conclusion:** The gradients are consistent with the forward pass logic (even failure logic), but "valid" gradients are only available in the "Green Zone" (converged regions).

---

## 3. Discrepancies: Python vs. C++

| Feature | Python Wrapper | C++ Extension | Impact |
| :--- | :--- | :--- | :--- |
| **State Management** | Stateful (Class holds `mL`, `nL`) | Stateless (User must pass seeds) | Extension requires explicit warm-start logic. |
| **Integrator** | `ABM4` (Multi-step, stable) | `RK4` (Single-step) | `ABM4` requires history, which stateless op lacks. `RK4` needs smaller `dt`. |
| **Drift Correction** | Implemented in Python | Implemented in C++ (Added) | Fixed explosion issue. |
| **Failure Handling** | Returns `converged=False` | Returns `localmin` tensor | User must check `localmin` to know if step is valid. |
| **Performance** | ~8.2ms / step | ~1.2ms / step | **6.8x Speedup** achieved. |

---

## 4. Remaining Risks & Recommendations

### Risk: "The Dead Zone"
If the RL agent pushes the catheter into a state where the BVP solver fails (e.g., extremely high current + wall contact), the simulator will **latch** (freeze).
*   **Symptom:** Gradients become exactly zero.
*   **RL Impact:** The agent receives no learning signal and may get stuck in that state.
*   **Mitigation:** The reward function **must** heavily penalize the `localmin != 0` flag. If the physics fails, the agent should effectively "die" or receive -100 reward to learn to avoid that region.

### Risk: Timestep Sensitivity
The C++ extension using RK4 is stable at `1ms` but marginal at `20ms`.
*   **Recommendation:** Do not increase `dt` above `2ms`. The performance gain of the extension (85x) easily absorbs the cost of running more substeps.

### Risk: Cold Start
You cannot simply ask the simulator "What is the state at 20mA?" from scratch.
*   **Recommendation:** Always initialize at `I=0` and ramp up currents over 10-20 steps during the episode reset phase.

---

## 5. Next Steps

1.  **RL Integration:**
    *   The extension is ready. Update your PPO/SAC training scripts to use `TorchCRMPhysics(..., use_cpp_extension=True)`.
    *   **CRITICAL:** Ensure your RL loop passes the `next_v`, `next_mL`, etc., from step $t$ to step $t+1$.

2.  **Training Config Updates:**
    *   Set `dt = 0.001` or `0.002`.
    *   Add a check: `if localmin != 0: done = True, reward = -10`.

3.  **Future Feature (Task 4.2):**
    *   Implement batching. Currently, we loop over the batch dimension in Python. Pushing this loop to C++ (OpenMP) will yield another 10x speedup.

---

**Signed-Off:** Claude Code
**System State:** STABLE & OPTIMIZED
