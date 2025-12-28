# Code Audit Phase 3 & 4: Python Integration & Examples

**Date:** 2025-12-25
**Auditor:** Gemini Agent
**Status:** Completed

## 1. Executive Summary
The Python-level integration (Gym environments and Torch wrappers) is well-aligned with the stabilized C++ core. The "Gold Standard" iLQR demo correctly avoids known simulator limitations. However, a major limitation in the Torch autograd wrapper (missing insertion length gradients for dynamics) and inconsistencies in multi-actuator reporting in the environment remain.

## 2. Detailed Findings

### 2.1 ML/RL Environment (`catheter_env.py`)
*   **Status:** ✅ **STABLE**
*   **Evidence:**
    - `CatheterEnv` correctly uses `simulator.step()`, which uses the stable `stepDynamics` C++ API.
    - **Robustness:** `CRMWrapper` includes a "re-initialization on failure" logic that automatically attempts to recover from simulator divergence by re-seeding from Kinematics.
*   **Issue:** The environment state only reports tip position/velocity. For multi-actuator control, the observation space should be expanded to include all actuator states to maintain the Markov property.

### 2.2 Torch Physics Wrapper (`torch_physics.py`)
*   **Status:** ⚠️ **PARTIAL GRADIENTS**
*   **Issue:** `CRMDynamicsStepFunction.backward` returns `None` for `grad_insertion`.
*   **Impact:** Any RL or control algorithm attempting to optimize the *insertion length* via backpropagation through time (BPTT) will fail or receive zero gradients. Curiously, `CRMFKFunction` *does* implement this gradient, creating an inconsistency between kinematics and dynamics wrappers.

### 2.3 iLQR Demo (`ilqr_catheter_demo.py`)
*   **Status:** ✅ **GOLD STANDARD**
*   **Evidence:**
    - Correctly uses `step()` for trajectory rollouts.
    - Correctly uses `linearize_*_from_seed` for one-off Jacobian calculations.
    - Implements proper regularization (`R_action = 0.1`) to avoid stiff regimes.
*   **Optimization Note:** Uses `A_t = I` (Identity) for state propagation. While stable, this ignores the rich coupling information available in the 15D state Jacobian returned by the implicit linearizer.

### 2.4 Obsolete / Debug Scripts
*   **Audit Task:** Identify candidates for `docs/archive/`.
*   **Candidates:**
    - `examples/debug_consecutive_stepping.py` (Issue documented in `CONSECUTIVE_STEPPING_LIMITATION.md`)
    - `examples/test_bvp_seed.py` (Redundant with stabilization docs)
    - `scripts/baseline_taskA1_capture.py` (Legacy task artifact)

## 3. Recommendations

1.  **Unified Gradient Support:** Implement `grad_insertion` in `torch_physics.py` for dynamics to match the kinematics wrapper.
2.  **Actuator-Aware Observations:** Update `CatheterEnv` to include all actuator velocities in the observation vector when `NUM_ACT_SET > 1`.
3.  **iLQR Refinement:** Experiment with mapping the full state Jacobian `A` to the 6D tip state in iLQR to improve convergence rate in complex trajectories.
