# Comprehensive Architecture Comparison: CRM Dynamics

**Date:** December 28, 2025
**Scope:** Core Physics, Python Wrapper (Option A), and Torch Extension (Option C).

## 1. Baseline: The Core C++ Physics (`src/`)
The `src/` directory contains the fundamental implementation of the Cosserat Rod Model (CRM).

*   **Mathematics:** Solves the Boundary Value Problem (BVP) for the rod's shape using a Shooting Method.
*   **Integration:** Supports Runge-Kutta 4 (RK4) and Adams-Bashforth-Moulton 4 (ABM4).
*   **Differentiation:** Implements **Analytical Sensitivity Propagation**.
    *   See `CRM_IVPJacobian.cpp`.
    *   It propagates partial derivatives ($\partial x / \partial p$) alongside the state during integration.
    *   This provides exact gradients but requires complex, manually derived derivative code.

## 2. Option A: The Python Wrapper (`crm_python`)
This is the legacy interface used for initial validation.

*   **Architecture:** **Stateful Class** (`CRMDynamicsWrapper`).
    *   The C++ object persists in memory.
    *   It maintains the history of previous states.
*   **Integration:** Uses **ABM4** (Multi-step).
    *   Leverages the stored history to predict the next state with high accuracy.
    *   **Advantage:** Extremely robust at high insertion lengths ($94.3\text{mm}$) because the predictor keeps the BVP solver in the convergence basin.
*   **Initialization:** **Robust.**
    *   Implements a "Zero-Current Retry" loop in Python: if FK fails, it solves for 0A (straight rod) to bootstrap the state.
*   **Differentiation:**
    *   Primary: Finite Differences (Numerical Perturbation). Safe but slow.
    *   Secondary: Exposes Core Analytical Jacobians (`linearize_...`).

## 3. Option C: The Torch Extension (`crm_torch_ext`)
The new high-performance interface for RL.

*   **Architecture:** **Stateless Function** (`crm_step`).
    *   Pure function: $State_{t+1} = F(State_t, Control_t)$.
    *   No internal memory.
*   **Integration:** **Sub-stepped RK4** (Single-step).
    *   Cannot use ABM4 effectively because it lacks history.
    *   Uses 20 internal sub-steps ($1\text{ms}$) per user step ($20\text{ms}$) to maintain stability.
    *   **Disadvantage:** Lacks the "predictive momentum" of ABM4. This makes it numerically stricter/weaker at $94.3\text{mm}$ for large jumps.
*   **Initialization:** **Robust (Ported).**
    *   We ported the "Zero-Current Retry" logic into C++ (`initialize_from_fk`). It now matches the wrapper's ability to safe-start.
*   **Differentiation:** **Implicit Differentiation.**
    *   Uses the Implicit Function Theorem (IFT) on the BVP Residual.
    *   Calculates $\partial x / \partial u = -(\partial R / \partial x)^{-1} (\partial R / \partial u)$.
    *   **Advantage:** Faster than sensitivity propagation for BVPs because it only differentiates the final residual, not the entire path. Supports `autodiff` library.

## 4. Key Discrepancy & Resolution

| Feature | Option A (Wrapper) | Option C (Extension) | Status |
| :--- | :--- | :--- | :--- |
| **Robustness** | High (ABM4 Predictor) | Medium (RK4 Stateless) | **Accepted Trade-off.** RK4 is required for statelessness. |
| **Safety** | High (Retry Logic) | High (Retry + Latching) | **Parity Achieved.** |
| **Gradient Speed** | Slow (FD) | Fast (Implicit) | **Extension Superior.** |
| **Initialization** | Automatic Retry | Automatic Retry | **Parity Achieved.** |

## 5. Conclusion
The Extension (Option C) successfully replicates the **safety** features of the Wrapper (Option A) but fundamentally trades **numerical leniency** (ABM4) for **performance and differentiability** (Stateless RK4 + Implicit Gradients). This makes it the correct choice for large-scale RL, provided the agent learns to respect physical constraints (avoiding "teleportation" jumps).
