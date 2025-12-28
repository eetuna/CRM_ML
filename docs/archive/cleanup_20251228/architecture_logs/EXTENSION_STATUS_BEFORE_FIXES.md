# Extension Status: Pre-Remediation Analysis

**Date:** December 28, 2025
**Subject:** State of `crm_torch_ext` vs `crm_python` before remediation.

## 1. The Discrepancy
- **Legacy Wrapper (`crm_python`):** Robust. Successfully initialized and tracked trajectories at high insertion lengths ($94.3\text{mm}$). Failed gracefully.
- **New Extension (`crm_torch_ext`):** Fragile. Failed instantly on initialization at $94.3\text{mm}$ or diverged (exploded to infinity) during the first dynamics step.

## 2. Root Cause Analysis: Why did the Wrapper work?
The Python wrapper's robustness was not due to superior C++ physics code, but rather a "protective shell" of logic:

1.  **Zero-Current Retry (Initialization):**
    - When `initializeFromKinematics` was called with high currents (e.g., 0.25A), the FK solver often failed.
    - The Wrapper detected this and **silently retried with 0 Amps**.
    - Solving for 0A (Straight Rod) is mathematically trivial and always succeeds.
    - The simulator effectively started "straight" even if the user asked for "bent".

2.  **Stateful Integrator (ABM4):**
    - The Wrapper used the **Adams-Bashforth-Moulton (ABM4)** integrator.
    - ABM4 is a **multi-step** method that uses the history of previous derivatives to take large, stable steps ($20\text{ms}$).
    - The Wrapper class persisted this history in memory between Python calls.

3.  **Python-Side Ramping:**
    - The success scripts (e.g., `lemniscate...py`) manually generated a **120-step ramp** in Python to gently bend the catheter before starting the trajectory.

## 3. The Problem with the Extension
The Extension was designed to be **stateless** (functional) to support PyTorch Autograd, which stripped away the Wrapper's protections:

1.  **No Retry Logic:** `initialize_from_fk` tried to solve the difficult high-current problem once. If it failed, it threw an error or returned garbage.
2.  **Stateless Integrator (RK4):**
    - Being stateless, it could not use ABM4 (which requires history).
    - It was forced to use **RK4** (single-step).
    - RK4 is unstable at $20\text{ms}$ for this stiff system. It requires $\approx 1\text{ms}$.
    - Forcing $1\text{ms}$ made the BVP solver (which finds the forces) extremely "stiff" and prone to failure on large jumps.
3.  **Garbage Memory (Hidden Bug):**
    - The C++ struct for initialization parameters was not zeroed. It contained random memory values, causing immediate divergence even for easy problems.

## 4. Suggestions for Remediation
To achieve parity, the Extension needed to internalize the Wrapper's logic:
1.  **Implement Zero-Current Retry** in C++ to guarantee initialization.
2.  **Implement Internal Sub-stepping:** Execute 20 small RK4 steps ($1\text{ms}$) for every user step ($20\text{ms}$) to emulate the stability of ABM4 without needing history.
3.  **Decouple Timesteps:** Use $20\text{ms}$ for the "Static Ramp" (recovery) solver to make it lenient, while using $1\text{ms}$ for the physics integration.
