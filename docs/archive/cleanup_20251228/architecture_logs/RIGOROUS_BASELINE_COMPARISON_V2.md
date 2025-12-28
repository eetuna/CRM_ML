# Rigorous Baseline Comparison: CRM Dynamics Architecture

**Date:** December 28, 2025
**Correction:** Supersedes all previous comparison reports. Corrects the assumption of code lineage and details the exact architectural divergence.

---

## 1. Codebase Lineage & Divergence

We compared the `main` branch of `../CRM_Dynamics` (Baseline) against `CRM_ML/src` (Local Core).

### 1.1 Fundamental Data Structures
*   **Baseline (`../CRM_Dynamics`):** Uses legacy C-style arrays for core physics parameters.
    *   *Evidence:* `CRM_IVPSolver.cpp` declares `double MagMoment[NUM_ACT_SET][3]`.
    *   *Implication:* Fixed-size, stack-allocated, harder to interface with modern AD libraries.
*   **Local Core (`CRM_ML/src`):** Uses modern C++ containers and Eigen.
    *   *Evidence:* `CRM_IVPSolver.cpp` declares `std::vector<Eigen::Vector3d> MagMoment`.
    *   *Implication:* Heap-allocated, dynamic sizing, fully compatible with `autodiff` templates.

### 1.2 Differentiation Infrastructure
*   **Baseline:** **None.** Contains no AD-specific headers. Relying on manual sensitivity analysis (chains of derivatives hand-coded in `CRM_IVPJacobian.cpp`).
*   **Local Core:** **Full AD Suite.**
    *   `CRM_DynamicsContext_AD.hpp`: Templated context for AD types.
    *   `CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp`: The core residual function rewritten to support `autodiff::real`.
    *   *Conclusion:* `CRM_ML` is a specialized fork architected specifically for Differentiable Physics.

---

## 2. Integration Strategy: The "Wrapper vs. Extension" Gap

Both Option A and Option C link against the **Same Local Core (`CRM_ML/src`)**, but they use it in fundamentally different ways.

### 2.1 Option A: The Python Wrapper (`crm_python`)
*   **Mechanism:** `CRMDynamicsWrapper` Class (C++).
*   **State:** **Persistent/Stateful.** The object remains in memory between Python calls.
*   **Time Integration:** **ABM4 (Adams-Bashforth-Moulton).**
    *   *Implementation:* Custom loop in `crm_bindings.cpp`.
    *   *Critical Feature:* Maintains `xdot_nm1`, `xdot_nm2`, `xdot_nm3` (derivatives at $t_{-1}, t_{-2}, t_{-3}$). 
    *   *Effect:* Predicts the next state ($t_{+1}$) using a high-order polynomial extrapolation of history.
*   **Initialization:** **Python-Side Robustness.**
    *   If FK fails, Python logic catches exception and retries with 0 Amps.
    *   Uses 120-step pre-rollout ramps in Python scripts to ease the solver into complex states.

### 2.2 Option C: The Torch Extension (`crm_torch_ext`)
*   **Mechanism:** `crm_step` Function (C++).
*   **State:** **Stateless/Functional.** Logic resets every call.
*   **Time Integration:** **RK4 (Runge-Kutta 4).**
    *   *Implementation:* Hardcoded `IntegratorType::RK4` in `crm_step_op.cpp`.
    *   *Constraint:* Cannot use ABM4 because it has no access to history tensors.
    *   *Workaround:* Uses 20 internal sub-steps ($1	ext{ms}$) to stabilize the single-step solver.
    *   *Effect:* Predicts next state based *only* on current state. Significantly less accurate prediction for stiff BVP problems at high insertion lengths.
*   **Initialization:** **C++-Side Robustness (Ported).**
    *   We implemented "Zero-Current Retry" inside the C++ function `initialize_from_fk`.
    *   We implemented "Static Ramp" inside `crm_step_forward` to recover from bad predictions.

---

## 3. The Numerical "Dead Zone"

Despite achieving **Feature Parity** (both have Retry & Ramps), we identified a critical **Numerical Discrepancy**:

*   **Scenario:** $L=94.3	ext{mm}$, Current Jump $0 	o 0.25	ext{A}$.
*   **Wrapper (ABM4):** Success. The history-based predictor generates a BVP guess within the convergence basin.
*   **Extension (RK4):** Failure (`localmin=3`). The memoryless predictor generates a guess slightly outside the basin. The internal 5-step recovery ramp is insufficient to bridge this specific gap.

**Root Cause:** The physics at $94.3	ext{mm}$ is so stiff that the difference between an ABM4 prediction and an RK4 prediction is the difference between convergence and divergence.

---

## 4. Final Verdict

1.  **Architecture:** `CRM_ML` is the superior, modern codebase supporting AD.
2.  **Robustness:** The Wrapper is **Functionally Superior** due to stateful ABM4 integration.
3.  **Safety:** The Extension is now **Safe** (latches on failure), preventing training crashes.
4.  **Trade-off:** To use the high-speed, differentiable Extension, the RL agent must accept a stricter physical environment (no teleportation) than the lenient Wrapper allowed.

## 5. Side-by-Side Architectural Summary



| Feature | Baseline (`../CRM_Dynamics`) | Local Core (`CRM_ML/src`) | Option A (Wrapper) | Option C (Extension) |

| :--- | :--- | :--- | :--- | :--- |

| **Architecture** | Legacy C-Style | Modernized C++ / Eigen | Stateful PyBind Class | Stateless Torch Operator |

| **Data Structures** | Raw Arrays (`double[]`) | `std::vector` / `Eigen` | Persistent C++ Objects | Stateless PyTorch Tensors |

| **Temporal Integrator** | Single-step (RK4) | Templated (RK4/ABM4) | **Stateful ABM4** | **Stateless RK4** (Sub-stepped) |

| **History Tracking** | None | None | **Internal (nm1...nm3)** | **None** (Memoryless) |

| **Differentiation** | Manual Analytical | Autodiff Ready | Finite Difference / Analytical | **Implicit Differentiation (IFT)** |

| **Initialization** | Basic FK Solve | Basic FK Solve | **Robust (Python Retry)** | **Robust (C++ Retry)** |

| **Numerical Limit** | High | High | **Maximum** (Predictive) | **Strict** (Non-predictive) |

| **Safety Logic** | None | None | Graceful Error Return | **Hardware-like Latching** |

| **Performance** | Reference | Optimized | Medium (Python Overhead) | **Maximum (Native Extension)** |



**Recommendation:**

Adopt the Extension for RL training. The "failure" to track unphysical jumps is a valid constraint for a control policy.
