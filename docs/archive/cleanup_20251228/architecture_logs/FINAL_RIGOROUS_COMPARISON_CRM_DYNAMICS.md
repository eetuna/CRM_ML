# Final Rigorous Comparison: CRM Dynamics Ecosystem

**Date:** December 28, 2025
**Subject:** Comparative Analysis of Baseline, Option A, and Option C architectures.

---

## 1. The Core Baseline (`../CRM_Dynamics`)
The baseline is the historical "ground truth" but is **functionally obsolete** for this project's goals.
*   **Architecture:** Procedural C code wrapped in `.cpp` files.
*   **Data Structures:** Raw C-arrays (`double arr[N][3]`).
*   **Gradients:** None / Manual derivation only.
*   **Status:** A reference point for physics validation, but lacks the infrastructure for ML integration.

## 2. Option A: The AD-Enhanced Hybrid Architecture (`CRM_ML/src`)
Option A is a **complete mathematical rework** of the baseline to support end-to-end differentiability. It is currently the only version that reliably finds solutions for high-load trajectories.

### 2.1 The Autodiff Rework
Option A introduces a specific C++ AD infrastructure missing from the baseline:
*   `CRM_DynamicsContext_AD.hpp` / `impl.hpp`: Templated containers that allow the physics equations to be solved using `autodiff::real` types.
*   `CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp`: A rewrite of the core Dynamics residual using `Eigen` types to support automatic Jacobian computation.
*   **Linearization:** It implements `linearize_full_seed_action_from_seed_implicit` which uses the Implicit Function Theorem (IFT) to compute $\partial y / \partial u$.

### 2.2 Numerical Advantage: Stateful ABM4
The "magic" that makes Option A work so well at $94.3\text{mm}$ is its **Stateful Time Integration**.
*   It maintains a persistent C++ class instance.
*   It stores derivative history ($t_{-1}, t_{-2}, t_{-3}$).
*   It uses **ABM4** to predict the next state. This high-order prediction ensures the Newton-Raphson solver (using the AD Jacobian) starts inside the convergence basin.

## 3. Option C: The Native Torch Extension (`crm_torch_ext`)
Option C is a **packaging evolution**. It ports the AD math and IFT logic from Option A into a native PyTorch C++ Extension.

*   **Architecture:** Stateless functional operator (`crm_step`).
*   **Math:** Identical to Option A (IFT + `autodiff` library + Eigen residual).
*   **The Conflict:** To maintain a stateless API (required for clean Torch integration), it currently uses **Sub-stepped RK4**.
*   **The Convergence Gap:** Because it lacks the ABM4 history found in Option A, its "prediction" for the BVP solver is cruder. At extreme insertion lengths, this leads to `localmin=3` (failure to converge) where Option A's predictive ABM4 succeeds.

---

## 4. Side-by-Side Comparison

| Feature | Baseline | **Option A** | **Option C** |
| :--- | :--- | :--- | :--- |
| **Differentiation** | None | **Implicit AD** | **Implicit AD** |
| **AD Implementation** | N/A | `src/CRMDYN_..._autodiff_eigen.hpp` | `src/CRMDYN_..._autodiff_eigen.hpp` |
| **Integration** | RK4 | **Stateful ABM4** | **Stateless RK4 (Sub-stepped)** |
| **Convergence Radius** | Low | **High** (via Predictor) | **Medium** (via Sub-stepping) |
| **Initialization** | Simple FK | **Zero-Current Retry** | **Zero-Current Retry** |
| **Memory** | N/A | C++ Class Memory | Torch Tensors |

## 5. Conclusion
Option A represents the **functional peak** of the simulator due to its stateful multi-step integration. Option C represents the **performance/integration peak** but is currently limited by the numerical "stiffness" of its stateless single-step integration. The software engineering bugs (uninitialized memory, damping drift) have been fixed in Option C, but the numerical gap remains an architectural trade-off of statelessness.