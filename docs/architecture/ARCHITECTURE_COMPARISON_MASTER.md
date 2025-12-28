# Detailed 5-Way Architectural Comparison: CRM Dynamics

**Date:** December 28, 2025
**Scope:** Exhaustive technical comparison of the 5 distinct entities in the simulator ecosystem.

---

## 1. The Baseline (`../CRM_Dynamics`)
**Identity:** The upstream legacy reference implementation.
**Path:** `../CRM_Dynamics/src/`

*   **Data Structures:** Pure C-style arrays (`double[N][3]`). No usage of `std::vector` or `Eigen` in the public API.
*   **Math Library:** Manual implementation of vector operations (`mCopy`, `mAdd`, `mMult`).
*   **Differentiation:** **None.** No headers for Automatic Differentiation. Jacobian functions (`CRM_FKJacobian.cpp`) implement manually derived sensitivity equations.
*   **Role:** Strictly a historical reference. It is **NOT** used in the build process of `CRM_ML`.

## 2. The Local Core (`CRM_ML/src`)
**Identity:** The active C++ physics engine used by this project.
**Path:** `CRM_ML/src/`

*   **Architecture:** A heavy fork/rewrite of the Baseline.
*   **Modernization:**
    *   Replaced C-arrays with `std::vector` and `Eigen::Matrix/Vector`.
    *   **Evidence:** `CRM_IVPSolver.cpp` uses `Eigen::Map` and `Eigen::Vector3d` for internal math.
*   **Differentiation Core:**
    *   **New File:** `CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp`. This file reimplements the core residual logic using templates that support `autodiff::real`.
    *   **New File:** `CRM_DynamicsContext_AD.hpp`. Provides type-safe containers for AD variables.
*   **Integrator Support:** Contains templates for **both** RK4 and ABM4 in `CRM_IVP_NumericalIntegrationTemplates.hpp`, but they are designed to be called by a higher-level manager.

## 3. The Python Wrapper (`crm_python`)
**Identity:** The C++ bindings library.
**Path:** `crm_ml_rl/wrappers/crm_bindings.cpp`

*   **Structure:** A PyBind11 module exporting the `CRMDynamicsWrapper` class.
*   **State Management:** **Persistent C++ Object.**
    *   The `CRMDynamicsWrapper` class holds member variables (`xdot_nm1`, `xdot_nm2`, etc.) that persist across Python calls.
*   **Integration Logic (The "Magic"):**
    *   It implements a **Custom Time-Stepping Loop** inside `step_from_seed`.
    *   It explicitly manages the **ABM4 History Buffer**.
    *   **Crucial Logic:** `predict_next_state()` uses this history to generate a high-quality initial guess for the BVP solver.
*   **Initialization:** Implements `linearize_full_seed_action_from_seed_implicit` which calculates Jacobians using the Local Core's AD infrastructure.

### Option A: The Python Wrapper (`crm_python`)
*   **Definition:** A PyBind11 class (`CRMDynamicsWrapper`) that exposes the C++ solver.
*   **Dual Nature:**
    1.  **Simulation Mode (`step()`):** **Stateful.** Uses private member history to run **ABM4**. Extremely robust. Used for forward rollout.
    2.  **Gradient Mode (`step_from_seed()`):** **Stateless.** Accepts explicit state but has no history. Uses **RK4** (or cold ABM4). Documented as "Not reliable for consecutive stepping." Used for backward pass.
*   **Safety:** **Python-Managed.** Initializes with "Zero-Current Retry" logic implemented in Python.

### Option C: The Torch Extension (`crm_torch_ext`)
*   **Definition:** A native PyTorch C++ Extension exposing a functional operator (`crm_step`).
*   **The Trap:** It was implemented by porting the **Gradient Mode** logic of Option A.
*   **Consequence:** It tries to use the stateless, history-less engine for simulation. It fails exactly as predicted by the Option A documentation.
*   **Integrator:** **RK4 (Runge-Kutta 4).**
    *   Forced to use a single-step method because it has no history.

---

## Summary of Dependencies

*   **Baseline** is standalone (ignored).
*   **Local Core** is the engine.
*   **Wrapper** wraps Core + adds **Memory** (ABM4).
*   **Option A** wraps Wrapper (Python Autograd).
*   **Option C** wraps Core + adds **Sub-stepping** (RK4).

**Final Verdict:** Option A works better physically because it has Memory. Option C works faster/cleaner software-wise but has Amnesia.
