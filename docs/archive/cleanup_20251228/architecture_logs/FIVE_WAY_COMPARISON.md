# 5-Way Architectural Comparison: CRM Dynamics Ecosystem

**Date:** December 28, 2025
**Scope:** Strict comparison of the 5 distinct entities in the codebase evolution.

---

## 1. The Entities Defined

1.  **Baseline (`../CRM_Dynamics`)**
    *   **Identity:** The upstream legacy C++ codebase.
    *   **Role:** Historical reference. Not used in build.
    *   **Tech:** Raw C-arrays, manual math, no AD.

2.  **Local Core (`CRM_ML/src`)**
    *   **Identity:** The active C++ library in this repo.
    *   **Role:** The shared physics engine for everything below.
    *   **Tech:** Modern C++ (`std::vector`), `Eigen` linear algebra, Templated for `autodiff` library.
    *   **Key File:** `CRM_IVPSolver.cpp` (The integrator).

3.  **Python Wrapper (`crm_python`)**
    *   **Identity:** The C++ PyBind11 bindings (`crm_bindings.cpp`).
    *   **Role:** Exposes the Local Core to Python as a stateful object (`CRMDynamicsWrapper`).
    *   **Tech:** Holds persistent state in C++ memory. Implements **ABM4** time integration loop.

4.  **Option A (The Strategy)**
    *   **Identity:** The end-to-end differentiable pipeline using the Python Wrapper.
    *   **Implementation:** `crm_ml_rl/wrappers/torch_physics.py` + `linearize_full_seed_action_from_seed_implicit`.
    *   **Tech:** Hybrid. Python orchestrates the `backward` pass, calling C++ helper functions to get AD-computed Jacobians.
    *   **Status:** Validated & Robust (due to ABM4).

5.  **Option C (`crm_torch_ext`)**
    *   **Identity:** The Native PyTorch C++ Extension.
    *   **Role:** A single C++ shared library implementing a Torch Operator.
    *   **Tech:** Stateless. Internalizes the AD/Implicit logic of Option A into a pure C++ `backward` function. Uses **RK4** (sub-stepped) because it lacks persistent state.

---

## 2. Five-Way Comparison Matrix

| Feature | 1. Baseline | 2. Local Core | 3. Python Wrapper | 4. Option A | 5. Option C |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Location** | `../CRM_Dynamics` | `CRM_ML/src` | `crm_bindings.cpp` | `torch_physics.py` | `crm_torch_ext/` |
| **Data Types** | `double[]` | `Eigen`, `vector` | `py::array` | `torch.Tensor` | `torch.Tensor` |
| **Differentiation**| None | **Autodiff Ready** | FD or AD helpers | **Hybrid AD/FD** | **Pure Implicit AD** |
| **State** | N/A | Raw Structs | **Persistent Class** | Python Variables | **Stateless Tensors** |
| **Integration** | RK4 | Generic (Templated) | **Stateful ABM4** | Uses Wrapper | **Stateless RK4** |
| **History** | No | No | **Yes (4 steps)** | Yes (Implicitly) | **No** |
| **Initialization**| Manual | Manual | Manual | **Robust Retry** | **Robust Retry** |
| **Convergence** | Low | High (if ABM4) | **Maximum** | **Maximum** | **Medium** |

---

## 3. The Critical Convergence Link

*   **Local Core (#2)** provides the math.
*   **Python Wrapper (#3)** adds **Memory (History)** to that math, enabling the powerful **ABM4 Integrator**.
*   **Option A (#4)** uses #3, inheriting the power of ABM4, making it robust.
*   **Option C (#5)** links directly to #2 but **cannot use the Memory** of #3. It is forced to use the stateless **RK4**, losing the predictive advantage.

**Verdict:**
*   **Option A** works because it builds on the Stateful Wrapper (#3).
*   **Option C** struggles because it bypasses the Wrapper and tries to be a pure function, losing the history needed for stiff convergence.
