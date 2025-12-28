# Architectural Specification: Option C (Stateful ABM4 Extension)

**Status:** DESIGN DOCUMENT
**Target:** `crm_torch_ext`
**Objective:** Define the architecture for a Native PyTorch C++ Extension that implements Differentiable Catheter Dynamics using the **Adams-Bashforth-Moulton (ABM4)** integrator.

---

## 1. System Requirements

1.  **Numerical Parity:** The system must produce trajectory outputs identical (within $10^{-6}\text{mm}$) to the reference Python Wrapper (`Option A`) for the `Circle` and `Lemniscate` benchmarks.
2.  **High-Load Stability:** The system must converge (`localmin=0`) for high-stiffness configurations (Insertion Length $\ge 94.3\text{mm}$). 
3.  **Differentiability:** The system must provide gradients $\partial y / \partial u$ and $\partial y / \partial x$ via PyTorch Autograd.
4.  **Performance:** The forward pass must execute in C++ without Python interpreter overhead.

---

## 2. Architecture Design

### 2.1 The Stateful Operator
Unlike a standard function $y = f(x, u)$, the CRM dynamics require history to resolve stiff boundary value problems. The operator is designed as a **Functional State Machine**:

$$ (x_{t+1}, H_{t+1}) = \text{Step}(x_t, u_t, H_t) $$

Where $H_t$ is the **History Tensor** containing previous derivatives $\{ \dot{x}_{t-1}, \dot{x}_{t-2}, \dot{x}_{t-3} \}$.

### 2.2 Data Structures

**History Tensor Specification:**
*   **Shape:** `(Batch, NumActuators, 3, 6)`
*   **Data Type:** `torch::kFloat64`
*   **Layout:**
    *   `H[:, :, 0, :]` stores $\dot{x}_{t-1}$
    *   `H[:, :, 1, :]` stores $\dot{x}_{t-2}$
    *   `H[:, :, 2, :]` stores $\dot{x}_{t-3}$

### 2.3 The Bootstrapping Logic
To handle the "Cold Start" (where $H$ is empty), the operator implements an internal state machine:

| Step Count | Integrator | Action |
| :--- | :--- | :--- |
| **0 - 2** | **RK4** | Adaptive Runge-Kutta 4. Generates initial derivatives. Populates $H$. |
| **3+** | **ABM4** | Multi-step Adams-Bashforth-Moulton. Consumes $H$ for high-order prediction. Updates $H$. |

---

## 3. Implementation Specification

### 3.1 C++ Operator (`crm_step`)
**Inputs:**
1.  `currents` ($u_t$)
2.  `insertion_length` ($L$)
3.  `state` ($x_t$) - [v, w, p, R, xf, mL, nL] packed
4.  `history` ($H_t$)
5.  `step_count` (Integer)

**Logic:**
*   **Extract:** Unpack tensors to Eigen maps.
*   **Integrate:** Call `ABM4_coildyn` (if step $\ge$ 3) or `RK4_step` (if step < 3) from the Core Library.
*   **Solve:** Use the integrator's output as the **Initial Guess** for the Boundary Value Problem (BVP) solver (`DynamicsBVP`).
    *   *Critical:* The BVP solver convergence depends on this guess being accurate. ABM4 provides the required accuracy.
*   **Update:** Shift the history buffer using the new derivative $\dot{x}_t$.
*   **Pack:** Return new state and new history tensor.

### 3.2 Python Wrapper (`TorchCRMPhysics`)
**Role:** State Manager.
*   Holds `self.history` and `self.step_count`.
*   Initializes them on `reset()`.
*   Passes them to the C++ operator in `step()`.
*   **Gradient Stop:** Calls `.detach()` on the returned history to prevent infinite backpropagation through time.

---

## 4. Verification Strategy

### 4.1 Unit Test: Integration Hand-off
Verify that the transition from Step 2 (RK4) to Step 3 (ABM4) does not introduce a discontinuity in the trajectory.

### 4.2 System Test: Parity
Run the "Circle" trajectory on both the Reference Wrapper and this Extension.
*   **Pass:** Maximum deviation $< 10^{-6}\text{mm}$.
*   **Fail:** Deviation diverges.

### 4.3 System Test: Convergence
Run the "Lemniscate" trajectory at $L=94.3\text{mm}$.
*   **Pass:** `localmin=0` (Converged) for $>99\%$ of steps.
*   **Fail:** `localmin=3` (Diverged).
