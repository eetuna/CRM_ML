# Rigorous Execution Plan: Upgrading Option C to Stateful ABM4

**Status:** READY FOR EXECUTION
**Current Artifact:** `crm_torch_ext` (Stateless RK4)
**Target Artifact:** `crm_torch_ext` (Stateful ABM4)
**Objective:** Transform the existing stateless extension into a stateful operator by injecting history management logic, thereby solving the convergence failure at high insertion lengths.

---

## 1. C++ Source Modifications (`crm_torch_ext/csrc/`)

### Task 1.1: Audit & Bridge (`crm_step_op.h` / `.cpp`)
**Goal:** Verify access to the Core Integrator.
*   **Step 1:** Check `src/CRMDYN.hpp` inclusion.
*   **Step 2:** Verify `ABM4_coildyn` signature availability.
*   **Step 3:** If hidden, add `extern` declaration in `crm_step_op.cpp`.

### Task 1.2: Rewrite `crm_step_forward` (`crm_step_op.cpp`)
**Target:** Replace the entire body of `crm_step_forward`.

**Implementation Detail:**
1.  **Signature Update:**
    ```cpp
    std::vector<torch::Tensor> crm_step_forward(
        ...,
        torch::Tensor history, // [Batch, N, 3, 6]
        int64_t step_count
    )
    ```
2.  **History Unpacking:**
    ```cpp
    auto h_acc = history.accessor<double, 4>();
    double xdot_nm1[NUM_ACT_SET][6], xdot_nm2[NUM_ACT_SET][6], xdot_nm3[NUM_ACT_SET][6];
    if (step_count >= 3) {
        // Copy data from tensor to C++ arrays
        // Dimension 2 index 0 -> nm1
        // Dimension 2 index 1 -> nm2
        // Dimension 2 index 2 -> nm3
    }
    ```
3.  **Integrator Switch:**
    ```cpp
    if (step_count < 3) {
        // Use RK4 (Adaptive)
        // Store result in history[step_count]
    } else {
        // Use ABM4
        ABM4_coildyn(..., xdot_nm1, xdot_nm2, xdot_nm3, ...);
        // Shift history buffer: [0]<-[1], [1]<-[2], [2]<-current
    }
    ```
4.  **BVP Seed Injection:**
    *   **Crucial:** Pass the ABM4 *predicted* state (from `ABM4_coildyn` output) as the initial guess to `DynamicsBVP`. This is what fixes the convergence.

### Task 1.3: Update Bindings (`bindings.cpp`)
**Target:** `PYBIND11_MODULE`
*   **Action:** Update `m.def("crm_step", ...)` to reflect the new arguments.

---

## 2. Python Wrapper Modifications (`crm_ml_rl/wrappers/`)

### Task 2.1: Update `TorchCRMPhysics` (`torch_physics.py`)
**Target:** `CRMDynamicsStepFunction` class.

**Step 1: Forward Signature**
*   Update `forward` to accept `history` and `step_count`.
*   Pass them to the C++ function.
*   Return `next_history`.

**Step 2: Backward Signature**
*   Update `backward` to accept `grad_next_history`.
*   Return `None` for `grad_history` (Implementing the Gradient Stop).

**Step 3: State Management**
*   **`__init__`**: `self.history = None`, `self.step_count = 0`.
*   **`reset()`**: `self.history = torch.zeros(...)`, `self.step_count = 0`.
*   **`dyn_step()`**:
    *   Lazy allocation: `if self.history is None: allocate(...)`.
    *   Call: `out, hist = self.func.apply(..., self.history, self.step_count)`.
    *   Update: `self.history = hist.detach()`.
    *   Update: `self.step_count += 1`.

---

## 3. Verification & Validation

### Task 3.1: Unit Test (History Flow)
**Script:** `crm_torch_ext/test/test_history.py`
*   **Action:** Run 4 steps.
*   **Verify:**
    *   Step 0-2: History buffer fills up.
    *   Step 3: History buffer shifts correctly (Step 0 data discarded, Step 1 moves to slot 0).

### Task 3.2: Parity Test (The Standard)
**Script:** `scripts/verify_abm4_parity.py`
*   **Action:** Run Lemniscate on Wrapper and Extension.
*   **Verify:** Trajectories match within $10^{-10}\text{mm}$.

---

## 4. Execution Order

1.  **Core Audit:** Ensure we can call the function.
2.  **C++ Rewrite:** Modify `crm_step_op.cpp` and `bindings.cpp`.
3.  **Build:** Verify compilation.
4.  **Python Update:** Modify `torch_physics.py`.
5.  **Test:** Run validation scripts.