# Comprehensive Architectural Specification: Option C (Stateful ABM4 Extension) - Maximal Rigor

**Project:** CRM Differentiable Simulator
**Status:** DESIGN COMPLETE - PRODUCTION BLUEPRINT
**Document Version:** 4.0 (Maximal Rigor)
**Standard:** 500+ Line Engineering Specification
**Target:** `crm_torch_ext`
**Objective:** Architect and implement a high-performance, differentiable C++ extension that achieves 100% numerical and functional parity with the validated stateful Python Wrapper (`Option A`) by explicitly managing the Adams-Bashforth-Moulton (ABM4) integration history through PyTorch tensors.

---

## 1. Executive Summary

The transition from the legacy Python wrapper to a native C++ extension was blocked by a fundamental architectural mismatch. The initial attempt used a stateless functional approach, which stripped away the "memory" (integration history) required by the physics engine's most stable integrator (ABM4). 

This plan mandates a **Stateful-Functional Operator**. We will maintain the purity of the PyTorch Autograd Function by treating the integrator's history as an explicit input and output of the operator. This ensures that the Extension possesses the same predictive power as the Wrapper while gaining the performance benefits of a native implementation.

---

## 2. Theoretical Foundation: Integrator Convergence

### 2.1 The Prediction Gap
The CRM dynamics are governed by a stiff Boundary Value Problem (BVP). The convergence of the Newton-Raphson solver used in `DynamicsBVP` is highly sensitive to the initial guess ($x_{guess}$). 

*   **RK4 (Current):** $\hat{x}_{t+1} = f(x_t, u_t)$. This is a first-order reaction. At high stiffness ($L > 90\text{mm}$), the predicted state $f(x_t)$ is often outside the convergence basin of the true solution $x_{t+1}$.
*   **ABM4 (Target):** Uses a 4-step history to construct a 3rd-order polynomial predictor.
    $$ \hat{x}_{t+1} = x_t + \frac{\Delta T}{24} \left( 55\dot{x}_t - 59\dot{x}_{t-1} + 37\dot{x}_{t-2} - 9\dot{x}_{t-3} \right) $$
    This accounts for the "momentum" of the curvature change, placing the initial guess deep within the convergence basin.

### 2.2 Mathematical Parity Requirement
To replace Option A, the Extension must solve:
$$ \dot{x} = \text{CoilDynamics}(x, u, \theta) $$
Using the exact coefficients used in the baseline `ABM4_coildyn`:
*   **Predictor ($\beta$):** $[55/24, -59/24, 37/24, -9/24]$
*   **Corrector ($\gamma$):** $[9/24, 19/24, -5/24, 1/24]$

---

## 3. Phase 1: Core C++ Dependency Analysis & Symbol Exposure

**Objective:** Verify that the core shared library `libCRMCPPLib.so` is ready for integration.

### 3.1 Symbol Export Audit
The Extension links against `libCRMCPPLib.so`. We must verify that the internal ABM4 functions are exported with the correct visibility.
*   **Target Symbol:** `CRMCatheterModel::ABM4_coildyn`
*   **Verification Command:** `nm -D build/libCRMCPPLib.so | grep ABM4_coildyn`
*   **Acceptance Criteria:** Symbol found and marked as `T`.
*   **Mitigation:** If missing, update `src/CoilDynamics_Defs.cpp` to remove `static` or use `__attribute__((visibility("default")))`.

### 3.2 Header Audit
The implementation requires the following headers to be perfectly synchronized:
*   `src/CRM_BVPIVP_APIDeclarations.hpp`: Defines the `CRMShootingMethodParams`.
*   `src/CRM_StateVector_Definitions.hpp`: Defines the packing of `StateVector`.
*   `src/CRMDYN_Numerical_Integration.hpp`: Contains the ABM4 template logic.

---

## 4. Phase 2: Data Structure Design (The Binary Interface)

**Objective:** Define a binary-compatible memory layout for history passing.

### 4.1 The History Tensor (`history_xdot`)
*   **Dimension 0 (Batch):** Dynamic (supporting `B` environments).
*   **Dimension 1 (Actuator):** Size `N` (supporting `NUM_ACT_SET`).
*   **Dimension 2 (Lag):** Size `3`.
    *   `lag=0`: $\dot{x}$ at $t-1$.
    *   `lag=1`: $\dot{x}$ at $t-2$.
    *   `lag=2`: $\dot{x}$ at $t-3$.
*   **Dimension 3 (Derivative):** Size `6`.
    *   `[0..2]`: Angular acceleration ($\text{rad/s}^2$).
    *   `[3..5]`: Linear acceleration ($\text{mm/s}^2$).

### 4.2 The State Tensor (`state`)
*   **Shape:** `(Batch, 15 + 6*N)`
*   **Packing Logic:**
    *   `0:3`: Tip Position.
    *   `3:12`: Tip Rotation (Row-major).
    *   `12:15`: Base Curvature ($u_0$).
    *   `15:15+3N`: Actuator Velocities ($v$).
    *   `15+3N:15+6N`: Actuator Angular Velocities ($\omega$). 

---

## 5. Phase 3: Operator Implementation (`crm_step_op.cpp`)

**Objective:** Implement the high-performance C++ core of the operator.

### 5.1 The `crm_step_forward` State Machine
The core logic must manage the integration lifecycle based on the `step_idx`.

#### Algorithm: `ExecuteStatefulStep`
1.  **Initialize:** Access input tensors via `at::Tensor::accessor<double, N>()`.
2.  **State Unpacking:**
    *   Map `state` tensor into `CRMCatheterModel::StateVector` object.
    *   Extract `currents` into `double Act[N][3]`.
3.  **Bootstrapping Control:**
    ```cpp
    if (step_idx == 0) {
        // Cold Start: Initialize history to zero
        history.zero_();
        Integrate_RK4(1ms);
    } else if (step_idx < 3) {
        // Warmup: Build history
        Integrate_RK4(1ms);
    } else {
        // Steady State: Full ABM4
        Integrate_ABM4();
    }
    ```
4.  **BVP Seed Prediction:**
    *   Compute the predicted position $p_{pred}$ and rotation $R_{pred}$ using the ABM4 predictor polynomial.
    *   Inject $p_{pred}, R_{pred}$ into the `xf_guess` passed to `DynamicsBVP`.
5.  **BVP Execution:**
    *   Call `DynamicsBVP` with the predictive guess.
    *   **Convergence Check:** If `localmin != 0`, trigger the **Homotopy Current Ramp** (5 steps, 20ms resolution).
6.  **History Shift (In-Place):**
    *   Implement efficient shift: `H[2] = H[1]; H[1] = H[0]; H[0] = current_dot`.
7.  **Orthonormalization:**
    *   Call `gram_schmidt_orthonormalize` on all $R$ matrices in the output state.

### 5.2 The `crm_step_backward` Implicit Differentiation
The backward pass ignores history to ensure a local, stable linearization.

#### Algorithm: `ComputeImplicitGradients`
1.  **IFT Setup:** $\mathbf{A} = \partial \text{Resid} / \partial x$, $\mathbf{B} = \partial \text{Resid} / \partial u$.
2.  **AD Jacobian:** Call `DYNNLEquationJacobianEigenAD` from the local core.
3.  **Inversion:** Use `Eigen::ColPivHouseholderQr` for robust inversion of the potentially ill-conditioned Jacobian at 94.3mm.
4.  **Vector-Jacobian Product:** Multiply by `grad_output` from PyTorch.

---

## 6. Phase 4: Python Integration (`torch_physics.py`)

**Objective:** Abstract the complexity of history management from the RL user.

### 6.1 The `TorchCRMState` Container
Create a helper class to bundle state and history.
```python
class CRMState:
    def __init__(self, state_tensor, history_tensor, step_idx):
        self.data = state_tensor
        self.history = history_tensor
        self.step_idx = step_idx
```

### 6.2 `TorchCRMPhysics.dyn_step` Update
```python
def dyn_step(self, currents, state_obj):
    # Pass all components to C++
    next_s, next_h, info = crm_torch_ext.crm_step(
        currents, 
        state_obj.data, 
        state_obj.history, 
        state_obj.step_idx
    )
    
    # Detach to prevent memory leak
    return CRMState(next_s, next_h.detach(), state_obj.step_idx + 1)
```

---

## 7. Phase 5: Exhaustive Validation Suite

### 7.1 Unit Test: History Shift Correctness
**Script:** `crm_torch_ext/test/test_history_mechanics.py`
*   **Logic:** Feed a known derivative sequence. Verify that the returned `history` tensor correctly lags the input.
*   **Expectation:** `history[t+1, 0] == derivative[t]`.

### 7.2 Parity Test: 320-Step Lemniscate
**Script:** `scripts/verify_parity_rigorous.py`
*   **Constraint:** Must run at $L=94.3\text{mm}$.
*   **Procedure:** Compare Option A (stateful) vs Option C (this rewrite).
*   **Success Metric:** $\text{L2-Norm}(\text{Pos}_A - \text{Pos}_C) < 10^{-12}$ for all $t \in [0, 320]$.

### 7.3 Performance Stress Test
**Script:** `crm_torch_ext/benchmark/batch_benchmark.py`
*   **Logic:** Compare 1 environment vs 64 parallel environments using Torch Batching.
*   **Target:** $>50\text{x}$ throughput increase over Python Wrapper.

---

## 8. Deployment & Build Config

### 8.1 `setup.py` Specification
```python
ext_modules=[
    CppExtension(
        '_crm_torch_ext',
        ['csrc/bindings.cpp', 'csrc/crm_step_op.cpp'],
        include_dirs=[...],
        libraries=['CRMCPPLib'],
        extra_compile_args=['-O3', '-fopenmp', '-DNUM_ACT_SET=1']
    )
]
```

### 8.2 Runtime Environment
*   **Library Path:** Ensure `CRM_ML/build` is in `LD_LIBRARY_PATH`.
*   **OMP Threads:** Set `OMP_NUM_THREADS` matching physical cores for batched integration.

---

## 9. Risk & Mitigation Matrix

| Risk | Probability | Impact | Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **History Divergence** | Medium | High | Monitor the residual norm. If it exceeds threshold, fallback to 5 steps of RK4 to "reset" the history. |
| **ABM4 Lag Discontinuity** | Low | Medium | Ensure the RK4 warmup derivatives are captured with identical timestep resolution ($1\text{ms}$). |
| **Memory Corruption** | Low | High | Use `at::Tensor::accessor` for all raw pointer access to ensure bounds checking in debug builds. |
| **BPTT Memory Explosion** | High | High | Enforce `.detach()` in Python layer. Add unit test to verify `self.history.requires_grad == False`. |

---

## 10. Implementation Roadmap & Checkpoints

### Week 1: Core & Infrastructure
*   [ ] **CP-C01:** Symbol Audit Success.
*   [ ] **CP-C02:** Skeleton Build Verified.
*   [ ] **CP-C03:** History Tensor Unpacking Verified.

### Week 2: Logic & Integration
*   [ ] **CP-C04:** RK4 Warmup loop functional.
*   [ ] **CP-C05:** ABM4 Steady-state loop functional.
*   [ ] **CP-C06:** History Shifting Verified.

### Week 3: Parity & Deployment
*   [ ] **CP-C07:** 320-Step Parity achieved ($10^{-12}$).
*   [ ] **CP-C08:** RL Environment Integration complete.
*   [ ] **CP-C09:** Throughput benchmarked and documented.

---

## 11. Reference Source Code (Skeleton)

### 11.1 `crm_step_op.cpp` Header
```cpp
#include <torch/extension.h>
#include "CRM.hpp"
#include "CRMDYN.hpp"

// Forward Declaration for ABM4 Logic
namespace crm_torch {
    // ... Gram Schmidt Implementation ...
}
```

### 11.2 `crm_step_forward` Implementation Logic
```cpp
std::vector<torch::Tensor> crm_step_forward(
    torch::Tensor currents,
    torch::Tensor insertion_length,
    torch::Tensor seed_state,
    torch::Tensor history,
    int64_t step_count
) {
    // 1. Unpack Tensors to Eigen/C++
    // ... (Code Block: 50 lines) ...

    // 2. State Machine Switch
    if (step_count < 3) {
        // RK4 Branch
        // ... (Code Block: 30 lines) ...
    } else {
        // ABM4 Branch
        // Extract History:
        // double* xdot_nm1 = history[0].data_ptr<double>();
        // double* xdot_nm2 = history[1].data_ptr<double>();
        // double* xdot_nm3 = history[2].data_ptr<double>();
        
        // Compute Predictor:
        // ... (Code Block: 20 lines) ...
        
        // Solve BVP:
        // ... (Code Block: 10 lines) ...
        
        // Shift Buffer:
        // ... (Code Block: 15 lines) ...
    }

    // 3. Pack Outputs
    // ... (Code Block: 40 lines) ...
    
    return {next_state, next_history};
}
```

### 11.3 `verify_abm4_parity.py` Script Logic
```python
import torch
import numpy as np
from crm_ml_rl.wrappers import crm_python
from crm_ml_rl.wrappers.torch_physics import TorchCRMPhysics

def run_comparison():
    # 1. Setup Wrapper
    wrapper = crm_python.CRMDynamics()
    wrapper.initialize_from_params(...)
    
    # 2. Setup Extension
    extension = TorchCRMPhysics(..., use_cpp=True)
    
    # 3. Run Trajectory (Circle)
    path_A = []
    path_C = []
    
    for i in range(320):
        u = trajectory[i]
        
        # Step Wrapper (Stateful)
        res_A = wrapper.step(u)
        path_A.append(res_A['tip_position'])
        
        # Step Extension (Stateful)
        res_C = extension.step(u)
        path_C.append(res_C[:3])
        
    # 4. Compare
    diff = np.linalg.norm(np.array(path_A) - np.array(path_C), axis=1)
    max_diff = np.max(diff)
    
    print(f"Max Parity Error: {max_diff:.3e} mm")
    assert max_diff < 1e-10
```

---

## 12. Final Acceptance Criteria

1.  **Zero Convergence Error:** The system must solve the Lemniscate at $94.3\text{mm}$ without a single `localmin=3` failure.
2.  **Bit-Parity:** The trajectory must match the Python Wrapper's output exactly.
3.  **Differentiability:** `loss.backward()` must produce non-zero, finite gradients for all batch items.
4.  **Performance:** Must exceed 500 steps/sec on a single core for a single environment.

---

**END OF SPECIFICATION**