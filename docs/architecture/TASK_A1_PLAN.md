# Implementation Plan: Task A1 - Parameter Gradients

**Goal:** Implement $\partial F/\partial \theta$ (parameter Jacobian) for all 16 learnable parameters in C++, exposed to Python for selective training.

## 1. Context & Strategy
The goal is to enable system identification by differentiating the dynamics residual $F(x, \theta) = 0$ with respect to physical parameters $\theta$.
*   **Strategy:** Implement gradients for ALL 16 parameters in C++ (write once), then selectively train in Python.
*   **Files:**
    *   `src/CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp`: Core autodiff implementation.
    *   `crm_ml_rl/wrappers/crm_bindings.cpp`: Python bindings.

## 2. Learnable Parameters (16 Total)
| Parameter | Dim | Notes |
| :--- | :--- | :--- |
| `damping` | 6 | Linear (3) + Angular (3) coefficients |
| `K_diag` | 3 | Stiffness matrix diagonal (Young's/Shear modulus) |
| `ustar` | 3 | Rest curvature vector |
| `actMass` | 1 | Actuator mass |
| `MagMoment` | 3 | Magnetic moment vector |

## 3. Implementation Steps

### Step 1: Shadow Parameter Struct
Create `DYNNLEqnParamsAD<Scalar>` in `src/CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp`:
```cpp
template <typename Scalar>
struct DYNNLEqnParamsAD {
    Eigen::Matrix<Scalar, 6, 1> damping;
    Eigen::Matrix<Scalar, 3, 3> K;
    // ... other learnable params as Scalar types
    const DYNNLEqnParams* base_params; // Pointer to fixed params
};
```

### Step 2: Pack/Unpack Utilities
Implement helper functions to convert between legacy params and AD struct:
*   `packLearnableParams`: `DYNNLEqnParams` -> `VectorXd` (flat 16 vector)
*   `unpackToADParams`: `VectorXreal` -> `DYNNLEqnParamsAD<real>`

### Step 3: Update Residual Function
Refactor `DYNNLEquationResidualEigenAD` to accept the templated parameter struct:
```cpp
template <typename Scalar>
Eigen::Matrix<Scalar, NUM_DYN_RESIDUAL, 1>
DYNNLEquationResidualEigenAD_WithParams(
    const Eigen::Matrix<Scalar, -1, 1>& x_scaled,
    const Eigen::Matrix<Scalar, -1, 1>& theta, 
    const DYNNLEqnParams* base_params);
```

### Step 4: Implement Parameter Jacobian
Add `DYNNLEquationParameterJacobianEigenAD`:
*   Uses `autodiff::jacobian` w.r.t `theta`.
*   Treats `x` (state) as fixed constants during this differentiation.

### Step 5: Python Bindings
Expose via `crm_bindings.cpp`:
*   `compute_parameter_jacobian(...)`: Returns `{"J_theta": Matrix, "theta": Vector}`.

### Step 6: Validation Tests
*   Create `tests/test_parameter_gradients.py`.
*   Validate $\partial F/\partial \theta$ against Finite Differences.

## 4. Constraints
*   **Do NOT** modify `src/numerical/` or `src/CRM_MatrixOperations.hpp`.
*   **Scope:** `NUM_ACT_SET == 1` only.
*   **Compatibility:** Ensure existing state Jacobian (`Jxx`) logic remains broken.
