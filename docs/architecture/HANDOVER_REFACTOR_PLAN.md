# Handover Plan: Modernizing C++ Core for Differentiable Dynamics

## 1. Objective
Refactor the legacy Cosserat Rod dynamics engine from raw C pointers (`double*`) to modern, memory-safe containers (`std::vector`, `Eigen::Matrix`) to enable stable Automatic Differentiation (AutoDiff) for Parameter Learning (Task A1).

## 2. Background & Challenges
- **Legacy State:** The code uses fixed-size buffers and raw pointer arithmetic. During AutoDiff (which uses dual numbers), the internal `double*` pointers can cause aliasing ("Ghost Values"), where gradients from one parameter leak into another.
- **The Failure Mode:** Previous attempts at a "Nuclear Rewrite" resulted in lost logic (e.g., rigid segment transport) and truncation errors.
- **Requirement:** The new implementation must match the legacy branch (`feature/autodiff-parameter-gradients`) output to within `1e-10` precision for the same inputs.

## 3. Step-by-Step Task List for the Implementing Agent

### Task 1: Establish Ground Truth (Regression Testing)
- **Do not refactor yet.** Check out the legacy branch.
- Write a script to run a variety of catheter configurations (1-3 segments, free tip and constrained).
- Save the results (tip position, orientation, and residuals) to a JSON/CSV file. This is the **Gold Standard**.

### Task 2: Incremental Type Migration
Instead of rewriting files, migrate data structures one by one:
1.  **`CRMCatheterModelParams`**: Change fixed arrays to `std::vector`. Update the constructor to resize based on `no_flex_seg`.
2.  **`CRMIVPCoreParams`**: Move from raw arrays to `Eigen::Vector3d` and `Eigen::Matrix3d`. 
3.  **Interface Alignment**: Keep the function signatures in `.hpp` files stable as long as possible, using `.data()` to bridge `std::vector` to legacy functions during the transition.

### Task 3: Porting the Physics Logic
- **`CRM_IVPSolver.cpp`**: 
    - Port `CRMSolverIVP_Core`. 
    - **Critical:** Ensure the loop handles both `FLEXIBLE` (integration via ABM4) and `RIGID` (linear transport: `p += L * R.col(2)`) segments.
- **`CoilDynamics_Defs.cpp`**: 
    - Implement the Newton-Euler equations using Eigen expressions for readability and AD compatibility.
    - Match the exact order of operations found in the legacy `CoilIntegrad`.

### Task 4: AutoDiff Integration
- Create a template-based residual function `DYNNLEquationResidualEigenAD<Scalar>`.
- Map the internal state to the `Scalar` type (which will be `autodiff::real` or `double`).
- Ensure all physical constants (Gravity, B-field) are correctly cast to the `Scalar` type.

### Task 5: Linker & Binding Audit
- Verify that `crm_bindings.cpp` (pybind11) returns a dictionary with:
    - `J_theta`: The Jacobian matrix.
    - `residual`: The current residual vector.
    - `base`: A sub-dictionary containing the state (`v`, `w`, `p`, `R`, `mL`, `nL`) to allow Finite Difference comparison.

### Task 6: Validation against Gold Standard
- Run the new code against the inputs saved in **Task 1**.
- Verify tip positions match.
- Verify that `J_theta` gradients match Finite Difference approximations to at least 4 decimal places.

## 4. Files to Audit Carefully
- `src/CRM_BVPIVP_APIDeclarations.hpp`: The source of truth for structs.
- `src/CRM_IVPSolver.cpp`: The core integration loop.
- `src/CoilDynamics_Defs.cpp`: The physics engine.
- `src/CRM_BVPSolver.cpp`: The shooting method implementation.
- `crm_ml_rl/wrappers/crm_bindings.cpp`: The Python gateway.
