# Handover Plan: Modernizing C++ Core for Task A1 (Parameter Learning)

## 1. Objective
Refactor the C++ Dynamics Core from raw pointers to `Eigen` and `std::vector` to enable stable AutoDiff. This is required to finalize Task A1.

## 2. The Ground Truth
- **Logic Source:** Branch `autodiff_eigen`. This is where Task A1 math was originally implemented and verified.
- **Physics Source:** `Mexfiles/CRMDYN_c.cpp`. This is the original, stable C++ code used by Matlab. It is the "Bible" for the dynamics equations.
- **Verification Source:** `tests/test_parameter_jacobian_autodiff.py`. This test must pass with analytical gradients ($J_{AD}$) matching Finite Difference ($J_{FD}$) within $1e-4$.

## 3. Step-by-Step Task List

### Task 1: Struct Modernization
- Modify `src/CRM_BVPIVP_APIDeclarations.hpp`.
- Replace all fixed-size arrays (e.g., `double K[5][9]`) with `std::vector<Eigen::Matrix3d>`.
- Use `.resize()` in constructors to ensure memory safety and eliminate "Ghost Value" bugs.

### Task 2: Porting the Core Loop (`CRM_IVPSolver.cpp`)
- Rewrite `CRMSolverIVP_Core` to use the new Eigen-based structs.
- **MANDATORY:** Ensure the loop handles `FLEXIBLE` segments (ABM4 integration) and `RIGID` segments (linear transport). Skipping rigid segments will break the catheter kinematic chain.

### Task 3: Physics Implementation (`CoilDynamics_Defs.cpp`)
- Re-implement Newton-Euler equations using Eigen.
- Ensure the mathematical steps match the logic in `Mexfiles/CRMDYN_c.cpp` exactly.

### Task 4: AutoDiff & Bindings
- Update `src/CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp` to match the new Eigen math.
- Update `crm_bindings.cpp` to return all keys: `J_theta`, `residual`, `base`, `next_mL`, `next_nL`.

## 4. Risks to Avoid
- **Silent Truncation:** Do not use `write_file` for large `.cpp` files. Use `replace` or incremental writes.
- **Nuclear Rewrite:** Do not delete legacy logic assuming you can "simplify" it. Every line of the original physics must have a verified Eigen equivalent.