# Post-Refactor Status and Immediate Next Steps

**Date:** December 22, 2025
**Current Branch:** `refactor/taskA1-modernize` (Validated & Approved)

## 1. Verified Achievements (What is FIXED)
The recent refactor has successfully resolved several critical blockers:
- **Kinematic Chain Continuity:** `src/CRM_IVPSolver.cpp` now correctly handles `RIGID` segments and marker updates. The catheter shape is no longer broken.
- **Python-C++ Stride Safety:** `crm_bindings.cpp` has been hardened against non-contiguous numpy arrays using explicit C-strides and `mutable_unchecked` accessors.
- **AutoDiff Math Parity:** The templated Newton-Euler logic in `CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp` matches the legacy `CoilDynamics_Defs.cpp` logic exactly.
- **Solver Compilation:** The BVP template ambiguity in `minpack.hpp` is resolved.

## 2. The Persistent Blocker: The "Ghost Value" Bug
While the refactor was a success for general memory safety, the **Dynamics Parameters** still suffer from memory aliasing.

### Evidence (from `pytest -s` output)
When running `compute_parameter_jacobian`, Python reports the learnable parameter vector (`theta`) as:
`Base theta: [0.018448, 0.018448, 0.018448, ...]`
- **Diagnosis:** `0.018448` is the calculated Magnetic Moment. The fact that it appears in the `damping` and `stiffness` slots proves that these variables are still overlapping in memory.
- **Root Cause:** In `src/CRM_BVPIVP_APIDeclarations.hpp`, the dynamics parameters are still defined as raw fixed-size arrays:
  ```cpp
  double actInertia[NUM_ACT_SET][9];
  double damping[NUM_ACT_SET][6];
  double v_L_pre[NUM_ACT_SET][3];
  ```
  These are not yet moved to `std::vector` or `Eigen` types, allowing the "Legacy Prep Overlap" bug to persist.

## 3. Immediate Next Steps (Task A1.8)

### Step 1: Complete Struct Modernization
Convert the following fields in `CRMShootingMethodParams` and `CRMIVPCoreParams` to modern containers:
- `v_L_pre`, `w_L_pre`, `p_pre`, `R_pre` -> `std::vector<Eigen::Vector3d>` / `std::vector<Eigen::Matrix3d>`.
- `actInertia` -> `std::vector<Eigen::Matrix3d>`.
- `damping` -> `std::vector<Eigen::Matrix<double, 6, 1>>`.
- **Goal:** Once these are moved to heap-allocated, `.resize()`-able vectors, the memory overlap will be impossible.

### Step 2: Integrator Stabilization (Task A1.6)
Replace the **ABM4** explicit integrator in `CoilDynamics_Defs.cpp` with **RK4**. 
- **Reason:** During parameter learning, AutoDiff often perturbs parameters into "stiff" regimes where ABM4 explodes ("Coil integration Unbounded!!"). RK4 is more robust for these perturbations.

### Step 3: Multi-Actuator AutoDiff (Task A2)
Generalize the residual function in `src/CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp`.
- **Action:** Replace the `static_assert(NUM_ACT_SET == 1)` with a loop over `Params.no_act_set`.
- **Action:** Update `unpackToADParams` to handle multiple parameter sets.

## 4. Final Validation Gate
The project is considered "Task A1 Complete" only when:
1. `pytest tests/test_parameter_jacobian_autodiff.py` reports the **true** physical values for theta (e.g., damping $\approx 12.17$, not $0.018$).
2. Analytical gradients match Finite Difference to within $1e-4$ relative error across all parameters.
