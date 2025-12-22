# Comprehensive Handover Plan: Modernizing C++ Core for Task A1 (Parameter Learning)

## 1. Primary Objective
Refactor the C++ Dynamics Core from raw pointer arrays (`double*`) to modern, memory-safe containers (`std::vector`, `Eigen::Matrix`). This is the **only** way to stabilize the AutoDiff engine for Task A1. 

**The Problem:** The current legacy implementation in `autodiff_eigen` uses raw pointer arithmetic which causes **Memory Aliasing ("Ghost Values")**. During AutoDiff dual-number propagation, the gradients of different parameters overwrite each other, leading to incorrect or zeroed-out Jacobians.

## 2. The Ground Truth Sources (Trust These ONLY)
- **Logic Source:** Branch `autodiff_eigen`. This is where Task A1 math was first implemented.
- **Physics Bible:** `Mexfiles/CRMDYN_c.cpp`. This is the original C++ code used by Matlab. It is verified and stable. Refer to this for any cross-product or sign ambiguity.
- **Regression Reference:** `matlab/linear_identification.m`. Use this to verify the Newton-Euler residual logic.
- **Final Validator:** `tests/test_parameter_jacobian_autodiff.py`. Analytical gradients ($J_{AD}$) must match Finite Difference ($J_{FD}$) within $1e-4$ relative error.

## 3. Step-by-Step Task List

### Task 1: Baseline Capture (Regression Testing)
- Checkout branch `autodiff_eigen`.
- Create a script to run the solver for multiple catheter configs (1, 2, and 3 segments).
- Save the resulting tip positions and coil residuals to a JSON file.
- **Goal:** The refactored code must match these outputs to within `1e-10` precision.

### Task 2: Incremental Struct Modernization
- **File:** `src/CRM_BVPIVP_APIDeclarations.hpp`.
- Replace all raw buffers (e.g., `double K[5][9]`, `double* xi`) with `std::vector<Eigen::Matrix3d>` and `Eigen::VectorXd`.
- **CRITICAL:** Update constructors to use `.resize()` based on `no_flex_seg` and `no_act_set`.
- **Why:** This ensures that each segment has its own private memory space, eliminating aliasing.

### Task 3: Porting the Core Loop (`CRM_IVPSolver.cpp`)
- Rewrite `CRMSolverIVP_Core`. 
- **The Rigid Segment Trap:** The initial refactor failed because it dropped the `else` block for `RIGID` segments. You **must** implement linear transport for rigid segments: `p_tip = p_base - Length * R.col(2)`.
- **The Marker Trap:** Ensure `LocMarkerUpdate` is called correctly as the integration passes the marker's `s` coordinate.

### Task 4: Physics Implementation (`CoilDynamics_Defs.cpp`)
- Re-implement the Newton-Euler equations using Eigen.
- **Requirement:** Do not "simplify." Match the exact sequence of the legacy `CoilIntegrad`.
- Verify `v_dot` and `w_dot` calculations against the `Mexfiles/CRMDYN_c.cpp` version.

### Task 5: BVP Solver & Template Synchronization (`CRM_BVPSolver.cpp`)
- **Template Error:** The `TrustRegionDogleg` template in `minpack.hpp` fails if `ParamType` is ambiguous (Reference vs Value).
- **Fix:** Declare `NLEquation` to take a pointer: `void NLEquation(..., NLEqnParams* Params)`.
- Pass parameters as pointers (`&Params`) to the solver.

### Task 6: Python Binding Audit (`crm_bindings.cpp`)
- Ensure the wrapper returns a dictionary with:
    - `J_theta`: Full Jacobian.
    - `residual`: Current residual.
    - `base`: Sub-dictionary containing `next_mL`, `next_nL`, `v`, `w`, `p`, `R`, `xf`. 
    - **Why:** The validation test uses these to perform state-locked Finite Difference comparisons.

## 4. Operational Safety (Agent Warnings)
- **NO SILENT TRUNCATION:** If writing large files, use `replace` or multiple writes. Tool token limits will cut off the bottom of your file.
- **NO NUCLEAR REWRITES:** Do not assume you can "clean up" the logic. Every line of deleted legacy code must be accounted for in the new Eigen version.
- **NON-ZERO VELOCITY:** To test damping gradients, you must inject non-zero velocity into the test case ($dF/dd = -v$). If $v=0$, gradients will be zero even if the code is correct.
