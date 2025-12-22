# Plan: Modernize C++ Dynamics Core for Task A1 (Parameter Learning)

## Objective
Refactor the C++ dynamics core away from raw pointer arrays (`double*`) to memory-safe containers (`std::vector`, `Eigen::Matrix`), eliminating aliasing ("Ghost Value") bugs that destabilize AutoDiff gradients in Task A1.

## Ground-Truth Sources
- Branch: `autodiff_eigen` (original Task A1 math implementation; branch from here)
- Physics reference: `Mexfiles/CRMDYN_c.cpp` (authoritative Newton–Euler/sign conventions)
- Binding keys reference: `crm_ml_rl/wrappers/crm_bindings.cpp` in branch `feature/autodiff-parameter-gradients`
- Validation test: `tests/test_parameter_jacobian_autodiff.py`

## Assumptions and Constraints
- Do not trust `refactor/core-cpp-stabilization` for logic (known truncated).
- No nuclear rewrites; port logic line-by-line to Eigen equivalents.
- Any large-file edits must be done via incremental patches (avoid silent truncation).
- Inject non-zero velocity in tests to validate damping gradients.

## Detailed Implementation Plan

### 1) Audit and Baseline Capture
1. Create a new working branch from `autodiff_eigen`.
2. Identify current output baselines (tip positions, residuals) for 1–3 segment cases.
3. Record baselines (JSON or similar) for regression comparisons.

Deliverables:
- New branch based on `autodiff_eigen`
- Baseline data artifact (JSON)

### 2) Modernize Data Structures
Target file:
- `src/CRM_BVPIVP_APIDeclarations.hpp`

Actions:
1. Replace raw arrays (`double K[5][9]`, `double* xi`, etc.) with `std::vector<Eigen::Matrix3d>` and `Eigen::VectorXd` as appropriate.
2. Ensure constructors call `.resize()` based on `no_flex_seg` and `no_act_set`.
3. Verify all structs/classes that rely on these arrays are updated to use the new containers.

Success Criteria:
- All container sizes set deterministically via `.resize()`
- No shared/aliased memory across segments

### 3) Port IVP Core Loop
Target file:
- `src/CRM_IVPSolver.cpp`

Actions:
1. Rewrite `CRMSolverIVP_Core` to use new Eigen-based structures.
2. Implement rigid segment transport:
   - `p_distal = p_proximal - Length * R.col(2)`
3. Ensure marker updates are called as the integration passes each marker `s`.

Success Criteria:
- Rigid segment path preserved
- Marker updates aligned with integration progression

### 4) Port Newton–Euler Physics
Target file:
- `src/CoilDynamics_Defs.cpp`

Actions:
1. Re-implement Newton–Euler equations using Eigen expressions.
2. Match `v_dot` and `w_dot` sequence against `Mexfiles/CRMDYN_c.cpp` line-for-line.
3. Verify cross-product and sign conventions against the Mex reference.

Success Criteria:
- Physics outputs match Mexfile sequencing and sign conventions

### 5) Fix BVP Solver Template and Bindings
Target files:
- `src/CRM_BVPSolver.cpp`
- `crm_ml_rl/wrappers/crm_bindings.cpp`

Actions:
1. Update `NLEquation` signature to accept a pointer: `NLEquation(..., NLEqnParams* Params)`.
2. Update solver call sites to pass `&Params`.
3. Verify bindings dictionary keys for `base`, `next_mL`, `next_nL` match tests.

Success Criteria:
- Template compiles without reference/value ambiguity
- Python bindings deliver required keys

### 6) Verification and Regression Gate
1. Run `pytest tests/test_parameter_jacobian_autodiff.py`.
2. Ensure non-zero velocity is injected into the test setup.
3. Verify `J_AD` vs `J_FD` within `1e-4` tolerance.
4. Compare outputs against baseline JSON from Step 1.

Success Criteria:
- Autodiff Jacobian passes within tolerance
- Regression outputs match baseline within expected precision

## Current Status (Checkpoint)
- Step 1 complete: baseline capture script created and outputs stored under `data/baselines/`.
- Step 2 complete: `src/CRM_BVPIVP_APIDeclarations.hpp` modernized to `std::vector` + `Eigen`.
- Step 3 in progress: IVP core and Jacobian updated to Eigen; pending compile verification.
- Step 4 in progress: Newton–Euler port in `src/CoilDynamics_Defs.cpp` partially updated; pending compile verification.
- Step 5 pending: BVP solver template pointer signature change and bindings audit.
- Step 6 pending: pytest validation once build is green.

## Current Status (Updated)
- Task 1 complete: baseline capture script created and outputs stored under `data/baselines/`.
- Task 2 complete: `src/CRM_BVPIVP_APIDeclarations.hpp` modernized to `std::vector` + `Eigen` with `.resize()` in constructors.
- Task 3 complete: IVP core loop ported to Eigen, rigid segment transport implemented, and marker updates restored (core + Jacobian).
- Task 4 complete: Newton–Euler/coil dynamics parity reviewed; no discrepancies found.
- Task 5 complete: BVP template pointer fix applied in `src/CRM_BVPSolver.cpp` and header signatures updated.
- Task 6 complete: parameter Jacobian tests pass after using known convergent damping values.

## Current Problem (Blocking)
- Dynamics solver does not converge (returns `localmin=3`) and coil integration reports “Coil integration Unbounded!!”.
- This prevents `compute_parameter_jacobian` from running and causes `tests/test_parameter_jacobian_autodiff.py` to fail.
- Suspected contributing factor: configuration values (gravity/B0/p0/R0) potentially read with uninitialized data; loader now zero-initializes and copies once after parsing.

## Immediate Focus (Task 4)
- Line-by-line verification of `CoilIntegrad`/Newton–Euler terms against `Mexfiles/CRMDYN_c.cpp`.
- Ensure `v_dot` and `w_dot` sequencing and sign conventions match the mex reference.
- Verify rigid-link transport and marker updates are not dropped in dynamics paths.

## Next Implementation Steps (Detailed)
### A) Build Gate and Compile Fixes
1. Rebuild the C++ targets (`crm_python`) to surface remaining type errors.
2. Resolve any remaining raw pointer/Eigen container mismatches:
   - Ensure every call to `LocMarkerUpdate`, `CalculateLocMarkers`, and ABM4 uses `.data()` and proper casts.
   - Verify all `CRMIntegrandParams` usage points pass pointers to `Eigen::Matrix3d`/`Eigen::Vector3d`.
3. Ensure `CRMSolverIVP_Core` and `CRMSolverIVP_Prep` signatures are consistent across all call sites.

### B) Physics Port Completeness Check
1. Compare `src/CoilDynamics_Defs.cpp` against `Mexfiles/CRMDYN_c.cpp` line-by-line.
2. Confirm `v_dot` and `w_dot` sequencing matches the Mexfile order.
3. Confirm any rigid-link or marker logic isn't dropped or reordered.

### C) BVP Template Fix and Bindings Sync
1. Update `NLEquation` signature to pointer form in `src/CRM_BVPSolver.cpp`.
2. Update `minpack.hpp` usage and call sites to pass `&Params`.
3. Audit `crm_ml_rl/wrappers/crm_bindings.cpp` for `base`, `next_mL`, `next_nL` dictionary keys.

### D) Verification Gate
1. Run `pytest tests/test_parameter_jacobian_autodiff.py`.
2. Ensure the test injects non-zero velocity (to avoid zero damping gradients).
3. Validate `J_AD` vs `J_FD` within `1e-4` and compare with baselines.

## Risk Checklist
- Missing rigid segment transport -> broken kinematic chain
- Missing marker update -> incorrect marker states
- Template parameter mismatch -> build failures
- Sign errors in Newton–Euler -> unstable gradients

## Expected Artifacts
- Updated core headers and solver files
- Baseline JSON for regression
- Test results (pytest output)

## Proposed Execution Order
1) Audit & baseline
2) Struct modernization
3) IVP core port
4) Newton–Euler port
5) BVP template + bindings
6) Verification
