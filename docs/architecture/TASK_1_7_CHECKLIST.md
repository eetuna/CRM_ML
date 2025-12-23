# Task A1.7 Checklist: Core C++ Refactor

**Goal**: Replace the brittle legacy C++ core with a robust, modern foundation that supports AutoDiff natively and uses stable integration. This supersedes the temporary "Manual Sync" and "Acceleration Clamp" patches.

**Scope**: Major refactoring task with three phases:
1. Phase 1: Memory & Type Modernization (eliminate "Ghost Value" bug)
2. Phase 2: Solver Templatization (native AD support)
3. Phase 3: Integrator Stabilization (prevent "Coil integration Unbounded")

**Risk Level**: HIGH - This modifies core physics code used throughout the system.

---

## Status Notes (Verification Context)

- Phase 3 added RK4 + soft-failure support, but ABM4 remained the default integrator for backwards compatibility.
- The RK4 option exists to improve stability during stiff regimes (e.g., AD sweeps), not because ABM4 is universally broken.
- As of current verification, instability often originates from BVP solver convergence (large residuals) before integrator choice takes effect.
- Treat "Phase 3 complete" as "implemented and wired" until stability is validated against real failing cases and BVP convergence metrics.
- Seed-based `m_L/n_L` initialization fallback (bindings) is now in place; validation pending.
- Original recommended next steps (Claude) tracking:
  - Benchmark ABM4 vs RK4: **not done**.
  - Test RK4 on failing “Unbounded” cases: **done** (still fails).
  - Validate soft failure mode in production scenarios: **not done**.
  - Consider making RK4 default: **not done**.
  - Merge branch to main: **not done**.

## Pre-requisites

### Required Reading
- [ ] `docs/architecture/DEVELOPMENT_TASKS.md` - Section 3 ("Legacy Prep Overlap" Bug)
- [ ] `docs/architecture/DEVELOPMENT_TASKS.md` - Section 4 (Task A1.7 Overview)
- [ ] `src/CRM_BVPIVP_APIDeclarations.hpp` - `CRMIVPCoreParams` and `CRMShootingMethodParams` classes
- [ ] `src/CoilDynamics_Defs.cpp` - Current ABM4/RK2 integrator
- [ ] `src/CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp` - Existing templatized AD code

### Current State Analysis

**Mixed Container Types in `CRMIVPCoreParams` (lines 186-243 in CRM_BVPIVP_APIDeclarations.hpp)**:

| Type | Container | Example |
|------|-----------|---------|
| Modern (safe) | `std::vector<Eigen::Matrix3d>` | `K`, `Kinv`, `CoilAlignmentTurnAreaMatrix` |
| Modern (safe) | `std::vector<Eigen::Vector3d>` | `ustar`, `MagMoment`, `fcumlambda` |
| Modern (safe) | `std::vector<double>` | `ActMass`, `rho`, `LocMarkers` |
| **LEGACY (unsafe)** | `double [NUM_ACT_SET][N]` | `damping[6]`, `v_L_pre[3]`, `w_L_pre[3]`, `p_pre[3]`, `R_pre[9]`, `actInertia[9]`, `m_L[3]`, `n_L[3]` |

**Root Cause of "Ghost Value" Bug**: Fixed-size arrays with `NUM_ACT_SET` dimension cause memory layout issues when `NUM_ACT_SET` differs between compilation units or when struct padding varies.

---

## Phase 1: Memory & Type Modernization

### Step 1.1: Create New Container Types
**Goal**: Define type-safe containers for dynamics parameters.
**Status**: COMPLETED (implemented in task/1.7-core-refactor)

- [x] Create `src/CRM_DynamicsContext.hpp`:
  ```cpp
  #pragma once
  #include <vector>
  #include <Eigen/Dense>

  namespace CRMCatheterModel {

  struct ActuatorDynamicsParams {
      Eigen::Matrix<double, 6, 1> damping;   // Linear (3) + angular (3)
      Eigen::Matrix3d inertia;                // 3x3 inertia matrix
      double mass;                            // Actuator mass
      Eigen::Vector3d v_L_pre;                // Linear velocity (previous)
      Eigen::Vector3d w_L_pre;                // Angular velocity (previous)
      Eigen::Vector3d p_pre;                  // Position (previous)
      Eigen::Matrix3d R_pre;                  // Rotation (previous)
      Eigen::Vector3d m_L;                    // Moment at coil
      Eigen::Vector3d n_L;                    // Force at coil
  };

  struct DynamicsContext {
      std::vector<ActuatorDynamicsParams> actuators;  // [no_act_set]
      double DELTA_T;                                  // Time step

      void resize(int no_act_set) {
          actuators.resize(no_act_set);
      }
  };

  } // namespace CRMCatheterModel
  ```

- [x] Add unit test `tests/cpp/test_dynamics_context.cpp`

### Step 1.2: Migrate CRMIVPCoreParams
**Goal**: Replace fixed-size arrays with `DynamicsContext`.
**Status**: COMPLETED (implemented in task/1.7-core-refactor)

- [x] In `CRM_BVPIVP_APIDeclarations.hpp`, add `DynamicsContext` member to `CRMIVPCoreParams`:
  ```cpp
  class CRMIVPCoreParams {
  public:
      // ... existing members ...

      // NEW: Modern dynamics storage (replaces fixed-size arrays)
      DynamicsContext dynamics;

      // DEPRECATED: Keep for backwards compatibility during migration
      // Remove after Phase 2 completion
      double v_L_pre[NUM_ACT_SET][3];
      // ... etc ...
  };
  ```

- [x] Add `sync_dynamics_context()` method to copy from legacy arrays to new container
- [x] Add `sync_legacy_arrays()` method to copy back (for legacy code compatibility)

### Step 1.3: Migrate CRMShootingMethodParams
**Goal**: Apply same pattern to shooting method params.
**Status**: COMPLETED (implemented in task/1.7-core-refactor)

- [x] Add `DynamicsContext` member
- [x] Add sync methods
- [x] Update `CRMConstructShootingMethodParamSet()` to populate new container

### Step 1.4: Update Prep Functions
**Goal**: Ensure new containers are populated correctly.
**Status**: COMPLETED (implemented in task/1.7-core-refactor)

- [x] Modify `CRMSolverIVP_Prep()` (CRM_IVPSolver.cpp) to populate `DynamicsContext`
- [x] Modify `CRMDYNSolverIVP_Prep()` (CoilDynamics_Defs.cpp) to populate `DynamicsContext`
- [x] Verify no memory aliasing with test: compare `dynamics.actuators[0].damping` vs legacy `damping[0]`

### Step 1.5: Validation
**Status**: COMPLETED (tests re-run on current branch)

- [x] Run all existing tests: `pytest -q` (32 passed)
- [x] Verify "Ghost Value" bug is resolved: check that `packLearnableParams` returns correct values without manual sync
- [x] Run AD tests: `pytest tests/test_parameter_jacobian_autodiff.py -v` (3 passed)

---

## Phase 2: Solver Templatization

### Step 2.1: Create Templatized DynamicsContext
**Goal**: Allow `DynamicsContext` to hold `autodiff::real` types.
**Status**: COMPLETED (implemented in task/1.7-core-refactor)

- [x] Create `src/CRM_DynamicsContext_AD.hpp`:
  ```cpp
  #pragma once
  #include <autodiff/forward/real.hpp>
  #include <Eigen/Dense>

  namespace CRMCatheterModel {
  namespace dynnl_ad_eigen {

  template <typename Scalar>
  struct ActuatorDynamicsParamsAD {
      Eigen::Matrix<Scalar, 6, 1> damping;
      Eigen::Matrix<Scalar, 3, 3> inertia;
      Scalar mass;
      Eigen::Matrix<Scalar, 3, 1> v_L_pre;
      Eigen::Matrix<Scalar, 3, 1> w_L_pre;
      Eigen::Matrix<Scalar, 3, 1> p_pre;
      Eigen::Matrix<Scalar, 3, 3> R_pre;
      Eigen::Matrix<Scalar, 3, 1> m_L;
      Eigen::Matrix<Scalar, 3, 1> n_L;
  };

  template <typename Scalar>
  struct DynamicsContextAD {
      std::vector<ActuatorDynamicsParamsAD<Scalar>> actuators;
      double DELTA_T;  // Time step (not differentiated)

      void resize(int no_act_set) {
          actuators.resize(no_act_set);
      }
  };

  // Convert double context to AD context
  template <typename Scalar>
  DynamicsContextAD<Scalar> convertToAD(const DynamicsContext& ctx);

  } // namespace dynnl_ad_eigen
  } // namespace CRMCatheterModel
  ```

### Step 2.2: Update Existing AD Residual
**Goal**: Use new `DynamicsContextAD` in existing AD code.
**Status**: PARTIAL (parameter-gradient path updated)

- [ ] Refactor `DYNNLEquationResidualEigenAD` to accept `DynamicsContextAD<Scalar>`
- [x] Update `DYNNLEquationResidualWithParamsAD` similarly
- [ ] Update `DYNNLEquationResidualWithControlsAD` similarly

### Step 2.3: Templatize CRMFlexible_IVP_Back
**Goal**: Make backward integration AD-compatible.
**Status**: COMPLETED

- [x] Create `CRMFlexible_IVP_Back<Scalar>` template in `CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp`
- [x] Replace hard-coded `double` with `Scalar` template parameter
- [x] Use `DynamicsContextAD<Scalar>` for dynamics params

### Step 2.4: Templatize CoilIntegrand (if not already)
**Goal**: Ensure full AD path through coil dynamics.
**Status**: COMPLETED

- [x] Verify `CoilIntegrand<Scalar>` accepts all AD types
- [x] Verify `DYNSE3_TimeSpace<Scalar>` accepts all AD types
- [x] Verify `CoilDynamics<Scalar>` properly propagates gradients

### Step 2.5: Remove Shadow Structs
**Goal**: Eliminate `DYNNLEqnParamsAD` shadow struct.
**Status**: COMPLETED

- [x] Once `DynamicsContextAD` is working, deprecate `DYNNLEqnParamsAD`
- [x] Update all callsites to use new container
- [x] Remove manual sync layer from `crm_bindings.cpp`

### Step 2.6: Validation
**Status**: PARTIAL (parameter Jacobian AD tests run)
- [ ] Run: `pytest tests/test_dynnlequation_residual_eigen_autodiff.py -v`
- [x] Run: `pytest tests/test_parameter_jacobian_autodiff.py -v`
- [ ] Run: `pytest tests/test_dynamics_implicit_linearization.py -v`
- [x] Verify AD gradients match FD within tolerance

---

## Phase 3: Integrator Stabilization

### Step 3.1: Analyze Instability Sources
**Goal**: Understand when and why "Coil integration Unbounded" occurs.
**Status**: IN PROGRESS (diagnostics added; formal stability doc pending)

- [x] Add instrumentation to `CoilDynamics<Scalar>` to log max acceleration per step
- [ ] Identify parameter regimes that cause instability (partial; default damping vs stable damping noted)
- [ ] Document findings in `docs/architecture/INTEGRATOR_STABILITY.md` (pending)

### Step 3.2: Implement RK4 Option
**Goal**: Provide a more stable integration option.
**Status**: COMPLETED (implemented + wired)

- [ ] Create `CoilDynamicsRK4<Scalar>` in `CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp`:
  ```cpp
  template <typename Scalar>
  inline void CoilDynamicsRK4(
      const Vec6<Scalar>& v0w0,
      const Vec3<Scalar>& p0,
      const Mat3<Scalar>& R0,
      /* ... same interface as CoilDynamics ... */
      Vec6<Scalar>& v1w1,
      Vec3<Scalar>& p1,
      Mat3<Scalar>& R1,
      Vec6<Scalar>& out_xdot_n)
  {
      const int N = static_cast<int>(std::ceil(DELTA_T / kCoilTStep));
      Vec6<Scalar> twist = v0w0;
      Vec3<Scalar> p = p0;
      Mat3<Scalar> R = R0;

      for (int i = 0; i < N; ++i) {
          const Scalar h = Scalar(kCoilTStep);

          // RK4 stage 1
          Vec6<Scalar> k1;
          CoilIntegrand(twist, n_L, g, R, actMass, actInertia, damping, B0, muhat, m_L, k1);

          // RK4 stage 2
          Vec6<Scalar> twist_half1 = twist + h * Scalar(0.5) * k1;
          Mat3<Scalar> R_half1; Vec3<Scalar> p_half1;
          DYNSE3_TimeSpace(R, p, h * Scalar(0.5), twist, R_half1, p_half1);
          Vec6<Scalar> k2;
          CoilIntegrand(twist_half1, n_L, g, R_half1, actMass, actInertia, damping, B0, muhat, m_L, k2);

          // RK4 stage 3
          Vec6<Scalar> twist_half2 = twist + h * Scalar(0.5) * k2;
          Vec6<Scalar> k3;
          CoilIntegrand(twist_half2, n_L, g, R_half1, actMass, actInertia, damping, B0, muhat, m_L, k3);

          // RK4 stage 4
          Vec6<Scalar> twist_end = twist + h * k3;
          Mat3<Scalar> R_end; Vec3<Scalar> p_end;
          DYNSE3_TimeSpace(R, p, h, twist, R_end, p_end);
          Vec6<Scalar> k4;
          CoilIntegrand(twist_end, n_L, g, R_end, actMass, actInertia, damping, B0, muhat, m_L, k4);

          // RK4 update
          twist = twist + h * (k1 + Scalar(2)*k2 + Scalar(2)*k3 + k4) / Scalar(6);
          DYNSE3_TimeSpace(R, p, h, (twist + twist) * Scalar(0.5), R, p);  // Midpoint rule for SE3
      }

      v1w1 = twist;
      p1 = p;
      R1 = R;
      out_xdot_n = k1;  // Approximate final derivative
  }
  ```

### Step 3.3: Add Integrator Selection
**Goal**: Allow runtime selection of integrator.
**Status**: COMPLETED (C++ + Python set/get)

- [ ] Add `enum class IntegratorType { ABM4, RK4 }` to `DynamicsContext`
- [ ] Modify `CoilDynamics<Scalar>` to dispatch based on integrator type
- [ ] Add Python binding to set integrator type

### Step 3.4: Implement Adaptive Stepping (Optional)
**Goal**: Reduce instability by adapting step size.
**Status**: NOT STARTED (optional)

- [ ] Add acceleration magnitude check after each integrand evaluation
- [ ] If acceleration exceeds threshold, subdivide step
- [ ] Add max subdivision limit to prevent infinite loops

### Step 3.5: Add Soft Failure Mode
**Goal**: Return penalty instead of crashing on divergence.
**Status**: PARTIAL (coil + BVP residual fallback; production validation pending)

- [ ] Check for NaN/Inf in `CoilDynamics` output
- [ ] If detected, set output to large but finite values
- [ ] Return error flag that propagates to Python
- [ ] Update `crm_bindings.cpp` to handle error flag gracefully

### Step 3.6: Validation
**Status**: PARTIAL (targeted tests + failing case validation done)
- [x] Run stability test across parameter sweep (failing case now converges with new damping defaults)
- [~] Verify RK4 produces same results as ABM4 for stable regimes (10-case comparison shows differences but no divergence)
- [ ] Verify RK4 remains stable where ABM4 fails (no ABM4-fail/RK4-pass found yet)
- [x] Benchmark RK4 vs ABM4 performance (RK4 ~1.21x slower on failing case)

---

## Post-Implementation

### Documentation Updates
- **Status**: NOT STARTED (other than task-specific notes/logs)
- [ ] Update `docs/architecture/DEVELOPMENT_TASKS.md` - mark Task A1.7 complete
- [ ] Update `docs/HANDOVER_AGENT_STATUS.md` with migration summary
- [ ] Create `docs/architecture/CORE_REFACTOR_MIGRATION.md` documenting breaking changes

### Cleanup
- **Status**: NOT STARTED
- [ ] Remove deprecated legacy arrays from `CRMIVPCoreParams` (after verification)
- [ ] Remove manual sync layer from `crm_bindings.cpp`
- [ ] Remove `DYNNLEqnParamsAD` shadow struct
- [ ] Update all documentation to reflect new architecture

### Final Validation
- **Status**: PARTIAL
- [x] Full test suite: `pytest -q`
- [x] Build all targets: `cmake --build build`
- [ ] Run C++ tests: `cd build && ctest` (no test config present)
- [ ] Performance benchmark: compare against pre-refactor baseline

---

## Risk Mitigation

### Risk 1: Breaking Existing Functionality
**Mitigation**:
- Keep legacy arrays during Phase 1, only deprecate in Phase 2
- Run full test suite after each step
- Use feature flags to enable/disable new code paths

### Risk 2: Performance Regression
**Mitigation**:
- Benchmark critical paths before and after changes
- Use `std::vector::reserve()` to avoid reallocations
- Profile memory allocation patterns

### Risk 3: AD Gradient Accuracy
**Mitigation**:
- Always validate AD gradients against FD
- Keep tolerance tests: `rtol=1e-3`, `atol=1e-6`
- Add gradient magnitude sanity checks

### Risk 4: Memory Safety Regressions
**Mitigation**:
- Run with AddressSanitizer: `cmake -DCMAKE_CXX_FLAGS="-fsanitize=address"`
- Use Valgrind for memory leak detection
- Enable `-Wall -Wextra` warnings

---

## Estimated Complexity

| Phase | Effort | Risk | Files Affected |
|-------|--------|------|----------------|
| Phase 1 (Memory Modernization) | Medium | Low | 4-5 core files |
| Phase 2 (Templatization) | High | Medium | 6-8 files |
| Phase 3 (Integrator) | Medium | Medium | 2-3 files |

**Recommended Approach**: Complete phases sequentially with full validation between each phase. Do NOT attempt all three phases in a single PR.

---

## Commit Strategy

1. **Phase 1 Commits**:
   - `Add DynamicsContext container struct`
   - `Integrate DynamicsContext into CRMIVPCoreParams`
   - `Update Prep functions to populate DynamicsContext`
   - `Validate Phase 1: all tests pass`

2. **Phase 2 Commits**:
   - `Add DynamicsContextAD template`
   - `Refactor AD residual to use DynamicsContextAD`
   - `Remove shadow struct DYNNLEqnParamsAD`
   - `Validate Phase 2: AD tests pass`

3. **Phase 3 Commits**:
   - `Add CoilDynamicsRK4 integrator`
   - `Add integrator selection to DynamicsContext`
   - `Add soft failure mode for divergence`
   - `Validate Phase 3: stability tests pass`
