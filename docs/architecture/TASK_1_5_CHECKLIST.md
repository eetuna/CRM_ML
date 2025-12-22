# Task 1.5 Checklist: Gradients w.r.t Control Inputs (∂F/∂u)

**Goal**: Replace finite-difference computation of `B = ∂(next_state)/∂(currents)` with AutoDiff-based implicit differentiation.

**Key Challenge**: Actuation currents don't appear directly in the dynamics residual. They affect magnetic moment (`MagMoment = μ_scale * currents`), which enters via `CoilIntegrand`. This requires implementing an AD-differentiable path: `currents → MagMoment → CoilDynamics → residual`.

**Approach**:
1. Add `currentsToMagMoment<Scalar>` helper
2. Create AD residual variant that accepts currents as differentiable inputs
3. Compute `Jxu = ∂F/∂u` via autodiff
4. Use implicit differentiation: `B = dx/du = -[∂F/∂x]^(-1) * [∂F/∂u]`

---

## Step 0: Create a fresh working branch
1. `git checkout refactor/taskA1-modernize`
2. `git checkout -b task/1.5-control-input-gradients`

## Step 1: Read current plan/status (context & goals)
- `docs/architecture/DEVELOPMENT_TASKS.md` (Task 1.5 definition)
- `docs/architecture/PLAN_end_to_end_differentiable_simulator_options_A_B_C.md` (Option A context)
- `docs/architecture/PLAN_CHECK.md` (current state + remaining tasks)
- `docs/HANDOVER_AGENT_STATUS.md` (latest validations, benchmarks)

## Step 2: Identify the control inputs and expected output
- Determine which control inputs are in scope:
  - Actuation currents (3 values)
  - Insertion length (optional — decide whether to include)
- Confirm the output dimension:
  - `y = [tip_position(3), tip_velocity(3)]` (6D)

## Step 3: Inspect current bindings (entry points)
- File: `crm_ml_rl/wrappers/crm_bindings.cpp`
- Locate existing linearization APIs:
  - `linearize_action_from_seed` (FD on currents only) — line ~1351
  - `linearize_full_seed_action_from_seed` (full FD) — line ~1436
  - `linearize_full_seed_action_from_seed_implicit` (implicit AD for A, FD for B) — line ~1671
- Note the current `B` output behavior:
  - Currently: `B = ∂(next_state)/∂(currents)` via FD even in implicit mode
  - Goal: Replace FD `B` with AD-based computation

## Step 3.5: Verify existing baseline (before changes)
- Run existing implicit linearization test to confirm current FD behavior:
  ```bash
  CRM_DYN_LINEARIZATION_METHOD=implicit pytest tests/test_dynamics_implicit_linearization.py -v
  ```
- This establishes a baseline for comparison after AD implementation

## Step 4: Inspect AD residual implementation
- File: `src/CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp`
- Find:
  - The AD residual function used for `Jxx` (∂F/∂x) — line ~723 (`DYNNLEquationResidualEigenAD`)
  - Parameter unpacking logic — line ~126 (`unpackToADParams`)
  - **Critical**: Trace currents flow:
    - Currents do NOT appear directly in the residual
    - Currents → `MagMoment` transform happens in legacy `CoilDynamics_Defs.cpp:Prep`
    - `CoilIntegrand` (line ~271) uses `muhat` (skew-symmetric matrix of `MagMoment`)
    - For AD: Need to model `MagMoment = μ_scale * currents` explicitly

## Step 5: Design the ∂(next_state)/∂u path (control inputs)
- **Clarification**: Task 1.5 targets `B = ∂(next_state)/∂(currents)`, NOT `∂F_residual/∂u`
  - Output: `next_state = [tip_position(3), tip_velocity(3)]` (6D)
  - Input: `currents` (3D) or `[currents(3), insertion_length(1)]` (4D)
- Decide the AD input vector layout for controls:
  - **Recommended**: Start with currents only (size 3)
  - Insertion length can be added later if needed (size 4)
- Design the AD path:
  - Implement `currentsToMagMoment<Scalar>(currents, mu_scale) → MagMoment` helper
  - Create AD variant of residual that accepts currents as differentiable inputs
  - Compute `B = ∂(next_state)/∂(currents)` using implicit differentiation:
    - Given converged `x*` from `F(x*, u) = 0`
    - `dx/du = -[∂F/∂x]^(-1) * [∂F/∂u]`
    - Use existing `Jxx = ∂F/∂x` (already AD-based)
    - Compute new `Jxu = ∂F/∂u` via AD

## Step 6: Implement ∂F/∂u in C++
- **File**: `src/CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp`
  - Add `currentsToMagMoment<Scalar>(currents, mu_scale)` helper function:
    - Simple linear scaling: `MagMoment[i] = mu_scale * currents[i]`
    - Find `mu_scale` from existing parameter files (ratio of MagMoment to currents)
  - Create `DYNNLEquationResidualWithControlsAD<Scalar>(x, u_controls, Params)`:
    - Similar to `DYNNLEquationResidualWithParamsAD` (line ~893)
    - Unpack `u_controls` to `currents`, compute `MagMoment`, inject into `CoilDynamics`
  - Add `DYNNLEquationControlJacobianEigenAD(x_scaled, Params, ...)`:
    - Compute `Jxu = ∂F/∂u` using autodiff jacobian
    - Similar pattern to `DYNNLEquationParameterJacobianEigenAD` (line ~1067)
- **File**: `crm_ml_rl/wrappers/crm_bindings.cpp` (line ~1671)
  - Modify `linearize_full_seed_action_from_seed_implicit`:
    - After solving for converged `x*`, compute `Jxx` (already done)
    - Call new `DYNNLEquationControlJacobianEigenAD` to get `Jxu`
    - Compute `B = -Jxx^(-1) * Jxu` using implicit differentiation
    - Replace existing FD-based `B` computation with AD result
  - Keep FD fallback available if AD computation fails (optional but useful)

## Step 7: Update/extend tests
- **File**: `tests/test_dynamics_implicit_linearization.py`
  - Add test: `test_control_jacobian_ad_vs_fd`:
    - Compare AD-based `B` vs FD-based `B` within tolerance (e.g., `rtol=1e-3`)
    - **Critical**: Use `base_damping` fixture from `conftest.py` or validated damping values
    - Do NOT use arbitrary damping values (will cause "Coil integration Unbounded")
  - Verify existing tests still pass with AD `B`
- Run: `CRM_DYN_LINEARIZATION_METHOD=implicit pytest -q tests/test_dynamics_implicit_linearization.py`

## Step 8: Full test run
- `pytest -q`

## Step 9: Benchmark (recommended)
- Extend `scripts/benchmark_task1_4.py` or create `scripts/benchmark_task1_5.py`:
  - Add timings for AD-based `B` vs FD-based `B`
  - Test both `linearize_action_from_seed` (currents-only FD) and `linearize_full_seed_action_from_seed_implicit` (AD)
  - Compare against Task 1.4 baseline: `data/output/benchmark_task1_4.json`
  - Metrics to capture:
    - Time per `B` computation (AD vs FD)
    - Numerical accuracy (max absolute difference)
    - Success rate (convergence)
- Save results to `data/output/benchmark_task1_5.json` (ignored by git)
- Expected outcome: AD `B` should be faster than FD while maintaining accuracy

## Step 10: Documentation
- Update:
  - `docs/architecture/DEVELOPMENT_TASKS.md` (mark Task 1.5 complete)
  - `docs/HANDOVER_AGENT_STATUS.md` (tests + benchmark summary)

## Step 11: Commit and push
- Commit with a clear message (e.g., "Add AD ∂F/∂u for control inputs (Task 1.5)").
- Push branch and open PR against `refactor/taskA1-modernize`.

---

## Known Risks & Troubleshooting

### Risk 1: Dynamics divergence during AD perturbation
- **Symptom**: "Coil integration Unbounded!!" during AD jacobian computation
- **Mitigation**:
  - Use validated damping values from existing tests
  - Keep FD fallback in `linearize_full_seed_action_from_seed_implicit`
  - Consider clamping perturbations if needed

### Risk 2: `mu_scale` value unknown or varies
- **Symptom**: AD `B` doesn't match FD `B` numerically
- **Investigation**:
  - Check `data/catheter_params/CatheterParameterSet_1_dyn.txt` for MagMoment values
  - Run FK with known currents and verify MagMoment output
  - May need to extract `mu_scale` from legacy `Prep` function in `CoilDynamics_Defs.cpp`

### Risk 3: Performance regression vs FD
- **Symptom**: AD `B` is slower than FD `B`
- **Context**: Expected for small perturbations, but should be faster for large batches
- **Action**: Benchmark validates this; document trade-offs in HANDOVER_AGENT_STATUS.md

### Risk 4: Matrix inversion numerical issues
- **Symptom**: `B = -Jxx^(-1) * Jxu` produces NaN or Inf
- **Mitigation**:
  - Use Eigen's `ldlt().solve()` or `lu().solve()` instead of direct inverse
  - Check condition number of `Jxx` before inversion
  - Add regularization if needed (e.g., `Jxx + λI` for small λ)
