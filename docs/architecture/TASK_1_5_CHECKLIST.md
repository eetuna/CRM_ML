# Task 1.5 Checklist: Gradients w.r.t Control Inputs (∂F/∂u)

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
  - `linearize_action_from_seed` (FD on currents only)
  - `linearize_full_seed_action_from_seed` (full FD)
  - `linearize_full_seed_action_from_seed_implicit` (implicit AD)
- Note the current `B` output behavior (FD vs AD).

## Step 4: Inspect AD residual implementation
- File: `src/CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp`
- Find:
  - The AD residual function used for `Jxx` (∂F/∂x)
  - Any parameter unpacking logic
  - Where actuation currents enter the residual calculation

## Step 5: Design the ∂F/∂u path (control inputs)
- Decide the AD input vector layout for controls:
  - Currents only (size 3), or
  - Currents + insertion length (size 4)
- Define how `Jxu = ∂F/∂u` will be computed:
  - Reuse the AD residual with controls as differentiable inputs.
  - Ensure dimensions align: residual is (NUM_ACT_SET*6), u is (3 or 4).

## Step 6: Implement ∂F/∂u in C++
- Extend AD helpers to compute `Jxu`.
- Wire `Jxu` into `linearize_full_seed_action_from_seed_implicit` so `B` comes from AD.
- Keep FD fallback available if AD computation fails (optional but useful).

## Step 7: Update/extend tests
- Start with `tests/test_dynamics_implicit_linearization.py`:
  - Add a check that AD `B` matches FD `B` within tolerance.
  - Use known convergent damping values (same as other tests).
- Run: `pytest -q tests/test_dynamics_implicit_linearization.py`

## Step 7.1: Convergence handling
- Mirror the new convergence guards used in linearizers:
  - If any perturbation step does not converge, fail fast or skip that perturbation.
  - Ensure AD/FD comparison only uses converged steps.

## Step 8: Full test run
- `pytest -q`

## Step 9: Benchmark (recommended)
- Extend `scripts/benchmark_task1_4.py`:
  - Add timings for AD‐based `B` vs FD `B`.
- Save results to `data/output/` (ignored by git).

## Step 10: Documentation
- Update:
  - `docs/architecture/DEVELOPMENT_TASKS.md` (mark Task 1.5 complete)
  - `docs/HANDOVER_AGENT_STATUS.md` (tests + benchmark summary)

## Step 11: Commit and push
- Commit with a clear message (e.g., “Add AD ∂F/∂u for control inputs”).
- Push branch and open PR against `refactor/taskA1-modernize`.
