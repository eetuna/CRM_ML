# Task 1.5 Checklist: Gradients w.r.t Control Inputs (∂F/∂u)

## Step 0: Create a fresh working branch
1. `git checkout refactor/taskA1-modernize`
2. `git checkout -b task/1.5-control-input-gradients`

## Step 1: Read current plan/status
- `docs/architecture/DEVELOPMENT_TASKS.md`
- `docs/architecture/PLAN_end_to_end_differentiable_simulator_options_A_B_C.md`
- `docs/architecture/PLAN_CHECK.md`
- `docs/HANDOVER_AGENT_STATUS.md`

## Step 2: Inspect current bindings (entry points)
- `crm_ml_rl/wrappers/crm_bindings.cpp`
  - `linearize_action_from_seed` (FD on currents)
  - `linearize_full_seed_action_from_seed` (full FD)
  - `linearize_full_seed_action_from_seed_implicit` (implicit AD path)

## Step 3: Inspect AD residual implementation
- `src/CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp`
  - Find where `Jxx` (∂F/∂x) is computed.
  - Identify control inputs used in the residual (currents/insertion length).

## Step 4: Implement ∂F/∂u (control inputs)
- Extend the AD path to compute `Jxu = ∂F/∂u`.
- Ensure the implicit linearization returns `B` from AD, not FD.

## Step 5: Validation
- Update or add a test:
  - Start with `tests/test_dynamics_implicit_linearization.py`.
  - Compare AD `B` vs FD `B` within tolerance.
- Run: `pytest -q tests/test_dynamics_implicit_linearization.py`

## Step 6: Full test run
- `pytest -q`

## Step 7: (Optional) Benchmark
- Extend `scripts/benchmark_task1_4.py` or create a new benchmark entry for AD vs FD `B`.
- Save results to `data/output/` (ignored by git).

## Step 8: Documentation
- Update:
  - `docs/architecture/DEVELOPMENT_TASKS.md` (mark Task 1.5 complete)
  - `docs/HANDOVER_AGENT_STATUS.md` (summary + tests)

## Step 9: Commit and push
- Commit with a clear message.
- Push branch and open PR against `refactor/taskA1-modernize`.
