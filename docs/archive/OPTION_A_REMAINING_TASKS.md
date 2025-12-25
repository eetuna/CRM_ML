# Option A Remaining Work (C++ AD + Implicit Diff)

This document enumerates the remaining work needed to declare **Option A** complete, based on `docs/architecture/PLAN_end_to_end_differentiable_simulator_options_A_B_C.md` and the current A1 completion evidence.

## Status Snapshot

- **Completed already**: AD `Jxx` and AD control gradients `Jxu` (Task 1.5) are implemented and validated; see `docs/archive/TASK_1_5_CHECKLIST.md` and `docs/archive/TASK_1_7_STATUS.md`.
- **Partially complete**: A1, A3, A4, A5 (implicit Jacobian completeness, torch wrapper coverage, tests, and paper-facing scripts); see `docs/architecture/PLAN_end_to_end_differentiable_simulator_options_A_B_C.md` for the partial status details.
- **Remaining**: A2 plus the unfinished portions of A1/A3/A4/A5.

## A1) Implicit Linearization Completeness

Goal: minimize FD usage in `linearize_full_seed_action_from_seed_implicit` so implicit `A,B` are trustworthy for paper-quality gradients.

### Subtasks
1. **AD residual partials beyond currents**
   - Add AD for residual partials with respect to seed blocks (and any other theta inputs used in `F`).
   - File focus: `src/CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp`.
2. **Output mapping partials**
   - Implement AD for output mapping `y = g(x*, theta)` where `x*` solves `F(x,theta)=0`.
   - If keeping FD for `g`, add explicit error characterization and document rationale.
   - Files: `crm_ml_rl/wrappers/crm_bindings.cpp`, plus any helper routines in `src/`.
3. **Scaling / conditioning**
   - Align scaling between solver variables and exposed Jacobians.
   - Ensure `A,B` are reported in the same scale as `linearize_full_seed_action_from_seed`.

### Acceptance Criteria
- For a stable sample set, implicit `A,B` match full FD within tolerance (e.g., relative Frobenius error < 1e-2).
- Document the test harness and results in `docs/architecture/OPTION_A_REMAINING_TASKS.md` or a follow-on report.

## A2) Multi-Actuator Support (`NUM_ACT_SET > 1`)

Goal: extend the AD residual and Jacobian logic to multiple actuator sets.

### Subtasks
1. Generalize residual construction and indexing in `src/CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp`.
2. Update Jacobian helpers to return block-structured derivatives sized to `(NUM_ACT_SET * 6)`.
3. Add coverage tests for multi-actuator configurations (build + jacobian sanity checks).

### Acceptance Criteria
- Build succeeds with `NUM_ACT_SET > 1`.
- AD vs FD residual Jacobians pass in the multi-actuator configuration.

## A3) PyTorch Wrapper End-to-End Gradients

Goal: expose gradients for the full differentiable step through the torch wrapper.

### Subtasks
1. Update `crm_ml_rl/wrappers/torch_physics.py` so `CRMDynamicsStepFunction.backward` returns gradients for:
   - currents
   - full seed tensors (`v, w, p, R, xf, mL, nL` if present)
2. Decide whether to differentiate w.r.t insertion length and document it.
3. Add `gradcheck` or FD-based checks for the wrapper on stable regimes.

### Acceptance Criteria
- Torch backward covers the specified inputs and passes the gradient checks on stable cases.

## A4) Option A Test Suite

Goal: add/extend tests proving forward correctness, gradient correctness, and implicit-vs-FD agreement.

### Subtasks
1. **Forward correctness**
   - Compare torch wrapper output vs `crm_python.CRMDynamics.step_from_seed`.
2. **Gradient correctness**
   - `∂y/∂u`: compare torch gradients to FD (central difference).
   - `∂y/∂seed`: compare torch gradients to FD for seed perturbations.
3. **Implicit vs baselines**
   - Compare implicit `A,B` to `linearize_full_seed_action_from_seed`.
   - Add directional FD checks (`y(u+δ)` vs `y(u-δ)`).
4. **Cross-implementation consistency**
   - Compare `autodiff_eigen` vs `autodiff_template` results.

### Acceptance Criteria
- Tests pass on stable envelopes and emit JSON summaries with relative errors and convergence stats.

## A5) Paper-Facing Experiment Scripts

Goal: provide reproducible evidence for the paper-quality story.

### Subtasks
1. **AD vs FD Jacobian quality**
   - Script to report runtime + error distributions for `A,B`.
2. **Controller demo**
   - Simple iLQR/MPC on short horizon using implicit Jacobians.

### Acceptance Criteria
- Scripts complete in a bounded time and emit figures/tables or JSON reports.

## Proposed Validation Envelope (shared for A1–A5)

- Use stabilized damping defaults (see `docs/architecture/INTEGRATOR_STABILITY.md`).
- Define bounded current and insertion ranges; filter for converged samples.
- Log `converged` flags and residual norms per sample.

## Evidence / References

- Option A plan: `docs/architecture/PLAN_end_to_end_differentiable_simulator_options_A_B_C.md`
- A1 completion evidence: `docs/architecture/TASK_A1_COMPLETION_REPORT.md`
- Control-gradient implementation: `docs/archive/TASK_1_5_CHECKLIST.md`
- Integrator stability notes: `docs/architecture/INTEGRATOR_STABILITY.md`
