# Gemini Review Checklist: Option A Work (Audit + Verification)

This checklist is for an external reviewer (Gemini) to audit and validate the work completed so far toward **Option A (C++ AD + implicit diff)**, confirm what is complete, and verify the remaining partial items.

## Scope

- Validate completed Task 1.5 (AD control gradients), Task A1.6 (stabilization), Task A1.7 (core refactor).
- Audit Option A plan items A1–A5 and confirm which are partial vs not implemented.
- Verify tests and scripts produce expected outputs on stable regimes.

## Evidence Index (start here)

- Option A plan and partial status: `docs/architecture/PLAN_end_to_end_differentiable_simulator_options_A_B_C.md`
- A1 completion report: `docs/architecture/TASK_A1_COMPLETION_REPORT.md`
- Task 1.5 implementation checklist: `docs/archive/TASK_1_5_CHECKLIST.md`
- Task 1.7 refactor + stabilization log: `docs/archive/TASK_1_7_STATUS.md`
- Integrator stability findings: `docs/architecture/INTEGRATOR_STABILITY.md`
- Task 1.4 benchmark output: `data/output/benchmark_task1_4.json`

## Review Tasks

### 0) Tasks 1.1–1.4 (Foundations)
1. Confirm framework selection and usage:
   - `docs/architecture/PLAN_end_to_end_differentiable_simulator_options_A_B_C.md`
   - `src/CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp`
2. Confirm prototype residual/Jacobian exists:
   - `src/CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp`
   - `crm_ml_rl/wrappers/crm_bindings.cpp`
3. Confirm gradient‑accuracy test coverage:
   - `tests/test_dynamics_implicit_linearization.py`
   - `docs/archive/TASK_1_5_CHECKLIST.md`
4. Confirm benchmark script/output for Task 1.4:
   - `scripts/benchmark_task1_4.py`
   - `data/output/benchmark_task1_4.json`

### 1) Code Audit: Task 1.5 (∂F/∂u)
1. Confirm AD control residual and Jacobian are implemented in:
   - `src/CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp`
2. Confirm implicit linearization path uses the AD `Jxu`:
   - `crm_ml_rl/wrappers/crm_bindings.cpp`
3. Verify the test exists and passes:
   - `tests/test_dynamics_implicit_linearization.py`
4. Record findings: shape (6x3), finite values, and AD flags.

### 2) Code Audit: Task A1.7 (Core Refactor)
1. Confirm `DynamicsContext` replaces legacy arrays:
   - `src/CRM_DynamicsContext.hpp`
   - `src/CRM_DynamicsContext_AD.hpp`
2. Verify refactor touches in AD residuals and solver paths:
   - `src/CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp`
   - `src/CoilDynamics_Defs.cpp`
3. Confirm legacy array sync paths are removed or dormant.

### 3) Code Audit: Task A1.6 (Stabilization)
1. Confirm RK4 + adaptive stepping integration:
   - `src/CoilDynamics_Defs.cpp`
2. Confirm divergence propagation to Python:
   - `crm_ml_rl/wrappers/crm_bindings.cpp`
3. Review stability tests:
   - `tests/test_adaptive_stepping_regression.py`
   - `tests/test_adaptive_stepping_stability.py`

### 4) Option A Plan A1–A5 Status
Check each section in `docs/architecture/PLAN_end_to_end_differentiable_simulator_options_A_B_C.md`:
- **A1 (Implicit completeness)**: verify AD residuals for seed blocks and output map are still FD/partial.
- **A2 (Multi‑actuator)**: confirm `NUM_ACT_SET == 1` static_assert remains in AD residual.
- **A3 (Torch wrapper)**: confirm seed gradients return; insertion gradients missing.
- **A4 (Tests)**: confirm only implicit linearization + AD residual tests exist; no torch gradcheck tests.
- **A5 (Experiments)**: confirm benchmarks exist; iLQR/MPC demo with implicit `A,B` is missing.

### 5) Validation Runs (recommended)
Run these on stable regimes and log the results:
1. `pytest -q`
2. `CRM_DYN_LINEARIZATION_METHOD=implicit pytest -q tests/test_dynamics_implicit_linearization.py`
3. `pytest -q tests/test_adaptive_stepping_regression.py tests/test_adaptive_stepping_stability.py`
4. `python scripts/benchmark_task1_4.py --help` (confirm runnable + expected outputs)

### 6) Deliverable from Reviewer
- A short report summarizing:
  - Verified completed items.
  - Any code regressions or inconsistencies.
  - Remaining tasks and their estimated effort.
  - Whether Option A is ready for paper‑grade gradients (yes/no, with justification).
