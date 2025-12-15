# Review Checklist for Claude Code

## 1) Dynamics Wrapper/Bindings Parity
- Inspect: `crm_ml_rl/wrappers/crm_bindings.cpp`, `crm_ml_rl/wrappers/crm_wrapper.py`
- Goals: Verify `initialize_from_seed`, `bvp_initialize_with_seed`, `disable_cpp_fallback`, and `localmin` reporting; ensure step logic aligns with C++ behavior.

## 2) Sweep Diagnostics
- Inspect: `examples/sweep_cpp_dynamics.py`, `main/CRMDYN_grid_sweep.cpp`
- Artifacts: `sweep_failures_py.json`, `sweep_failures_cpp.json`
- Goals: Confirm sweep setup (currents grid, insertion 94.3 mm, step sizes [0.01, 0.05, 0.1, 0.2]), validate failure sets, and analyze convergence gaps.

## 3) Outstanding Solver Issues
- Reference: “Dynamics Sweep Status” in `next_steps.md`
- Goals: Investigate solver tolerances/step control for failing cases; consider direct BVP/IVP binding for diagnostics; decide fallback strategy; document/clamp non-convergent regions.

## 4) ML/RL Implementations
- Inspect code: `crm_ml_rl/models`, `crm_ml_rl/envs`, `crm_ml_rl/training`
- Docs: `USAGE_GUIDE.md`, `docs/modeling_guide.md`
- Examples/tests: `examples/ml_examples.py`, `examples/rl_examples.py`, `scripts/test_ml_models.py`, `scripts/test_rl_integration.py`
- Goals: Verify interfaces, training loops, and integration with the CRM wrapper.

## 5) Experimental Data and Validation
- Inspect: `crm_ml_rl/data/experimental_loader.py`, `crm_ml_rl/evaluation/validation_metrics.py`, `crm_ml_rl/evaluation/cpp_validation.py`
- Goals: Ensure experimental loading/validation aligns with model inputs; check assumptions and metrics.

## 6) End-to-End ML/RL Pipelines
- Run/inspect: `crm_ml_rl/training/train_rl.py`, `crm_ml_rl/training/train_dynamics.py`, `scripts/test_rl_integration.py`, `scripts/test_ml_models.py`
- MPC/model-based: `crm_ml_rl/training/mpc_controller.py`, `crm_ml_rl/training/model_based_rl.py`
- Policies/networks: `crm_ml_rl/training/custom_policies.py`, `crm_ml_rl/models/networks.py`
- Envs: `crm_ml_rl/envs/*` (reset/step, obs shapes, reward/termination)
- Requirements/examples: `requirements.txt`, `crm_ml_rl/requirements.txt`, `examples/ml_examples.py`, `examples/rl_examples.py`
- Goals: Verify end-to-end execution with CRM wrapper enabled, consistency of observation/action spaces, and that examples/training scripts run to completion.

## 7) Tests to Run
- Pytest targets: `tests/test_crmdyn_binding_vs_cpp.py`, `tests/test_ml_models.py`, `tests/test_rl_models.py`, `tests/test_experimental_loader.py`
- Optional scripts: `scripts/test_rl_integration.py`, `scripts/test_ml_models.py` (already listed above), to sanity-check pipelines.
- Goals: Catch regressions in bindings, ML/RL models, and data loading.

## 8) Additional Checks
- Numeric stability: review clamping/tolerances in solvers/wrappers to guard against NaNs/huge velocities.
- Logging: ensure non-convergence logs include currents, insertion length, step size, and `localmin` for troubleshooting.
- Performance: look for avoidable Python↔C++ crossings in training loops; profile CRM calls if needed.
- Parameters/units: verify radii/mass/damping units and dataset-specific current flips (`flip_third_current`) are consistent.
- Docs accuracy: confirm `USAGE_GUIDE.md` and `docs/modeling_guide.md` reflect current APIs/defaults.

## 9) FK Parity Check
- Script: `scripts/compare_fk_cpp_vs_wrapper.py`
- Goal: Verify the C++ FK and wrapper FK outputs match (when both use C++) across a few currents.
