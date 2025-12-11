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
