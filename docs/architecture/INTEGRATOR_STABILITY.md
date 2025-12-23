# Integrator Stability Findings (Phase 3)

## Summary
- Instability manifests as "Coil integration Unbounded" / non-convergence during BVP + flexible-segment IVP.
- Divergence originates in coil dynamics integration (angular velocity blows up early).
- RK4 does not consistently fix the failing case; both ABM4 and RK4 can diverge under default damping.
- Small actuator inertia and large angular accelerations are a primary driver of blow-up.

## Reproduction (Known Failing Case)
Input file:
- `docs/architecture/TASK_1_7_FAILING_CASE.json`

Expected behavior:
- BVP fails to converge; residual norms become large (order 1e6).
- Coil dynamics diverges early (angular velocity spikes before rotation matrix becomes non-finite).

Steps (Python bindings):
1. Load parameters and configuration used in the failing case JSON.
2. Call `step_from_seed` or the equivalent dynamics step with the recorded currents/insertion length.
3. Observe divergence / non-convergence (localmin != 0).

## Observed Signals
- ABM4 path: divergence often begins during RK2 warmup (step 1).
- RK4 path: divergence begins at step 0 in some cases.
- `|w|` grows rapidly while `|R|` remains near 1 initially; rotation matrix is not the first to go non-finite.

## Notes
- Default damping values are a common trigger; known stable damping helps convergence.
- No ABM4-fail / RK4-pass case was found in prior sweeps (see TASK_1_7_STATUS).

## References
- `docs/architecture/TASK_1_7_STATUS.md` (detailed diagnostics and logs)
- `docs/architecture/TASK_1_7_FAILING_CASE.json` (repro input)
