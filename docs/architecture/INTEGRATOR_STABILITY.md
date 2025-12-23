# Integrator Stability Findings (Phase 3)

## Summary
- Instability manifests as "Coil integration Unbounded" / non-convergence during BVP + flexible-segment IVP.
- Divergence originates in coil dynamics integration (angular velocity blows up early).
- RK4 does not consistently fix the failing case; both ABM4 and RK4 can diverge under default damping.
- Small actuator inertia and large angular accelerations are a primary driver of blow-up.

## Known Parameter Regimes

| Regime | Damping | Insertion | Currents | BVP Converged | ABM4 Coil | RK4 Coil | Notes |
|--------|---------|-----------|----------|--------------|-----------|----------|-------|
| **Default (Stabilized Params)** | CRMWrapper defaults | 85.86 | [0.043, 0.021, 0.009] | ✅ | ✅ Converged | ✅ Converged | Harness uses tuned damping from `CatheterParameterSet_1_dyn.txt` |
| **Known Stable** | [0.015, 0.015, 0.015] | 50 | [0.0, 0.0, 0.0] | ✅ | ✅ OK | ✅ OK | From test suite defaults; no angular acceleration spike |
| **Historical Failing Case** | None (default) | 85.86 | [0.043, 0.021, 0.009] | ❌ | ❌ Diverges | ❌ Diverges | See `TASK_1_7_STATUS.md` diagnostics for default damping failure |
| **Seed Sweep (10 cases)** | Default | [30-100] | random ±0.05 | 0/10 | Same failure | Same failure | No ABM4-only or RK4-only convergence mismatch (historical) |

### Key Findings
- **Damping Impact**: Default (no explicit damping set) triggers divergence in coil integration; known-stable damping from test suite prevents blow-up.
- **Integrator Trade-off**: RK4 is more stable numerically but slower (~1.21x); neither fixes divergence with default damping.
- **Inertia Scaling**: Small actInertia (e.g., 2.38e-4) combined with torque spikes leads to angular velocity explosion.
- **BVP Solver**: Fails to converge due to non-finite residuals from coil dynamics; solver tuning alone cannot fix structural instability.

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
- Default damping values are a common trigger; tuned damping (as used in the wrapper) improves convergence.
- No ABM4-fail / RK4-pass case was found in prior sweeps (see TASK_1_7_STATUS).
- Current harness results show both ABM4 and RK4 converge under tuned damping.

## References
- `docs/architecture/TASK_1_7_STATUS.md` (detailed diagnostics and logs)
- `docs/architecture/TASK_1_7_FAILING_CASE.json` (repro input)
