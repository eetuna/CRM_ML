# Integrator Stability Findings (Phase 3)

## Summary
- Instability manifests as "Coil integration Unbounded" / non-convergence during BVP + flexible-segment IVP.
- Divergence originates in coil dynamics integration (angular velocity blows up early).
- RK4 does not consistently fix the historical failing case when damping is unset/unstable; both ABM4 and RK4 can diverge under those conditions.
- Small actuator inertia and large angular accelerations are a primary driver of blow-up.

## Known Parameter Regimes

| Regime | Damping | Insertion | Currents | BVP Converged | ABM4 Coil | RK4 Coil | Notes |
|--------|---------|-----------|----------|--------------|-----------|----------|-------|
| **Default (Stabilized Params)** | CRMWrapper defaults | 85.86 | [0.043, 0.021, 0.009] | ✅ | ✅ Converged | ✅ Converged | Harness uses tuned damping from `CatheterParameterSet_1_dyn.txt` |
| **Known Stable** | [0.015, 0.015, 0.015] | 50 | [0.0, 0.0, 0.0] | ✅ | ✅ OK | ✅ OK | From test suite defaults; no angular acceleration spike |
| **Historical Failing Case (pre-fix)** | None (unstable defaults) | 85.86 | [0.043, 0.021, 0.009] | ❌ | ❌ Diverges | ❌ Diverges | Diverged before damping defaults were stabilized |
| **Seed Sweep (10 cases)** | Default | [30-100] | random ±0.05 | 0/10 | Same failure | Same failure | No ABM4-only or RK4-only convergence mismatch (historical) |

### Key Findings
- **Damping Impact**: Unstable/default damping triggers divergence in coil integration; known-stable damping prevents blow-up.
- **Integrator Trade-off**: RK4 is more stable numerically but slower (~1.21x); neither fixes divergence with default damping.
- **Inertia Scaling**: Small actInertia (e.g., 2.38e-4) combined with torque spikes leads to angular velocity explosion.
- **BVP Solver**: Fails to converge due to non-finite residuals from coil dynamics; solver tuning alone cannot fix structural instability.

## Adaptive Stepping (Phase 3.4)

### Implementation
- RK4 integrator supports adaptive step-size subdivision when angular acceleration exceeds a threshold.
- Threshold: 1000 rad/s^2 (2.5x above critical acceleration for typical inertia ~2.4e-4).
- Max subdivision depth: 4 levels (minimum step size 62.5 microseconds).

### Mechanism
- Monitor angular acceleration magnitude during RK4 stage evaluation.
- If the threshold is exceeded, subdivide the step into two half-steps recursively.
- If max subdivisions are exceeded while still unsafe, the step is flagged as diverged.

### Performance Notes
- Stable cases: No subdivisions triggered, minimal overhead beyond the check.
- Transient spikes: Localized 2x-16x slowdown only during the spike window.
- Pathological cases: Exit gracefully with diverged flag instead of hanging.

### Tests
- `tests/test_adaptive_stepping_regression.py`
- `tests/test_adaptive_stepping_stability.py`

## Reproduction (Known Failing Case)
Input file:
- `docs/architecture/TASK_1_7_FAILING_CASE.json`

Expected behavior:
- With historical (unstable) damping: BVP fails to converge; residual norms become large (order 1e6).
- With current defaults: converges (historical failure no longer reproduces).

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
- `docs/archive/TASK_1_7_STATUS.md` (detailed diagnostics and logs)
- `docs/architecture/TASK_1_7_FAILING_CASE.json` (repro input)
