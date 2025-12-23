# Task A1.7 Stabilization Plan (BVP First)

## Goal
Identify and fix the root cause of BVP non-convergence that prevents stable dynamics integration, then re-evaluate ABM4 vs RK4 stability.

## Current Status (Baseline)
- Baseline case fails at BVP iter=1 with `info=4` (slow progress).
- Metrics: `fnorm ~ 7.2e+06`, `actred < 0`, `prered = 0`, `ratio = 0`.
- Interpretation: solver makes no progress from initial guess; integrator choice is not the primary bottleneck.

## Relevance Audit
- This plan is now a historical log of the stabilization attempts (steps 1–4 completed).
- Remaining work should be tracked in `docs/architecture/TASK_1_7_CHECKLIST.md` and Claude’s recommended next steps.

## Plan

1) **Baseline Repro (Locked Case)** — STATUS: DONE
   - Run a fixed seed/currents case with:
     - `CRM_DEBUG_BVP=1`
     - `CRM_DEBUG_BVP_SOLVER=all`
   - Record first-iteration metrics: `fnorm`, `fnorm1`, `actred`, `prered`, `ratio`, `delta`.

2) **BVP Parameter Tuning (One Change at a Time)** — STATUS: DONE (no improvement)
   - **2A. Reduce trust region `factor`** (e.g., 100 → 10 → 1).
   - **2B. Increase finite-difference step `epsfcn`** (e.g., 1e-6 or 1e-4).
   - **2C. Adjust `tol`** (if needed, to avoid premature slow-progress termination).
   - After each change, re-run the locked case and compare first-iteration metrics.
   - Keep only changes that yield positive progress (`fnorm1 < fnorm`, `actred > 0`, `ratio > 0`).

3) **Initial Guess / Scaling Review** — STATUS: DONE (no improvement)
   - Inspect and, if needed, adjust `IVALUE_SCALE_*` and initial `m_L/n_L` guesses.
   - Confirm that residual magnitudes are within a reasonable numeric range.

4) **ABM4 vs RK4 Re-evaluation** — STATUS: DONE (no ABM4-fail/RK4-pass found)
   - Re-run the sweep only after BVP progress is confirmed.
   - Look for “ABM4 fails / RK4 succeeds” cases and document them.

## Output Artifacts
- Debug logs from the locked case for each tuning attempt.
- Summary table of BVP metrics before/after each change.
- Decision on solver parameter updates to keep.
- Track original “Recommended Next Steps” from Claude:
  - Benchmark ABM4 vs RK4 (speed/accuracy): **pending**.
  - Test RK4 on failing “Unbounded” cases: **done** (failing case still fails under RK4).
  - Validate soft failure mode in production scenarios: **pending**.
  - Consider making RK4 default (once validated): **pending**.
  - Merge branch to main: **pending**.

## Tuning Log

- 2025-01-13: **2A factor=10** (from 100).
  - Result: no improvement; `actred=-1`, `prered=0`, `ratio=0` at iter=1; `info=4` (slow progress).
- 2025-01-13: **2A factor=1** (from 10).
  - Result: no improvement; `actred=-1`, `prered=0`, `ratio=0` at iter=1; `info=4` (slow progress).
- 2025-01-13: **2B epsfcn=1e-6**.
  - Result: regression; `fnorm1` became NaN and `pnorm` became NaN at iter=1; reverted.
- 2025-01-13: **2C tol=1e-3** (from 1e-4).
  - Result: no improvement; `actred=-1`, `prered=0`, `ratio=0` at iter=1; `info=4`.
- 2025-01-13: **3A IVALUE_SCALE_M/N=100**.
  - Result: no improvement; residuals remained O(1e6); saw alternating zero residuals on some n_L perturbations.
- 2025-01-13: **3B IVALUE_SCALE_M/N=10000**.
  - Result: no improvement; residuals still O(1e6); zero-residual pattern persists on n_L perturbations.
- 2025-01-13: **3C Non-finite tracing**.
  - Found non-finite outputs in `CRMFlexible_IVP_Back` (segment 0), originating from `ABM4_dyn` (step 4-7).
  - Added finite-penalty fallback for non-finite states in `ABM4_dyn`; residuals are now finite (1e6) but solver still stalls.
- 2025-01-13: **3D Flexible segment step size x4**.
  - Result: non-finite in `ABM4_dyn` persists, now at later steps (~8-10); no convergence.
- 2025-01-13: **3E Flexible segment step size x8**.
  - Result: non-finite in `ABM4_dyn` persists, now at later steps (~17-23 for some cases); no convergence.
- 2025-01-13: **3F Flexible segment RK4 fallback**.
  - Result: RK4 fallback triggered but also produced non-finite states; solver still stalls.
- 2025-01-13: **3G Seed-based m_L/n_L initialization**.
  - Result: default m_L/n_L guesses now fall back to internal seed state when inputs are empty or all zeros.
  - Validation: rebuilt `crm_python` and confirmed `step_from_seed` returns internal mL/nL when inputs are empty and BVP fails.
- 2025-01-13: **3H Locked-case BVP re-run with seed-based m_L/n_L fallback**.
  - Result: initial iteration improved (`fnorm` dropped from `2.26e7` to `1.73e6` with `ratio=0.994`), then stalled at iter=2 with `ratio=0` and `info=4`.
- 2025-01-13: **3I Locked-case BVP re-run after review fixes**.
  - Result: no change in metrics; still stalls at iter=2 (`fnorm=1.73e6`, `ratio=0`, `info=4`).
- 2025-01-13: **3J Integrator diagnostics (flexible-segment)**.
  - Result: `CRMIntegrand_dyn` observes `|R|~1e6` while `|u|` is still small; `ABM4_dyn` shows `|u|` exploding later. K/Kinv/ustar magnitudes are normal.
- 2025-01-13: **3K Optional SE(3) projection (CRM_CLAMP_SE3=1)**.
  - Result: initial `fnorm` reduced (~1.42e7), but solver still stalls; non-finite `u` persists.
- 2025-01-13: **3L Trace R blow-up source (tip vs coil inputs)**.
  - Added `CRM_DEBUG_BVP_SCALE` logs in `DYNNLEquation` to flag large/non-finite `R` at the tip input and coil output feeding `CRMFlexible_IVP_Back`.
- 2025-01-13: **3M Re-run locked case with known damping (tests baseline)**.
  - Run `step_from_seed` using `base_damping` from `tests/test_parameter_jacobian_autodiff.py`.
  - Result: BVP converged (`localmin=0`), no `DYNNLEquation tip/coil R input` logs triggered; no rotation blow-up observed.
- 2025-01-13: **3N Generate failing case + trace R blow-up source**.
  - Created `docs/architecture/TASK_1_7_FAILING_CASE.json` via sweep (default damping, trial 0).
  - Re-run with `CRM_DEBUG_BVP_SCALE=1`: `DYNNLEquation coil R output |R|=1e+06` logged at segment 0, actno 0.
  - Indicates large/non-finite `R` originates from coil integration output (not tip seed `Params.xf`).
- 2025-01-13: **3O Trace coil integrator inputs (pre-dispatch)**.
  - Added `CRM_DEBUG_BVP_SCALE` log before `CoilDynamicsDispatchLegacy` to flag large/non-finite coil input states.
  - Re-run failing case: no `coil input state` log triggered; inputs appear finite while outputs blow up.
- 2025-01-13: **3P Trace coil integrator step outputs**.
  - Added `CRM_DEBUG_BVP_SCALE` log inside `CoilDynamics`/`CoilDynamicsRK4` to flag large/non-finite per-step outputs.
  - Failing case (ABM4): `coil step output step=1 method=rk2 |w|=1.65135e+09` while `|R|~1`; blow-up begins in early RK2 init step.
- 2025-01-13: **3Q RK4 vs ABM4 on failing case**.
  - With `integrator=rk4`, still fails; `coil step output step=0 method=rk4 |w|~2.75e+08`.
  - Integrator switch does not prevent coil blow-up for this case.
- 2025-01-13: **3R Fix RK4 SE(3) stage inputs (consistency)**.
  - Updated RK4 to use stage-consistent twists for `DYNSE3_TimeSpace` (twist_2/3/4).
  - Failing case still fails under RK4; `|w|` remains ~2.75e+08 at step 0.
- 2025-01-13: **3S Instrument CoilIntegrad (torque/inertia diagnostics)**.
  - Added `CRM_DEBUG_BVP_SCALE` log in `CoilIntegrad` to capture `|w|`, `|tau|`, `|res_w|`, and diagonal `actInertia`.
  - Failing case: `|w|≈2.4e+06`, `|tau|≈0.27`, `|res_w|≈2.4e+07`, `actInertia≈[2.38e-4, 2.38e-4, 1.45e-5]`.
  - Suggests small inertia + large angular velocity drive huge `wdot` early.
- 2025-01-13: **3T Trace CoilIntegrad stage inputs**.
  - Logged first 3 `CoilIntegrad` input twists: `w` jumps from `~0.001` to `~10` to `~-3485` within RK4 stages.
  - Confirms angular velocity escalates inside the integrator stages even from a small initial state.
- 2025-01-13: **3U Validate actInertia loading + early wdot scale**.
  - `CRM_DEBUG_PARAM_LOAD=1` shows `mass=8.2859e-06`, `r_out=1.5875`, `r_in=0.9906`, `seg_len=18.3`,
    computed `I_xx=2.38492e-4`, `I_zz=1.45063e-5` (as expected from formula).
  - Early RK4 stage: `tau_z≈0.304`, `res_w_z≈0.294`, `wdot_z≈2.03e+04` (small inertia → large wdot).
- 2025-01-13: **3V Verify damping values and defaults**.
  - Failing case uses default damping from bindings: `d=[10,10,10,10,10,10]` (not loaded from files).
  - Known stable tests use `base_damping=[12.176, 12.176, 284.43, 0.0305, 0.0305, 0.00503]`.
  - Damping is not data-driven by default; stability depends on explicitly setting validated values.
