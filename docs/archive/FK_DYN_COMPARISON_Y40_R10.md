# FK vs Dynamics Comparison (Y=±40mm circles, r=10mm)

This document summarizes what was generated/tested in this repo for the “circle at y=+40 and y=-40” experiment, what parameters were used, and the resulting FK-vs-dynamics tracking errors.

Repo revision: `c25fb0bbc68a2c10eecd7c6258a2144934fc2090`

## Goal

- Generate catheter actuation currents that realize two circular tip trajectories:
  - circle centered at `(x=0, y=+40)` mm, radius `10` mm
  - circle centered at `(x=0, y=-40)` mm, radius `10` mm
- Because desired circles may be partially unreachable, first **project** desired points onto a dense FK workspace and solve IK against the **projected** points.
- Compare **static FK tip** vs **dynamic tip** when applying the generated currents, including ramps and a transition between circles.

## What was used (C++ vs Python)

All core model computations used the **C++ bindings**:

- Kinematics/FK/Jacobian: `crm_ml_rl.wrappers.crm_python.CRMKinematics`
- Dynamics: `crm_ml_rl.wrappers.crm_python.CRMDynamics` (via `crm_ml_rl.wrappers.crm_wrapper.CRMWrapper(use_cpp=True, disable_cpp_fallback=True)`)

## Common model.parameters

- `param_file`: `data/catheter_params/CatheterParameterSet_1_dyn.txt`
- `config_file`: `data/catheter_params/CatheterSpatialConfiguration_1.txt`
- `insertion_length`: `94.3` mm
- Along-rod spatial integration step (FK and Dyn): `integration_step_size = 0.2` mm
- Dynamics time step: `dt = 0.05` s

## Workspace generation (dense FK “reachable set”)

Workspace cache (generated once, then reused):

- `data/output/workspace_fk_ins94.3_b0.3_step0.01_int0.2.npz`

How it was generated:

- Script: `scripts/fk_workspace_grid.py`
- Currents grid: `c1,c2,c3 ∈ [-0.3, +0.3]` with step `0.01` A (10mA)
- Total FK samples: `61^3 = 226,981`
- Converged: `220,310 / 226,981` (`97.1%`)
- Tip range (mm) over converged samples:
  - x: `[-40.54, 43.18]`
  - y: `[-75.39, 75.39]`
  - z: `[-19.88, 94.30]`

## Circle point projection to workspace (reachability handling)

Desired circle points `p_des[i]` are **not** used directly as IK targets. Instead, we project them to the closest reachable points in the sampled workspace:

- Method: DP/Viterbi over K nearest workspace points, penalizing current jumps
  - minimize: `Σ ||p_ws(i) - p_des(i)|| + α ||u_ws(i) - u_ws(i-1)||`
- Script: `scripts/fk_ik_test_projected_workspace_dp.py`
- Projection parameters used:
  - `k_nearest = 300`
  - `alpha_u = 0.4`

Projection residual (this is *not* an IK error; it measures “how far desired is from the reachable projection”):

- Circle1 `||des - proj||`: mean `3.381` mm, p95 `6.176` mm, max `6.241` mm
- Circle2 `||des - proj||`: mean `3.373` mm, p95 `6.150` mm, max `6.210` mm

## IK current generation (matching projected targets)

IK targets: the **projected** points `p_proj[i]`.

- IK update: normalized pseudoinverse (MATLAB-style)
  - `u <- u + step_size * (du / ||du||)` where `du = pinv(J_xyz) * (p_target - p_fk)`
- Script: `scripts/fk_ik_test_projected_workspace_dp.py`
- Params (tightened to hit 0.1mm):
  - `step_size = 1e-3`
  - `threshold_start_mm = 0.1`
  - `threshold_traj_mm = 0.1`
  - `itrmax = 500`
- Points per circle: `200` (one lap)

FK/IK consistency error vs projected targets (what IK actually solves):

- Circle1 `||proj - FK||`: mean `0.072` mm, p95 `0.098` mm, max `0.100` mm
- Circle2 `||proj - FK||`: mean `0.072` mm, p95 `0.099` mm, max `0.100` mm

Exported currents:

- `data/output/circle1_currents_y40_r10_dp_0p1mm_n200.csv`
- `data/output/circle2_currents_y-40_r10_dp_0p1mm_n200.csv`
- Combined: `data/output/circle_currents_ypm40_r10_dp_0p1mm_n200.csv`

## Dynamics rollouts and FK-vs-Dyn comparison

The applied current timeline includes:

1) ramp from `u=[0,0,0.01]` to the selected circle1 start current (linear interpolation in **current space**)
2) traverse circle1 currents
3) transition/bridge from circle1 end to circle2 start (linear interpolation in **current space**, not forcing `c1=c2=0`)
4) traverse circle2 currents

### Hold=1 vs Hold=2 definition

We keep `dt=0.05` everywhere and implement “hold” by repeating rows:

- Hold=1: each waypoint applied for `0.05s`
- Hold=2: each waypoint applied for `0.10s` (rows repeated twice)

### Full sequence (ramp + circle1 + bridge + circle2)

Hold=2 rerun bundle:

- Data: `data/output/dyn_fk_ramped_circles_hold2_rerun.npz`
- Figures:
  - `plots/dyn_fk_ramped_circles_multiview_hold2_rerun.png`
  - `plots/dyn_fk_ramped_circles_timeseries_hold2_rerun.png`

Hold=2 segment lengths: `[240, 400, 480, 400]` = `[ramp, circle1, bridge, circle2]`

FK–Dyn tip position error `||FK - Dyn||` (mm), hold=2:

- Overall: mean `0.280`, p95 `0.559`, max `2.080`, dyn_fail `0`
- Per segment:
  - ramp_to_circle1: mean `0.414`, p95 `1.582`, max `2.080`
  - circle1: mean `0.271`, p95 `0.470`, max `0.793`
  - bridge_to_circle2: mean `0.202`, p95 `0.252`, max `0.252`
  - circle2: mean `0.302`, p95 `0.541`, max `0.704`

Hold=1 bundle:

- Data: `data/output/dyn_fk_ramped_circles_hold1.npz`
- Figures:
  - `plots/dyn_fk_ramped_circles_multiview_hold1.png`
  - `plots/dyn_fk_ramped_circles_timeseries_hold1.png`

Hold=1 segment lengths: `[120, 200, 480, 200]`

FK–Dyn error (mm), hold=1:

- Overall: mean `0.409`, p95 `0.929`, max `3.443`, dyn_fail `0`
- Per segment:
  - ramp_to_circle1: mean `0.804`, p95 `3.086`, max `3.443`
  - circle1: mean `0.502`, p95 `0.828`, max `1.138`
  - bridge_to_circle2: mean `0.203`, p95 `0.252`, max `0.516`
  - circle2: mean `0.571`, p95 `0.954`, max `1.060`

### Ramp + circle1 only (no bridge, no circle2)

Script: `scripts/dynamics_fk_validation/dyn_fk_compare_ramp_circle1_only.py`

Hold=1:

- Data: `data/output/dyn_fk_ramp_circle1_hold1.npz`
- Figures:
  - `plots/dyn_fk_ramp_circle1_hold1_multiview.png`
  - `plots/dyn_fk_ramp_circle1_hold1_timeseries.png`
- Segment lengths: `[120, 200]` = `[ramp, circle1]`
- Error (mm):
  - ramp: mean `0.804`, p95 `3.086`, max `3.443`
  - circle1: mean `0.502`, p95 `0.828`, max `1.138`

Hold=2:

- Data: `data/output/dyn_fk_ramp_circle1_hold2.npz`
- Figures:
  - `plots/dyn_fk_ramp_circle1_hold2_multiview.png`
  - `plots/dyn_fk_ramp_circle1_hold2_timeseries.png`
- Segment lengths: `[240, 400]`
- Error (mm):
  - ramp: mean `0.414`, p95 `1.582`, max `2.080`
  - circle1: mean `0.271`, p95 `0.470`, max `0.793`

## Exported “plot-ready” CSVs (tip-space)

- Desired circles:
  - `data/output/circle1_desired_y40_r10.csv`
  - `data/output/circle2_desired_y-40_r10.csv`
  - `data/output/circle_desired_ypm40_r10.csv`
- Projected circles (onto workspace):
  - `data/output/circle1_projected_ws_y40_r10.csv`
  - `data/output/circle2_projected_ws_y-40_r10.csv`
  - `data/output/circle_projected_ws_ypm40_r10.csv`
- Combined (desired + projected + ramps, hold=2):
  - `data/output/circle_des_proj_and_ramps_hold2_labeled.csv`
  - `data/output/circle_proj_and_ramps_hold2_labeled.csv`
- Combined (projected + ramps, hold=1):
  - `data/output/circle_proj_and_ramps_hold1_labeled.csv`

## Replaying the current files

Replay doc:

- `data/output/RUN_INSTRUCTIONS_currents_y40_r10.md`

Metadata snapshot:

- `data/output/run_metadata_y40_r10_dp_0p1mm_hold2.json`
- `data/output/run_metadata_y40_r10_dp_0p1mm_hold2.txt`

## Recommended additional tests (next)

These are the most useful follow-ups to validate robustness and understand remaining error sources.

### 0) Branch / hysteresis sensitivity (FK “multiple equilibria”)

Goal: confirm when/where FK becomes history-dependent (solver can converge to different equilibria for the same currents).

- Regenerate the FK workspace with different sweep order (e.g., reverse `c3`, random order, or no warm-start) and compare:
  - how many points converge
  - whether the same `u` maps to multiple distinct `tip_position`
- Store and reuse `delta_u0` (`DU0`) when you need reproducible “same-branch” FK replay.

### 1) Time-step sensitivity (dynamics `dt`)

Goal: ensure FK–Dyn errors and convergence are not artifacts of a coarse timestep.

- Sweep `dt` while keeping the *physical waypoint time* constant (i.e., if you halve `dt`, also double hold steps or repeat rows accordingly):
  - Example: `dt ∈ {0.05, 0.025, 0.01}` s
- Metrics to compare:
  - `dyn_fail` count
  - FK–Dyn error mean/p95/max per segment
  - “worst-step” timestamps: do peaks move or shrink?

### 2) Along-rod integration step sensitivity (`integration_step_size`)

Goal: check numerical discretization effects in both FK and Dyn.

- Try `integration_step_size ∈ {0.2, 0.1, 0.05}` mm for:
  - FK only (for FK-vs-projected consistency)
  - Dyn only (for time integration behavior)
  - both together (for fair FK–Dyn comparisons)
- Compare runtime vs accuracy and whether convergence rates change.

### 3) Input slew limiting vs “hold”

Goal: reduce tracking error without simply slowing time.

- Apply a cap on `||Δu||` per step (or per-channel caps) by inserting intermediate current points.
- Compare to “hold=2” at equal total runtime/trajectory duration.

### 4) “Settle-to-threshold” waypoint advancement

Goal: replace fixed hold steps with a convergence criterion.

- For each waypoint, step dynamics until:
  - tip speed is below threshold, or
  - `||p_dyn(t)-p_dyn(t-Δt)||` is below threshold for K consecutive steps
- Then advance to next waypoint.
- This directly measures the settling behavior and avoids over/under-holding.

### 5) Initialization robustness

Goal: ensure results do not depend on a lucky initial condition.

## Lemniscate Addendum (Y=±40mm, a=10mm)

This section adds the “figure-8 / lemniscate” test requested after the circle results. It reuses the same model.parameters (insertion length, spatial integration step, dt) and the same DP projection approach, but it exposed an important FK solver behavior (“branching”) that must be handled explicitly for consistent FK↔workspace replay.

### Goal

- Lemniscate 1 centered at `(x=0, y=+40)` mm, scale `a=10` mm
- Lemniscate 2 centered at `(x=0, y=-40)` mm, scale `a=10` mm
- Run:
  - ramp1 + lem1, hold=1 and hold=2
  - ramp1 + lem1 + bridge + lem2, hold=1 and hold=2

Script:

- `scripts/lemniscate_ypm40_a10_dp_ik_fk_dyn_suite.py`

### Key finding: FK “branching” (why `DU0` matters)

The C++ FK routine is an iterative nonlinear solve; it is not a closed-form map. For some actuation ranges it can converge to different valid equilibria depending on the initial guess / warm-start history.

In practice this showed up as:

- The dense workspace stores tip positions `P(u)` produced during a sweep that warm-starts each FK solve from the previous one.
- Replaying FK later at the same current `u` (but with a different initial guess) can converge to a different equilibrium, producing `FK(u)` that does not match the previously cached workspace tip.

Fix:

- During workspace generation, store `delta_u0` per sample (`DU0`) and reuse it when re-evaluating FK for those workspace-selected currents.
- Workspace cache with `DU0`:
  - `data/output/workspace_fk_ins94.3_b0.3_step0.01_int0.2_withDU0.npz`

### IK / projection consistency (against projected targets)

For lemniscate, targets are the projected points (not the raw desired points). With `DU0` reuse, the projected workspace seed currents replay to essentially exact FK matches (up to numerical tolerance).

- `||proj - FK||` (lem1): mean/p95/max ≈ `0` mm (numerically ~`1e-5` mm)
- `||proj - FK||` (lem2): same (measured inside the combined rollout files)

### Dynamics tracking results (FK vs Dyn)

Ramp + lem1:

- Hold=1: `data/output/dyn_fk_lem1_y40_a10_hold1.npz`
  - `||FK - Dyn||` mean `0.896` mm, p95 `3.010` mm, max `3.510` mm, dyn_fail `0`
- Hold=2: `data/output/dyn_fk_lem1_y40_a10_hold2.npz`
  - `||FK - Dyn||` mean `0.482` mm, p95 `1.580` mm, max `2.050` mm, dyn_fail `0`

Full sequence (ramp + lem1 + bridge + lem2):

- Hold=1: `data/output/dyn_fk_lem12_ypm40_a10_hold1.npz`
  - `||FK - Dyn||` mean `0.649` mm, p95 `1.185` mm, max `3.510` mm, dyn_fail `0`
- Hold=2: `data/output/dyn_fk_lem12_ypm40_a10_hold2.npz`
  - `||FK - Dyn||` mean `0.445` mm, p95 `0.881` mm, max `2.050` mm, dyn_fail `0`

Figures (desired trajectory removed for clarity; only workspace/projected/FK/Dyn are shown):

- Projection to workspace:
  - `plots/lemniscate_ypm40_a10_workspace_projection_xy.png`
- Ramp + lem1:
  - `plots/dyn_fk_lem1_y40_a10_hold1_multiview.png`
  - `plots/dyn_fk_lem1_y40_a10_hold1_timeseries.png`
  - `plots/dyn_fk_lem1_y40_a10_hold2_multiview.png`
  - `plots/dyn_fk_lem1_y40_a10_hold2_timeseries.png`
- Full sequence:
  - `plots/dyn_fk_lem12_ypm40_a10_hold1_multiview.png`
  - `plots/dyn_fk_lem12_ypm40_a10_hold1_timeseries.png`
  - `plots/dyn_fk_lem12_ypm40_a10_hold2_multiview.png`
  - `plots/dyn_fk_lem12_ypm40_a10_hold2_timeseries.png`

### Additional implementation note (lem2 alignment bugfix)

During the combined rollout, lem2 is rotated to start near a chosen current (closest to `-u_init`) so the bridge connects sensibly. A bug in the earlier alignment code could desynchronize the rotated `u_seq2` vs rotated `p_des_seq2/p_proj_seq2`, making it look like projected-vs-FK was bad even when IK was correct. This was fixed by using the same computed start index for all three sequences.

- Vary start current:
  - `u_init = [0,0,c3_start]` with `c3_start ∈ {0.005, 0.01, 0.02}`
- Vary ramp length:
  - `ramp_to_circle_steps ∈ {60, 120, 240}`
- Confirm no convergence regressions and track peak ramp error.

### 6) Trajectory coverage / different circles

Goal: verify this isn’t specific to y=±40, r=10.

- Repeat with:
  - different radii (e.g. r=5, r=15)
  - different centers (e.g. y=±30, ±50) within workspace
  - varying z-plane selection (fixed z, or z chosen by workspace median vs nearest)
- Report projection residual `||des-proj||` and FK–Dyn errors.

### 7) Workspace projection quality checks

Goal: understand whether DP projection is the limiting factor.

- Compare projection methods:
  - greedy vs DP (`k_nearest`, `alpha_u`)
  - DP with stronger/weaker smoothness penalty
- Track:
  - projection residual `||des-proj||` statistics
  - current smoothness `||Δu||` statistics
  - whether improved projection reduces dynamics error (it may, indirectly, by reducing jerk).

### 8) Add projected-target overlays to FK/Dyn plots

Goal: make failure modes visually obvious.

- For the dyn plots, overlay:
  - projected circle points (the IK target in tip space)
  - optionally the desired circle
- This separates:
  - IK error (proj vs FK under IK currents)
  - dynamic tracking error (FK vs Dyn)

### 9) Stress tests (optional)

Goal: check solver robustness near boundaries / fast excitation.

- Larger current bounds (if physically allowed): `[-0.35, 0.35]`, `[-0.4, 0.4]`
- Inject small current noise on top of the circle (band-limited)
- Increase traversal speed until tracking breaks (estimate effective bandwidth).
