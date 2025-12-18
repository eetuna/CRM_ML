# Autodiff Dynamics Review: `autodiff_eigen` vs `autodiff_template`

This document captures what was implemented, why, how to use it, what was tested, what failed, and what still needs improvement — in a way that a different agent can reproduce and review.

## 0) Goal / Context

Goal: enable **autodiff-based residual Jacobians** for the dynamics nonlinear equation (DYNNLEquation residual) and use them in the **implicit differentiation** linearization path used by the `crm_ml_rl` pipeline.

Two implementations were compared:
- **`autodiff_template`**: templated “math path” for DYNNLEquation-style residuals (dual-number/autodiff).
- **`autodiff_eigen`**: reimplementation of DYNNLEquation **residual only** using **Eigen** types + autodiff scalars.

Both are evaluated against baselines from the existing C++/pybind dynamics wrappers by running repeated converged trials and checking prediction accuracy under perturbations.

## 1) What Was Done (Branch-by-Branch)

### 1.1 `autodiff_eigen` (this branch)

**Purpose**
- Provide a **clean Eigen-based residual implementation** for DYNNLEquation so we can get `Jxx = dF/dx` via autodiff (dual numbers).
- Use `Jxx` in the implicit-diff linearization of dynamics.

**Key implementation**
- **Eigen+autodiff residual (residual only)**:
  - `src/CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp`
  - Implements (for `NUM_ACT_SET == 1`) the DYNNLEquation residual using:
    - coil dynamics integrator (ABM4 + RK2 warmup),
    - backward flexible segment integration (ABM4 + RK2 warmup),
    - analytic SE(3) updates,
    - residual assembly identical in shape to `DYNNLEquation` output (6D for one actuator set).
  - Exposes:
    - `DYNNLEquationJacobianEigenAD(x_scaled, Params, &residual)`

**Bindings usage**
- `crm_ml_rl/wrappers/crm_bindings.cpp`:
  - `linearize_full_seed_action_from_seed_implicit(...)` computes `Jxx` via `DYNNLEquationJacobianEigenAD(...)` when possible, otherwise falls back to FD `Jxx`.
  - Added optional `return_debug` argument to expose:
    - `have_ad_jxx`, `Jxx`, `Jxx_fd`, `residual_norm`

**Build**
- `CMakeLists.txt` adds autodiff include path to `crm_python`:
  - `third_party/autodiff`

**Known limitations**
- The Eigen residual implementation is **explicitly scoped to `NUM_ACT_SET==1`** (`static_assert`).
- Current implicit linearization still uses **FD** for:
  - `Jxθ` (residual partials wrt currents+seed),
  - `gx` and `gθ` (output mapping partials),
  - only `Jxx` is autodiff.

### 1.2 `autodiff_template` (worktree build)

**Purpose**
- Provide a templated residual evaluation where the residual math path is templated on scalar type (dual).

**How it was reviewed without changing it**
- Used `git worktree` to check out the branch in a separate folder:
  - `.worktrees/autodiff_template`
- Built and executed comparison scripts against the produced `.so` from that worktree.

**Build caveat discovered**
- The branch expects autodiff headers (`<autodiff/...>`) but the branch CMake may not add the include directory.
- Workaround used for comparison build (no source changes to the branch):
  - Configure with an additional include in compiler flags:
    - `-DCMAKE_CXX_FLAGS='-I/workspaces/catheter/CRM_ML/third_party/autodiff'`

## 2) How To Reproduce (Task-Based)

### 2.1 Create worktrees (optional but recommended for comparisons)

From repo root:
```bash
mkdir -p .worktrees
git worktree add .worktrees/autodiff_template autodiff_template
git worktree add .worktrees/main main
```

### 2.2 Build Python module for each branch

**Build current branch (`autodiff_eigen`)**
```bash
cmake -S . -B build_py -DBUILD_PYTHON_BINDINGS=ON
cmake --build build_py -j 4
```

**Build `autodiff_template` worktree**
```bash
cmake -S . -B build_py -DBUILD_PYTHON_BINDINGS=ON \
  -DCMAKE_CXX_FLAGS='-I/workspaces/catheter/CRM_ML/third_party/autodiff'
cmake --build build_py -j 4
```

Outputs:
- `autodiff_eigen` module:
  - `crm_ml_rl/wrappers/crm_python.cpython-310-x86_64-linux-gnu.so`
- `autodiff_template` module:
  - `.worktrees/autodiff_template/crm_ml_rl/wrappers/crm_python.cpython-310-x86_64-linux-gnu.so`

### 2.3 Run targeted AD-vs-FD `Jxx` checks

These tests are gated (slow):
```bash
CRM_RUN_DYNNLEQUATION_AD_TESTS=1 pytest -q tests/test_dynnlequation_residual_eigen_autodiff.py
```

### 2.4 Run the comparison sweeps (Eigen vs Template vs Baselines)

The comparison is split into two scripts:
- `scripts/run_dyn_linearization_benchmark.py`: runs a single `.so` and writes JSONL trials.
- `scripts/compare_autodiff_eigen_template_report.py`: orchestrates both modules, aligns common converged samples, aggregates results.

Run:
```bash
python3 scripts/compare_autodiff_eigen_template_report.py --candidates 60 --trials 15 --dirs 6 --seed 0
```

Outputs (written under `output_data/`):
- `output_data/bench_eigen.jsonl`, `output_data/bench_template.jsonl`
- `output_data/bench_eigen.log`, `output_data/bench_template.log`

### 2.5 Trajectory replay (ramp + circle) paper-style report

For the y=40mm, r=10mm “ramp_then_circle1” sequences, we added a “paper-style” report that:
- steps through the full CSV from the beginning (so the **ramp** produces the correct internal dynamics state/seed),
- samples **K** indices from the **circle** segment,
- evaluates implicit linearization (`B_imp`) and validates it with a robust **directional two-solve** baseline (no matrix FD baseline).

Scripts:
- Single-module report: `scripts/dynamics_fk_validation/run_circle_linearization_report_one_module.py`
- Eigen vs Template orchestrator: `scripts/dynamics_fk_validation/compare_circle_linearization_report.py`

Example runs:
```bash
# Hold1: circle has 200 rows (start=120)
python3 scripts/dynamics_fk_validation/compare_circle_linearization_report.py \
  --currents-csv output_data/ramp_then_circle1_currents_hold1.csv \
  --out-json output_data/paper_circle_report_hold1_k100.json \
  --k 100 --dirs 5 --du-sigma 1e-5 --seed-sigma 0.0 --seed 0

# Hold2: circle has 400 rows (start=240)
python3 scripts/dynamics_fk_validation/compare_circle_linearization_report.py \
  --currents-csv output_data/ramp_then_circle1_currents_hold2.csv \
  --out-json output_data/paper_circle_report_hold2_k200.json \
  --k 200 --dirs 5 --du-sigma 1e-5 --seed-sigma 0.0 --seed 0
```

Outputs:
- `output_data/paper_circle_report_hold1_k100.json` (+ per-module `.eigen.json` / `.template.json`)
- `output_data/paper_circle_report_hold2_k200.json` (+ per-module `.eigen.json` / `.template.json`)

## 3) What “Baselines” and “Checks” Mean (Exact Definitions)

Each trial chooses a `(currents, insertion_length, seed)` and requires:
- `step_from_seed(...)` returns `converged=True`

Then it computes:

### 3.1 `B_fd` (FD on currents only)

Computed via:
- `crm_python.CRMDynamics.linearize_action_from_seed(...)`

Meaning:
- Fix full seed `(v,w,p,R,xf,mL,nL)`
- Perturb currents `u ∈ R^3` by `±ε e_i`
- Estimate `B_fd[:, i] ≈ (y(u+εe_i) - y(u-εe_i)) / (2ε)`

Here output is:
- `y = [tip_position(3), tip_velocity(3)] ∈ R^6`

### 3.2 Implicit linearization `(A_imp, B_imp)`

Computed via:
- `crm_python.CRMDynamics.linearize_full_seed_action_from_seed_implicit(...)`

Meaning:
- Internal unknowns `x` (typically `mL,nL` in scaled form) satisfy `F(x, u, seed) = 0`.
- We want `dy/du` and `dy/dseed` while accounting for `x` changing implicitly with `(u, seed)`.
- Current state estimate:
  - `y_pred ≈ y0 + B_imp δu + A_imp δseed`

**Important**: only `Jxx = ∂F/∂x` is autodiff; other required partials are currently FD.

### 3.3 “Directional validation errors”

Two kinds:
- **Currents-only check of `B_fd`**:
  - sample small `δu`, compare `y0 + B_fd δu` vs `y(u+δu)` from `step_from_seed`.
- **Full (currents+seed) directional check**:
  - sample `δu` and `δseed`
  - compute `y_true = y(u+δu, seed+δseed)` via `step_from_seed`
  - compute `y_pred = y0 + B_imp δu + A_imp δseed`
  - report normalized error `||y_pred - y_true|| / (||y_true|| + 1e-9)`

Additionally, we computed a “directional FD baseline” (two-solve):
- Evaluate both `y(u+δ)` and `y(u-δ)` (with symmetric seed perturbations) and use:
  - `y0 + (y_plus - y_minus)/2`
This gives a very strong baseline for “how linear” the solver mapping is at that operating point.

## 4) Tests / Comparisons That Were Run and Why

### 4.1 Unit/Integration tests

**Why**: ensure builds and core RL pipeline didn’t break; ensure the new autodiff residual Jacobian is numerically sane.

Commands (examples):
```bash
pytest -q
CRM_RUN_DYNNLEQUATION_AD_TESTS=1 pytest -q tests/test_dynnlequation_residual_eigen_autodiff.py
```

### 4.2 Cross-branch comparison sweep

**Why**:
- verify `autodiff_eigen` and `autodiff_template` produce equivalent implicit linearizations (at least for `B_imp`),
- see whether either matches the old FD baseline (`B_fd`),
- validate both using “directional truth” checks.

Commands executed (examples):
```bash
python3 scripts/compare_autodiff_eigen_template_report.py --candidates 60 --trials 15 --dirs 6 --seed 0
python3 scripts/compare_autodiff_eigen_template_report.py --candidates 80 --trials 25 --dirs 6 --seed 1
python3 scripts/compare_autodiff_eigen_template_report.py --candidates 60 --trials 15 --dirs 6 --seed 2
```

## 5) Results Summary (Key Takeaways)

Across multiple runs:
- `autodiff_eigen` and `autodiff_template` produced **numerically identical `B_imp`** (relative Frobenius differences ~`1e-11`).
- Both **strongly disagreed** with the current-only FD baseline `B_fd` (`rel||B|| ≈ 1`).
- Under a stronger “directional FD two-solve” baseline, the solver behaves locally linear with errors around `~1e-4`–`1e-3`, while implicit linearization errors were `~1e-3`–`1e-2` (and worse in harder regimes).

Example output (seed=0, candidates=60, trials=15, dirs=6):
- `eigen_imp vs tpl:  rel||B|| mean/max 9.208e-12 / 3.196e-11`
- `eigen_imp vs B_fd: rel||B|| mean/max 1.000e+00 / 1.001e+00`
- Currents-only `B_fd` prediction error mean/max `2.415e-01 / 4.437e-01`
- Directional FD baseline mean/max `8.842e-05 / 2.754e-04`
- Directional implicit mean/max `4.507e-03 / 9.108e-03`

### 5.1 Trajectory replay (ramp included, metrics on circle)

We validated on the real “ramp_then_circle1” trajectories (from `output_data/RUN_INSTRUCTIONS_currents_y40_r10.md`) using the paper-style report in §2.5.

Key outcomes:
- Eigen vs Template `B_imp` match remained extremely tight (order `1e-10`–`1e-8` relative).
- No “Coil integration Unbounded!!” warnings occurred in these paper-style reports because they do **not** call the matrix FD baseline (`linearize_action_from_seed`).

Scaled sample-count runs:
- Hold1 (`k=100`, `dirs=5`, `du_sigma=1e-5`, `seed_sigma=0`)
  - `rel||B_imp||` eigen vs template mean/max: `2.022e-10 / 9.177e-09`
  - directional two-solve baseline error mean/max: `1.253e-04 / 5.753e-03`
  - directional implicit error mean/max: `2.622e-04 / 9.325e-03`
  - report: `output_data/paper_circle_report_hold1_k100.json`
- Hold2 (`k=200`, `dirs=5`, `du_sigma=1e-5`, `seed_sigma=0`)
  - `rel||B_imp||` eigen vs template mean/max: `6.493e-10 / 2.989e-08`
  - directional two-solve baseline error mean/max: `1.161e-04 / 4.017e-03`
  - directional implicit error mean/max: `2.884e-04 / 1.902e-02`
  - report: `output_data/paper_circle_report_hold2_k200.json`

## 6) What Failed / Issues Encountered

### 6.1 Multiple `.so` imports in one Python process
- CPython extension modules define an init symbol like `PyInit_crm_python`.
- You cannot safely import the same extension binary under a different module name (`PyInit_crm_python_eigen` does not exist).
- **Fix used**: run each module’s benchmarks in separate subprocesses and compare their JSONL outputs.

### 6.2 “Coil integration Unbounded!!” spam / numerical instability
- Many random currents cause blow-ups in the coil integrator path, flooding stdout and slowing comparisons.
- **Mitigation used**: sample smaller currents and bias `c3 += 0.02` to stay in a more stable region.

#### 6.2.1 Isolation finding: unbounded spam is dominated by the FD baseline path

We added diagnostic switches to `scripts/dynamics_fk_validation/run_dyn_linearization_sequence_benchmark.py` and found:
- `linearize_action_from_seed` (currents-only matrix FD baseline) is the primary source of `"Coil integration Unbounded!!"` spam on the circle trajectory slice.
- Implicit-only runs (`linearize_full_seed_action_from_seed_implicit` only) showed **zero** unbounded warnings on the same slice.

Evidence (hold1 circle slice, 40 sampled steps, no validation perturbations):
- FD-only (`--skip-implicit`) and “FD+implicit” produced identical unbounded counts and identical hotspot steps.
- Implicit-only (`--skip-linearize-action-fd`) produced 0 unbounded warnings.

Practical implication:
- do not treat `B_fd` from `linearize_action_from_seed` as a reliable “truth” baseline in these regimes; prefer directional two-solve validation.

### 6.3 Matplotlib cache permission warning
- Some environments can’t write to `~/.cache/matplotlib`.
- **Mitigation used**: set `MPLCONFIGDIR=/tmp/mpl` in subprocess env.

### 6.4 FD baseline `B_fd` can be extremely unstable
- In some seeds, “currents-only” linearization error became astronomically large (indicative of solver instability / extremely nonlinear behavior / sensitivity).
- This suggests `B_fd` is not a reliable “truth baseline” for the implicit method at arbitrary sampled points.

## 7) What Needs Improvement Next

High-impact improvements (in priority order):
1. **Improve implicit linearization accuracy**
   - Autodiff `Jxθ` (residual partials wrt currents + seed), not just `Jxx`.
   - Reduce FD use in `gx`/`gθ` partials (or compute via autodiff through a differentiable output mapping).
2. **Stabilize the coil dynamics integration**
   - Investigate why `CoilDynamics` becomes unbounded for modest currents; add safeguards or narrower operating envelope for training.
3. **Define a better baseline**
   - Prefer “two-solve directional FD baseline” over `B_fd` for validating linearizations.
   - Add automatic detection and logging of outlier regimes (currents, residual norms, etc.).
   - Consider de-emphasizing or removing matrix FD `linearize_action_from_seed` from benchmark suites, or run it only on “safe” operating envelopes.
4. **Extend beyond `NUM_ACT_SET==1`**
   - Remove the `static_assert` limitation and generalize the Eigen residual to multiple actuators.
5. **Fix build ergonomics in `autodiff_template`**
   - Add proper `third_party/autodiff` include path in its `CMakeLists.txt` so no flag hack is needed.
