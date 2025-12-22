# Plan: End-to-End Differentiable Simulator (Options A / B / C)

This document is a **task-based implementation plan** for three “end-to-end differentiable simulator” options, grounded in the current state of this repo (notably the `autodiff_eigen` and `autodiff_template` branches/worktrees).

It is written so another AI agent can:
- understand where we are today,
- pick an option (A/B/C),
- implement incrementally with clear acceptance criteria,
- validate against baselines (FD and existing C++),
- and report results suitable for an academic paper.

## Executive Summary (Read Order for a New Agent)

If you are handing this repo to another agent, this is the recommended review sequence:

1. **Implementation + results (what exists today)**
   - Read `docs/autodiff_dynamics_eigen_vs_template_review.md`
   - Focus on:
     - what `autodiff_eigen` implements (Eigen residual + AD `Jxx`),
     - what `autodiff_template` provides (templated residual),
     - how implicit linearization is wired in the pybind layer,
     - and the trajectory-based “paper-style” reports (ramp included, metrics on circle).

2. **End-to-end options (what to build next)**
   - Read this doc: `docs/PLAN_end_to_end_differentiable_simulator_options_A_B_C.md`
   - Focus on:
     - Option A (extend implicit+AD partials),
     - Option C (Torch C++ extension packaging),
     - Option B (full torch physics, highest rewrite risk).

3. **Raw artifacts (repro + paper evidence)**
   - Inspect the produced reports:
     - `data/output/paper_circle_report_hold1_k100.json`
     - `data/output/paper_circle_report_hold2_k200.json`
   - These are the most paper-relevant “does it work on the real trajectory?” summaries.

4. **Optional deeper dive (script-level / debugging)**
   - If investigating stability issues or FD baselines:
     - `scripts/dynamics_fk_validation/run_dyn_linearization_sequence_benchmark.py`
     - `scripts/dynamics_fk_validation/compare_circle_linearization_report.py`
     - and the log isolation showing FD baseline triggers `"Coil integration Unbounded!!"`.

## 0) Definitions / Current State (What exists today)

### Dynamics “seed”
In this repo, “seed” means the full internal state required to compute one dynamics step as a pure function:
- `v, w, p, R` for each actuator set (coil/segment states)
- `xf` (15D tip state pack: `u,R,p`)
- optional `mL, nL`

Python-facing entry points (pybind, `crm_python.CRMDynamics`):
- `get_seed_state() -> dict`
- `step_from_seed(currents, insertion_length, v,w,p,R,xf, mL=None,nL=None, dt=None) -> dict`

These **are not** upstream `../CRM_Dynamics` APIs; they are added in `CRM_ML` to make the solver callable as a pure function from Python and to support ML/control tooling.

### Linearization utilities (pybind)
Python-facing linearization utilities exist in `crm_ml_rl/wrappers/crm_bindings.cpp`:
- `linearize_action_from_seed(...)`: returns `B = ∂y/∂u` via FD on currents only.
- `linearize_full_seed_action_from_seed(...)`: returns `A = ∂y/∂seed` and `B` via FD (expensive baseline).
- `linearize_full_seed_action_from_seed_implicit(...)`: returns `A,B` via implicit differentiation of the internal nonlinear solve; **currently uses AD only for `Jxx = ∂F/∂x`**, and FD for other partials.

Torch integration (Python-level autograd wrappers):
- `crm_ml_rl/wrappers/torch_physics.py`
  - FK: analytical Jacobian from C++ (already differentiable w.r.t currents/insertion in Torch).
  - Dynamics: differentiable w.r.t currents, and optionally seed, by using the above `linearize_*` outputs in `backward()`.

### Autodiff dynamics residual implementations (two styles)
1) **Eigen residual + autodiff jacobian** (current branch `autodiff_eigen`)
- File: `src/CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp`
- Provides residual-only implementation of DYNNLEquation and:
  - `DYNNLEquationJacobianEigenAD(x_scaled, Params, &residual)`
- Limitation: `NUM_ACT_SET == 1` only.

2) **Template residual path** (`autodiff_template` worktree/branch)
- Templated residual evaluation with autodiff scalar types.
- In comparisons, produced essentially identical `B_imp` to the Eigen approach.

### Validation baselines already available in this repo
- “Current-only FD baseline”: `linearize_action_from_seed` (cheap, but often a weak “truth” baseline).
- “Full FD baseline”: `linearize_full_seed_action_from_seed` (expensive, but useful for sanity).
- “Directional FD two-solve baseline”: compute `y(u+δ)`, `y(u-δ)` via `step_from_seed` and use the symmetric difference to validate linearizations.
- Cross-branch comparison harness:
  - `scripts/run_dyn_linearization_benchmark.py`
  - `scripts/compare_autodiff_eigen_template_report.py`

## 1) The three options (what they mean operationally)

### Option A — “C++ AD + implicit diff, expose Jacobians” (what we’re currently closest to)
End-to-end differentiability is achieved by:
1) keeping the C++ simulator as the forward pass,
2) computing and exposing the needed Jacobians (A/B/optionally parameter grads),
3) using those Jacobians in a Torch custom backward (Python `torch.autograd.Function` or a Torch C++ extension).

This is “end-to-end differentiable” **with respect to whichever inputs you provide gradients for**.

### Option B — “Full PyTorch physics implementation”
End-to-end differentiability is achieved by rewriting the dynamics step in PyTorch so autograd traces the entire computation graph.

### Option C — “Torch C++ extension custom op”
End-to-end differentiability is achieved by writing a Torch extension op whose forward is the C++ simulator call and whose backward is implemented in C++ (using either implicit-diff Jacobians or other differentiation strategies).

This removes the *Python-level* linearization dependency but still uses the same math ideas as Option A.

## 2) Recommendation: starting point (Eigen vs Template)

For Options A/C, prefer **starting from `autodiff_eigen`**:
- it already has a clear, isolated AD Jacobian implementation (`Jxx`) without requiring templating the entire double-only math stack,
- it was validated against `autodiff_template` and matched closely,
- incremental extension is easier (add partials one by one).

Keep `autodiff_template` as:
- a reference/baseline,
- or a future refactor target if you decide to template the original residual codebase end-to-end.

For Option B (pure PyTorch), branch choice matters less; it’s effectively a new implementation.

## 3) Shared prerequisites (all options)

### 3.1 Decide what “end-to-end” means (inputs/outputs)
Define a single canonical differentiable step:
- Inputs: `(seed, u, insertion_length, dt, θ)`
  - `seed` includes at least `v,w,p,R,xf` and optionally `mL,nL`
  - `θ` are physical parameters you may want to identify (damping, stiffness, magnetization scalings, etc.)
- Output: `y = next_state6 = [tip_position(3), tip_velocity(3)]` (current repo convention)

Acceptance criteria:
- deterministically produces the same output as the existing C++ stepping for the same inputs (within tolerance).

### 3.2 Decide gradient targets
Minimum useful gradients for paper-quality “differentiable planning/control”:
- `B = ∂y/∂u` (currents)
- `A = ∂y/∂seed` (state)

Optional (higher effort, high paper value):
- `G = ∂y/∂θ` (system identification / sim-to-real calibration)

### 3.3 Stabilize/define operating envelope
The C++ dynamics can blow up (“Coil integration Unbounded!!”) for some random currents.

Tasks:
- define bounded action ranges for evaluation (currents and slew-rate),
- define “converged sample” selection criteria:
  - `step_from_seed(...).get("converged", True)` must be true,
  - residual norm thresholds and max iterations thresholds.

Acceptance criteria:
- benchmark scripts can sample N points and obtain at least K converged points consistently.

## 4) Option A: Detailed task plan (C++ AD + implicit diff, expose Jacobians)

### A0) Scope (what “done” means)
Deliver an end-to-end differentiable step usable from torch via Python:
- Forward: call C++ dynamics step (seeded).
- Backward: return correct gradients for at least:
  - currents (`∂y/∂u`)
  - seed (`∂y/∂seed`)
- Validate gradients with:
  - FD checks,
  - cross-branch consistency (`autodiff_template`),
  - directional multi-solve checks.

### A1) Make implicit linearization “complete enough”
Current situation: `linearize_full_seed_action_from_seed_implicit` uses AD for `Jxx` only and FD for the rest.

Tasks:
1. Implement AD for **residual partials wrt θ** (“theta” = currents + seed components that enter `F`):
   - `Jxθ = ∂F/∂θ`
   - In practice, start with `∂F/∂u` and then expand to seed blocks.
2. Implement (or reduce FD in) **output mapping partials**:
   - `gx = ∂g/∂x` and `gθ = ∂g/∂θ` where `y = g(x*, θ)` and `x*` solves `F(x,θ)=0`.
   - If `g` is obtained by running IVP after computing `u0/tau`, decide where AD applies:
     - either make `g` differentiable and compute its partials,
     - or keep FD but quantify error (paper story: implicit residual AD vs FD output map).
3. Conditioning and scaling:
   - ensure consistent scaling between the solver’s scaled variables and reported Jacobians.

Acceptance criteria:
- For a set of converged points, implicit `A,B` match full FD (`linearize_full_seed_action_from_seed`) within a specified tolerance:
  - e.g., `rel_frob(A_imp - A_fd) < 1e-2` and `rel_frob(B_imp - B_fd_full) < 1e-2` on stable regimes.

### A2) Expand the Eigen residual beyond `NUM_ACT_SET==1` (optional but important)
Tasks:
- Generalize residual construction for `NUM_ACT_SET > 1`:
  - block-structure of residual is `(NUM_ACT_SET*6)` in the current DYNNLEquation convention (mL,nL per set).
- Update Jacobian helpers and tests accordingly.

Acceptance criteria:
- Build succeeds for higher `NUM_ACT_SET` configurations and AD-vs-FD residual Jacobians pass.

### A3) PyTorch “end-to-end” wrapper (Python-level)
Current wrapper exists: `crm_ml_rl/wrappers/torch_physics.py` (`CRMDynamicsStepFunction`).

Tasks:
1. Ensure `CRMDynamicsStepFunction.backward` can return gradients for:
   - `currents`
   - full seed tensors (`seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL`)
2. Decide whether `insertion_length` participates in gradients (optional).
3. Provide a higher-level module API:
   - `TorchCRMPhysics.dyn_step(...)` already exists; keep it the canonical interface.

Acceptance criteria:
- `torch.autograd.gradcheck` passes (or a custom FD check passes) on small batches and stable regimes.

### A4) Tests for Option A
Add/extend tests under `tests/`:

1) **Forward correctness**
- Compare torch wrapper forward output vs direct `crm_python.CRMDynamics.step_from_seed` output.

2) **Gradient correctness**
- `∂y/∂u`:
  - compare torch gradient to FD on `step_from_seed` (central diff).
- `∂y/∂seed`:
  - compare torch gradient to FD on `step_from_seed` where you perturb the seed components.

3) **Implicit vs baselines**
- Compare implicit `A,B` to:
  - full FD (`linearize_full_seed_action_from_seed`)
  - directional FD two-solve baseline

4) **Cross-implementation consistency**
- Compare `autodiff_eigen` vs `autodiff_template` (subprocess harness) for `A_imp,B_imp`.

Acceptance criteria:
- All tests pass in stable envelopes and produce a structured report (JSONL) with:
  - relative Frobenius errors,
  - directional validation errors,
  - convergence stats.

### A5) Paper-facing experiment recipes (minimal, not overly ambitious)
Provide scripts that reproduce:
- **AD vs FD Jacobian quality** (runtime + error):
  - plot/print distributions of `rel_frob(A)`, `rel_frob(B)` and directional prediction error.
- **Controller demo** (simple):
  - iLQR/MPC on short horizon with tip-position tracking, using:
    - FD Jacobians vs implicit Jacobians vs (optional) partial-AD improvements.

Acceptance criteria:
- A script that runs in under X minutes and outputs figures/tables.

## 5) Option B: Detailed task plan (Full PyTorch physics)

### B0) Scope (what “done” means)
Implement `torch`-native `y = f_torch(seed,u,θ)` where:
- forward matches C++ (within tolerance),
- autograd provides gradients automatically (no `linearize_*` calls),
- stable enough for training-time usage.

### B1) Implement the dynamics step in torch
Tasks:
1. Choose state representation in torch (match C++):
   - `v,w,p,R` per actuator, plus tip state `xf`.
2. Reimplement integration:
   - coil dynamics update (`CoilDynamics`) with stable integrator,
   - flexible segment integration (ABM4/RK2 warmup as in C++ if you want close matching),
   - se(3)/SE(3) update (ensure orthonormalization strategy matches).
3. Implement the nonlinear solve:
   - if the C++ step relies on solving for `mL,nL` via a root finder, replicate in torch:
     - prefer a differentiable root solve (e.g., fixed-point / Newton with differentiable iterations),
     - or use implicit differentiation (still possible in torch, but now entirely in torch).

Acceptance criteria:
- Forward match on a curated stable dataset of seeds/currents:
  - `||y_torch - y_cpp||` within tolerance.

### B2) Tests for Option B
1) Forward matching tests vs C++ (`step_from_seed`).
2) Autograd sanity:
   - compare `torch.autograd` gradients to FD on `f_torch`.
3) Cross-check vs C++ linearizations:
   - compare autograd-computed `B` to `linearize_action_from_seed` and/or implicit `B`.

### B3) Risks / why this is expensive
- large surface area rewrite,
- numerical mismatch risk (especially around implicit solves and stabilization),
- performance (Python-level torch loops can be slow unless vectorized).

## 6) Option C: Detailed task plan (Torch C++ extension custom op)

### C0) Scope (what “done” means)
Provide a Torch operator `crm_dyn_step(seed,u,θ) -> y` such that:
- forward uses the existing C++ step implementation,
- backward uses C++-computed gradients (implicit/AD/FD),
- install/build is reproducible and importable from Python,
- performance is better than Python-calling-pybind in a loop.

### C1) Implement a Torch extension
Tasks:
1. Create a new extension package, e.g.:
   - crm_ml_rl/wrappers/crm_torch_ext/ (planned; not present in this repo today), or `crm_torch/`
2. Build system:
   - use `torch.utils.cpp_extension` (Python `setup.py`) or integrate into existing CMake with Torch.
3. Implement the custom autograd Function in C++:
   - forward: call the same underlying C++ stepping as `step_from_seed`.
   - backward: compute `A,B,(G)` via:
     - `linearize_full_seed_action_from_seed_implicit` (preferred), or
     - FD fallback in debug mode.
4. Minimize overhead:
   - accept batched tensors and do batched stepping where possible.

Acceptance criteria:
- `import crm_torch_ext` works.
- `torch.autograd.gradcheck` (or FD check) passes on stable regimes.
- A microbenchmark shows fewer Python overheads vs `torch_physics.py` loop approach.

### C2) Tests for Option C
- Same as Option A, plus:
  - “operator parity” test:
    - `TorchCRMPhysics.dyn_step` output/gradients match the C++ extension op output/gradients.

## 7) Comparison matrix (what to measure and report)

For paper-quality reporting, implement a single evaluation harness that can test any option:

### Metrics (per converged sample)
- Forward agreement:
  - `||y_candidate - y_cpp||`
- Jacobian agreement:
  - vs full FD: `rel_frob(A - A_fd)`, `rel_frob(B - B_fd_full)`
  - vs directional two-solve: directional prediction error distributions
- Stability:
  - convergence rate (% converged)
  - residual norms
  - frequency of “unbounded” warnings
- Performance:
  - forward runtime per step
  - backward runtime per step

### Baselines to include
- `B_fd` (currents-only FD; keep but do not treat as ground truth)
- full FD (`A_fd_full, B_fd_full`)
- directional two-solve baseline (strong local linearity proxy)
- cross-implementation (`autodiff_eigen` vs `autodiff_template`)

## 8) What you might be missing (common pitfalls)

1) **Scaling consistency**
- The DYNNLEquation residual uses scaled variables (`IVALUE_SCALE_*`, `RESIDUAL_SCALE_*`). Ensure Jacobians are reported in unscaled coordinates expected by controllers/Torch.

2) **Nondifferentiabilities**
- Branches/guards (“unbounded”, clamps, early stopping) create gradient discontinuities. For paper experiments, report operating envelope and convergence filters.

3) **State definition mismatch**
- Many pipelines use `tip position + velocity` as “state”, but the solver’s true state is far higher-dimensional. Be explicit in docs/experiments about what `A` means (derivative wrt *seed*, not just tip pose).

4) **POMDP vs full-state**
- For SB3 RL (model-free), physics gradients are unused; don’t over-invest in “end-to-end” there unless you’re explicitly doing gradient-based planning or system ID.

5) **Multi-actuator sets**
- The current AD residual is limited to `NUM_ACT_SET==1`. If your real hardware uses more, plan this upgrade early.

## 9) Concrete next-step sequence (suggested execution order)

If the goal is “most value soonest” with minimal risk:
1) Finish Option A to the point that implicit `A,B` are validated vs full FD and directional FD.
2) Add a simple iLQR/MPC demo that consumes `A,B` (paper-friendly).
3) Only then consider Option C (performance/packaging).
4) Option B only if the paper’s core contribution requires a fully torch-native simulator.
