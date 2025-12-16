# Plan: Implicit Differentiation + Residual Derivatives (Dynamics)

Objective: provide stable, differentiable dynamics linearizations for (b) iLQR/MPC and (c) end-to-end learning:

- `B = d(next_state6)/d(currents3)`
- `A = d(next_state6)/d(full_seed)` where `full_seed = [v,w,p,R,xf,mL,nL]` flattened consistently with `torch_physics.py`.

`next_state6 = [tip_position(3), tip_velocity(3)]`.

## Why implicit differentiation

The dynamics step solves an inner nonlinear system each step (BVP root solve in `DynamicsBVP` via `TrustRegionDogleg_dyn`), then runs an IVP (`DYNSolverIVP`).
Differentiating through solver iterations is brittle; implicit differentiation differentiates the *equilibrium equations*.

## Phase 0: Baseline (already present)

- Full-step finite differences: `CRMDynamics.linearize_full_seed_action_from_seed` (A,B by perturb-and-rerun).
- Currents-only finite differences: `CRMDynamics.linearize_action_from_seed` (B only).
- Torch backprop uses cached A,B for gradients into currents/seed.

## Phase 1: Implicit differentiation (residual Jacobians) + extensive validation

### 1) Add an implicit linearization API in C++

Add `CRMDynamics.linearize_full_seed_action_from_seed_implicit(...)` returning:

- `next_state` (6,)
- `B` (6,3)
- `A` (6,seed_dim)
- debugging fields: `residual_norm`, `converged`, timing, and optionally `x_star`

Implementation outline:

1. Run the existing step once (or call `DynamicsBVP`) to get the converged root variables:
   - `x* := [mL*, nL*]` (the 6*NUM_ACT_SET unknowns).
2. Define the equilibrium residual function:
   - `F(x; theta) = DYNNLEquation(x, Params(theta))` where `theta` includes currents + seed inputs.
3. Compute residual Jacobians at `(x*, theta)`:
   - `Jxx = ∂F/∂x`
   - `Jxθ = ∂F/∂θ` for the subset of θ we need (currents + full seed).
4. Implicit sensitivity:
   - `dx/dθ = - Jxx^{-1} Jxθ`
5. Define the *post-solve* mapping (no root-solve):
   - `y = g(x, theta)` computed by evaluating `DYNNLEquation` once (to get `u0,tau`) then `DYNSolverIVP`.
6. Compute partials:
   - `gθ = ∂g/∂θ |x fixed`
   - `gx = ∂g/∂x |theta fixed`
7. Assemble:
   - `dy/dθ = gθ + gx * dx/dθ`
8. Split `dy/dθ` into `A` (seed part) and `B` (currents part).

Note: for “seed” components that only influence *solver initialization* (e.g., `mL,nL` guesses), implicit math yields ~0 sensitivity if the root is unique; FD baselines can differ because they capture algorithmic dependence on the initial guess. Comparisons will primarily focus on the physical seed components (`v,w,p,R,xf`) and currents.

### 2) Residual derivatives backend

Target backend: autodiff of residuals (forward-mode) to compute `Jxx` and `Jxθ` robustly.

However, the current dynamics residual uses `double`-only matrix utilities (`CRM_MatrixOperations.hpp`), so enabling true autodiff requires templating or reimplementing the needed math for `DYNNLEquation` with AD scalar types.

Implementation will proceed in two steps:

1. **Scaffold + correctness**: implement Phase 1 with central-difference residual Jacobians as a reference.
2. **Upgrade**: replace the residual Jacobian computation with autodiff once the residual path is templated.

### 3) Validation & testing

Add:

- Unit test comparing implicit vs full-step FD:
  - shapes, finiteness, and relative agreement for `B` and for the `v,w,p,R,xf` blocks of `A`.
  - allow larger tolerance near non-converged points; record `converged/localmin`.
- Sweep script producing CSV summaries:
  - random seeds/currents around realistic operating ranges
  - metrics: max_abs, max_rel, Frobenius, cosine similarity per column; stability vs FD step sizes.

Optional comparisons:

- Compare implicit `B` vs the existing currents-only FD `B` (`linearize_action_from_seed`).
- Compare directional derivatives of scalar losses `L = rᵀ next_state6` using central differences.

## Phase 2: Autodiff residuals (true AD backend)

Refactor options:

- Template the matrix ops used by `DYNNLEquation` to accept `autodiff::real`, or
- Reimplement the residual function in Eigen with AD scalars.

Then swap the residual Jacobian backend in Phase 1 from FD → autodiff and keep the same API/tests.

