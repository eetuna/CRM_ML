# Plan: Route (1) – Template DYNNLEquation Math Path for Autodiff

Goal: enable **autodiff residual Jacobians** for dynamics by making the math path inside `DYNNLEquation` evaluable with autodiff scalars.

## Why this exists

The inner dynamics solve is defined by the residual equation:

- `F(x; θ) = 0` where `F` is computed by `DYNNLEquation(...)`
- `x` is the solver unknown vector (scaled `mL,nL`), size `6 * NUM_ACT_SET`

Implicit differentiation needs `Jxx = ∂F/∂x` (and optionally `Jxθ`).

## Step 1 — Template low-level math utilities

`DYNNLEquation` uses low-level math helpers (matrix multiplies, hat operators, SE(3) stepping).
Many are currently `double`-only via `src/CRM_MatrixOperations.hpp`.

Action:
- Generalize these helpers to accept arbitrary scalar types and mixed-type inputs:
  - `double` for parameters/constants
  - `autodiff::real` for differentiable variables

## Step 2 — Template DYNNLEquation evaluation

Add a templated version:

- `template <class Scalar> DYNNLEquationT(...)`

Keep the existing `double` entrypoint unchanged by calling `DYNNLEquationT<double>(...)`.

## Step 3 — Compute `Jxx` using autodiff

In the implicit linearization API:

- Replace finite differences for `Jxx = ∂F/∂x` with autodiff:
  - Evaluate `F(x)` with `autodiff::real` inputs at the converged solution `x*`
  - Use `autodiff::jacobian` to obtain `Jxx`
- Keep `Jxθ` as FD initially (θ includes currents + full seed, large dimensional).

## Step 4 — Validation

Add tests/tooling:

- Compare `Jxx_ad` vs `Jxx_fd` at converged points:
  - max_abs/max_rel/frobenius agreement
  - ensure both finite
- Compare downstream `A,B` (implicit method) against full-step FD baseline at stable operating points.

## Step 5 — Performance & safety

- Guard AD code behind a separate C++ API or environment flag (so normal simulation stays fast).
- Keep FD fallback if convergence fails or `Jxx` becomes rank-deficient.

