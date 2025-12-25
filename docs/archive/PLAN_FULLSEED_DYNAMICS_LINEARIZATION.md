# Plan: Full-Seed Dynamics Linearization (crm_ml_rl)

Goal: enable (b) iLQR/MPC and (c) end-to-end learning by providing gradients of one-step dynamics outputs w.r.t. **currents** and the **full seed** (`v,w,p,R,xf,mL,nL`).

## Definitions

- Output we differentiate: `next_state6 = [tip_position(3), tip_velocity(3)]`.
- Full seed (inputs to one-step dynamics): `v,w,p,R,xf,mL,nL` (as already carried by the C++ bindings + `DynamicsSeed` in `torch_physics.py`).
- Jacobians:
  - `B = d(next_state6) / d(currents3)` (already exists via finite differences).
  - `A = d(next_state6) / d(seed_flat)` where `seed_flat = [v,w,p,R,xf,mL,nL]` flattened in a fixed order.

## Phase 1 (this implementation): finite-difference A + B

1) **C++ binding: compute full-seed linearization**
   - Add `CRMDynamics.linearize_full_seed_action_from_seed(...)` in `crm_ml_rl/wrappers/crm_bindings.cpp`.
   - Use central differences to compute:
     - `B` as before (currents perturbations).
     - `A` by perturbing each element of the full seed vector.
   - Return `{next_state, B, A, base}` where `base` is the existing `step_from_seed(...)` dict.

2) **Torch autograd: propagate gradients into seed**
   - Update `crm_ml_rl/wrappers/torch_physics.py`:
     - Call the new C++ API when any seed tensor requires grad.
     - Cache `A` and `B` for backward.
     - In backward: return gradients for `seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL` using `Aᵀ * dL/dnext_state`.

3) **Tests**
   - Extend `tests/test_torch_physics_gradients.py` to assert:
     - finite gradients for currents (existing)
     - finite gradients for full seed tensors (new)

## Phase 2 (future): replace finite differences with implicit differentiation + autodiff of residuals

- Replace FD `A/B` with Jacobians computed via implicit differentiation of the inner dynamics solve:
  - Compute `∂F/∂x` and `∂F/∂inputs` at the converged solution using autodiff on the residual function, then solve linear systems for sensitivities.
- Keep FD fallback for non-converged / mode-switch points (discontinuities).

