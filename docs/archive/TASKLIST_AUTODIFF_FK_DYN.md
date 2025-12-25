# Concrete Task List: Differentiable FK + DYN (CRM_ML)

This checklist is the implementation plan, broken down per file, in the recommended refactor/implementation order.

## 0) Ground truth (no code changes)
- `src/CRM_ForwardKinematics.cpp`: FK forward path (free + contact)
- `src/CoilDynamics_Defs.cpp`: dynamics residual + stepping
- `crm_ml_rl/wrappers/crm_bindings.cpp`: existing pybind11 module exposing FK and DYN forward
- `crm_ml_rl/models/hybrid_kinematics.py` / `crm_ml_rl/models/hybrid_dynamics.py`: currently detach tensors to numpy before calling C++ physics (breaks gradients)

## 1) C++ / pybind: add differentiable-friendly, *pure* stepping APIs
### `crm_ml_rl/wrappers/crm_bindings.cpp`
Add APIs that avoid hidden mutation so gradients/linearizations are well-defined:
- [x] **Seed IO**
  - `CRMDynamics.get_seed_state() -> dict` returning `v,w,p,R,xf,mL,nL` as numpy arrays.
  - `CRMDynamics.set_seed_state(v,w,p,R,xf,mL=None,nL=None)` to set internal state deterministically.
- [x] **Pure step**
  - `CRMDynamics.step_from_seed(currents, insertion_length, v,w,p,R,xf, mL=None,nL=None, dt=None) -> dict`
    - Runs one dynamics step using local copies of the seed (does not mutate internal state).
    - Returns `next_v,next_w,next_p,next_R,next_xf,next_mL,next_nL` + `tip_position, tip_velocity, converged, localmin`.
- [x] **Linearization**
  - `CRMDynamics.linearize_action_from_seed(..., eps=1e-4, method="central") -> dict`
    - Computes `B = d(next_state6)/d(currents3)` via finite differences around the seed.
    - Returns `next_state6` and `B` (6x3) plus the `next_*` seed.

Notes:
- FK already has an analytical Jacobian in C++; for Torch autograd we can use `CRMKinematics.compute_jacobian` and slice `dp/d(currents,insertion)`.
- A full autodiff refactor of dynamics (templated scalar types for `autodiff::real`) is a separate phase; this change still unblocks iLQR/MPC by providing `B` reliably.

## 2) Python/Torch: expose differentiable FK and DYN for ML/MPC
### `crm_ml_rl/wrappers/torch_physics.py` (new)
Implement Torch autograd Functions that call `crm_python`:
- [x] `crm_fk(currents, insertion_length) -> tip_pos`
  - Forward: calls `CRMKinematics.forward_kinematics` per sample.
  - Backward: calls `CRMKinematics.compute_jacobian` (or a new minimal Jacobian API) and returns gradients w.r.t currents/insertion.
- [x] `crm_dyn_step(seed, currents, insertion_length) -> next_state6`
  - Forward: calls `CRMDynamics.linearize_action_from_seed` and returns `next_state6` (and caches `B`).
  - Backward: uses cached `B` to compute `dL/dcurrents = B^T * dL/dnext_state`.

### `crm_ml_rl/models/hybrid_kinematics.py`
- [x] Add a flag `use_torch_physics` (default False).
- [x] When enabled: call `crm_fk(...)` directly on tensors (no detach-to-numpy), so supervised training can backprop into currents/insertion if desired.

### `crm_ml_rl/models/hybrid_dynamics.py`
- [x] Add a flag `use_torch_physics` (default False) for training/planning use.
- [x] When enabled: call `crm_dyn_step(...)` (or use `B` for MPC) instead of `CRMSimulator` numpy stepping.

## 3) Tests / sanity checks
### `tests/test_torch_physics_gradients.py` (new)
- [x] FK: verify `tip_pos.sum().backward()` produces finite gradients for currents (and insertion if tensor).
- [x] DYN: verify `next_state.sum().backward()` produces finite gradients for currents (via cached `B`).

## 4) Build
- [x] Rebuild pybind module: configure with `-DBUILD_PYTHON_BINDINGS=ON` and ensure `crm_ml_rl/wrappers/crm_python*.so` is produced and importable.
