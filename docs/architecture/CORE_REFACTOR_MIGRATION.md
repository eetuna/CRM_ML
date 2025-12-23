# Core Refactor Migration Notes (Task A1.7)

## Scope
This document captures the migration steps after removing legacy dynamics arrays
from `CRMIVPCoreParams`/`CRMShootingMethodParams` in favor of `DynamicsContext`.

## What Changed
- Legacy fixed-size dynamics arrays (`v_L_pre`, `w_L_pre`, `p_pre`, `R_pre`,
  `actInertia`, `damping`, `m_L`, `n_L`, `DELTA_T`) are no longer stored on the
  parameter structs.
- All dynamics data now flows through `DynamicsContext` and
  `DynamicsContextAD`/`LearnableParamsAD`.
- Manual damping overrides in `crm_bindings.cpp` were removed; parameter vectors
  now reflect the values stored in `DynamicsContext`.
- CTest is now enabled for the `TestDynamicsContext` target.

## Migration Guidance
If you were previously reading or writing legacy arrays on:
- `CRMIVPCoreParams`
- `CRMShootingMethodParams`

use the equivalent fields on `params.dynamics.actuators[i]` instead.

### Field Mapping
- `v_L_pre[i][j]` -> `dynamics.actuators[i].v_L_pre(j)`
- `w_L_pre[i][j]` -> `dynamics.actuators[i].w_L_pre(j)`
- `p_pre[i][j]` -> `dynamics.actuators[i].p_pre(j)`
- `R_pre[i][r*3+c]` -> `dynamics.actuators[i].R_pre(r, c)`
- `actInertia[i][r*3+c]` -> `dynamics.actuators[i].inertia(r, c)`
- `damping[i][j]` -> `dynamics.actuators[i].damping(j)`
- `m_L[i][j]` -> `dynamics.actuators[i].m_L(j)`
- `n_L[i][j]` -> `dynamics.actuators[i].n_L(j)`
- `DELTA_T` -> `dynamics.DELTA_T`

## Behavioral Notes
- Dynamics prep (`CRMDYNSolverIVP_Prep` and `CRMDYNConstructShootingMethodParamSet`)
  now populate `DynamicsContext` directly.
- AD contexts now read `dynamics` for damping and inertia, so the AD parameter
  vector is consistent with the simulation state.

## Verification
- Run `pytest -q` for the Python suite.
- Run `ctest` (after building) to execute `TestDynamicsContext`.
