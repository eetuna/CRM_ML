# Gradient Bug Analysis - gx Computation

## Problem Statement
The AD gradient has **194-401% error** compared to finite differences.

Debug output shows:
- `gx norm: 4.50346e+07` (AD)
- `gx_fd norm: 5.26e+04` (FD estimate, may be inaccurate)
- **~857x difference**

The component `gx * dx/dθ` dominates the final gradient, causing the huge error.

## What is gx?

`gx = ∂y/∂x` where:
- `y = [tip_pos, tip_vel]` (6D output)
- `x = [mL, nL]` (6D BVP solution - internal forces/moments)

This represents: **how does the output change when we perturb the BVP solution?**

## The Forward Pass Architecture

### Actual Forward Pass (from crm_bindings.cpp)

```cpp
// Step 1: Solve BVP for internal forces
DynamicsBVP(params, xf, mL_guess, nL_guess, ftip_guess,
            out_u0, out_mL, out_nL, out_tau, ftip_calc, localmin);
// Returns: mL*, nL*, tau*, u0*

// Step 2: Forward integrate with solved values
DYNSolverIVP(params, out_u0, out_mL, out_nL, out_tau, ftip_calc,
             true, xf_new, x_coil, ReportedMarkerPos);
// Returns: tip_position, tip_velocity
```

**Key insight**: The forward pass takes `(mL, nL, tau)` as inputs to `DYNSolverIVP`.

### How tau is computed

From `DYNNLEquation` (BVP residual function), lines 888-892:

```cpp
// Backward integrate through flex segment
CRMFlexible_IVP_Back(segi, p_t, R_t, Params, u_t, n_0, u_tau, p_, R_);

// Compute tau from backward integration result
tau = K * (u_tau - ustar)
```

So: **`tau = tau(geometry, n_0, tip_state)` - it does NOT directly depend on mL!**

Then line 896:
```cpp
net_mL = m_L - tau
```

This `net_mL` is used in the coil dynamics.

## The AD Implementation (eval_output_AD)

Location: `src/CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp:1157-1310`

What it does:
1. Unpack `x = [mL, nL]` (lines 1172-1177)
2. Backward integrate to compute `tau` (lines 1213-1224)
3. Compute `net_mL = m_L - tau` (line 1250)
4. Integrate actuator dynamics (lines 1257-1261)
5. Forward integrate through flex segment (lines 1263-1271)
6. Return tip position and velocity

## The Discrepancy

### What eval_output_AD computes:

Line 1265:
```cpp
Vec3<Scalar> u_boundary = ustar + Kinv * tau;
```

Where `tau` was computed from backward integration (line 1224):
```cpp
tau = K * (u_tau - ustar)
```

**Problem**: When autodiff differentiates through this, it computes:
```
∂(u_boundary)/∂(mL) = ∂(ustar + Kinv * tau)/∂(mL)
                     = Kinv * ∂tau/∂(mL)
                     = Kinv * K * ∂(u_tau)/∂(mL)
```

But `u_tau` comes from backward integration which DOESN'T directly depend on `mL`!

### What the actual forward pass does:

Looking at `DYNSolverIVP` (line 1683):
```cpp
CRMIVP_DYN(CoreParams, u_0, p0, R0, m_L, n_L, tau, ftip,
           out_coil_state, u_new, p_new, R_new, ...);
```

It uses `tau` that was computed separately and passed in.

## Hypothesis: The Chain is Broken

The issue is that `eval_output_AD` is trying to compute `∂y/∂x` by differentiating through a **simulation** of what happens given `x = [mL, nL]`.

But this simulation includes:
1. Backward integration (which computes tau from geometry/boundary conditions)
2. Forward integration (which uses mL, nL, and tau)

When we perturb `mL` in the AD, it affects:
- `net_mL = mL - tau` ✓ Correct
- But `tau` also gets spurious derivatives through the backward integration ✗ Wrong!

## What Should Happen

To properly compute `∂y/∂x`, we need to know: **if we change mL and nL, holding everything else fixed, how does y change?**

But "everything else" includes `tau`, which in the actual forward pass is computed INDEPENDENTLY from the BVP solve.

The correct dependency graph:
```
BVP Solve:
  (currents, geometry, boundary conditions) → (mL*, nL*, tau*, u0*)

Forward Pass:
  (mL*, nL*, tau*, u0*, initial_state) → (tip_pos, tip_vel)
```

So `∂y/∂(mL, nL)` should be computed **holding tau fixed** at the value it had from the BVP solve.

But `eval_output_AD` RE-COMPUTES tau from backward integration, creating spurious derivatives.

## The Fix

Option 1: **Pass tau as an input to eval_output_AD**

```cpp
inline Eigen::Matrix<Scalar, Eigen::Dynamic, 1> eval_output_AD(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x_scaled,
    const Vec3<Scalar>& tau_fixed,  // NEW: tau from BVP solve
    const DynamicsContextAD<Scalar>& ctx)
```

Then use `tau_fixed` instead of recomputing it from backward integration.

Option 2: **Use the actual forward pass implementation**

Instead of re-implementing the physics in `eval_output_AD`, call the actual `DYNSolverIVP` equivalent.

## ROOT CAUSE CONFIRMED

**Location**: `src/CoilDynamics_Defs.cpp:1343, 1392, 1419-1420`

The actual forward pass `CRMIVP_DYN`:
1. **Copies tau from input** (line 1343): `mCopy_ABm<NUM_ACT_SET, 3>(in_tau, tau);`
2. **Uses it directly** (line 1392): `net_mL = m_L - tau`
3. **Uses it for curvature** (lines 1419-1420): `u_0 = ustar + Kinv * tau`

**tau is a FIXED parameter**, not recomputed!

But `eval_output_AD` RECOMPUTES tau from backward integration (lines 1213-1224 in autodiff header), creating spurious derivatives ∂tau/∂(mL,nL) that shouldn't exist.

## The Fix

**Modify `eval_output_AD` to accept tau as a fixed input parameter**:

1. Add `tau_fixed` parameter to `eval_output_AD`
2. Remove backward integration that computes tau
3. Use `tau_fixed` directly (like the actual forward pass does)
4. Update `DYNNLEquationOutputJacobianEigenAD` to compute and pass tau

This will make gx correctly represent ∂y/∂(mL,nL) **with tau held constant**.
