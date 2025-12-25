# Task 4.7: Implement Direct Output Jacobian (g_θ) AD Logic

## Current Status

**File:** `src/CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp:1119-1167`

**Current Implementation:**
- `DYNNLEquationOutputJacobianEigenAD` computes `out_gx` (Jacobian w.r.t. x) correctly using AD
- `out_gth` (Jacobian w.r.t. θ = currents + seed) is set to zero (line 1160)
- Function receives `currents` and `seed_flat` parameters but doesn't use them for differentiation

## Problem

The `eval_output_AD` function (lines 1000-1115) currently:
1. Takes `x_scaled` (BVP solution) and `ctx` (context with fixed parameters)
2. Reads currents, seed state, and other parameters from `ctx.geometry` (Params) as **fixed doubles**
3. These fixed values cannot be differentiated w.r.t

To compute ∂y/∂θ, we need currents and seed to be **AD variables** that can be differentiated.

## Required Changes

###1. Create New Overload of `eval_output_AD`

Add a new template function that accepts parameters as AD variables:

```cpp
template <typename Scalar>
inline Eigen::Matrix<Scalar, 6, 1> eval_output_AD_with_params(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x_scaled,
    const Eigen::Matrix<Scalar, 3, 1>& currents_ad,      // NEW: currents as AD
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& seed_flat_ad,  // NEW: seed as AD
    const DynamicsContextAD<Scalar>& ctx)
{
    // Unpack x_scaled to get m_L, n_L (same as before)
    std::array<Vec3<Scalar>, NUM_ACT_SET> m_L_all, n_L_all;
    for (int j = 0; j < num_sets && j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; ++i) {
            m_L_all[j](i) = Scalar(IVALUE_SCALE_M) * x_scaled(i + j * 6);
            n_L_all[j](i) = Scalar(IVALUE_SCALE_N) * x_scaled(i + j * 6 + 3);
        }
    }

    // CHANGE: Extract seed state from seed_flat_ad instead of Params
    // seed_flat layout: [v (num_sets*3), w (num_sets*3), p (num_sets*3),
    //                    R (num_sets*9), xf (15), mL (num_sets*3), nL (num_sets*3)]

    int offset = 0;
    std::array<Vec3<Scalar>, NUM_ACT_SET> v_seed, w_seed, p_seed;
    std::array<Mat3<Scalar>, NUM_ACT_SET> R_seed;
    Vec3<Scalar> xf_seed; // Only need first 3 elements for position
    // ... etc

    // Extract from seed_flat_ad using template indexing

    // CHANGE: Use currents_ad instead of Params.dynamics.currents
    // This affects magnetic field computation in CoilDynamicsDispatch

    // Continue with rest of computation using AD variables
    // ...

    return y;
}
```

### 2. Update `DYNNLEquationOutputJacobianEigenAD`

Replace the TODO section (lines 1157-1167) with:

```cpp
// Compute Jacobian w.r.t. θ (parameters: currents + seed)
VectorXreal theta_ad(theta_dim);
for (int i = 0; i < curr_dim; ++i) theta_ad(i) = currents(i);
for (int i = 0; i < seed_dim; ++i) theta_ad(curr_dim + i) = seed_flat(i);

VectorXreal y_theta;
auto output_fn_theta = [&ctx_base, &x_scaled](const VectorXreal& th_) -> VectorXreal {
    dynnl_ad_eigen::DynamicsContextAD<real> ctx_ad(ctx_base);

    // Split theta into currents and seed
    Eigen::Matrix<real, 3, 1> curr_ad;
    for (int i = 0; i < 3; ++i) curr_ad(i) = th_(i);

    VectorXreal seed_ad(th_.size() - 3);
    for (int i = 3; i < th_.size(); ++i) seed_ad(i - 3) = th_(i);

    // Convert x_scaled to AD type
    VectorXreal x_ad_local(x_scaled.size());
    for (int i = 0; i < x_scaled.size(); ++i) x_ad_local(i) = x_scaled(i);

    return dynnl_ad_eigen::eval_output_AD_with_params(x_ad_local, curr_ad, seed_ad, ctx_ad);
};

jacobian(output_fn_theta, wrt(theta_ad), at(theta_ad), y_theta, out_gth);
```

### 3. Challenges

1. **Seed State Complexity**: The seed_flat vector contains v, w, p, R, xf, mL, nL for all actuators. Need to carefully unpack this in template code.

2. **Currents Usage**: Currents affect magnetic fields in `CoilDynamicsDispatch`. This function needs to accept template Scalar type for currents.

3. **BVP Solution Dependency**: The output y depends on x* (BVP solution), which itself depends on θ. The current approach assumes x* is given and computes ∂y/∂θ holding x* fixed. For full end-to-end differentiation, we'd need to account for ∂x*/∂θ (which is what implicit differentiation gives us via the A, B matrices).

4. **Performance**: Adding this parameter Jacobian will increase computation time. May need to make it optional.

## Simplified Alternative (Recommended for Phase 4)

Instead of full implementation, document that:
1. `out_gx` is correctly computed via AD
2. `out_gth` would require significant refactoring to pass parameters as AD variables
3. For current iLQR use case, the A and B matrices (from implicit differentiation) provide the necessary gradients
4. Parameter Jacobian (g_θ) would be needed for:
   - End-to-end learning of catheter parameters
   - Trajectory optimization w.r.t. seed state
   - These are future features beyond current stabilization scope

## Testing

Once implemented, test with:
```python
result = dyn.linearize_full_seed_action_from_seed_implicit(..., return_debug=True)
gx = result['gx']  # Should exist and be non-zero
gth = result['gth']  # Should exist and be non-zero (after implementation)
```

## Estimated Effort

- Full implementation: 4-6 hours (complex template refactoring)
- Documentation/deferral: 30 minutes (current task scope)

## Recommendation

Given that:
1. The A and B matrices already provide necessary gradients for iLQR
2. This feature is not blocking the stabilization goals (Phase 4-6)
3. The implementation requires significant refactoring

**Recommend:** Document current state and defer full implementation to future work focused on parameter learning/optimization.
