# Task 4.9: Generalize Output Vector for Multi-Actuator

## Current Status

**File:** `src/CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp`

**Current Implementation (lines 1105-1114):**
```cpp
// Build output: [u_tip, v_coil]
Eigen::Matrix<Scalar, 6, 1> y;
y(0) = u_tau(0);  // Tip curvature (3D)
y(1) = u_tau(1);
y(2) = u_tau(2);
y(3) = vw_out(0);  // Actuator 0 linear velocity (3D)
y(4) = vw_out(1);
y(5) = vw_out(2);
return y;
```

- **Output dimension:** Fixed 6D (tip curvature + first actuator velocity)
- **Actuators handled:** Only actuator 0 (lines 1069-1103)
- **Return type:** `Eigen::Matrix<Scalar, 6, 1>` (fixed size)

## Problem

The function `eval_output_AD`:
1. Only processes actuator 0 (line 1069: `const auto& act0 = Params.dynamics.actuators[0]`)
2. Returns fixed 6D output regardless of `NUM_ACT_SET`
3. For systems with multiple actuators, we'd want output for all actuators

## Required Changes for Full Multi-Actuator Support

### 1. Dynamic Output Size

Change return type to support variable dimensions:
```cpp
template <typename Scalar>
inline Eigen::Matrix<Scalar, Eigen::Dynamic, 1> eval_output_AD(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x_scaled,
    const DynamicsContextAD<Scalar>& ctx)
{
    const int num_sets = Params.dynamics.size();
    const int output_dim = 3 + 3 * num_sets;  // tip curvature + velocity per actuator
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> y(output_dim);

    // ... computation ...

    return y;
}
```

### 2. Loop Through All Actuators

Replace single actuator logic with loop:
```cpp
// Process all actuators
for (int act_idx = 0; act_idx < num_sets && act_idx < NUM_ACT_SET; ++act_idx) {
    const auto& act = Params.dynamics.actuators[act_idx];

    // Extract actuator state
    Vec6<Scalar> vw_coil;
    Vec3<Scalar> p_coil;
    Mat3<Scalar> R_coil;
    // ... (extract from act)

    // Compute net forces
    Vec3<Scalar> net_nL, net_mL;
    if (act_idx < num_sets - 1) {
        net_nL = n_L_all[act_idx] - n_L_all[act_idx + 1];
    } else {
        net_nL = n_L_all[act_idx] - n_0;  // Last actuator uses tip force
    }
    // ... (compute net_mL based on actuator chain)

    // Integrate dynamics
    Vec6<Scalar> vw_out;
    Vec3<Scalar> p_out;
    Mat3<Scalar> R_out;
    CoilDynamicsDispatch(/* ... */);

    // Store in output vector
    y(3 + act_idx*3 + 0) = vw_out(0);
    y(3 + act_idx*3 + 1) = vw_out(1);
    y(3 + act_idx*3 + 2) = vw_out(2);
}
```

### 3. Update Output Jacobian Function

Update `DYNNLEquationOutputJacobianEigenAD` to handle dynamic output size:
```cpp
inline void DYNNLEquationOutputJacobianEigenAD(
    const Eigen::VectorXd& x_scaled,
    const Eigen::Vector3d& currents,
    const Eigen::VectorXd& seed_flat,
    DYNNLEqnParams& Params,
    Eigen::MatrixXd& out_gx,    // (3 + 3*num_sets) × x_dim
    Eigen::MatrixXd& out_gth)   // (3 + 3*num_sets) × theta_dim
{
    const int num_sets = Params.dynamics.size();
    const int output_dim = 3 + 3 * num_sets;

    out_gx.resize(output_dim, x_dim);
    out_gth.resize(output_dim, theta_dim);

    // ... rest of computation ...
}
```

### 4. Challenges

1. **Breaking Change:** All callers expecting 6D output need updating
2. **Actuator Chaining:** Need to properly compute forces/moments through actuator chain
3. **Testing:** Would require recompiling with `NUM_ACT_SET > 1` and extensive testing
4. **Performance:** More actuators = larger Jacobians = slower computation

## Current System Constraints

- **Compile-time setting:** `NUM_ACT_SET = 1` (src/CRM.hpp:14)
- **All tests:** Use single actuator configuration
- **iLQR demo:** Designed for single actuator

## Recommendation

**Defer this task** because:
1. Current system compiled with `NUM_ACT_SET=1`
2. No immediate use case for multi-actuator output
3. Significant refactoring required for a feature not currently needed
4. Stabilization goals (Phase 4-6) focus on single-actuator iLQR

**When to implement:**
- When hardware with multiple actuators becomes available
- When control algorithms need per-actuator feedback
- After recompiling with `NUM_ACT_SET > 1`

## Alternative: Minimal Change for Documentation

For now, add comments documenting the limitation:
```cpp
// Build output: [u_tip, v_coil_0]
// NOTE: Currently only returns output for first actuator (actuator 0).
// For multi-actuator support (NUM_ACT_SET > 1), this function would need to:
//   1. Change return type to Eigen::Matrix<Scalar, Eigen::Dynamic, 1>
//   2. Loop through all actuators computing vw_out for each
//   3. Return vector of size (3 + 3*num_sets)
// See docs/TASK_4_9_MULTI_ACTUATOR_OUTPUT_PLAN.md for details.
Eigen::Matrix<Scalar, 6, 1> y;
y(0) = u_tau(0);  // Tip curvature
y(1) = u_tau(1);
y(2) = u_tau(2);
y(3) = vw_out(0);  // Actuator 0 linear velocity
y(4) = vw_out(1);
y(5) = vw_out(2);
```

## Testing Plan (Future)

Once implemented:
1. Recompile with `NUM_ACT_SET=2`
2. Create test with 2 actuators
3. Verify output dimension is 9 (3 tip + 3*2 actuators)
4. Verify Jacobian dimensions are correct
5. Test gradient flow through all actuators

## Estimated Effort

- Full implementation: 3-4 hours
- Testing with multi-actuator: 2-3 hours
- Documentation only: 15 minutes (recommended for current scope)
