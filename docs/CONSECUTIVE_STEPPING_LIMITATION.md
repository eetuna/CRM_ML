# Consecutive Stepping Limitation in `step_from_seed`

## Executive Summary

**The `step_from_seed` API is NOT reliable for consecutive stepping** (using step N output as step N+1 input). This is a fundamental limitation of the underlying BVP trust region solver, not a bug that can be easily fixed.

## The Problem

When attempting to perform consecutive forward integration using `step_from_seed`:

```python
# Step 1: Works fine
result1 = dyn.step_from_seed(currents, insertion_length, seed1['v'], seed1['w'], ...)

# Step 2: Often FAILS with divergence
seed2 = {'v': result1['next_v'], 'w': result1['next_w'], ...}  # Use output from step 1
result2 = dyn.step_from_seed(currents, insertion_length, seed2['v'], seed2['w'], ...)
# Result: converged=False, diverged=True, localmin=3
```

**Symptoms:**
- Step 1 succeeds
- Step 2+ fail with `diverged=True` and `localmin=3` (trust region iteration limit)
- The BVP solver cannot converge when using arbitrary seed states from previous steps

## Root Cause

The boundary value problem (BVP) solver (`DynamicsBVP` in `src/CoilDynamics_Defs.cpp`) uses a MINPACK trust region dogleg algorithm that:

1. **Has hidden static/global state** that gets initialized on first call
2. **Is sensitive to initial guesses** in ways that make consecutive arbitrary-seed stepping fail
3. **Was designed for single-shot evaluations**, not sequential stepping

### Investigation Results

Extensive investigation (see `STABILIZATION_DEBUG_PLAN.md` Phase 1-3) revealed:

- ✅ State variables (v, w, mL, nL) are all within normal ranges
- ✅ Orthonormality of rotation matrices is perfect (error ~10⁻¹⁵)
- ✅ Issue is NOT integrator-specific (both ABM4 and RK4 fail identically)
- ✅ Issue is NOT numerical blow-up or NaN propagation
- ❌ Issue IS in the BVP solver's trust region algorithm
- ❌ Even "warm-up" calls don't fix consecutive stepping

**Debug output shows:**
```
DynamicsBVP call #1 (Step 1):  localmin=0, diverged=0  ✅ SUCCESS
DynamicsBVP call #2 (Step 2):  localmin=3, diverged=1  ❌ FAILURE
```

The solver hits its iteration limit (`localmin=3`) and cannot find a solution when starting from the seed produced by Step 1.

## Workarounds & Recommended Usage

### ✅ DO: Use `step()` for Sequential Integration

```python
dyn.initialize_from_kinematics(currents, insertion_length)

# Step forward sequentially - this maintains internal state and works reliably
for i in range(num_steps):
    result = dyn.step(currents, insertion_length)
    # Internal state automatically updated
```

### ✅ DO: Use `step_from_seed` for Single-Shot Evaluations

```python
# Linearization around a specific state - this works fine
jacobians = dyn.linearize_full_seed_action_from_seed_implicit(
    currents, insertion_length,
    seed['v'], seed['w'], seed['p'], seed['R'], seed['xf'],
    seed['mL'], seed['nL']
)
```

### ❌ DON'T: Use `step_from_seed` for Consecutive Steps

```python
# This pattern DOES NOT WORK reliably
seed = initial_seed
for i in range(num_steps):
    result = dyn.step_from_seed(currents, insertion_length, seed['v'], ...)
    seed = result  # ❌ Step 2+ will likely fail!
```

## iLQR and Trajectory Optimization

The iLQR demo (`examples/ilqr_catheter_demo.py`) uses a different pattern that DOES work:

1. It uses `step()` for forward rollouts (not `step_from_seed`)
2. It uses `linearize_*_from_seed` only for computing Jacobians around trajectory points
3. The linearization is single-shot evaluation at each point, not consecutive stepping

## Technical Details

### BVP Solver Architecture

The dynamics stepping involves:
1. **Boundary Value Problem (BVP)**: Solve for internal forces/moments (mL, nL) that satisfy equilibrium
2. **Initial Value Problem (IVP)**: Forward integrate with solved mL/nL

The BVP step uses MINPACK's trust region dogleg algorithm which:
- Maintains static state across calls
- Is highly sensitive to initial guess quality
- Converges poorly when initial guess is "far" from solution

### Why Step 1 → Step 2 Fails

When stepping from a kinematic initialization:
- **Step 1**: Starts from zero velocity, static equilibrium → solver converges
- **Step 2**: Starts from moving state with non-zero velocities → solver sees this as "far" from equilibrium and fails

The mL/nL values from Step 1 don't provide a good enough initial guess for the trust region algorithm at Step 2's state.

## Future Work

To fully fix this would require:

1. **Deep solver refactoring**: Replace or modify the trust region algorithm to handle sequential states
2. **Better initial guess generation**: Develop heuristics for mL/nL initialization based on state
3. **Alternative BVP formulation**: Redesign to avoid trust region sensitivity

**Estimated effort**: Several weeks to months, high risk of breaking existing functionality.

**Recommendation**: Accept the limitation and use `step()` for sequential integration.

## See Also

- `examples/debug_consecutive_stepping.py` - Demonstrates the failure
- `STABILIZATION_DEBUG_PLAN.md` - Detailed investigation plan and results
- `examples/ilqr_catheter_demo.py` - Correct usage pattern for trajectory optimization
