# Task 3.2: Gradient Correctness Status Report

**Date:** 2025-12-26
**Status:** 🚧 IN PROGRESS - Backward infrastructure complete, gradient computation not yet implemented
**Task:** CP-C07 - Verify extension gradients match Python wrapper

---

## Executive Summary

The C++ extension has **complete backward pass infrastructure** but currently returns **zero gradients**. This is a documented limitation, not a bug. To implement actual gradient computation, the implicit linearization code from Python bindings must be ported to C++.

**Current State:**
- ✅ Forward pass: **PERFECT** (0.00e+00 error)
- ✅ Autograd integration: **WORKING** (grad_fn created, backward callable)
- ❌ Gradient computation: **NOT IMPLEMENTED** (returns zeros)

---

## Test Results

### Current Behavior

```python
# Forward pass
output = ext.crm_step(currents, ...)  # ✓ Works perfectly
loss = output.sum()

# Backward pass
loss.backward()  # ✓ Runs without error
print(currents.grad)  # Outputs: [0., 0., 0.]  ❌ Wrong (should be non-zero)
```

### Numerical vs Analytical Gradients

**Test case:** Zero currents, standard insertion

```
Numerical gradient (FD, eps=1e-6):  16.78
Analytical gradient (backward):      0.00  ❌
Relative error:                     100.00%
```

**Expected behavior:** Analytical gradient should match numerical gradient within ~1% error.

---

## Why Gradients Are Zero

### Current Implementation

**File:** `crm_torch_ext/csrc/crm_step_op.cpp:376-421`

```cpp
std::vector<torch::Tensor> crm_step_backward(
    torch::Tensor grad_output,
    torch::Tensor currents,
    // ... saved tensors
) {
    // Extract dimensions
    auto grad_acc = grad_output.accessor<double, 1>();
    auto curr_acc = currents.accessor<double, 1>();

    // Create zero gradients
    torch::Tensor grad_currents = torch::zeros_like(currents);
    torch::Tensor grad_insertion = torch::zeros_like(insertion_length);
    // ... zero gradients for all seed components

    return {grad_currents, grad_insertion, grad_seed_v, ...};  // ❌ All zeros!
}
```

**Problem:** The function creates zero tensors and returns them without computing actual gradients.

### What's Needed: Implicit Linearization

The Python bindings use **implicit differentiation with autodiff** to compute gradients:

**File:** `crm_ml_rl/wrappers/crm_bindings.cpp:1969-2180`

```cpp
py::dict linearize_full_seed_action_from_seed_implicit(...) {
    // 1. Solve forward BVP to get base solution (already done in forward pass)
    // 2. Compute residual Jacobians using autodiff:
    //    - dF/dx (state Jacobian)
    //    - dF/du (control Jacobian)
    // 3. Apply implicit function theorem:
    //    dy/du = -(dF/dx)^-1 @ (dF/du)
    // 4. Return A, B matrices for chain rule:
    //    grad_inputs = grad_outputs @ A  (or B depending on convention)
}
```

**Key components:**
- Uses `CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp` for autodiff Jacobians
- Computes implicit sensitivity via linear solve (not finite differences)
- Returns structured gradients for all inputs (currents, seed state)

---

## Implementation Plan

To implement proper gradients, we need to:

### Step 1: Port Implicit Linearization (~4-6 hours)

**Add to `crm_step_op.cpp`:**

```cpp
#include "CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp"

std::vector<torch::Tensor> crm_step_backward(...) {
    // 1. Reconstruct BVP parameters from saved tensors
    CRMShootingMethodParams BVPParams = ...;

    // 2. Call existing C++ implicit linearization
    Eigen::MatrixXd A, B;  // Jacobian matrices
    computeImplicitLinearization(BVPParams, saved_state, A, B);

    // 3. Chain rule: grad_inputs = grad_outputs @ Jacobians
    torch::Tensor grad_currents = matmul(grad_output, B_currents);
    torch::Tensor grad_seed_v = matmul(grad_output, B_seed_v);
    // ...

    return {grad_currents, grad_insertion, grad_seed_v, ...};
}
```

**Challenges:**
- Need to expose/port `linearize_full_seed_action_from_seed_implicit()`
- Handle Eigen ↔ Torch tensor conversions
- Manage autodiff context and dual number computations
- Scale gradients appropriately

### Step 2: Test Gradient Correctness

1. **Compare to Python wrapper** (Task 3.2 requirement):
   ```python
   # Extension gradients
   ext_loss.backward()
   ext_grad = currents.grad.clone()

   # Python wrapper gradients
   py_loss.backward()
   py_grad = currents.grad.clone()

   assert torch.allclose(ext_grad, py_grad, atol=1e-6)
   ```

2. **Compare to finite differences**:
   ```python
   numerical_grad = compute_fd_gradient(eps=1e-7)
   assert relative_error(analytical_grad, numerical_grad) < 0.01  # <1%
   ```

3. **Test `torch.autograd.gradcheck()`**:
   ```python
   # May not pass due to BVP solver sensitivity
   # But should be close
   gradcheck(ext.crm_step, inputs, eps=1e-6)
   ```

### Step 3: Optimize Performance

- Cache Jacobian factorizations if possible
- Use efficient Eigen operations
- Minimize Torch ↔ Eigen conversions

---

## Acceptance Criteria (CP-C07)

**From implementation plan:**

- [ ] Gradients match Python wrapper within 1e-6
- [ ] Gradients match FD within 1% (at eps=1e-7)
- [ ] `torch.autograd.gradcheck` advisory (may not pass due to FD sensitivity)

**Current status:** 0/3 criteria met (gradients not implemented)

---

## Workarounds for Current Limitations

Until gradients are implemented, users can:

### 1. Use Zero-Order Optimization

```python
from scipy.optimize import differential_evolution, minimize

def objective(currents_flat):
    currents = torch.tensor(currents_flat, dtype=torch.float64)
    output = ext.crm_step(currents, ...)
    return loss(output).item()

# Genetic algorithm (no gradients needed)
result = differential_evolution(objective, bounds=[(-1, 1)]*3)

# Nelder-Mead (no gradients needed)
result = minimize(objective, x0, method='Nelder-Mead')
```

### 2. Use Python Wrapper for Gradient-Based Methods

```python
# Use extension for forward-only tasks
output = ext.crm_step(...)  # Fast C++ forward pass

# Use Python wrapper when gradients needed
from crm_ml_rl.wrappers.torch_physics import TorchCRMPhysics
physics = TorchCRMPhysics()
output = physics.step(...)  # Slower but has gradients
loss.backward()  # ✓ Works
```

### 3. Finite-Difference Approximation

```python
# Manual gradient computation
def compute_gradient(f, x, eps=1e-6):
    grad = torch.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.clone()
        x_plus[i] += eps
        grad[i] = (f(x_plus) - f(x)) / eps
    return grad
```

---

## Recommendations

### For Immediate Use

**The C++ extension is READY for:**
- ✅ Forward simulation
- ✅ Trajectory generation
- ✅ Zero-order optimization (CMA-ES, genetic algorithms, Nelder-Mead)
- ✅ Performance-critical forward passes

**NOT ready for:**
- ❌ Gradient-based optimization (Adam, SGD, L-BFGS)
- ❌ Backpropagation through dynamics
- ❌ iLQR with analytical gradients

### For Future Development

**Priority:** Medium-High (enables gradient-based optimization)

**Estimated effort:** 4-6 hours for experienced developer familiar with:
- C++ autodiff (autodiff library)
- Eigen linear algebra
- PyTorch C++ API
- Implicit function theorem

**Alternative:** Use Python wrapper for gradient-based tasks (functional workaround)

---

## Files Referenced

- **Extension:** `crm_torch_ext/csrc/crm_step_op.cpp`
- **Python wrapper:** `crm_ml_rl/wrappers/crm_bindings.cpp`
- **Autodiff header:** `src/CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp`
- **Test:** `crm_torch_ext/test/test_gradients.py`

---

## Conclusion

**Forward pass:** ✅ **COMPLETE AND PERFECT** (0.00e+00 error)

**Backward pass:** 🚧 **INFRASTRUCTURE COMPLETE, COMPUTATION PENDING**

The extension provides a solid foundation with perfect forward accuracy. Gradient computation is the next enhancement to unlock gradient-based optimization workflows. Until then, zero-order methods or the Python wrapper provide functional alternatives.

---

**Author:** Claude Sonnet 4.5
**Date:** 2025-12-26
**Status:** Forward complete, gradients deferred for future work
