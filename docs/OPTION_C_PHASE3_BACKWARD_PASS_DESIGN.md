# Option C Phase 3: Backward Pass Design

**Date:** 2025-12-28
**Goal:** Implement backward pass (gradient computation) for Option C PyTorch extension

---

## Design Overview

### Approach: Call Option A's Implicit Linearization

**Strategy:** Leverage Option A's existing `linearize_full_seed_action_from_seed_implicit()` function to compute gradients.

**Why this approach:**
1. ✅ Option A already implements implicit differentiation with AD
2. ✅ Validated and working (used by iLQR controller)
3. ✅ Minimal code duplication
4. ✅ Faster implementation (1-2 hours vs 8+ hours for full C++ reimplementation)

---

## Mathematical Background

### Forward Pass
```
y = f(u, x_seed)  where:
  u = currents
  x_seed = [v, w, p, R, xf, mL, nL]  (seed state)
  y = [tip_position(3), tip_velocity(3)]  (output)
```

### Backward Pass (Chain Rule)
```
Given: grad_y (∂Loss/∂y from upstream)
Compute:
  grad_u = (∂Loss/∂u) = (∂Loss/∂y) * (∂y/∂u) = grad_y^T * B
  grad_x_seed = (∂Loss/∂x_seed) = (∂Loss/∂y) * (∂y/∂x_seed) = grad_y^T * A
```

Where Option A provides:
- **A = ∂y/∂x_seed** (6 × seed_dim) - state Jacobian
- **B = ∂y/∂u** (6 × 3) - control Jacobian

---

## Implementation Plan

### Step 1: Update C++ Backend (`dynamics_backward` in `dynamics_op.cpp`)

**Current (stub):**
```cpp
std::tuple<...> dynamics_backward(...) {
    // Returns zeros
    return std::make_tuple(grad_currents_zero, grad_insertion_zero, ...);
}
```

**New implementation:**
```cpp
std::tuple<...> dynamics_backward(
    torch::Tensor grad_output,  // (B, 6) - incoming gradients
    torch::Tensor currents,
    torch::Tensor insertion_length,
    torch::Tensor seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL,
    const std::string& param_file,
    const std::string& config_file,
    double eps_seed
) {
    int64_t batch_size = get_batch_size(currents);

    // Allocate gradient tensors
    auto grad_currents = torch::zeros_like(currents);
    auto grad_seed_v = torch::zeros_like(seed_v);
    // ... (allocate all seed gradients)

    // Get raw pointers
    double* grad_curr_ptr = grad_currents.data_ptr<double>();
    double* grad_v_ptr = grad_seed_v.data_ptr<double>();
    // ...

    // Acquire GIL for Python calls
    py::gil_scoped_acquire acquire;
    py::module_ crm_python = py::module_::import("crm_ml_rl.wrappers.crm_python");
    py::object CRMDynamics = crm_python.attr("CRMDynamics");
    py::object dyn = CRMDynamics();
    dyn.attr("load_parameters")(param_file, config_file);

    // For each batch element:
    for (int64_t i = 0; i < batch_size; ++i) {
        // Extract batch element i
        py::array_t<double> curr_i = extract_sample(currents, i);
        py::array_t<double> v_i = extract_sample(seed_v, i);
        // ... (extract all inputs for sample i)

        // Call Option A implicit linearization
        py::dict result = dyn.attr("linearize_full_seed_action_from_seed_implicit")(
            curr_i, insertion_i,
            v_i, w_i, p_i, R_i, xf_i, mL_i, nL_i,
            eps_seed,  // eps_residual_x
            eps_seed,  // eps_residual_theta
            eps_seed,  // eps_g_x
            eps_seed,  // eps_g_theta
            false      // return_debug
        ).cast<py::dict>();

        // Extract Jacobians
        py::array_t<double> A_np = result["A"].cast<py::array_t<double>>();  // (6, seed_dim)
        py::array_t<double> B_np = result["B"].cast<py::array_t<double>>();  // (6, 3)

        // Get grad_output for this sample: grad_y = (6,)
        double grad_y[6];
        for (int j = 0; j < 6; ++j) {
            grad_y[j] = grad_output[i][j].item<double>();
        }

        // Compute grad_currents = B^T @ grad_y (vector-Jacobian product)
        // B is (6, 3), grad_y is (6,) -> result is (3,)
        auto B_buf = B_np.unchecked<2>();
        for (int j = 0; j < 3; ++j) {  // For each current dimension
            double grad_u_j = 0.0;
            for (int k = 0; k < 6; ++k) {  // For each output dimension
                grad_u_j += B_buf(k, j) * grad_y[k];  // B[k,j] * grad_y[k]
            }
            grad_curr_ptr[i * 3 + j] = grad_u_j;
        }

        // Compute grad_seed = A^T @ grad_y
        // A is (6, seed_dim), grad_y is (6,) -> result is (seed_dim,)
        auto A_buf = A_np.unchecked<2>();
        // Unpack seed_dim = 3*num_sets (v) + 3*num_sets (w) + 3*num_sets (p) + 9*num_sets (R) + 15 (xf) + 3*num_sets (mL) + 3*num_sets (nL)
        // For simplicity, extract each component based on A column index

        // grad_v: columns 0:3*num_sets
        // grad_w: columns 3*num_sets:6*num_sets
        // ... (map A columns to seed components)

        // This is tedious but straightforward indexing
    }

    // Return gradients (insertion_length not differentiable -> None)
    return std::make_tuple(
        grad_currents, torch::Tensor(),  // grad_insertion = None
        grad_seed_v, grad_seed_w, grad_seed_p, grad_seed_R,
        grad_seed_xf, grad_seed_mL, grad_seed_nL
    );
}
```

### Step 2: Update Python Autograd (`CRMDynamicsStep.backward()`)

**Current:**
```python
def backward(ctx, grad_output):
    grads = _crm_torch_ext.dynamics_backward(...)
    return grads + (None, None, None)
```

**Updated:**
```python
def backward(ctx, grad_output):
    """
    Backward pass using implicit differentiation via Option A.

    Returns gradients for:
        currents, insertion_length, seed_v, seed_w, seed_p, seed_R,
        seed_xf, seed_mL, seed_nL, param_file, config_file, eps_seed
    """
    if not _extension_available:
        raise RuntimeError("C++ extension not available")

    # Retrieve saved tensors
    saved = ctx.saved_tensors
    currents, insertion_length = saved[0], saved[1]
    seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL = saved[2:]

    # Call C++ backward function
    grads = _crm_torch_ext.dynamics_backward(
        grad_output.detach().cpu().double().contiguous(),
        currents, insertion_length,
        seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL,
        ctx.param_file, ctx.config_file, ctx.eps_seed
    )

    # grads is a tuple: (grad_currents, grad_insertion, grad_v, grad_w, grad_p, grad_R, grad_xf, grad_mL, grad_nL)
    # Return gradients for all forward() inputs (+ None for non-tensor args)
    # Order must match forward() signature
    return grads + (None, None, None)  # None for param_file, config_file, eps_seed
```

**This is already correct!** Just need to implement the C++ part.

---

## Seed State Dimension Mapping

From Option A, the seed state components map to A Jacobian columns as:

```python
seed_dim = 3*num_sets + 3*num_sets + 3*num_sets + 9*num_sets + 15 + 3*num_sets + 3*num_sets
         = 21*num_sets + 15

For num_sets=1: seed_dim = 36

Column indices in A:
  v:  [0:3]      (3 elements)
  w:  [3:6]      (3 elements)
  p:  [6:9]      (3 elements)
  R:  [9:18]     (9 elements)
  xf: [18:33]    (15 elements)
  mL: [33:36]    (3 elements)
  nL: [36:39]    (3 elements)

Total: 39 columns
```

**Wait, discrepancy!** Let me check Option A's actual implementation...

Actually, Option A may not include mL/nL in the gradient since they're typically BVP solver outputs, not inputs. Need to verify.

For Phase 3 MVP, we can:
1. **Compute gradients for currents only** (most important for control)
2. **Set seed gradients to zero** initially
3. **Validate current gradients work**
4. **Then add seed gradients** if needed

---

## Simplified MVP Implementation

### Phase 3A: Currents Gradients Only

**Rationale:**
- Control applications primarily need ∂Loss/∂currents
- Seed state is often not optimized (comes from trajectory rollout)
- Faster to implement and validate

**Implementation:**
```cpp
std::tuple<...> dynamics_backward(...) {
    // Allocate gradients
    auto grad_currents = torch::zeros_like(currents);
    auto grad_seed_v = torch::zeros_like(seed_v);  // Will be zero
    // ... (all seed grads = zero)

    // For each batch element:
    for (int64_t i = 0; i < batch_size; ++i) {
        // Call linearize_full_seed_action_from_seed_implicit
        py::dict result = dyn.attr("linearize_full_seed_action_from_seed_implicit")(...);

        // Extract B Jacobian (∂y/∂currents)
        py::array_t<double> B_np = result["B"].cast<py::array_t<double>>();  // (6, 3)

        // Compute grad_currents = B^T @ grad_output
        // ... (as shown above)
    }

    return std::make_tuple(grad_currents, grad_insertion_zero, grad_v_zero, ...);
}
```

### Phase 3B: Full Seed Gradients (Optional)

**If needed later:**
- Extract A Jacobian
- Map A columns to seed components
- Compute grad_seed = A^T @ grad_output

---

## Testing Strategy

### Test 1: Gradient Check (Finite Differences)

Compare autograd gradients to finite differences:

```python
def test_gradient_currents():
    # Forward pass
    y = CRMDynamicsStep.apply(currents, ...)
    loss = y.sum()

    # Autograd gradient
    grad_auto = torch.autograd.grad(loss, currents)[0]

    # Finite difference gradient
    eps = 1e-4
    grad_fd = torch.zeros_like(currents)
    for i in range(currents.numel()):
        currents_plus = currents.clone()
        currents_plus.view(-1)[i] += eps
        y_plus = CRMDynamicsStep.apply(currents_plus, ...)

        currents_minus = currents.clone()
        currents_minus.view(-1)[i] -= eps
        y_minus = CRMDynamicsStep.apply(currents_minus, ...)

        grad_fd.view(-1)[i] = (y_plus.sum() - y_minus.sum()) / (2 * eps)

    # Compare
    diff = (grad_auto - grad_fd).abs()
    assert diff.max() < 1e-2, "Gradient check failed"
```

### Test 2: PyTorch Autograd Integration

```python
def test_torch_autograd():
    currents = torch.randn(1, 3, requires_grad=True, dtype=torch.float64)
    # ... (setup seed state)

    output = CRMDynamicsStep.apply(currents, insertion, seed_v, ...)
    loss = output.pow(2).sum()

    # Backward should work without error
    loss.backward()

    assert currents.grad is not None
    assert not torch.isnan(currents.grad).any()
    assert not torch.isinf(currents.grad).any()
```

### Test 3: Performance Benchmark

Compare to Option A:

```python
def benchmark():
    # Option A (implicit linearization directly)
    t0 = time.time()
    for _ in range(100):
        result = dyn.linearize_full_seed_action_from_seed_implicit(...)
        A, B = result['A'], result['B']
    t_option_a = time.time() - t0

    # Option C (via PyTorch extension)
    t0 = time.time()
    for _ in range(100):
        output = CRMDynamicsStep.apply(...)
        loss = output.sum()
        loss.backward()
    t_option_c = time.time() - t0

    print(f"Option A: {t_option_a:.2f}s")
    print(f"Option C: {t_option_c:.2f}s")
    print(f"Overhead: {(t_option_c / t_option_a - 1) * 100:.1f}%")
```

---

## Expected Performance

**Phase 2A (forward only):**
- Expected: 2-5x speedup vs pure Python (marginal, due to GIL)
- Actual: Need to measure

**Phase 3 (forward + backward):**
- Expected: Similar to Option A (calls same linearization function)
- Overhead: ~10-20% from tensor conversions and batch loop

**If performance insufficient:**
- Phase 2B: Implement native C++ forward/backward (no GIL, parallelizable)
- Expected: 10x+ speedup

---

## Risk Assessment

**Low Risk:**
- ✅ Reusing validated Option A implementation
- ✅ Clear mathematical foundation (implicit differentiation)
- ✅ Incremental testing (currents first, then seeds)

**Medium Risk:**
- ⚠️ Tensor indexing errors (batch dimension, seed components)
- ⚠️ Performance may not be significantly better than Option A
- **Mitigation:** Careful testing, performance measurement

**High Risk:**
- ❌ None identified

---

## Implementation Timeline

**Phase 3A: Currents Gradients (MVP)**
- Task 3: Implement `dynamics_backward()` in C++ (1 hour)
- Task 4: Update Python wrapper (already done) (0 hours)
- Task 5: Create gradient validation test (30 min)
- Task 6: Run validation (30 min)
- Task 7: Test PyTorch autograd (30 min)
- Task 8: Measure performance (30 min)
- **Total: 3 hours**

**Phase 3B: Seed Gradients (Optional)**
- Add A Jacobian extraction and mapping (1 hour)
- Validate seed gradients (1 hour)
- **Total: 2 hours**

---

## Success Criteria

**Phase 3A (MVP):**
1. ✅ `dynamics_backward()` computes current gradients without errors
2. ✅ Gradient check passes (FD error < 1%)
3. ✅ PyTorch autograd integration works
4. ✅ No NaN/Inf in gradients
5. ✅ Performance measured vs Option A

**Phase 3B (Full):**
6. ✅ Seed gradients computed and validated
7. ✅ End-to-end gradient flow through full seed state

---

**Design Complete - Ready for Implementation**

**Next Step:** Implement Task 3 (C++ backward pass)
