# Option C Implementation Plan: Torch C++ Extension Custom Op

**Status:** AWAITING APPROVAL
**Author:** Claude Code
**Date:** 2025-12-26
**Branch:** `claude/option-c-plan`
**Prerequisites:** Option A complete (verified)

---

## Executive Summary

Option C packages the existing Option A infrastructure (C++ dynamics + AD Jacobians) into a native PyTorch C++ extension operator. This eliminates Python-level overhead and provides a standard distribution mechanism.

### Scope
- **Forward**: Call existing C++ `step_from_seed()`
- **Backward**: Use existing `linearize_full_seed_action_from_seed_implicit()` for gradients
- **Package**: Installable via `pip` with standard Torch extension workflow
- **Batch**: Support for batched tensor inputs where possible

### Non-Scope (Explicit Exclusions)
- CUDA kernels (future work)
- New physics implementations
- Changes to existing AD math

---

## Phase 1: Extension Package Structure

### Task 1.1: Create Extension Directory Structure

**Objective:** Set up the Torch C++ extension package skeleton.

**Actions:**
1. Create `crm_torch_ext/` directory at repo root
2. Create standard extension layout:
   ```
   crm_torch_ext/
   ├── __init__.py          # Python entry point
   ├── setup.py             # torch.utils.cpp_extension build
   ├── CMakeLists.txt       # Optional CMake integration
   ├── csrc/
   │   ├── crm_step_op.cpp  # Custom autograd Function in C++
   │   ├── crm_step_op.h    # Header declarations
   │   └── bindings.cpp     # pybind11 module registration
   └── test/
       └── test_extension.py
   ```

**Acceptance Criteria:**
- [ ] Directory structure exists
- [ ] Empty skeleton files created
- [ ] `.gitignore` updated to exclude build artifacts

**Checkpoint:** CP-C01 - Package skeleton created

---

### Task 1.2: Configure Build System

**Objective:** Set up `setup.py` with `torch.utils.cpp_extension.CppExtension`.

**Actions:**
1. Create `setup.py` with:
   - `CppExtension` for CPU build
   - Include paths for existing CRM headers
   - Link against existing CRM library
   - Compiler flags matching existing build
2. Create `pyproject.toml` for modern pip compatibility
3. Test that `pip install -e crm_torch_ext/` runs (even if compile fails initially)

**Dependencies:**
- Must link against: `crm_dynamics` (existing library)
- Must include: `src/`, `extern/autodiff/`

**Acceptance Criteria:**
- [ ] `python crm_torch_ext/setup.py build_ext --inplace` runs
- [ ] Build finds all required headers
- [ ] Link step finds CRM library

**Checkpoint:** CP-C02 - Build system functional

---

## Phase 2: Core Operator Implementation

### Task 2.1: Implement Forward Pass

**Objective:** Create `crm_step_forward()` that wraps existing `step_from_seed`.

**Actions:**
1. In `csrc/crm_step_op.cpp`, implement:
   ```cpp
   std::tuple<torch::Tensor, torch::Tensor> crm_step_forward(
       torch::Tensor currents,        // [3] or [B, 3]
       torch::Tensor insertion_length, // [1] or [B, 1]
       torch::Tensor seed_v,          // [num_sets, 3] or [B, num_sets, 3]
       torch::Tensor seed_w,          // [num_sets, 3] or [B, num_sets, 3]
       torch::Tensor seed_p,          // [num_sets, 3] or [B, num_sets, 3]
       torch::Tensor seed_R,          // [num_sets, 9] or [B, num_sets, 9]
       torch::Tensor seed_xf,         // [15] or [B, 15]
       torch::Tensor seed_mL,         // [num_sets, 3] or [B, num_sets, 3]
       torch::Tensor seed_nL          // [num_sets, 3] or [B, num_sets, 3]
   );
   ```
2. Convert Torch tensors to Eigen types
3. Call existing `CRMDynamics::step_from_seed()`
4. Return: `(next_state, new_seed_tuple)`

**Technical Notes:**
- Start with unbatched (single sample) implementation
- Use `at::Tensor::accessor<>()` for efficient data access
- Handle contiguity requirements

**Acceptance Criteria:**
- [ ] Forward pass compiles
- [ ] Output matches `crm_python.CRMDynamics.step_from_seed()` exactly
- [ ] Test: `torch.allclose(cpp_result, python_result, atol=1e-10)`

**Checkpoint:** CP-C03 - Forward pass matches Python bindings

---

### Task 2.2: Implement Backward Pass

**Objective:** Create `crm_step_backward()` using existing implicit linearization.

**Actions:**
1. Implement backward function:
   ```cpp
   std::tuple<torch::Tensor, ...> crm_step_backward(
       torch::Tensor grad_output,     // gradient w.r.t. next_state
       torch::Tensor saved_currents,  // from forward
       torch::Tensor saved_insertion,
       // ... all saved seed tensors
   );
   ```
2. Call existing `linearize_full_seed_action_from_seed_implicit()`
3. Compute gradients:
   - `grad_currents = A.T @ grad_output` (or B.T depending on convention)
   - `grad_seed_* = ...` via chain rule with Jacobians
4. Return gradient tuple matching forward inputs

**Technical Notes:**
- Must handle the seed dimensionality correctly
- Jacobians A, B are already computed by Option A infrastructure
- Use same scaling/unscaling as existing Python wrapper

**Acceptance Criteria:**
- [ ] Backward pass compiles
- [ ] Gradients match `torch.autograd.Function` wrapper from `torch_physics.py`
- [ ] Test: Compare to existing `CRMDynamicsStepFunction.backward()`

**Checkpoint:** CP-C04 - Backward pass matches Python wrapper

---

### Task 2.3: Register as Custom Autograd Function

**Objective:** Wire forward/backward into Torch's autograd system.

**Actions:**
1. Create C++ autograd Function:
   ```cpp
   class CRMStepFunction : public torch::autograd::Function<CRMStepFunction> {
   public:
       static torch::Tensor forward(
           torch::autograd::AutogradContext *ctx,
           torch::Tensor currents, ...);

       static torch::autograd::tensor_list backward(
           torch::autograd::AutogradContext *ctx,
           torch::autograd::tensor_list grad_outputs);
   };
   ```
2. Save tensors for backward in `ctx->save_for_backward()`
3. Properly handle `requires_grad` flags
4. Register with pybind11 in `bindings.cpp`

**Acceptance Criteria:**
- [ ] `import crm_torch_ext` works
- [ ] `crm_torch_ext.crm_step(...)` is callable
- [ ] Autograd graph builds correctly (`.grad_fn` populated)

**Checkpoint:** CP-C05 - Autograd Function registered

---

## Phase 3: Validation & Testing

### Task 3.1: Forward Correctness Tests

**Objective:** Verify extension forward matches existing Python bindings exactly.

**Actions:**
1. Create `crm_torch_ext/test/test_forward.py`:
   ```python
   def test_forward_matches_python():
       # Compare crm_torch_ext.crm_step() vs crm_python.step_from_seed()
       # At 10+ random stable points
       # Assert allclose with atol=1e-10
   ```
2. Test edge cases:
   - Zero currents
   - Maximum safe currents
   - Various insertion lengths

**Acceptance Criteria:**
- [ ] All forward tests pass
- [ ] Maximum difference < 1e-10

**Checkpoint:** CP-C06 - Forward validation complete

---

### Task 3.2: Gradient Correctness Tests

**Objective:** Verify extension gradients match existing implementation.

**Actions:**
1. Create `crm_torch_ext/test/test_gradients.py`:
   ```python
   def test_gradients_match_python_wrapper():
       # Compare crm_torch_ext gradients vs CRMDynamicsStepFunction
       # At 10+ random stable points

   def test_gradients_vs_finite_diff():
       # Compare to FD with eps=1e-7
       # Tolerance: <1% relative error (matching CP-08)
   ```
2. Test gradient computation for:
   - Currents only
   - Full seed
   - Mixed requires_grad patterns

**Acceptance Criteria:**
- [ ] Gradients match Python wrapper within 1e-6
- [ ] Gradients match FD within 1% (at eps=1e-7)
- [ ] `torch.autograd.gradcheck` advisory (may not pass due to FD sensitivity)

**Checkpoint:** CP-C07 - Gradient validation complete

---

### Task 3.3: Operator Parity Test

**Objective:** End-to-end comparison with Python wrapper.

**Actions:**
1. Create comprehensive parity test:
   ```python
   def test_full_parity():
       # Run 100 steps with both implementations
       # Compare trajectories
       # Compare accumulated gradients
   ```
2. Verify in iLQR context (optional):
   - Run iLQR demo with extension
   - Compare convergence behavior

**Acceptance Criteria:**
- [ ] Trajectory match over 100 steps
- [ ] Gradient accumulation matches

**Checkpoint:** CP-C08 - Full parity validated

---

## Phase 4: Performance Optimization

### Task 4.1: Benchmark vs Python Wrapper

**Objective:** Measure performance improvement from C++ extension.

**Actions:**
1. Create `crm_torch_ext/benchmark/benchmark_overhead.py`:
   ```python
   def benchmark_single_step():
       # Time: crm_torch_ext.crm_step() vs CRMDynamicsStepFunction
       # Report: forward time, backward time, total time

   def benchmark_trajectory():
       # Time: 100-step trajectory with both
   ```
2. Measure:
   - Per-step overhead reduction
   - Memory allocation patterns
   - Total iLQR iteration time

**Targets:**
- Forward: <0.1ms overhead per call (vs ~1ms for Python wrapper)
- Backward: Similar to Python (dominated by linearization compute)

**Acceptance Criteria:**
- [ ] Benchmark script runs
- [ ] Results documented in report
- [ ] Overhead reduction quantified

**Checkpoint:** CP-C09 - Performance benchmarked

---

### Task 4.2: Add Batched Support (Optional)

**Objective:** Support batched inputs for parallel stepping.

**Actions:**
1. Extend forward to handle batch dimension:
   ```cpp
   // If currents.dim() == 2, treat as batch
   if (currents.dim() == 2) {
       // Loop or parallelize over batch
   }
   ```
2. Extend backward similarly
3. Use OpenMP or std::async for parallel execution

**Note:** This is optional and depends on use case requirements.

**Acceptance Criteria:**
- [ ] Batched forward works
- [ ] Batched backward works
- [ ] Speedup measured for batch sizes 1, 4, 16

**Checkpoint:** CP-C10 - Batched support (optional)

---

## Phase 5: Packaging & Documentation

### Task 5.1: Package for pip Install

**Objective:** Make extension installable via `pip install .`

**Actions:**
1. Finalize `setup.py` with:
   - Version number
   - Dependencies (`torch`, `numpy`)
   - Package metadata
2. Create `MANIFEST.in` for source distribution
3. Test: `pip install ./crm_torch_ext && python -c "import crm_torch_ext"`

**Acceptance Criteria:**
- [ ] `pip install .` works in fresh virtualenv
- [ ] `import crm_torch_ext` succeeds
- [ ] Basic smoke test passes

**Checkpoint:** CP-C11 - pip installable

---

### Task 5.2: Documentation & Examples

**Objective:** Document extension usage.

**Actions:**
1. Add docstrings to Python bindings
2. Create `crm_torch_ext/README.md`:
   - Installation instructions
   - API reference
   - Example usage
   - Performance notes
3. Add example script: `examples/torch_extension_demo.py`
4. Update main `README.md` to mention extension

**Acceptance Criteria:**
- [ ] README exists with clear instructions
- [ ] Example script runs successfully
- [ ] Main README updated

**Checkpoint:** CP-C12 - Documentation complete

---

### Task 5.3: Integration with Existing Codebase

**Objective:** Make extension usable as drop-in replacement.

**Actions:**
1. Add conditional import in `torch_physics.py`:
   ```python
   try:
       from crm_torch_ext import crm_step as _crm_step_cpp
       USE_CPP_EXTENSION = True
   except ImportError:
       USE_CPP_EXTENSION = False
   ```
2. Add `use_cpp_extension` flag to `TorchCRMPhysics`
3. Allow runtime switching between Python and C++ backends

**Acceptance Criteria:**
- [ ] Existing code works unchanged
- [ ] Can opt-in to C++ extension with flag
- [ ] iLQR demo works with extension backend

**Checkpoint:** CP-C13 - Integration complete

---

## Summary: Checkpoint List

| Checkpoint | Description | Phase | Dependencies |
|------------|-------------|-------|--------------|
| CP-C01 | Package skeleton created | 1 | None |
| CP-C02 | Build system functional | 1 | CP-C01 |
| CP-C03 | Forward pass matches | 2 | CP-C02 |
| CP-C04 | Backward pass matches | 2 | CP-C03 |
| CP-C05 | Autograd registered | 2 | CP-C04 |
| CP-C06 | Forward validation | 3 | CP-C05 |
| CP-C07 | Gradient validation | 3 | CP-C06 |
| CP-C08 | Full parity validated | 3 | CP-C07 |
| CP-C09 | Performance benchmarked | 4 | CP-C08 |
| CP-C10 | Batched support (opt) | 4 | CP-C09 |
| CP-C11 | pip installable | 5 | CP-C08 |
| CP-C12 | Documentation complete | 5 | CP-C11 |
| CP-C13 | Integration complete | 5 | CP-C12 |

---

## Estimated Effort

| Phase | Tasks | Effort |
|-------|-------|--------|
| Phase 1: Structure | 1.1, 1.2 | Medium |
| Phase 2: Core Implementation | 2.1, 2.2, 2.3 | High |
| Phase 3: Validation | 3.1, 3.2, 3.3 | Medium |
| Phase 4: Performance | 4.1, 4.2 (opt) | Low-Medium |
| Phase 5: Packaging | 5.1, 5.2, 5.3 | Low |

**Critical Path:** 1.1 → 1.2 → 2.1 → 2.2 → 2.3 → 3.1 → 3.2 → 3.3 → 5.1 → 5.3

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Build system complexity | Medium | Medium | Start with minimal CppExtension; fallback to CMake |
| Tensor↔Eigen conversion bugs | Medium | Medium | Extensive testing at each step |
| Gradient mismatch | Low | High | Compare to validated Python wrapper |
| Library linking issues | Medium | Medium | Document exact build requirements |
| Performance not improved | Low | Low | Python wrapper still works; extension is bonus |

---

## Rollback Strategy

If Option C encounters blocking issues:
1. **Fallback:** Python wrapper from Option A remains fully functional
2. **Partial value:** Even a forward-only extension reduces some overhead
3. **Debug path:** Extension build artifacts help diagnose linking/header issues

---

## Approval Required

**Before proceeding with implementation:**
- [ ] User approves this plan
- [ ] User confirms priority (all phases or subset)
- [ ] User confirms batched support requirement (Task 4.2)

---

**Plan Status:** AWAITING USER APPROVAL
**Next Action:** Await explicit permission to begin implementation
