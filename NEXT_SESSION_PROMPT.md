# Next Session Prompt for Option C Phase 3 Completion

**Copy and paste this prompt to start the next session:**

---

I'm continuing the Option C implementation for CRM_ML catheter physics - Phase 3 backward pass.

**Context:** Phase 3A backward pass is implemented and functional. Please read the handoff report:
- `/workspaces/catheter/CRM_ML/docs/OPTION_C_PHASE3_HANDOFF_REPORT.md`

**Current Status:**
- ✅ Phase 3A complete: Backward pass computes current gradients correctly
- ✅ Smoke test passes: No NaN/Inf, non-zero gradients (norm ~207)
- ✅ Extension builds and loads successfully
- 🔄 Need to complete: Tasks 5, 6, 7 (gradient validation, testing, benchmarking)

**Remaining Tasks (~1.5 hours):**

**Task 5: Gradient Validation (30 min)**
- Create `crm_torch/test/test_gradient_validation.py`
- Compare autograd gradients vs finite differences
- Target: Relative error < 1% (match Option A tolerance)

**Task 6: PyTorch Autograd Testing (30 min)**
- Create `crm_torch/test/test_pytorch_autograd.py`
- Test single sample, batch, gradient accumulation
- Verify edge cases

**Task 7: Performance Benchmark (30 min)**
- Create `crm_torch/test/benchmark_performance.py`
- Measure Option C vs Option A performance
- Document speedup/overhead

**Guidelines:**
- Stop at each task completion and ask my permission to proceed
- If context runs low, generate handoff report and new prompt for next session
- Keep tests concise and focused
- Follow design in `docs/OPTION_C_PHASE3_BACKWARD_PASS_DESIGN.md`

**Success Criteria:**
- Gradient validation passes with <1% error
- All autograd tests pass
- Performance documented
- Phase 3 marked complete

**Files to reference:**
- Handoff: `docs/OPTION_C_PHASE3_HANDOFF_REPORT.md`
- Design: `docs/OPTION_C_PHASE3_BACKWARD_PASS_DESIGN.md`
- Smoke test: `crm_torch/test/test_backward_smoke.py`

**Branch:** `claude/option-c-implementation`

Start by reading the handoff report and confirming you understand the current status.
