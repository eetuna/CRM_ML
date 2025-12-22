# Implementation Plan Check

**Status:** Sound

**Feasibility:**
*   **Templating `DYNNLEqnParams`:** This is feasible and necessary. The current struct uses `double` pointers (e.g., `double (*K)[9]`). A templated version `DYNNLEqnParamsT<Scalar>` will need to use `Scalar (*K)[9]` or, preferably, `Eigen::Matrix<Scalar, 3, 3>` to be fully compatible with `autodiff::real`. Since `CRMIVPCoreParams` uses raw pointers, we might need a "shadow" struct for the AD path to avoid rewriting the entire core legacy codebase (which is risky).
*   **Shadow Struct Strategy:** Instead of intrusive templating of the legacy `CRMIVPCoreParams`, creating a `DYNNLEqnParamsAD` that mirrors the structure but uses `autodiff::real` is a safer, low-risk approach. The `DYNNLEquationResidualEigenAD` function can then be updated to take this AD-friendly struct.

**Dependencies:**
*   **AutoDiff:** `autodiff` library is present (`third_party/autodiff`) and already used in `autodiff_eigen`.
*   **Eigen:** Present and used.

**Risks:**
*   **Maintenance:** A shadow struct means two places to update if parameters change. However, given the "Legacy" status of the core C++, this is acceptable.
*   **Performance:** `autodiff::real` overhead is non-trivial. Benchmarking (Task 1.4 in Plan) will be crucial.

**Conclusion:**
The plan in `docs/architecture/PLAN_end_to_end_differentiable_simulator_options_A_B_C.md` is sound and actionable. `DEVELOPMENT_TASKS.md` correctly identifies the immediate next steps (Task A1).
