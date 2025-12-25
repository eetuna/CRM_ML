# Code Audit & Gap Analysis Report

**Date:** December 21, 2025
**Scope:** Full Repository Audit (C++, Python, MATLAB, Build System)

## 1. Critical Issues (✅ Fixed)

### 🚨 ~~Broken Data Paths~~ (Resolved)
*   **Status:** **FIXED**
*   **Action Taken:** Moved all data to `data/` and updated hardcoded paths in `crm_ml_rl/`, `matlab/`, and `tests/`. Verified with `pytest`.

### 🐍 ~~Optional Python Bindings~~ (Resolved)
*   **Status:** **FIXED**
*   **Action Taken:** `CMakeLists.txt` updated to default `BUILD_PYTHON_BINDINGS` to `ON`. Users can now simply `pip install .` or build via cmake and get the bindings automatically.

### 🧪 ~~Legacy "Main" Directory~~ (Resolved)
*   **Status:** **FIXED**
*   **Action Taken:** Renamed `main/` to `tests/cpp/` and updated build scripts.

## 2. Code Quality & Architecture (⚠️ Remaining)

### 🔄 Logic Duplication
**Severity: Medium**
There is significant duplication of mathematical primitives across languages:
*   **Matrix Operations:**
    *   C++: `src/CRM_MatrixOperations.hpp` (Custom template implementations)
    *   MATLAB: `matlab/rotation_matrix.m`, `matlab/skewsym2vector.m`
*   **Risk:** If a bug is fixed in the C++ core, the MATLAB verification layer might remain broken (or vice-versa), leading to false positives in validation.
*   **Recommendation:** Mark MATLAB scripts as "Reference Only" or consider porting the C++ logic to use a unified interface.

### ⚠️ C++ Safety
**Severity: Medium**
The C++ code in `src/CRM_MatrixOperations.hpp` uses raw pointer arrays (`double in_A[D1 * D2]`) extensively.
*   **Risk:** No bounds checking. While efficient for embedded use, this is risky for a research code.
*   **Recommendation:** Consider wrapping these in a light structure or using `Eigen::Map` more consistently to leverage Eigen's safety checks in debug mode.

### 📄 Documentation Gaps
*   **Redundant Plans:** `docs/architecture/AUTODIFF_IMPLEMENTATION_PLAN.md` is outdated and superseded by `docs/architecture/PLAN_end_to_end_differentiable_simulator_options_A_B_C.md`.
*   **Minpack Integration:** The files in `src/numerical/` appear to be a copy of the Minpack library. It is unclear if these are modified.

## 3. Testing Gaps (⚠️ Remaining)

*   **Numerical Stability:** There are no explicit tests for the *failure modes* of the numerical solver. What happens if the catheter enters a singular configuration?
*   **Sim-to-Real:** While there are scripts to load data, there isn't a continuous integration test that verifies the *accuracy* of the C++ model against the ground truth data (e.g., "Error must be < 2mm for circle trajectory").
