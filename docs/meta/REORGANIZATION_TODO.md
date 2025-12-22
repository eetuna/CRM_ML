# Repository Reorganization: Remaining Tasks

This document outlines the final steps to complete the repository cleanup and structural alignment.

## 1. Source Code Consolidation (`numerical` → `src`)
The `numerical/` directory contains core mathematical solver logic (Minpack) that is closely coupled with the CRM model implementation.
- [ ] Move `numerical/` to `src/numerical/`.
- [ ] Update `#include` paths in `src/CRMDYN.hpp` and other relevant files (e.g., change `"../numerical/minpack.hpp"` to `"numerical/minpack.hpp"`).
- [ ] Update `CMakeLists.txt` to reflect the new source directory structure for the `numerical` components.

## 2. Test Infrastructure Alignment (`main` → `tests`)
The `main/` directory currently contains C++ test drivers and experimental entry points that are better suited for the `tests` directory.
- [ ] Create `tests/cpp/` and move the contents of `main/` into it.
- [ ] (Optional) Rename existing `tests/` to `tests/python/` for clear language separation.
- [ ] Update `CRMCPPTest.vcxproj` and `CMakeLists.txt` to point to the new test source locations.
- [ ] Verify that `CRMDYN_test.cpp` and `CRMTest.cpp` still compile and run correctly.

## 3. Build & Path Verification
Moving directories often breaks relative paths in build scripts and data loaders.
- [ ] **CMake Update**: Audit `CMakeLists.txt` for all hardcoded paths to `numerical/`, `data/catheter_params/`, and `data/simulation_parameters/`.
- [ ] **Python Data Loaders**: Check `crm_ml_rl/` and `scripts/` for any hardcoded references to root-level data folders. Update them to use the new `data/` subdirectories.
- [ ] **MATLAB Paths**: Ensure `matlab/startup.m` (now in the `matlab/` folder) is updated if it adds root-level folders to the path.

## 4. Cleanup
- [ ] Remove any empty directories or `.bak` files (e.g., `main/CRMDYN_grid_sweep.cpp.bak`).
- [ ] Update `README.md` to reflect the new directory structure.
