REPO Entry Guide

**Last Updated:** December 21, 2025
**Branch:** `feature/autodiff-parameter-gradients`

## 1. Repository Purpose
This repository (`CRM_ML`) implements a **Continuum Robot Model (CRM)** simulator for catheter control, combining:
1.  **C++ Physics Engine:** A Cosserat Rod Model using Minpack/Eigen for nonlinear BVP/IVP solving.
2.  **Python Wrappers:** `pybind11` bindings exposing the physics to Python.
3.  **ML/RL Integration:** OpenAI Gym environments and training scripts for reinforcement learning and system identification.

## 2. Directory Structure (Key Paths)
*   `src/`: Core C++ physics code (`CRM.hpp`, `CRMDYN.hpp`) and solvers (`numerical/`).
*   `crm_ml_rl/`: Python package for ML/RL.
    *   `wrappers/`: C++ bindings (`crm_bindings.cpp`).
    *   `envs/`: Gym environments.
*   `data/`: Datasets and configurations (split into `catheter_params`, `experimental`, `simulation_parameters`).
*   `tests/`: `pytest` suite (Python) and `cpp/` (C++ executables).
*   `docs/`: Project documentation (Architecture, Guides, Reports).

## 3. Current Status
*   **Repository Health:** Clean, organized, and fully verified. All broken paths from recent reorganization are fixed.
*   **Active Development:** We are implementing **End-to-End Differentiable Physics**.
    *   The project is currently capable of running Forward Kinematics (FK) and Dynamics simulations.
    *   Basic AutoDiff (state gradients $\partial F/\partial x$) is implemented (`autodiff_eigen`).
    *   **Goal:** Enable training by computing parameter gradients ($\partial F/\partial \theta$).

## 4. Key Documents (Read Order)
1.  **`docs/architecture/REPOSITORY_OVERVIEW.md`**: High-level system map and data flow.
2.  **`docs/architecture/PLAN_end_to_end_differentiable_simulator_options_A_B_C.md`**: The master strategy document. We are executing **Option A**.
3.  **`docs/architecture/DEVELOPMENT_TASKS.md`**: The specific "ToDo" list for the current phase.
4.  **`docs/architecture/CODE_AUDIT_REPORT.md`**: Known technical debt and quality gaps.

## 5. Immediate Next Steps (For the Next Agent)
You are on branch `feature/autodiff-parameter-gradients`. Your primary task is **Task A1** from `DEVELOPMENT_TASKS.md`:

1.  **Refactor `DYNNLEqnParams`:** Update `src/CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp` (or creating a new header) to use a templated parameter structure that supports `autodiff::real`.
2.  **Implement Gradients:** Extend the residual function to calculate gradients with respect to physical parameters (damping, stiffness, etc.).
3.  **Expose to Python:** Update `crm_bindings.cpp` to expose these new gradients.

**Do NOT** modify the core C++ logic (`src/numerical/`, `src/CRM_MatrixOperations.hpp`) unless absolutely necessary. Focus on the AutoDiff layer.
