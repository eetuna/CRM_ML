# CRM ML/RL Integration Test Guide

## Overview
This document outlines the steps to run the integration tests for the Catheter Robot Model (CRM) Machine Learning and Reinforcement Learning pipeline. It verifies the interaction between the Python wrappers, the training loop, and the RL feature extractors.

## Prerequisites
*   Python 3.8+ (repo currently targets Python 3.10 in the devcontainer)
*   PyTorch
*   NumPy
*   Gymnasium
*   (Optional) C++ Compiler (GCC/Clang) and CMake for building the physics engine bindings.

## Running Integration Tests

Tests are run with `pytest` and live under `tests/`. Coverage includes:
1.  **Wrapper/Bindings Parity**: C++ vs Python wrapper behavior, seed-based stepping, and diagnostics.
2.  **Training Loop**: ML model training/evaluation smoke tests.
3.  **RL Integration**: Environment + model-based RL smoke tests.

### Command
To run the full test suite from the workspace root:

```bash
python3 -m pytest -q
```

To run a smaller “integration-ish” subset:
- End-to-end RL smoke: `python3 -m pytest -q tests/test_end_to_end_rl.py`
- RL model-based components: `python3 -m pytest -q tests/test_rl_models.py`
- C++ bindings parity: `python3 -m pytest -q tests/test_crmdyn_binding_vs_cpp.py`

## Plan for Next Agent

The following tasks need to be completed to move from the current testing state to a fully functional training pipeline.

### 1. Implement Data Loader
*   **Status**: Implemented.
*   **Files**: `crm_ml_rl/data/data_loader.py`, `crm_ml_rl/data/experimental_loader.py`
*   **Tests**: `tests/test_experimental_loader.py`

### 2. Implement Neural Networks
*   **Status**: Implemented.
*   **File**: `crm_ml_rl/models/networks.py`
*   **Tests**: `tests/test_ml_models.py`, `scripts/check_ml_models.py`

### 3. Build C++ Bindings
*   **Task**: Compile the C++ physics engine to replace the Python mocks/fallbacks.
*   **Steps**:
    ```bash
    mkdir build
    cd build
    cmake .. -DBUILD_PYTHON_BINDINGS=ON
    make
    ```
*   **Verification**: Ensure `import crm_ml_rl.wrappers.crm_python` succeeds in a Python shell.

### 4. Run Real Training
*   **Task**: Train the dynamics model using real experimental data.
*   **Command**:
    ```bash
    python3 crm_ml_rl/training/train_dynamics.py --model-type residual --epochs 50
    ```

### 5. RL Environment Setup
*   **Task**: Finalize the Gymnasium environment using `CRMSimulator`.
*   **Goal**: Train an RL agent (PPO/SAC) using the trained dynamics model as the environment (Model-Based RL) or using the simulator directly.
