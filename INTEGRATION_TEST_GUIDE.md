# CRM ML/RL Integration Test Guide

## Overview
This document outlines the steps to run the integration tests for the Catheter Robot Model (CRM) Machine Learning and Reinforcement Learning pipeline. It verifies the interaction between the Python wrappers, the training loop, and the RL feature extractors.

## Prerequisites
*   Python 3.8+
*   PyTorch
*   NumPy
*   Gymnasium
*   (Optional) C++ Compiler (GCC/Clang) and CMake for building the physics engine bindings.

## Running Integration Tests

The integration tests are located in `tests/test_crm_integration.py`. These tests verify:
1.  **Wrapper Logic**: `CRMWrapper` functionality (handling fallback to simplified physics if C++ is missing).
2.  **Training Loop**: `DynamicsTrainer` batching and data handling for both Residual (MLP) and Sequence (Transformer) models.
3.  **RL Integration**: `PhysicsInformedExtractor` feature extraction logic.

### Command
To run the tests, execute the following command from the workspace root:

```bash
python3 tests/test_crm_integration.py
```

### Expected Output
You should see output indicating that tests passed. Note that the test script automatically mocks missing C++ bindings and data loaders to ensure logic correctness even without the full environment.

```text
Mocking crm_python for testing...
...
Ran 5 tests in 0.xxx s

OK
```

## Plan for Next Agent

The following tasks need to be completed to move from the current testing state to a fully functional training pipeline.

### 1. Implement Data Loader
*   **File**: `crm_ml_rl/data/data_loader.py`
*   **Task**: Implement `CRMDataLoader` to parse the experimental data formats (MATLAB `.mat` or text files in `3D_dynamic_response_data_0124`).
*   **Requirement**: Ensure it returns trajectory objects containing `tip_positions`, `currents`, and `sampling_time_ms`.

### 2. Implement Neural Networks
*   **File**: `crm_ml_rl/models/networks.py`
*   **Task**: Implement the actual PyTorch modules referenced in the models: `MLP`, `LSTM_MLP`, `DeepResidualMLP`, and `ResidualBlock`.

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
