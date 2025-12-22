# Repository Overview: Continuum Robot Model (CRM) & ML

This document provides a high-level map of the repository, explaining the architecture, data flow, and key entry points for the Catheter CRM project.

## 1. System Architecture

The project is a hybrid system combining high-performance C++ physics modeling with Python-based Machine Learning/Reinforcement Learning and legacy MATLAB validation tools.

```mermaid
graph TD
    subgraph "Data Layer"
        ExpData[Experimental Data] -->|Loader| PyData[Python Data Loader]
        ExpData -->|Loader| MatData[MATLAB Data Loader]
    end

    subgraph "Core Physics (C++)"
        CRM[CRM Engine (src/)]
        Num[Numerical Solvers (numerical/)]
        CRM --> Num
    end

    subgraph "Python / ML Layer"
        PyBind[PyBind11 Bindings]
        Gym[Gym Environments]
        RL[RL Agents (PPO/SAC)]
        
        PyBind --> CRM
        Gym --> PyBind
        RL --> Gym
        Gym --> PyData
    end

    subgraph "MATLAB Layer"
        MEX[MEX Wrappers]
        Scripts[Validation Scripts]
        
        MEX --> CRM
        Scripts --> MEX
        Scripts --> MatData
    end
```

### Key Components
1.  **C++ Core (`src/`, `numerical/`)**: The "source of truth" for the physics. Implements the Cosserat Rod Model kinematics and dynamics using Eigen for linear algebra and Minpack for numerical optimization.
2.  **Python Bindings (`crm_ml_rl/wrappers/`)**: Uses `pybind11` to expose the C++ classes to Python. This allows Python to step the physics simulation at C++ speeds.
3.  **ML/RL Framework (`crm_ml_rl/`)**:
    *   `envs/`: OpenAI Gym-compatible environments (`ReachingEnv`, `TrackingEnv`) that use the physics engine.
    *   `data/`: Loaders for experimental datasets.
    *   `models/`: Neural network definitions (if custom).
4.  **MATLAB (`matlab/`, `Mexfiles/`)**: Used for prototyping, validation, and system identification. It mirrors the C++ logic via MEX files.

---

## 2. Directory Structure

| Directory | Purpose |
| :--- | :--- |
| `src/` | **Core C++ Library**. Contains `CRM.hpp` (Kinematics) and `CRMDYN.hpp` (Dynamics). |
| `numerical/` | **Solvers**. Custom C++ implementation of Minpack (Levenberg-Marquardt). |
| `crm_ml_rl/` | **Python Package**. The main library for ML/RL tasks. |
| `scripts/` | **Entry Points**. Runnable Python scripts for training and analysis. |
| `tests/` | **Python Tests**. Pytest suite for the Python layer. |
| `main/` | **C++ Tests**. Test drivers for the C++ library (e.g., `CRMDYN_test.cpp`). |
| `matlab/` | **MATLAB Scripts**. Utilities and algorithms in MATLAB code. |
| `Mexfiles/` | **MEX Interfaces**. C++ files that bridge MATLAB and the Core Library. |
| `data/` | **Datasets**. Experimental logs, configuration JSONs, and model.parameters. |

---

## 3. Key Entry Points

### 🐍 Python (Machine Learning & RL)
*   **Train an RL Agent:**
    ```bash
    python scripts/example_train_rl.py --algorithm ppo --timesteps 100000
    ```
*   **Quick Demo:**
    ```bash
    python scripts/example_train_rl.py --demo
    ```
*   **Compare Controllers:**
    ```bash
    python scripts/compare_controllers.py
    ```

### ⚙️ C++ (Physics Validation)
*   **Build the project:**
    ```bash
    mkdir build && cd build
    cmake .. -DBUILD_PYTHON_BINDINGS=ON
    make -j4
    ```
*   **Run Dynamics Test:**
    ```bash
    ./CRMDYNTest
    ```

### 📉 MATLAB (Data Analysis)
*   **Verify MEX Bindings:**
    Run `Mexfiles/TestCppMEXFiles.m` in MATLAB.

---

## 4. Data Flow

1.  **Input:** Experimental data is stored in `data/experimental/` (formerly `data/experimental/`).
    *   Format: Text files (`circle100_01.txt`) containing timestamps, currents, and 3D positions.
2.  **Loading:** `crm_ml_rl/data/data_loader.py` reads these files, standardizes units (m vs mm), and creates PyTorch/Numpy datasets.
3.  **Simulation:** The C++ engine consumes catheter parameters (length, stiffness) from `data/catheter_params/`.
4.  **Output:** Training scripts output models to `trained_models/`.

## 5. Testing

*   **Python:** Run `pytest` at the root.
    *   `tests/test_crm_ml_rl_unit.py`: Unit tests.
    *   `tests/test_end_to_end_rl.py`: Integration tests for training loops.
*   **C++:** Run the executables built in `build/` (e.g., `CRMTest`).
