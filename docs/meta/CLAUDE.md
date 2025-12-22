# CRM_ML - Cosserat Rod Model for Magnetically-Actuated Catheters

## Project Overview

This project implements a **hybrid physics-based and system identification framework** for modeling magnetically-actuated catheter dynamics. It combines continuum mechanics (Cosserat Rod Model) with experimental parameter estimation to achieve accurate predictive models for medical catheter control and navigation.

The system bridges rigorous physics simulation (C++) with data-driven parameter identification (MATLAB) to model flexible catheters under magnetic actuation.

## Tech Stack

**Languages:**
- C++17 - Core physics engine and numerical solvers
- MATLAB - System identification, parameter estimation, visualization
- CMake - Build configuration

**Key Dependencies:**
- **Eigen3** - Linear algebra operations (matrices, vectors, transformations)
- **MINPACK** - Trust region methods for nonlinear equation solving
- **MATLAB Toolboxes**: System Identification, Control System, Optimization

**Mathematical Frameworks:**
- Cosserat Rod Theory - Continuum mechanics for slender flexible structures
- SE(3) Lie Group - Special Euclidean group for rigid body transformations
- Trust Region Methods - Nonlinear boundary value problem solving
- Grey-box System Identification - Physics-informed parameter estimation

## Architecture

### High-Level System Flow

```
Experimental Data → MATLAB System ID → Parameter Files → C++ Physics Engine → Predictions
                         ↑                                        ↓
                         └────────── Validation Loop ────────────┘
```

### Core Components

#### 1. C++ Physics Engine (`/src/`)
The computational backbone implementing Cosserat rod mechanics:

- **[CRM.hpp](./src/CRM.hpp)** (23KB) - Core data structures
  - Catheter parameter definitions
  - Segment types (flexible, rigid, actuators)
  - 15-state representation: u[0..2] (curvature), R[0..9] (rotation), p[0..2] (position)

- **[CRMDYN.hpp](./src/CRMDYN.hpp)** (10KB) - Dynamic solver
  - Time-domain simulation with coil dynamics
  - Inertia, damping, and gravity effects

- **[CRM_BVPSolver.cpp](./src/CRM_BVPSolver.cpp)** (26KB) - Boundary Value Problem solver
  - Shooting method for static configurations
  - Trust region optimization

- **[CRM_IVPSolver.cpp](./src/CRM_IVPSolver.cpp)** (38KB) - Initial Value Problem solver
  - Numerical integration (Runge-Kutta variants)
  - Integrates differential equations from base to tip

- **[CRM_IVPJacobian.cpp](./src/CRM_IVPJacobian.cpp)** (29KB) - Analytical Jacobian
  - Sensitivity analysis for optimization

- **[CoilDynamics_Defs.cpp](./src/CoilDynamics_Defs.cpp)** (64KB) - Magnetic actuation
  - Magnetic forces and torques on coils
  - Coil inertia and dynamics integration

#### 2. Numerical Solvers (`/numerical/`)
Modified MINPACK library for nonlinear equation solving:

- **minpack.cpp/hpp** - Trust region with numerical Jacobian
- **minpack_DYN.hpp/cpp** - Dynamic version for time-domain problems
- **minpack_wGivenJac_*.hpp** - Trust region with analytical Jacobian (faster)
- Tolerance: 1e-5, dogleg trust region method

#### 3. MATLAB System Identification (`/matlab/`)
Parameter estimation from experimental data:

- **[greybox_modeling.m](./matlab/greybox_modeling.m)** - Grey-box identification
  - Uses physics structure with unknown data/simulation_parameters
  - Estimates parameters from trajectory data

- **[linear_identification.m](./matlab/linear_identification.m)** - ARX models
  - Linear approximations for comparison

- **[read_input_output.m](./matlab/read_input_output.m)** - Data loading
- **[data_stitch.m](./matlab/data_stitch.m)** - Combine multiple experimental runs
- **Rotation utilities** - so3rot.m, twistr.m, twistt.m for SE(3) operations

#### 4. MATLAB-C++ Bridge (`/Mexfiles/`)
MEX functions for calling C++ from MATLAB:

- **CRMDYN_c_mex.cpp/.mexa64** - Dynamic solver interface
- **CRM_ForwardKinematics_matlab.cpp** - Static kinematics
- **CRM_FKJacobian_Analytical_matlab.cpp** - Jacobian computation
- **Load_CRMCatheterModelParams_matlab.cpp** - Parameter loading
- **DrawCRMCatheterModel.m** - 3D visualization

#### 5. Test Programs (`/main/`)
Standalone executables for testing:

- **CRMDYN_test.cpp** - Dynamic solver validation
- **CRMTest.cpp** - Static kinematics tests
- **CRM_KinematicsTestFunctions.cpp/hpp** - Test utilities

## Directory Structure

```
CRM_ML/
├── src/                          # Core C++ physics engine (328KB, 19 files)
├── numerical/                    # MINPACK-based solvers
├── main/                         # Test executables
├── matlab/                       # System identification scripts
├── Mexfiles/                     # MATLAB-C++ interface
├── data/catheter_params/                 # Physical parameter configs
├── data/simulation_parameters/                   # Identified parameters (.mat files)
├── data/experimental/  # Experimental trajectory data (6.7MB)
│   ├── input_currents/           # Actuation signals
│   ├── desired_input_trajectories/  # Target paths
│   └── output_trajectories/      # Measured responses (1-100 Hz)
├── data/hybrid/                  # Hybrid model data
├── data/output/                  # Simulation outputs
└── build/                        # CMake build artifacts
```

## Key Concepts

### Cosserat Rod Model
- **State Variables**: At each point along catheter
  - u[0..2]: Curvature components (bending/twist)
  - R[0..9]: Rotation matrix (orientation)
  - p[0..2]: Position (x, y, z)

- **Segments**: Multi-segment catheter
  - Flexible segments: Bending under loads
  - Rigid segments: No deformation
  - Actuator segments: Magnetic coils embedded

- **Contact Modes**:
  - FREE_TIP: Unconstrained tip
  - FIXED_TIP: Tip position/orientation constrained

### Magnetic Actuation
- External uniform magnetic field B0
- Magnetic dipole moments in coils
- Force = ∇(m · B), Torque = m × B
- Current-controlled magnetic moments

### Hybrid Approach
1. **Physics Model**: Provides mathematical structure (differential equations)
2. **Experimental Data**: Real catheter trajectories at various frequencies
3. **Parameter Identification**: Estimate unknown parameters (stiffness, damping, etc.)
4. **Validation**: Compare predictions vs. measurements

## Important Conventions

### Code Style
- C++17 standard features
- Eigen::VectorXd, Eigen::MatrixXd for linear algebra
- Hungarian notation for member variables (m_variableName)
- Inline comments for complex mathematical operations

### File Naming
- `CRM_*.cpp/hpp` - Cosserat Rod Model components
- `*_matlab.cpp` - MEX interface files
- `*_test.cpp` - Test programs
- `CatheterParameterSet_*.txt` - Physical data/simulation_parameters
- `CatheterSpatialConfiguration_*.txt` - Environment setup

### Parameter Files
- **data/catheter_params/**: Text files with physical properties
  - Young's modulus, cross-sectional radii
  - Coil mass, magnetic dipole moments
  - Segment lengths and types

- **data/simulation_parameters/**: MATLAB .mat files with identified data/simulation_parameters
  - model_1.mat, model_2.mat, model_3.mat
  - Results from grey-box identification

## Common Tasks

### Building the C++ Project
```bash
cd /workspaces/CRM_ML
mkdir -p build && cd build
cmake ..
make
```
**Outputs:**
- `libCRMCPPLib.a` - Static library
- `CRMTest` - Static kinematics test
- `CRMDYNTest` - Dynamic simulation test

### Building Python Bindings (pybind11)
```bash
cd /workspaces/CRM_ML
mkdir -p build && cd build
cmake -DBUILD_PYTHON_BINDINGS=ON ..
make crm_python
```
**Outputs:**
- `crm_ml_rl/wrappers/crm_python.cpython-*.so` - Python module

**Requirements:**
- pybind11: `apt install python3-pybind11 pybind11-dev` or `pip install pybind11`

**Usage in Python:**
```python
from crm_ml_rl.wrappers.crm_wrapper import CRMWrapper

wrapper = CRMWrapper(
    param_file='data/catheter_params/CatheterParameterSet_1.txt',
    config_file='data/catheter_params/CatheterSpatialConfiguration_1.txt',
    use_cpp=True
)
result = wrapper.forward_kinematics(currents, insertion_length)
```

### Building MEX Files
```matlab
cd /workspaces/CRM_ML/Mexfiles
mex CRMDYN_c_mex.cpp -I../src -I../numerical -L../build -lCRMCPPLib -I/usr/include/eigen3
```

### Running MATLAB System Identification
```matlab
cd /workspaces/CRM_ML
startup  % Add paths
cd matlab
greybox_modeling  % Run parameter estimation
```

### Running C++ Tests
```bash
cd /workspaces/CRM_ML/build
./CRMTest           # Static kinematics
./CRMDYNTest        # Dynamic simulation
```

### Loading Experimental Data
- Trajectories: `data/experimental/output_trajectories/`
- Currents: `data/experimental/input_currents/`
- Column format defined in `data_column_definitions.txt`

## Critical Implementation Details

### Numerical Tolerances
- Trust region solver: 1e-5
- Integration step size: typically 0.2 mm
- Incremental current application for convergence

### State Vector Organization
Each point has 15 states stored sequentially:
- Indices 0-2: u (curvature)
- Indices 3-11: R (rotation matrix, row-major)
- Indices 12-14: p (position)

### Jacobian Computation
Two options available:
1. **Numerical**: Finite differences (slower, always works)
2. **Analytical**: Closed-form derivatives (faster, requires careful implementation)

Choose analytical when available for 5-10x speedup.

### Coordinate Frames
- **World Frame**: Fixed laboratory frame
- **Body Frame**: Attached to catheter cross-section
- **Magnetic Frame**: Aligned with B0 field direction

Transformations via SE(3) matrices and exponential maps.

## Experimental Data Structure

### Trajectory Types
- **Circle**: Circular tip motion at varying frequencies
- **Lemniscate**: Figure-8 motion at varying frequencies

### Frequency Sweep
Data collected at: 01, 03, 05, 08, 10, 12, 15, 18, 20, 25, 50, 100 Hz

### Data Columns (see data_column_definitions.txt)
- Time stamp
- Arduino state
- Base position/orientation
- Coil position/tangent vectors
- Tip position

## Gotchas / Important Notes

### Recent Bug Fixes (from commit history)
1. **Torque calculation** (commit 86fd484): Added tau into forward kinematics
   - Ensure torque balance equations included in all solvers

2. **RL bug** (commit 392ef4b): Likely rotation/coil location issue
   - Verify rotation matrix orthogonality (det(R) = 1, R^T R = I)
   - Check coil position calculations

3. **Debugging complete** (commit 1bc04e1): Current problem fixed
   - Validate against this working state if issues arise

### Parameter Sensitivity
- Identified parameters are frequency-dependent
- Use appropriate parameter set for simulation frequency range
- Grey-box models perform better than pure physics or pure black-box

### Convergence Issues
If solver fails to converge:
1. Reduce initial current magnitude
2. Use incremental loading (ramp up current slowly)
3. Check for unphysical parameters (negative stiffness, etc.)
4. Verify boundary conditions match catheter configuration

### Memory Management
- Large state vectors for multi-segment catheters
- Jacobian matrices can be memory-intensive
- Use analytical Jacobian to reduce memory footprint

### MATLAB-C++ Integration
- MEX files must be rebuilt after C++ library changes
- Ensure matching Eigen3 versions between MATLAB and C++
- MATLAB uses column-major, Eigen defaults to column-major (compatible)

### Git Status Note
- `.devcontainer/` currently untracked
- Consider adding to repo for reproducible development environment

## Development Environment

**Container**: Ubuntu 22.04 (see `.devcontainer/devcontainer.json`)
**IDE**: VSCode with extensions:
- C/C++ (ms-vscode.cpptools)
- CMake Tools (ms-vscode.cmake-tools)
- Claude Code (anthropic.claude-code)

**MATLAB Setup**: Run `startup.m` to configure paths

## References & Resources

### Cosserat Rod Theory
- Geometric formulation with SE(3) transformations
- Local curvature-based representation
- Well-suited for slender flexible structures

### System Identification
- Grey-box models: `idnlgrey` (MATLAB)
- Combines physics structure with parameter flexibility
- Validates against frequency-domain characteristics

### Magnetic Actuation
- Uniform field assumption (far from field source)
- Dipole model for coil-field interaction
- Current control for force/torque generation

## Project Status

**Latest Updates** (as of commit 828bf8c, Feb 5 2024):
- Cleaned up obsolete parameters and folders
- Hybrid dynamic solver updated to latest version
- Bug fixes in dynamics calculation complete
- Parameter files organized in `/data/simulation_parameters/` directory

**Current Focus**:
- Refining hybrid solver combining physics and data-driven models
- Validating against experimental trajectory data
- Improving convergence and computational efficiency

---

## Quick Start Checklist

### For Python/ML Development (Recommended):
```bash
# 1. Install system dependencies
sudo apt-get install libeigen3-dev python3-pybind11 pybind11-dev python3-pip

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Build C++ Python bindings
mkdir -p build && cd build
cmake -DBUILD_PYTHON_BINDINGS=ON ..
make crm_python

# 4. Test the bindings
cd /workspaces/CRM_ML
python3 -c "from crm_ml_rl.wrappers.crm_wrapper import CRMWrapper; print('OK')"
```

### For C++/MATLAB Development:
1. Install Eigen3: `sudo apt-get install libeigen3-dev`
2. Build C++ library: `mkdir build && cd build && cmake .. && make`
3. Compile MEX files in MATLAB: `cd Mexfiles; mex CRMDYN_c_mex.cpp ...`
4. Run MATLAB startup: `startup.m`
5. Test basic kinematics: `./build/CRMTest`
6. Run system identification: `matlab/greybox_modeling.m`
7. Visualize results: `Mexfiles/DrawCRMCatheterModel.m`

### Using DevContainer (VS Code):
The `.devcontainer` configuration automatically:
- Installs all system dependencies
- Installs Python packages from `requirements.txt`
- Builds the C++ Python bindings

Just open the project in VS Code and select "Reopen in Container".

For questions or issues, refer to commit history for recent changes and bug fixes.
