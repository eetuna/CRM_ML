# CRM_ML Project - Next Steps Plan

## Overview
This document outlines the roadmap for completing the CRM_ML reinforcement learning pipeline integration with C++ physics bindings. The plan is organized by priority to ensure critical functionality is established before optimization and enhancement.

---

## 1. Fix Dynamics Solver Initialization (High Priority)

### Problem
The C++ dynamics solver (`CRMDynamics`) currently struggles with convergence because it lacks proper initialization from a valid kinematics solution.

### Tasks
- **Implement `initialize_from_kinematics()` in CRMDynamics**:
  - Run Forward Kinematics (FK) first to obtain a valid initial coil state
  - Use FK outputs (tip position, rotation matrix, curvature, etc.) to seed the dynamics state variables
  - Ensure proper transfer of:
    - Coil positions and orientations
    - Curvature vectors (u)
    - Rotation matrices (R)
    - Position vectors (p)

- **Eliminate convergence warnings**:
  - Fix `"Coil integration Unbounded!!"` errors
  - Resolve divergences during BVP/IVP solves
  - Add validation checks for physical constraints (orthogonality of R, valid curvatures)

- **Add initialization validation**:
  - Verify det(R) = 1 and R^T R = I for rotation matrices
  - Check that initial state satisfies boundary conditions
  - Add logging for initialization success/failure

### Success Criteria
- Dynamics solver successfully initializes from FK solution without warnings
- `CRMSimulator.step()` completes without convergence errors
- Initial state satisfies all physical constraints

### Files to Modify
- [src/CRMDYN.hpp](src/CRMDYN.hpp)
- [src/CRMDYN.cpp](src/CRMDYN.cpp) (if exists)
- [crm_ml_rl/wrappers/crm_wrapper.py](crm_ml_rl/wrappers/crm_wrapper.py)

---

## 2. Validate ML/RL Pipeline (High Priority)

### Problem
After fixing C++ bindings, need to ensure the full RL training loop works end-to-end.

### Tasks

#### 2.1 Test Catheter Environment
- **Run basic environment tests**:
  ```python
  from crm_ml_rl.envs.catheter_env import CatheterEnv
  env = CatheterEnv(use_crm=True)
  obs, info = env.reset()
  for _ in range(100):
      action = env.action_space.sample()
      obs, reward, terminated, truncated, info = env.step(action)
  ```
- Verify observation space matches expected dimensions
- Check reward calculation is reasonable
- Ensure episode termination works correctly

#### 2.2 Validate Simulator Data Generator
- Test data generation with C++ simulator:
  ```python
  from crm_ml_rl.data.sim_data_generator import SimDataGenerator
  gen = SimDataGenerator(use_crm=True)
  data = gen.generate_trajectory(...)
  ```
- Verify generated trajectories are physically plausible
- Check data format matches training expectations
- Validate against experimental data if available

#### 2.3 Run Short Training Test
- Execute short training run (1000 steps) with:
  - PPO algorithm
  - C++ simulator backend
  - Monitor for crashes, memory leaks, or convergence issues
- Check tensorboard logs for reasonable learning curves
- Verify model checkpoints are saved correctly

### Success Criteria
- Environment completes 1000+ step rollouts without errors
- Data generator produces valid trajectory data
- Training loop runs for 1000+ timesteps with stable performance
- No memory leaks or crashes during extended runs

### Files to Test
- [crm_ml_rl/envs/catheter_env.py](crm_ml_rl/envs/catheter_env.py)
- [crm_ml_rl/data/sim_data_generator.py](crm_ml_rl/data/sim_data_generator.py)
- [crm_ml_rl/training/train_rl.py](crm_ml_rl/training/train_rl.py)

---

## 3. Data Integration (Medium Priority)

### Problem
Experimental data exists but needs to be integrated with the training pipeline for validation and potentially hybrid modeling.

### Tasks

#### 3.1 Load Experimental Trajectories
- Create data loader for experimental data:
  ```
  3D_dynamic_response_data_0124/
  ├── input_currents/
  ├── desired_input_trajectories/
  └── output_trajectories/
  ```
- Parse trajectory files using column definitions from `data_column_definitions.txt`
- Handle different frequency sweeps (1-100 Hz)
- Support both circle and lemniscate trajectory types

#### 3.2 Create Validation Metrics
- Implement trajectory comparison metrics:
  - Position error (Euclidean distance)
  - Orientation error (rotation matrix difference)
  - Frequency response characteristics
- Compare simulation vs. experimental data
- Generate validation reports with plots

#### 3.3 Parameter Loading from MATLAB
- Load identified parameters from `parameters/model_*.mat`
- Integrate with C++ parameter loading system
- Support frequency-dependent parameter sets
- Add parameter validation and bounds checking

### Success Criteria
- Experimental data successfully loaded and parsed
- Validation metrics computed and visualized
- Parameter files from MATLAB integrated
- Validation report shows reasonable simulation accuracy

### Files to Create/Modify
- `crm_ml_rl/data/experimental_loader.py` (new)
- `crm_ml_rl/evaluation/validation_metrics.py` (new)
- [crm_ml_rl/data/sim_data_generator.py](crm_ml_rl/data/sim_data_generator.py)

---

## 4. Performance Optimization (Medium Priority)

### Problem
C++ bindings may be slow due to Python-C++ overhead; need to profile and optimize.

### Tasks

#### 4.1 Profiling
- Profile training loop to identify bottlenecks:
  ```python
  import cProfile
  cProfile.run('train_rl.main()', 'profile_stats')
  ```
- Measure time spent in:
  - `simulator.step()`
  - `env.reset()`
  - Model forward/backward passes
  - Data transfer between Python/C++

#### 4.2 Batch Processing
- Implement batch forward kinematics if possible
- Vectorize repeated computations
- Reduce Python-C++ boundary crossings

#### 4.3 Caching and Memoization
- Cache FK solutions for repeated configurations
- Memoize parameter loading
- Implement warm-start strategies for BVP solver

### Success Criteria
- 2-5x speedup in simulation step time
- Training throughput increased
- Profile shows optimized hotspots

### Tools
- cProfile, line_profiler for Python
- Valgrind, perf for C++
- PyTorch profiler for RL training

---

## 5. Testing & CI (Medium Priority)

### Problem
Need automated testing to catch regressions and ensure code quality.

### Tasks

#### 5.1 Unit Tests
- **C++ tests**:
  - Forward kinematics accuracy
  - Dynamics solver convergence
  - Jacobian correctness
  - Parameter loading

- **Python tests**:
  - Wrapper interface
  - Environment gym API compliance
  - Data generator output validation
  - Model save/load

#### 5.2 Integration Tests
- End-to-end training test (short run)
- Simulation vs. experimental data comparison
- Parameter identification workflow
- DevContainer build verification

#### 5.3 CI/CD Pipeline
- Set up GitHub Actions or similar:
  - Build C++ library
  - Build Python bindings
  - Run test suite
  - Check code formatting (clang-format, black)
  - Generate coverage reports

### Success Criteria
- 80%+ test coverage for critical paths
- CI pipeline runs on every commit
- All tests pass in DevContainer environment

### Files to Create
- `tests/test_cpp_bindings.py` (new)
- `tests/test_catheter_env.py` (new)
- `tests/test_simulator.py` (new)
- `.github/workflows/ci.yml` (new)
- `CMakeLists.txt` updates for testing

---

## 6. Documentation (Low Priority)

### Problem
Code needs better documentation for future development and collaboration.

### Tasks

#### 6.1 API Documentation
- Add docstrings to all Python modules:
  - CRMWrapper, CRMSimulator
  - CatheterEnv
  - Training scripts
  - Data loaders

- Generate Sphinx or MkDocs documentation:
  ```bash
  sphinx-quickstart docs/
  sphinx-apidoc -o docs/source crm_ml_rl/
  make html
  ```

#### 6.2 Usage Examples
- Create Jupyter notebook tutorials:
  - Basic simulation usage
  - Training an RL agent
  - Loading and validating experimental data
  - Parameter identification workflow

#### 6.3 Architecture Diagrams
- Update CLAUDE.md with:
  - Class diagrams (UML)
  - Data flow diagrams
  - RL training loop diagram
  - C++/Python interaction diagram

### Success Criteria
- Complete API documentation published
- 3+ tutorial notebooks available
- Architecture diagrams up to date
- README has clear quick-start guide

### Files to Create/Modify
- `docs/` directory (new)
- `notebooks/` directory (new)
- [CLAUDE.md](CLAUDE.md) updates
- [README.md](README.md) improvements

---

## Priority Order

1. **Week 1**: Fix Dynamics Solver Initialization (#1)
2. **Week 2**: Validate ML/RL Pipeline (#2)
3. **Week 3**: Data Integration (#3)
4. **Week 4**: Performance Optimization (#4)
5. **Week 5**: Testing & CI (#5)
6. **Week 6**: Documentation (#6)

---

## Current Status

### Completed ✓
- C++ Python bindings created and working
- Forward kinematics functional
- CRMWrapper and CRMSimulator interfaces defined
- DevContainer configuration updated
- Python dependencies installed
- `.gitignore` created
- Compatibility fixes in `catheter_env.py` and `sim_data_generator.py`
- **Dynamics solver initialization fixed** (FK-based initialization)
- **pybind11 array creation bug fixed** (proper `mutable_unchecked` usage)
- **State vector layout corrected** (p[0..2], R[3..11], u[12..14])
- **Damping parameter support** added to wrapper
- **ML/RL Pipeline Validated**:
  - CatheterEnv, ReachingEnv, TrackingEnv tested
  - Gymnasium API compliance verified
  - Vectorized environment working
  - C++ physics integration in environments working
  - PPO/SAC/TD3 training tested

### Test Scripts Created
- `scripts/test_cpp_bindings.py` - C++ bindings validation
- `scripts/test_environment.py` - Environment test suite
- `scripts/test_data_generation.py` - Data generation tests
- `scripts/example_train_rl.py` - Example RL training script

### In Progress ⚙️
- Data Integration (loading experimental data)
- Performance Optimization

### Blocked ⚠️
- None - core functionality complete!

---

## Notes

- **Git Status**: `.devcontainer/devcontainer.json` currently modified (consider committing)
- **Recent Bug Fixes**:
  - Torque calculation added (commit 86fd484)
  - RL rotation/coil location bug (commit 392ef4b)
  - Current problem fixed (commit 1bc04e1)
- **Parameter Sensitivity**: Use frequency-appropriate parameter sets
- **Convergence**: Use incremental loading if solver fails

---

## Questions / Decisions Needed

1. Should we use analytical or numerical Jacobian for dynamics? (analytical is 5-10x faster)
2. What validation threshold is acceptable for sim vs. experimental data?
3. Which RL algorithm to focus on? (PPO, SAC, TD3?)
4. Should we implement multi-threading for batch processing?
5. What CI/CD platform to use? (GitHub Actions, GitLab CI, other?)

---

**Last Updated**: 2025-12-06
**Maintainer**: CRM_ML Development Team
