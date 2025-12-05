# 🚀 Next Steps Plan for CRM_ML Project

Based on the current state of the project, the following next-step roadmap is recommended.

---

## 1. Fix Dynamics Solver Initialization (High Priority)

The C++ dynamics solver (`CRMDynamics`) currently struggles with convergence because it lacks proper initialization from a valid kinematics solution.

### Tasks
- Implement `initialize_from_kinematics()`:
  - Run Forward Kinematics (FK) first to obtain a valid initial coil state.
  - Use FK outputs (tip position, rotation matrix, curvature, etc.) to seed the dynamics state variables.
- Ensure dynamics integration begins from a feasible configuration to eliminate:
  - `"Coil integration Unbounded!!"` warnings
  - Divergences during BVP/IVP solves.

---

## 2. Validate ML/RL Pipeline (High Priority)

Test the full ML + RL training workflow end-to-end.

### Tasks
- Test `train_dynamics.py` with simulated data.
- Test `train_rl.py` using the `ReachingEnv` environment.
- Verify:
  - Model checkpoint saving
  - Checkpoint loading
  - Rollout evaluation
- Run evaluation and plotting scripts to confirm training quality.

---

## 3. Data Integration (Medium Priority)

Integrate experimental trajectory data stored in `3D_dynamic_response_data_0124/` into the training pipeline.

### Tasks
- Validate that `CRMDataLoader` properly loads all experimental trajectories.
- Build a robust preprocessing pipeline (normalization, trimming, filtering).
- Generate reproducible train/validation/test splits.
- Test hybrid model training:
  - Physics model + learned residual correction.

---

## 4. Performance Optimization (Medium Priority)

Improve simulation speed to make RL training practical.

### Tasks
- Profile the C++ bindings to identify bottlenecks.
- Implement batch forward kinematics for parallel RL environments.
- Add caching/memoization for repeated FK calls.
- Explore reducing state dimensionality or simplifying certain dynamics steps.

---

## 5. Testing & CI (Medium Priority)

Add formal testing and automation infrastructure.

### Tasks
- Create unit tests for:
  - Python API layer
  - Wrapper outputs
- Add integration tests for:
  - RL environments
  - Dynamics/FK consistency
- Configure `pytest`.
- Add GitHub Actions workflow for CI:
  - Build C++ bindings
  - Run Python tests
  - Linting

---

## 6. Documentation (Low Priority)

Improve clarity and ease of use for new users and collaborators.

### Tasks
- Add docstrings to all public-facing APIs.
- Create example Jupyter notebooks:
  - FK usage
  - Dynamics simulation
  - RL training walkthrough
- Document recommended RL hyperparameters.
- Add a "Getting Started" section to the README.

---

### ✅ Recommended Starting Point

**Item #1 — Fix Dynamics Solver Initialization**

This unlocks the full capability of the C++ physics engine for accurate simulation and RL integration.

Using the simplified dynamics fallback, RL development can proceed in parallel even before the full fix is complete.

---
