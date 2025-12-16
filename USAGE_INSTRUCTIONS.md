# Usage Overview

- **Data generation sanity check**  
  `python3 scripts/check_data_generation.py`  
  Uses CRM C++ if available, otherwise falls back to simplified dynamics; warnings about CasADi/CVXPY are OK.

- **Basic simulator**  
  ```python
  from crm_ml_rl.wrappers.crm_wrapper import CRMSimulator
  sim = CRMSimulator(dt=0.02, use_cpp=False)  # set use_cpp=True if bindings built
  traj = sim.simulate_trajectory([[0.1, 0, 0]] * 50, insertion_length=50.0)
  print(traj["positions"].shape)
  ```

- **Hybrid dynamics rollout**  
  ```python
  from crm_ml_rl.models.hybrid_dynamics import HybridDynamicsModel, HybridDynamicsConfig
  cfg = HybridDynamicsConfig(use_cpp=False, dt=0.02)
  model = HybridDynamicsModel(cfg)
  state = model.reset()
  next_state = model.step(state, action=[0.1, 0, 0])
  ```

- **RL agents**  
  SAC/TD3 automatically set proper `net_arch` when you pass `algorithm` through `create_policy_kwargs`; run your usual SB3 training scripts via `crm_ml_rl/training/rl_agents.py`.

- **Gym environment**  
  ```python
  from crm_ml_rl.envs.catheter_env import CatheterEnv, CatheterEnvConfig
  env = CatheterEnv(CatheterEnvConfig(use_cpp=False))
  obs, info = env.reset()
  obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
  ```

## Goal: catheter tip control
- The control tasks are reaching a static target or tracking a moving trajectory with a magnetic catheter. Actions are coil currents (3D), observations include tip position/velocity (plus target/history), and rewards penalize distance and large/jerky actions with a success bonus on target.

## Suggested workflow
- Generate data (if needed): `python3 scripts/check_data_generation.py` or a custom `SimDataGenerator` script to produce/save trajectories.
- Train models on your data: choose full/residual/hybrid kinematics/dynamics depending on how much physics prior you want.
- Validate models: run the provided test scripts under `scripts/` to sanity-check forward passes (e.g., `test_ml_models.py`, `test_hybrid_models.py`).
- Integrate into control: use the chosen dynamics model in MPC rollouts, or configure the RL env/policies to use CRM or learned dynamics; train SAC/TD3/etc.
- Deploy/sim-to-real: run the trained policy/model in your runtime stack.

## Comparing controllers
- PID baseline: `python3 scripts/pid_baseline.py --env reaching --episodes 5` (tune `--kp/--ki/--kd`).
- Quick comparison (PID vs model-free PPO vs simple model-based shooter):  
  `python3 scripts/compare_controllers.py --env reaching --episodes 3 --ppo-steps 2000 --mb-candidates 16`

## Notes
- If you want the C++ backend, ensure bindings are built and available; otherwise leave `use_cpp=False` (defaults are robust fallbacks).
- Install `requirements.txt` plus `pytest` if you want to run the full test suite with `pytest scripts`.
