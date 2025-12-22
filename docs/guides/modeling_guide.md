# CRM Modeling and RL Guide

This document summarizes how the residual and hybrid models inside `crm_ml_rl/models` relate to each other and how they support different reinforcement learning (RL) strategies. Equations are written in plain text so they can be read without LaTeX.

---

## Notation

- `u_t ∈ ℝ^3`: coil current command at time `t`.
- `ℓ`: insertion length (mm).
- `s_t`: any additional state features (e.g., base pose).
- `x_t = [p_t, v_t] ∈ ℝ^6`: concatenated tip position and velocity.
- `f_CRM`: C++ CRM physics model (forward kinematics or dynamics).
- `r_θ(·)`: neural residual network with parameters `θ`.
- `ε_res`: clamp limit that constrains the residual magnitude.
- `w`: learnable blend weight (`w = sigmoid(α)`).

---

## Residual Models

Residual models assume you already computed a physics prediction outside the module (e.g., from CRM, a ROM, or logged simulator output). They learn *only* the correction term and therefore remain lightweight, data-efficient, and easy to pair with alternative physics surrogates.

### Residual Forward Kinematics

1. Physics baseline: `y_phys = f_CRM_fk(u_t, ℓ)` (tip position).
2. Residual network input: `z = concat[u_t, ℓ, y_phys]`.
3. Prediction: `y_final = y_phys + r_θ(z)` with `r_θ` clamped to `[-ε_res, ε_res]` per axis.

Implementation: `ResidualKinematicsModel` and `ResidualKinematicsWithUncertainty` in `crm_ml_rl/models/residual_kinematics.py`.

Key points:
- Optional deep residual block stack (improves expressiveness without sacrificing stability).
- Learnable output scaling enables coarse-to-fine tuning.
- Uncertainty variant outputs `(mean_residual, variance)` so downstream planners can reason about aleatoric risk.

### Residual Dynamics

1. Physics rollout: `x_{t+1}^{phys} = f_CRM_dyn(x_t, u_t)`.
2. Residual input: `z = concat[x_t, u_t, x_{t+1}^{phys}]`.
3. Prediction: `x_{t+1} = x_{t+1}^{phys} + r_θ(z)`, clamped to `[-ε_res, ε_res]`.

Variants (`crm_ml_rl/models/residual_dynamics.py`):
- **MLP / DeepResidualMLP** backend.
- **LSTM residual** (`ResidualDynamicsLSTM`) that consumes sequences `[x_{t-k:t}, u_{t-k:t}, x_{t-k+1:t+1}^{phys}]`.
- **EnsembleResidualDynamics** for epistemic uncertainty via member disagreement.

Training tips:
- Because `r_θ` only learns the discrepancy, you can normalize the target residuals (`x_target - x_{t+1}^{phys}`) for faster convergence.
- Residual caps (`ε_res`) prevent the network from inventing behavior that contradicts the physics baseline.

When to use:
- You already logged CRM predictions alongside measurements.
- You need to swap physics engines (e.g., coarse FEM vs. CRM) without retraining the full network.
- Deployments with limited compute where CRM runs elsewhere (embedded, ROS node) and the ML piece is a small correction layer.

---

## Hybrid Models

Hybrid models tightly couple to the CRM simulator: the module itself calls the physics engine, converts tensors to NumPy, and blends the physics prediction with a learned residual. This guarantees that the latest catheter parameters, damping, and insertion settings are reflected in every forward pass, and it exposes simulator-style APIs for RL environments (reset, step, trajectory prediction).

### Hybrid Forward Kinematics

1. Internal physics: `y_phys = f_CRM_fk(u_t, ℓ)` computed per sample.
2. Residual input: `z = concat[u_t, ℓ]` (physics output appended inside `ResidualKinematicsModel` if `include_physics_in_input=True`).
3. Blend:
   - Simple addition: `y_final = y_phys + r_θ([z, y_phys])`.
   - Learnable blend: `w = sigmoid(α)` → `y_final = w * y_phys + (1 - w) * (y_phys + r_θ(...))`.

Implementation: `HybridKinematicsModel` and `HybridKinematicsWithUncertainty` in `crm_ml_rl/models/hybrid_kinematics.py`.

Features:
- Handles NumPy I/O via `predict()` for quick evaluation.
- Cacheable physics outputs (`enable_cache`) to accelerate batched training.
- Rich loss reporting (`final`, `physics`, `residual_reg`) for debugging.

### Hybrid Dynamics

1. Maintain CRM simulator state `CatheterState` internally (`reset`, `step`).
2. For each batch element, re-initialize CRM at the provided `x_t`, run one physics step to get `x_{t+1}^{phys}`.
3. Residual network (single model or ensemble) receives `[x_t, u_t, x_{t+1}^{phys}]`.
4. Blend as in kinematics (optional learnable `w`).

Implementation: `HybridDynamicsModel`, `HybridDynamicsLSTM`, and `HybridDynamicsConfig` in `crm_ml_rl/models/hybrid_dynamics.py`.

Capabilities:
- Trajectory rollout `predict_trajectory(actions)` for MPC-style planning.
- Per-component loss tracking (position vs. velocity error).
- Ensemble-based uncertainty hooks (`get_uncertainty`).
- History-dependent corrections via LSTM residuals with rolling buffers.

When to use:
- Need end-to-end differentiable components that include CRM every step.
- Desire for consistent physics parameters during both training and deployment.
- RL environments that must expose `reset`/`step` and maintain hidden physics state.

Trade-offs:
- Higher compute and tighter dependency on CRM paths/damping settings.
- GPU ↔ CPU transfers due to NumPy physics calls; batching helps amortize the cost.

---

## Sequence-Aware Learned Models

When you want fully learned simulators that retain long-term context or capture multimodal transitions, the repository now includes dedicated transformer and diffusion dynamics models (`crm_ml_rl/models/sequence_models.py`).

### TransformerDynamicsModel

- **Inputs:** A rolling window of concatenated `[x_t, u_t]` tokens (`context_len` configurable). Each token is linearly projected into the transformer’s `d_model` space with learned positional encodings.
- **Core:** Transformer encoder layers (multihead attention + feedforward) summarize the context and emit the representation of the latest token.
- **Output:** A linear head predicts either the delta `Δx_t` or the absolute next state. Internally the sequence model updates its history so it can be used for multi-step rollout.
- **Use cases:** History-dependent hysteresis, situations where you want the model to “remember” long bends or action bursts without explicitly feeding handcrafted features.
- **Configuration:** Pass custom kwargs (`transformer_kwargs`) through `TrainingConfig` or `ModelBasedConfig` to adjust `context_len`, `d_model`, `nhead`, etc.

### DiffusionDynamicsModel

- **Distributional modeling:** Learns a conditional distribution over next-state deltas using DDPM. Training noise schedule `β_t` is linearly spaced; the network predicts injected noise ε conditioned on `[x_t, u_t]` and the diffusion timestep.
- **Loss:** `L = E_{t,ε}[ ||ε - ε_θ(√ᾱ_t Δx + √(1-ᾱ_t) ε, x_t, u_t, t)||² ]`.
- **Sampling:** Reverse-diffusion loop generates stochastic next states; deterministic mode keeps the posterior variance term zero for a mean prediction. Call `sample_multiple` for ensembles of rollouts when running stochastic MPC or uncertainty-aware planning.
- **Config:** `DiffusionDynamicsConfig` (state/action dims, diffusion steps, β schedule, hidden width). Training uses `diffusion_loss`, while evaluation uses deterministic predictions for consistency.

These models behave like the “full” networks (no CRM dependency) but provide better temporal fidelity (transformer) or uncertainty-aware generation (diffusion).

---

## RL Strategy and Model Selection

| RL Approach        | Goal                                                | Suggested Model Stack                                         |
|--------------------|-----------------------------------------------------|----------------------------------------------------------------|
| Model-free RL      | Learn policy directly from data/interactions        | Residual models (lightweight critic) or hybrid wrappers as environment dynamics |
| Model-based RL     | Learn dynamics model, use it inside planner/policy  | Hybrid Dynamics (for accuracy) or Residual Dynamics (if physics predictions precomputed) |
| MPC / shooting     | Roll out candidate controls under learned dynamics  | Hybrid Dynamics trajectory rollout or FullDynamicsModel when no CRM access |

### Model-Free RL

- Treat the hybrid model as the *environment simulator*: call `HybridDynamicsModel.step()` inside the RL loop so the agent never sees the physics mismatch.
- Alternatively, run real hardware or CRM simulator externally and feed logged transitions to a policy/value network that uses residual models purely as function approximators for reward shaping or auxiliary predictions.
- Advantage: minimal modeling overhead for the policy. Disadvantage: sample inefficient unless combined with experience replay from hybrid simulators.

### Model-Based RL

1. Train `ResidualDynamicsModel` (or hybrid variant) on dataset `D = {(x_t, u_t, x_{t+1})}`.
2. Use learned model `\hat{f}(x_t, u_t)` inside planning (e.g., CEM, shooting) or inside imagination rollouts (e.g., MBPO, PETS-style).  
   - PETS-style = ensemble residual for epistemic uncertainty.  
   - MBPO-style = hybrid + uncertainty to bound rollout horizon.
3. Policy improvement objective: maximize expected return using synthetic rollouts under `\hat{f}`.

Model choice guidelines:
- If CRM accuracy is high and parameters are known, favor **hybrid dynamics** to minimize modeling error.
- If CRM predictions are unavailable or expensive at runtime, use **residual dynamics** trained on cached CRM outputs.
- When data is scarce, ensembles or uncertainty-aware variants help identify when the model leaves the training distribution.

### MPC and Optimal Control

- **Hybrid DynamicsModel** is the easiest drop-in for MPC:  
  ```
  x_{t+1} = HybridDynamicsModel.forward(x_t, u_t)
  cost = tracking_cost(x_{t+1}) + λ * residual_magnitude
  ```
  The built-in trajectory predictor accelerates shooting methods like iLQR, CEM, or random shooting.
- **Residual DynamicsModel** works with classical MPC pipelines if you can supply `x_{t+1}^{phys}` cheaply (e.g., precomputed CRM lookups). MPC then adds the residual correction to the baseline model.
- **FullDynamicsModel** (pure NN) remains a fallback when CRM is unavailable, but expect larger data requirements.
- **TransformerDynamicsModel** plugs into `MPCController` and `ModelBasedConfig(dynamics_model_type="transformer")` when you want longer context.
- **DiffusionDynamicsModel** pairs with stochastic MPC by sampling multiple rollouts; configure `dynamics_model_type="diffusion"` so MBPO/Dyna agents and MPC use the diffusion-aware world model.

### Implementation hooks

- `crm_ml_rl/training/train_dynamics.py`: set `TrainingConfig.model_type` to `"transformer"` or `"diffusion"` to train the new models. Diffusion training automatically swaps in the DDPM loss.
- `crm_ml_rl/training/model_based_rl.py`: choose `ModelBasedConfig.dynamics_model_type` ∈ {`"hybrid"`, `"full"`, `"transformer"`, `"diffusion"`}, optionally providing `transformer_kwargs` or a `DiffusionDynamicsConfig`.
- `crm_ml_rl/training/mpc_controller.py`: accepts any `torch.nn.Module` obeying `next_state = model(state, action)`—the new models slot in without changes. For diffusion you can sample (`deterministic=False`) to create stochastic shooting objectives.
- Experimental data: use `crm_ml_rl/data/experimental_loader.py` to load `data/experimental` currents/trajectories. The historical dataset flips the third current channel (`u[:,2]*=-1`) to match the MATLAB greybox convention; the loader applies this by default with `flip_third_current=True`.
- CRM validation: `crm_ml_rl/evaluation/cpp_validation.py` runs CRM FK/dynamics against the experimental trajectories and reports RMSE/MAE using `validation_metrics.py`.

### Decision Checklist

1. **Physics availability**  
   - Real-time CRM access → Hybrid models.  
   - Offline CRM logs only → Residual models.
2. **Need for simulator APIs (`reset`, `step`)**  
   - Yes → Hybrid dynamics/kinematics.  
   - No → Residual modules embedded inside your own simulator.
3. **Uncertainty requirements**  
   - Epistemic (model disagreement) → Ensemble residual/hybrid.  
   - Aleatoric (noise level) → `ResidualKinematicsWithUncertainty` or `ProbabilisticFullDynamics`.
4. **Computation budget**  
   - Tight real-time constraints → Residual models with cached physics predictions.  
   - Offline planning / training → Hybrids acceptable.

---

## Summary

- **Residual models** correct whatever physics signal you supply; they are modular, lightweight, and suitable when CRM outputs can be cached or streamed externally.
- **Hybrid models** own the physics call, stay synchronized with catheter parameters, and present simulator-like interfaces ideal for RL and MPC, at the cost of higher compute and tighter coupling to CRM bindings.
- **RL integration** hinges on how much structure you want: model-free RL can treat hybrids as the environment; model-based RL and MPC benefit from hybrid dynamics for fidelity or residual dynamics for modularity when CRM access is limited.

Choose the combination that balances physics fidelity, compute budget, and control strategy requirements. Document paths referenced above make it easy to dive deeper into the exact PyTorch implementations.
