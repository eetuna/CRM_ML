# CRM_ML Usage Guide: ML Models & RL Integration

This guide provides comprehensive usage instructions for the ML models and RL training pipeline in the CRM_ML project.

---

## Table of Contents

1. [ML Models Overview](#ml-models-overview)
2. [Using Individual ML Models](#using-individual-ml-models)
3. [Hybrid Models (CRM Physics + ML)](#hybrid-models-crm-physics--ml)
4. [RL Training with Custom Models](#rl-training-with-custom-models)
5. [Model-Based RL](#model-based-rl)
6. [Advanced Usage Examples](#advanced-usage-examples)
7. [Training and Evaluation](#training-and-evaluation)

---

## ML Models Overview

The `crm_ml_rl/models/` directory contains several types of neural network models:

### Base Networks
- **MLP**: Multi-layer perceptron with configurable architecture
- **LSTM_MLP**: LSTM followed by MLP for sequential data
- **DeepResidualMLP**: Deep MLP with residual connections
- **ResidualBlock**: Building block for residual networks

### Residual Models (Learn Corrections)
- **ResidualKinematicsModel**: Learns corrections to CRM FK predictions
- **ResidualDynamicsModel**: Learns corrections to CRM dynamics predictions
- **EnsembleResidualDynamics**: Ensemble for uncertainty estimation

### Full Models (Pure Neural Network)
- **FullKinematicsModel**: Complete NN-based forward kinematics
- **FullDynamicsModel**: Complete NN-based dynamics prediction
- **ProbabilisticFullDynamics**: With uncertainty estimation

### Hybrid Models (CRM Physics + Learned Residuals)
- **HybridKinematicsModel**: CRM FK + learned residual corrections
- **HybridDynamicsModel**: CRM dynamics + learned residual corrections
- **HybridDynamicsLSTM**: LSTM-based temporal corrections

---

## Using Individual ML Models

### 1. Basic MLP Network

```python
from crm_ml_rl.models import MLP
import torch

# Create MLP
model = MLP(
    input_dim=10,
    output_dim=3,
    hidden_dims=[256, 256, 128],
    activation="relu",
    dropout=0.1,
    batch_norm=False,
    layer_norm=True
)

# Forward pass
x = torch.randn(32, 10)  # batch of 32 samples
y = model(x)
print(y.shape)  # (32, 3)
```

### 2. LSTM-MLP for Sequential Data

```python
from crm_ml_rl.models import LSTM_MLP
import torch

# Create LSTM-MLP
model = LSTM_MLP(
    input_dim=6,
    output_dim=3,
    lstm_hidden_dim=128,
    lstm_num_layers=2,
    mlp_hidden_dims=[128, 64],
    dropout=0.1,
    bidirectional=False
)

# Forward pass with sequence
x = torch.randn(32, 10, 6)  # (batch, seq_len, features)
output, hidden = model(x)
print(output.shape)  # (32, 3)
```

### 3. Deep Residual MLP

```python
from crm_ml_rl.models import DeepResidualMLP
import torch

# Create deep residual network
model = DeepResidualMLP(
    input_dim=10,
    output_dim=3,
    hidden_dim=256,
    num_blocks=6,
    dropout=0.1
)

# Forward pass
x = torch.randn(32, 10)
y = model(x)
print(y.shape)  # (32, 3)
```

### 4. Residual Kinematics Model

```python
from crm_ml_rl.models import ResidualKinematicsModel
from crm_ml_rl.wrappers import CRMWrapper
import torch
import numpy as np

# Create residual kinematics model
model = ResidualKinematicsModel(
    input_dim=3,  # currents
    output_dim=6,  # position + orientation residuals
    hidden_dims=[256, 256, 128],
    max_correction=5.0
)

# Create physics engine
physics = CRMWrapper(
    param_file="catheterdata/CatheterParameterSet_1.txt",
    config_file="catheterdata/CatheterSpatialConfiguration_1.txt",
    use_cpp=True
)

# Predict with physics integration
currents = np.array([0.1, 0.0, 0.0])
currents_tensor = torch.tensor([currents], dtype=torch.float32)

# Get physics prediction
physics_result = physics.forward_kinematics(currents, insertion_length=50.0)
physics_pred = torch.tensor([physics_result['tip_position']], dtype=torch.float32)

# Get residual correction
residual = model(currents_tensor, physics_pred)

# Final prediction
final_prediction = physics_pred + residual
print(final_prediction.shape)  # (1, 6)
```

### 5. Full Dynamics Model

```python
from crm_ml_rl.models import FullDynamicsModel
import torch

# Create full dynamics model
model = FullDynamicsModel(
    state_dim=6,  # position + velocity
    action_dim=3,  # currents
    hidden_dims=[256, 256, 128],
    normalize_inputs=True,
    normalize_outputs=True
)

# Predict next state
state = torch.randn(32, 6)
action = torch.randn(32, 3)
next_state = model(state, action)
print(next_state.shape)  # (32, 6)

# Multi-step prediction
trajectory = model.predict_trajectory(state[0:1], action[:10])
print(trajectory.shape)  # (11, 6) - initial state + 10 steps
```

---

## Hybrid Models (CRM Physics + ML)

Hybrid models combine CRM C++ physics with learned neural network corrections for best accuracy.

### 1. Hybrid Kinematics Model

```python
from crm_ml_rl.models import HybridKinematicsModel, HybridKinematicsWithUncertainty
import torch
import numpy as np

# Create hybrid kinematics model
model = HybridKinematicsModel(
    param_file="catheterdata/CatheterParameterSet_1.txt",
    config_file="catheterdata/CatheterSpatialConfiguration_1.txt",
    use_cpp=True,
    hidden_dims=[256, 256],
    max_correction=3.0
)

# Forward pass
currents = torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float32)
insertion_length = 50.0

# Get prediction (physics + residual)
prediction = model.forward(currents, insertion_length)
print(prediction.shape)  # (1, 6)

# Get components separately
final, physics, residual = model.forward(
    currents, insertion_length, return_components=True
)
print(f"Physics: {physics[0]}")
print(f"Residual: {residual[0]}")
print(f"Final: {final[0]}")

# With uncertainty
uncertainty_model = HybridKinematicsWithUncertainty(
    param_file="catheterdata/CatheterParameterSet_1.txt",
    config_file="catheterdata/CatheterSpatialConfiguration_1.txt",
    use_cpp=True
)
prediction, uncertainty = uncertainty_model.forward(currents, insertion_length)
print(f"Prediction: {prediction.shape}, Uncertainty: {uncertainty.shape}")
```

### 2. Hybrid Dynamics Model

```python
from crm_ml_rl.models import HybridDynamicsModel, HybridDynamicsConfig
import torch
import numpy as np

# Configure hybrid dynamics
config = HybridDynamicsConfig(
    param_file="catheterdata/CatheterParameterSet_1.txt",
    config_file="catheterdata/CatheterSpatialConfiguration_1.txt",
    use_cpp=True,
    dt=0.02,
    insertion_length=50.0,
    hidden_dims=[256, 256, 128],
    max_correction=5.0,
    use_ensemble=False
)

# Create model
model = HybridDynamicsModel(config, device="cpu")

# Reset to initial state
initial_state = model.reset(
    initial_currents=np.zeros(3),
    insertion_length=50.0
)
print(f"Initial state: {initial_state}")  # (6,) - position + velocity

# Step forward
action = np.array([0.1, 0.0, 0.0])
next_state = model.step(action)
print(f"Next state: {next_state}")  # (6,)

# Predict trajectory
actions = np.random.randn(50, 3) * 0.1  # 50 timesteps
trajectory = model.predict_trajectory(actions)
print(f"Trajectory shape: {trajectory.shape}")  # (51, 6) - includes initial state

# Batch forward (for training)
batch_states = torch.randn(32, 6)
batch_actions = torch.randn(32, 3)
batch_next = model.forward(batch_states, batch_actions)
print(f"Batch output: {batch_next.shape}")  # (32, 6)
```

### 3. Training Hybrid Models

```python
from crm_ml_rl.models import HybridDynamicsModel, HybridDynamicsConfig
import torch
import torch.nn as nn

# Create model
config = HybridDynamicsConfig(use_cpp=True)
model = HybridDynamicsModel(config)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(100):
    # Sample batch (states, actions, next_states)
    states = torch.randn(64, 6)
    actions = torch.randn(64, 3)
    next_states = torch.randn(64, 6)

    # Forward pass
    optimizer.zero_grad()
    losses = model.compute_loss(states, actions, next_states)

    # Backward pass
    losses['total'].backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}:")
        print(f"  Total loss: {losses['total'].item():.4f}")
        print(f"  Physics baseline: {losses['physics'].item():.4f}")
        print(f"  Improvement: {losses['improvement'].item():.4f}")

# Save model
torch.save(model.state_dict(), "hybrid_dynamics.pt")

# Load model
model.load_state_dict(torch.load("hybrid_dynamics.pt"))
```

---

## RL Training with Custom Models

The RL agents now support custom feature extractors from your ML models.

### 1. Using Custom Feature Extractors

```python
from crm_ml_rl.training import SACAgent, PPOAgent, TD3Agent
from crm_ml_rl.envs import ReachingEnv

env = ReachingEnv()

# Option 1: MLP Feature Extractor
agent = SACAgent(
    env,
    feature_extractor="mlp",
    features_dim=128,
    extractor_kwargs={
        "hidden_dims": [256, 256],
        "dropout": 0.1,
        "activation": "relu"
    },
    learning_rate=3e-4,
    verbose=1
)

# Option 2: Deep Residual Feature Extractor
agent = PPOAgent(
    env,
    feature_extractor="deep_residual",
    features_dim=128,
    extractor_kwargs={
        "hidden_dim": 256,
        "num_blocks": 6,
        "dropout": 0.1
    }
)

# Option 3: LSTM Feature Extractor (for temporal features)
agent = TD3Agent(
    env,
    feature_extractor="lstm",
    features_dim=128,
    extractor_kwargs={
        "lstm_hidden_dim": 128,
        "lstm_num_layers": 2,
        "mlp_hidden_dims": [128, 64]
    }
)

# Option 4: Physics-Informed Extractor
agent = SACAgent(
    env,
    feature_extractor="physics",
    features_dim=128,
    extractor_kwargs={
        "use_cpp": True,
        "param_file": "catheterdata/CatheterParameterSet_1.txt",
        "config_file": "catheterdata/CatheterSpatialConfiguration_1.txt"
    }
)

# Option 5: Catheter-Specific Extractor
agent = PPOAgent(
    env,
    feature_extractor="catheter",
    features_dim=128,
    extractor_kwargs={
        "state_dim": 6,
        "target_dim": 3
    }
)
```

### 2. Training RL Agents

```python
from crm_ml_rl.training import SACAgent
from crm_ml_rl.envs import ReachingEnv

# Create environment
env = ReachingEnv()
eval_env = ReachingEnv()

# Create agent with custom extractor
agent = SACAgent(
    env,
    feature_extractor="deep_residual",
    features_dim=256,
    extractor_kwargs={"num_blocks": 4},
    learning_rate=3e-4,
    buffer_size=1_000_000,
    batch_size=256
)

# Train
results = agent.train(
    total_timesteps=500_000,
    eval_env=eval_env,
    eval_freq=10_000,
    n_eval_episodes=10,
    save_path="trained_models/sac_deep_residual",
    log_freq=1000
)

# Save agent
agent.save("trained_models/sac_final")

# Load agent
agent.load("trained_models/sac_final")

# Evaluate
obs, _ = env.reset()
for _ in range(1000):
    action = agent.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### 3. Training Configuration

```python
from crm_ml_rl.training import RLTrainingConfig, train_rl_agent

# Configure training
config = RLTrainingConfig(
    # Algorithm
    algorithm="sac",  # "sac", "ppo", "td3"

    # Environment
    env_type="reaching",  # "reaching", "tracking"
    max_steps=200,
    dt=0.02,

    # Training
    total_timesteps=500_000,
    n_envs=4,
    eval_freq=10_000,
    save_freq=50_000,

    # Hyperparameters
    learning_rate=3e-4,
    batch_size=256,
    gamma=0.99,

    # Feature Extractor
    feature_extractor="deep_residual",
    features_dim=256,
    extractor_num_blocks=6,
    extractor_dropout=0.1,

    # Normalization
    normalize_obs=True,
    normalize_reward=True
)

# Train
results = train_rl_agent(
    config=config,
    output_dir="trained_models/rl",
    seed=42,
    device="auto"
)

print(f"Training complete! Saved to: {results['run_dir']}")
```

### 4. Using Hybrid Dynamics in RL Environment

```python
from crm_ml_rl.envs import CatheterEnv, CatheterEnvConfig

# Configure environment with hybrid dynamics
env_config = CatheterEnvConfig(
    dt=0.02,
    max_steps=200,
    use_hybrid_dynamics=True,  # Enable hybrid model
    use_cpp=True,
    param_file="catheterdata/CatheterParameterSet_1.txt",
    config_file="catheterdata/CatheterSpatialConfiguration_1.txt",
    insertion_length=50.0
)

# Create environment
env = CatheterEnv(config=env_config)

# Reset and step
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

# The environment now uses HybridDynamicsModel internally!
```

---

## Model-Based RL

Use learned dynamics models for planning and imagination.

### 1. Dyna Agent (Real + Imagined Experience)

```python
from crm_ml_rl.training import DynaAgent, ModelBasedConfig
from crm_ml_rl.envs import ReachingEnv

# Configure model-based RL
mb_config = ModelBasedConfig(
    dynamics_model_type="hybrid",
    use_cpp_physics=True,
    param_file="catheterdata/CatheterParameterSet_1.txt",
    config_file="catheterdata/CatheterSpatialConfiguration_1.txt",
    imagined_rollout_horizon=10,
    imagined_data_ratio=0.5,
    model_update_freq=1000
)

# Create Dyna agent
env = ReachingEnv()
agent = DynaAgent(
    env,
    config=mb_config,
    base_algorithm="sac",
    feature_extractor="deep_residual",
    features_dim=128,
    learning_rate=3e-4,
    verbose=1
)

# Train (uses both real and imagined data)
results = agent.train(
    total_timesteps=500_000,
    eval_env=ReachingEnv(),
    eval_freq=10_000,
    save_path="trained_models/dyna"
)

print(f"Real steps: {results['total_real_steps']}")
print(f"Imagined steps: {results['total_imagined_steps']}")
```

### 2. MBPO Agent (Model-Based Policy Optimization)

```python
from crm_ml_rl.training import MBPOAgent, ModelBasedConfig
from crm_ml_rl.models import HybridDynamicsModel, HybridDynamicsConfig

# Create pre-trained dynamics model
dynamics_config = HybridDynamicsConfig(use_cpp=True)
dynamics_model = HybridDynamicsModel(dynamics_config)
# ... (load trained weights)

# Create MBPO agent
mb_config = ModelBasedConfig(
    imagined_rollout_horizon=10,
    model_train_epochs=10
)

env = ReachingEnv()
agent = MBPOAgent(
    env,
    dynamics_model=dynamics_model,  # Pre-trained model
    config=mb_config,
    feature_extractor="mlp",
    learning_rate=3e-4
)

# Train (short-horizon model rollouts)
results = agent.train(
    total_timesteps=500_000,
    eval_env=ReachingEnv(),
    save_path="trained_models/mbpo"
)
```

### 3. MPC Agent (Model Predictive Control)

```python
from crm_ml_rl.training import MPCAgent, ModelBasedConfig
from crm_ml_rl.models import HybridDynamicsModel, HybridDynamicsConfig

# Create dynamics model
dynamics_config = HybridDynamicsConfig(use_cpp=True)
dynamics_model = HybridDynamicsModel(dynamics_config)
# ... (load trained weights)

# Create MPC agent
mpc_config = ModelBasedConfig(
    mpc_horizon=20,
    mpc_num_samples=500,
    mpc_temperature=1.0
)

env = ReachingEnv()
agent = MPCAgent(
    env,
    dynamics_model=dynamics_model,
    config=mpc_config
)

# Use MPC for planning
obs, _ = env.reset()
action = agent.predict(obs)  # Plans optimal action sequence
obs, reward, term, trunc, info = env.step(action)
```

---

## Advanced Usage Examples

### 1. Ensemble Models for Uncertainty

```python
from crm_ml_rl.models import EnsembleResidualDynamics, HybridDynamicsModel, HybridDynamicsConfig
import torch

# Ensemble residual dynamics
ensemble = EnsembleResidualDynamics(
    state_dim=6,
    action_dim=3,
    hidden_dims=[256, 256],
    num_models=5
)

state = torch.randn(32, 6)
action = torch.randn(32, 3)
physics_pred = torch.randn(32, 6)

mean, std = ensemble(state, action, physics_pred)
print(f"Mean: {mean.shape}, Std: {std.shape}")  # (32, 6), (32, 6)

# Hybrid dynamics with ensemble
config = HybridDynamicsConfig(
    use_ensemble=True,
    num_ensemble=5,
    use_cpp=True
)
hybrid_ensemble = HybridDynamicsModel(config)

# Get uncertainty estimates
uncertainty = hybrid_ensemble.get_uncertainty(state, action)
print(f"Uncertainty: {uncertainty.shape}")  # (32, 6)
```

### 2. Custom Training Loop

```python
from crm_ml_rl.training import SACAgent
from crm_ml_rl.envs import ReachingEnv
import numpy as np

env = ReachingEnv()
agent = SACAgent(
    env,
    feature_extractor="deep_residual",
    features_dim=256
)

# Custom training loop
obs, _ = env.reset()
episode_rewards = []
current_reward = 0

for step in range(100_000):
    # Select action
    action, _ = agent.model.predict(obs, deterministic=False)

    # Environment step
    next_obs, reward, terminated, truncated, info = env.step(action)
    current_reward += reward

    # Store transition
    agent.model.replay_buffer.add(
        obs, next_obs, action, reward,
        terminated or truncated, [info]
    )

    # Train
    if agent.model.replay_buffer.size() > agent.model.batch_size:
        agent.model.train(gradient_steps=1)

    # Reset if done
    if terminated or truncated:
        episode_rewards.append(current_reward)
        current_reward = 0
        obs, _ = env.reset()
    else:
        obs = next_obs

    # Logging
    if step % 1000 == 0 and len(episode_rewards) > 0:
        mean_reward = np.mean(episode_rewards[-10:])
        print(f"Step {step}, Mean reward (last 10): {mean_reward:.2f}")

agent.save("trained_models/custom_trained")
```

### 3. Transfer Learning

```python
from crm_ml_rl.models import HybridDynamicsModel, HybridDynamicsConfig
import torch

# Train on Task A
config_A = HybridDynamicsConfig(use_cpp=True)
model = HybridDynamicsModel(config_A)
# ... train on task A ...
torch.save(model.state_dict(), "task_A_model.pt")

# Transfer to Task B
config_B = HybridDynamicsConfig(use_cpp=True)
model_B = HybridDynamicsModel(config_B)

# Load weights
state_dict = torch.load("task_A_model.pt")
model_B.load_state_dict(state_dict, strict=False)

# Fine-tune on Task B with lower learning rate
optimizer = torch.optim.Adam(model_B.parameters(), lr=1e-4)
# ... fine-tune ...
```

---

## Training and Evaluation

### Running Tests

```bash
# Test all ML models
python3 scripts/test_ml_models.py

# Quick test
python3 scripts/test_ml_models.py --quick

# Test RL integration
python3 scripts/test_rl_integration.py

# Test hybrid models
python3 scripts/test_hybrid_models.py
```

### Command-Line Training

```bash
# Train SAC with deep residual extractor
python3 crm_ml_rl/training/train_rl.py \
    --algorithm sac \
    --env-type reaching \
    --total-timesteps 500000 \
    --n-envs 4 \
    --output-dir trained_models

# Evaluate trained model
python3 crm_ml_rl/training/train_rl.py \
    --evaluate trained_models/sac_reaching_20231201_120000/final_model \
    --env-type reaching
```

### Monitoring Training

```bash
# View tensorboard logs (if using stable-baselines3 logging)
tensorboard --logdir trained_models/
```

---

## Summary

### Key Takeaways

1. **ML Models**: Use `crm_ml_rl/models/` for physics-based or pure neural models
2. **Hybrid Models**: Combine CRM C++ physics with learned residuals for best accuracy
3. **RL Integration**: All agents support custom feature extractors from your models
4. **Model-Based RL**: Use Dyna, MBPO, or MPC with learned dynamics
5. **Testing**: Comprehensive test suites verify all integrations (104 tests passing)

### File Locations

- **Models**: `crm_ml_rl/models/*.py`
- **RL Agents**: `crm_ml_rl/training/rl_agents.py`
- **Feature Extractors**: `crm_ml_rl/training/custom_policies.py`
- **Model-Based RL**: `crm_ml_rl/training/model_based_rl.py`
- **Environments**: `crm_ml_rl/envs/*.py`
- **Tests**: `scripts/test_*.py`

### Getting Help

- Check test files for more examples: `scripts/test_ml_models.py`, `scripts/test_rl_integration.py`
- Review source code docstrings for detailed API documentation
- See `CLAUDE.md` for project architecture overview
- See `next_steps.md` for recent updates and completed features

---

**Last Updated**: 2025-12-11
