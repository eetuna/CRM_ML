"""
CRM ML/RL Framework
===================

Machine Learning and Reinforcement Learning models for
MRI-actuated robotic catheter control.

Modules:
    - data: Data loading utilities for experimental and simulated data
    - models: Neural network architectures (residual and full models)
    - envs: Gymnasium environments for RL training
    - wrappers: C++ to Python bindings for physics engine
    - training: Training scripts, RL agents (SAC, PPO, TD3), and MPC
    - evaluation: Model evaluation and visualization tools

Example Usage:
    # Train a dynamics model
    from crm_ml_rl.training.train_dynamics import train_dynamics_model
    trainer, history = train_dynamics_model(
        data_dir="data/experimental",
        output_dir="trained_models",
        model_type="residual"
    )

    # Train an RL agent
    from crm_ml_rl.training.train_rl import train_rl_agent, RLTrainingConfig
    config = RLTrainingConfig(algorithm="sac", env_type="reaching")
    train_rl_agent(config, output_dir="trained_models/rl")
"""

__version__ = "0.1.0"
__author__ = "CRM_ML Team"

from . import data
from . import models
from . import envs
from . import wrappers
from . import training
from . import evaluation

__all__ = [
    'data',
    'models',
    'envs',
    'wrappers',
    'training',
    'evaluation',
]
