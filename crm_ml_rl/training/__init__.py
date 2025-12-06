"""
Training module for CRM ML/RL models.

Includes:
- RL agents (SAC, PPO, TD3) with custom feature extractor support
- Custom feature extractors for stable-baselines3
- Model-based RL agents (Dyna, MBPO, MPC)
- MPC controller
- Training utilities
"""

from .rl_agents import SACAgent, PPOAgent, TD3Agent
from .mpc_controller import MPCController
from .custom_policies import (
    MLPFeaturesExtractor,
    DeepResidualFeaturesExtractor,
    LSTMFeaturesExtractor,
    PhysicsInformedExtractor,
    CatheterMLPExtractor,
    FEATURE_EXTRACTORS,
    create_policy_kwargs,
    get_feature_extractor
)
from .model_based_rl import (
    DynaAgent,
    MBPOAgent,
    MPCAgent,
    ModelBasedConfig
)
from .train_rl import RLTrainingConfig, train_rl_agent, create_agent

__all__ = [
    # RL Agents
    'SACAgent',
    'PPOAgent',
    'TD3Agent',
    # Feature Extractors
    'MLPFeaturesExtractor',
    'DeepResidualFeaturesExtractor',
    'LSTMFeaturesExtractor',
    'PhysicsInformedExtractor',
    'CatheterMLPExtractor',
    'FEATURE_EXTRACTORS',
    'create_policy_kwargs',
    'get_feature_extractor',
    # Model-Based RL
    'DynaAgent',
    'MBPOAgent',
    'MPCAgent',
    'ModelBasedConfig',
    # Training
    'RLTrainingConfig',
    'train_rl_agent',
    'create_agent',
    # Controllers
    'MPCController',
]
