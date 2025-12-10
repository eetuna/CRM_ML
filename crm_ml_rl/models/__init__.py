"""
Neural network models for CRM ML/RL.

Models:
    - ResidualKinematicsModel: Learns residual corrections for FK
    - ResidualDynamicsModel: Learns residual corrections for dynamics
    - FullKinematicsModel: Full neural network FK model
    - FullDynamicsModel: Full neural network dynamics model
    - HybridKinematicsModel: Integrates CRM FK with learned residuals
    - HybridDynamicsModel: Integrates CRM dynamics with learned residuals
"""

from .networks import MLP, LSTM_MLP
from .residual_kinematics import ResidualKinematicsModel, ResidualKinematicsWithUncertainty
from .residual_dynamics import ResidualDynamicsModel, EnsembleResidualDynamics
from .full_kinematics import FullKinematicsModel
from .full_dynamics import FullDynamicsModel
from .hybrid_kinematics import HybridKinematicsModel, HybridKinematicsWithUncertainty
from .hybrid_dynamics import HybridDynamicsModel, HybridDynamicsConfig, HybridDynamicsLSTM
from .sequence_models import (
    TransformerDynamicsModel,
    DiffusionDynamicsModel,
    DiffusionDynamicsConfig
)

__all__ = [
    'MLP',
    'LSTM_MLP',
    'ResidualKinematicsModel',
    'ResidualKinematicsWithUncertainty',
    'ResidualDynamicsModel',
    'EnsembleResidualDynamics',
    'FullKinematicsModel',
    'FullDynamicsModel',
    'HybridKinematicsModel',
    'HybridKinematicsWithUncertainty',
    'HybridDynamicsModel',
    'HybridDynamicsConfig',
    'HybridDynamicsLSTM',
    'TransformerDynamicsModel',
    'DiffusionDynamicsModel',
    'DiffusionDynamicsConfig',
]
