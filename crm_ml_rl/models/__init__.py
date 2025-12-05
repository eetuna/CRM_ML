"""
Neural network models for CRM ML/RL.

Models:
    - ResidualKinematicsModel: Learns residual corrections for FK
    - ResidualDynamicsModel: Learns residual corrections for dynamics
    - FullKinematicsModel: Full neural network FK model
    - FullDynamicsModel: Full neural network dynamics model
"""

from .networks import MLP, LSTM_MLP
from .residual_kinematics import ResidualKinematicsModel
from .residual_dynamics import ResidualDynamicsModel
from .full_kinematics import FullKinematicsModel
from .full_dynamics import FullDynamicsModel

__all__ = [
    'MLP',
    'LSTM_MLP',
    'ResidualKinematicsModel',
    'ResidualDynamicsModel',
    'FullKinematicsModel',
    'FullDynamicsModel',
]
