
import sys
from pathlib import Path
import pytest
import numpy as np
import torch

# Add project root to path to allow importing from crm_ml_rl
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from crm_ml_rl.envs.catheter_env import CatheterEnv, CatheterEnvConfig
from crm_ml_rl.models.networks import MLP, LSTM_MLP, DeepResidualMLP
from crm_ml_rl.training.mpc_controller import MPCController, MPCConfig, LinearMPC

@pytest.fixture
def default_env():
    """Fixture for a default CatheterEnv."""
    return CatheterEnv()

@pytest.fixture
def custom_env():
    """Fixture for a CatheterEnv with a custom configuration."""
    config = CatheterEnvConfig(
        max_steps=50,
        include_velocity=False,
        history_length=2,
        use_cpp=False 
    )
    return CatheterEnv(config=config)

def test_env_initialization(default_env):
    """Test initialization of the default CatheterEnv."""
    assert default_env is not None
    assert default_env.config.max_steps == 200
    assert default_env.action_space.shape == (3,)
    assert default_env.observation_space.shape is not None

def test_custom_env_initialization(custom_env):
    """Test initialization with a custom configuration."""
    assert custom_env.config.max_steps == 50
    assert not custom_env.config.include_velocity
    assert custom_env.config.history_length == 2
    # obs_dim = (pos) + (target) + (current_action) = 3 + 3 + 3 = 9
    # history_length = 2, so obs_dim = 9 * 2 = 18
    assert custom_env.observation_space.shape == (18,)

def test_reset_method(default_env):
    """Test the reset method."""
    obs, info = default_env.reset()
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)
    assert 'target_position' in info
    assert obs.shape == default_env.observation_space.shape

def test_step_method(default_env):
    """Test the step method."""
    default_env.reset()
    action = default_env.action_space.sample()
    obs, reward, terminated, truncated, info = default_env.step(action)
    
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert obs.shape == default_env.observation_space.shape

def test_observation_and_action_spaces(default_env):
    """Test the observation and action spaces."""
    assert default_env.action_space.low.shape == (3,)
    assert default_env.action_space.high.shape == (3,)
    assert all(default_env.action_space.low == -default_env.config.max_current)
    assert all(default_env.action_space.high == default_env.config.max_current)
    
    # obs_dim = (pos) + (vel) + (target) + (current_action) = 3 + 3 + 3 + 3 = 12
    # history_length = 1, so obs_dim = 12
    assert default_env.observation_space.shape == (12,)
    

# Tests for networks.py

def test_mlp_forward():
    """Test the forward pass of the MLP."""
    mlp = MLP(input_dim=10, output_dim=3, hidden_dims=[32, 32])
    input_tensor = torch.randn(64, 10)
    output_tensor = mlp(input_tensor)
    assert output_tensor.shape == (64, 3)

def test_lstm_mlp_forward():
    """Test the forward pass of the LSTM_MLP."""
    lstm_mlp = LSTM_MLP(input_dim=5, output_dim=2, lstm_hidden_dim=16)
    input_tensor = torch.randn(32, 10, 5)  # (batch, seq_len, input_dim)
    output_tensor, _ = lstm_mlp(input_tensor)
    assert output_tensor.shape == (32, 2)

def test_deep_residual_mlp_forward():
    """Test the forward pass of the DeepResidualMLP."""
    deep_mlp = DeepResidualMLP(input_dim=20, output_dim=5, hidden_dim=64, num_blocks=2)
    input_tensor = torch.randn(128, 20)
    output_tensor = deep_mlp(input_tensor)
    assert output_tensor.shape == (128, 5)

# Tests for mpc_controller.py

@pytest.fixture
def mpc_controller():
    """Fixture for a default MPCController."""
    config = MPCConfig(horizon=5, max_iter=5)
    return MPCController(config=config)

@pytest.fixture
def linear_mpc_controller():
    """Fixture for a LinearMPC controller."""
    config = MPCConfig(horizon=5)
    mpc = LinearMPC(config)
    
    # Simple linear dynamics
    A = np.eye(6)
    A[:3, 3:6] = np.eye(3) * config.dt
    B = np.zeros((6, 3))
    B[3:6, :] = np.eye(3) * config.dt * 100
    mpc.set_dynamics(A,B)

    return mpc

def test_mpc_controller_init(mpc_controller):
    """Test initialization of the MPCController."""
    assert mpc_controller is not None
    assert mpc_controller.config.horizon == 5

@pytest.mark.parametrize("method", ["shooting", "cem", "mppi"])
def test_mpc_control_methods(mpc_controller, method):
    """Test the different control methods of the MPCController."""
    current_state = np.array([0.0, 0.0, 80.0, 0.0, 0.0, 0.0])
    target_trajectory = np.zeros((mpc_controller.config.horizon + 1, 6))
    target_trajectory[:, :3] = np.array([10.0, 5.0, 85.0])
    
    action, info = mpc_controller.control(current_state, target_trajectory, method=method)
    
    assert isinstance(action, np.ndarray)
    assert action.shape == (3,)
    assert isinstance(info, dict)
    assert 'cost' in info

def test_mpc_reset(mpc_controller):
    """Test the reset method of the MPCController."""
    mpc_controller.prev_solution = np.random.rand(mpc_controller.config.horizon, mpc_controller.config.action_dim)
    mpc_controller.reset()
    assert mpc_controller.prev_solution is None

def test_linear_mpc_solve(linear_mpc_controller):
    """Test the solve method of the LinearMPC."""
    current_state = np.array([0.0, 0.0, 80.0, 0.0, 0.0, 0.0])
    target = np.array([10.0, 5.0, 85.0, 0.0, 0.0, 0.0])
    
    try:
        action, info = linear_mpc_controller.solve(current_state, target)
        assert isinstance(action, np.ndarray)
        assert action.shape == (3,)
        assert isinstance(info, dict)
        assert 'cost' in info
    except ImportError:
        pytest.skip("CVXPY not available")

