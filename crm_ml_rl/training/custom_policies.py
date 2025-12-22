"""
Custom feature extractors and policies for RL agents.

This module provides SB3-compatible feature extractors that use the custom
neural network architectures from crm_ml_rl/models/.

Feature Extractors:
    - MLPFeaturesExtractor: Uses MLP with configurable architecture
    - DeepResidualFeaturesExtractor: Uses DeepResidualMLP for deeper networks
    - LSTMFeaturesExtractor: Uses LSTM_MLP for temporal/sequential features
    - PhysicsInformedExtractor: Includes CRM FK predictions as additional features
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, List, Type, Union
import gymnasium as gym

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from ..models.networks import MLP, LSTM_MLP, DeepResidualMLP, ResidualBlock


class MLPFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor using configurable MLP architecture.

    Wraps the MLP from crm_ml_rl/models/networks.py for use with SB3.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 128,
        hidden_dims: List[int] = None,
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
        layer_norm: bool = False
    ):
        """
        Initialize MLP feature extractor.

        Args:
            observation_space: Observation space
            features_dim: Output feature dimension
            hidden_dims: Hidden layer dimensions
            activation: Activation function name
            dropout: Dropout probability
            batch_norm: Use batch normalization
            layer_norm: Use layer normalization
        """
        super().__init__(observation_space, features_dim)

        if hidden_dims is None:
            hidden_dims = [256, 256]

        input_dim = int(np.prod(observation_space.shape))

        self.mlp = MLP(
            input_dim=input_dim,
            output_dim=features_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
            layer_norm=layer_norm
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Extract features from observations."""
        return self.mlp(observations)


class DeepResidualFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor using deep residual MLP.

    Better gradient flow for deeper networks through skip connections.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 128,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize deep residual feature extractor.

        Args:
            observation_space: Observation space
            features_dim: Output feature dimension
            hidden_dim: Hidden dimension in residual blocks
            num_blocks: Number of residual blocks
            dropout: Dropout probability
        """
        super().__init__(observation_space, features_dim)

        input_dim = int(np.prod(observation_space.shape))

        self.deep_mlp = DeepResidualMLP(
            input_dim=input_dim,
            output_dim=features_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout=dropout
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Extract features from observations."""
        return self.deep_mlp(observations)


class LSTMFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor using LSTM for temporal/sequential features.

    Maintains hidden state across steps for history-dependent feature extraction.
    Useful for environments with history_length > 1 or recurrent policies.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 128,
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 2,
        mlp_hidden_dims: List[int] = None,
        dropout: float = 0.1,
        sequence_length: int = 1
    ):
        """
        Initialize LSTM feature extractor.

        Args:
            observation_space: Observation space
            features_dim: Output feature dimension
            lstm_hidden_dim: LSTM hidden dimension
            lstm_num_layers: Number of LSTM layers
            mlp_hidden_dims: MLP hidden dimensions after LSTM
            dropout: Dropout probability
            sequence_length: Expected sequence length (for reshaping)
        """
        super().__init__(observation_space, features_dim)

        if mlp_hidden_dims is None:
            mlp_hidden_dims = [128, 64]

        self.sequence_length = sequence_length
        obs_shape = observation_space.shape

        # Calculate feature dimension per timestep
        if sequence_length > 1:
            self.feature_dim = int(np.prod(obs_shape)) // sequence_length
        else:
            self.feature_dim = int(np.prod(obs_shape))

        self.lstm_mlp = LSTM_MLP(
            input_dim=self.feature_dim,
            output_dim=features_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
            mlp_hidden_dims=mlp_hidden_dims,
            dropout=dropout
        )

        # Hidden state management
        self._hidden = None

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Extract features from observations."""
        batch_size = observations.shape[0]

        # Reshape for LSTM: (batch, seq, features)
        if self.sequence_length > 1:
            x = observations.view(batch_size, self.sequence_length, self.feature_dim)
        else:
            x = observations.view(batch_size, 1, self.feature_dim)

        # Forward through LSTM-MLP
        features, self._hidden = self.lstm_mlp(x, self._hidden)

        return features

    def reset_hidden(self, batch_size: int = 1):
        """Reset LSTM hidden state."""
        self._hidden = None


class PhysicsInformedExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that includes CRM physics predictions.

    Runs CRM forward kinematics to get physics-based features and
    concatenates them with learned features from observations.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 128,
        hidden_dims: List[int] = None,
        activation: str = "relu",
        use_cpp: bool = True,
        param_file: Optional[str] = None,
        config_file: Optional[str] = None,
        state_dim: int = 6,  # position + velocity
        action_dim: int = 3,  # currents in observation
    ):
        """
        Initialize physics-informed feature extractor.

        Args:
            observation_space: Observation space
            features_dim: Output feature dimension
            hidden_dims: Hidden layer dimensions
            activation: Activation function name
            use_cpp: Whether to use C++ physics bindings
            param_file: Path to catheter parameter file
            config_file: Path to catheter config file
            state_dim: State dimension in observation
            action_dim: Action dimension in observation
        """
        super().__init__(observation_space, features_dim)

        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_cpp = use_cpp

        # Initialize CRM wrapper for physics
        self.wrapper = None
        if use_cpp:
            try:
                from ..wrappers.crm_wrapper import CRMWrapper
                self.wrapper = CRMWrapper(
                    param_file=param_file,
                    config_file=config_file,
                    use_cpp=True
                )
            except ImportError:
                print("Warning: C++ bindings not available, physics features disabled")
                self.use_cpp = False

        # Input dimension: observation + physics features
        obs_dim = int(np.prod(observation_space.shape))
        physics_feature_dim = 6 if self.use_cpp else 0  # position (3) + jacobian info (3)
        total_input_dim = obs_dim + physics_feature_dim

        # Observation encoder
        self.obs_encoder = MLP(
            input_dim=obs_dim,
            output_dim=features_dim // 2,
            hidden_dims=hidden_dims,
            activation=activation
        )

        # Physics encoder (if available)
        if self.use_cpp:
            self.physics_encoder = MLP(
                input_dim=physics_feature_dim,
                output_dim=features_dim // 2,
                hidden_dims=[64, 64],
                activation=activation
            )

        # Combiner
        combiner_input = features_dim // 2 + (features_dim // 2 if self.use_cpp else 0)
        self.combiner = nn.Sequential(
            nn.Linear(combiner_input, features_dim),
            nn.ReLU()
        )

    def _get_physics_features(self, observations: torch.Tensor) -> torch.Tensor:
        """Extract physics features using CRM FK.

        Note: This loops over samples since C++ bindings don't support batch operations.
        For large batches, consider caching or using the non-physics extractor.
        """
        batch_size = observations.shape[0]
        device = observations.device

        if not self.use_cpp or self.wrapper is None:
            return torch.zeros(batch_size, 6, device=device)

        # Extract currents from observation (assumed to be last 3 dims)
        obs_np = observations.detach().cpu().numpy()
        currents = obs_np[:, -self.action_dim:]  # Last action_dim elements

        physics_features = []
        for i in range(batch_size):
            try:
                # Run forward kinematics
                result = self.wrapper.forward_kinematics(currents[i], insertion_length=94.3)
                tip_pos = result['tip_position']

                # Get Jacobian for additional physics info
                jac = self.wrapper.compute_jacobian(currents[i], insertion_length=94.3)
                jac_norm = np.linalg.norm(jac, axis=0)[:3]  # First 3 singular values

                physics_features.append(np.concatenate([tip_pos, jac_norm]))
            except Exception:
                physics_features.append(np.zeros(6))

        return torch.tensor(np.array(physics_features), dtype=torch.float32, device=device)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Extract features from observations with physics augmentation."""
        # Encode observations
        obs_features = self.obs_encoder(observations)

        if self.use_cpp:
            # Get physics features
            physics_feats = self._get_physics_features(observations)
            physics_encoded = self.physics_encoder(physics_feats)

            # Combine
            combined = torch.cat([obs_features, physics_encoded], dim=1)
        else:
            combined = obs_features

        return self.combiner(combined)


class CatheterMLPExtractor(BaseFeaturesExtractor):
    """
    Specialized feature extractor for catheter observations.

    Separates observation into semantic components (state, target, action)
    and processes each with dedicated encoders before combining.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 128,
        state_dim: int = 6,  # position + velocity
        target_dim: int = 3,  # target position
        state_hidden: List[int] = None,
        target_hidden: List[int] = None,
        activation: str = "relu"
    ):
        """
        Initialize catheter-specific feature extractor.

        Args:
            observation_space: Observation space
            features_dim: Output feature dimension
            state_dim: State dimension (position + velocity)
            target_dim: Target dimension
            state_hidden: Hidden dims for state encoder
            target_hidden: Hidden dims for target encoder
            activation: Activation function name
        """
        super().__init__(observation_space, features_dim)

        if state_hidden is None:
            state_hidden = [64, 64]
        if target_hidden is None:
            target_hidden = [32, 32]

        self.state_dim = state_dim
        self.target_dim = target_dim

        obs_dim = int(np.prod(observation_space.shape))
        remaining_dim = obs_dim - state_dim - target_dim

        # State encoder
        self.state_encoder = MLP(
            input_dim=state_dim,
            output_dim=64,
            hidden_dims=state_hidden,
            activation=activation
        )

        # Target encoder
        self.target_encoder = MLP(
            input_dim=target_dim,
            output_dim=32,
            hidden_dims=target_hidden,
            activation=activation
        )

        # Extra features encoder (action history, etc.)
        self.has_extra = remaining_dim > 0
        if self.has_extra:
            self.extra_encoder = nn.Sequential(
                nn.Linear(remaining_dim, 32),
                nn.ReLU()
            )
            combined_dim = 64 + 32 + 32
        else:
            combined_dim = 64 + 32

        # Combiner
        self.combiner = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Extract features from catheter observations."""
        # Split observation
        state = observations[:, :self.state_dim]
        target = observations[:, self.state_dim:self.state_dim + self.target_dim]

        # Encode
        state_features = self.state_encoder(state)
        target_features = self.target_encoder(target)

        if self.has_extra:
            extra = observations[:, self.state_dim + self.target_dim:]
            extra_features = self.extra_encoder(extra)
            combined = torch.cat([state_features, target_features, extra_features], dim=1)
        else:
            combined = torch.cat([state_features, target_features], dim=1)

        return self.combiner(combined)


# Registry of feature extractors
FEATURE_EXTRACTORS = {
    "default": None,  # Use SB3 default
    "mlp": MLPFeaturesExtractor,
    "deep_residual": DeepResidualFeaturesExtractor,
    "lstm": LSTMFeaturesExtractor,
    "physics": PhysicsInformedExtractor,
    "catheter": CatheterMLPExtractor,
}


def get_feature_extractor(
    name: str,
    observation_space: gym.spaces.Box,
    features_dim: int = 128,
    **kwargs
) -> Optional[Type[BaseFeaturesExtractor]]:
    """
    Get feature extractor class and kwargs by name.

    Args:
        name: Feature extractor name
        observation_space: Observation space
        features_dim: Output feature dimension
        **kwargs: Additional extractor-specific arguments

    Returns:
        Tuple of (extractor_class, extractor_kwargs) or (None, None) for default
    """
    if name not in FEATURE_EXTRACTORS:
        raise ValueError(f"Unknown feature extractor: {name}. "
                        f"Available: {list(FEATURE_EXTRACTORS.keys())}")

    extractor_class = FEATURE_EXTRACTORS[name]

    if extractor_class is None:
        return None, None

    extractor_kwargs = {
        "features_dim": features_dim,
        **kwargs
    }

    return extractor_class, extractor_kwargs


def create_policy_kwargs(
    feature_extractor: str = "default",
    features_dim: int = 128,
    net_arch: List[int] = None,
    activation_fn: Type[nn.Module] = nn.ReLU,
    algorithm: Optional[str] = None,
    **extractor_kwargs
) -> Dict:
    """
    Create policy kwargs dict for SB3 agents.

    Args:
        feature_extractor: Feature extractor name
        features_dim: Feature dimension
        net_arch: Network architecture after feature extractor
        activation_fn: Activation function class
        **extractor_kwargs: Additional extractor arguments

    Returns:
        Policy kwargs dict for SB3 agent initialization
    """
    if net_arch is None:
        net_arch = [256, 256]

    # SB3 >= 1.8 prefers dict-based net_arch; choose heads based on algorithm
    if isinstance(net_arch, list) and all(not isinstance(n, dict) for n in net_arch):
        algo = (algorithm or "").lower()
        if algo in {"sac", "td3"}:
            net_arch = {"pi": net_arch, "qf": net_arch}
        else:
            net_arch = {"pi": net_arch, "vf": net_arch}

    policy_kwargs = {
        "net_arch": net_arch,
        "activation_fn": activation_fn
    }

    if feature_extractor != "default":
        extractor_class = FEATURE_EXTRACTORS.get(feature_extractor)
        if extractor_class is not None:
            policy_kwargs["features_extractor_class"] = extractor_class
            policy_kwargs["features_extractor_kwargs"] = {
                "features_dim": features_dim,
                **extractor_kwargs
            }

    return policy_kwargs


if __name__ == "__main__":
    import gymnasium as gym

    print("Testing custom feature extractors...")

    # Create test observation space
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
    test_obs = torch.randn(32, 12)

    # Test MLP extractor
    print("\n1. Testing MLPFeaturesExtractor...")
    mlp_ext = MLPFeaturesExtractor(obs_space, features_dim=128)
    mlp_out = mlp_ext(test_obs)
    print(f"   Input: {test_obs.shape}, Output: {mlp_out.shape}")

    # Test Deep Residual extractor
    print("\n2. Testing DeepResidualFeaturesExtractor...")
    deep_ext = DeepResidualFeaturesExtractor(obs_space, features_dim=128, num_blocks=4)
    deep_out = deep_ext(test_obs)
    print(f"   Input: {test_obs.shape}, Output: {deep_out.shape}")

    # Test LSTM extractor
    print("\n3. Testing LSTMFeaturesExtractor...")
    lstm_ext = LSTMFeaturesExtractor(obs_space, features_dim=128)
    lstm_out = lstm_ext(test_obs)
    print(f"   Input: {test_obs.shape}, Output: {lstm_out.shape}")

    # Test Catheter extractor
    print("\n4. Testing CatheterMLPExtractor...")
    catheter_ext = CatheterMLPExtractor(obs_space, features_dim=128, state_dim=6, target_dim=3)
    catheter_out = catheter_ext(test_obs)
    print(f"   Input: {test_obs.shape}, Output: {catheter_out.shape}")

    # Test Physics extractor (without C++)
    print("\n5. Testing PhysicsInformedExtractor (CPU only)...")
    physics_ext = PhysicsInformedExtractor(obs_space, features_dim=128, use_cpp=False)
    physics_out = physics_ext(test_obs)
    print(f"   Input: {test_obs.shape}, Output: {physics_out.shape}")

    # Test policy kwargs creation
    print("\n6. Testing create_policy_kwargs...")
    for name in FEATURE_EXTRACTORS.keys():
        kwargs = create_policy_kwargs(name, features_dim=64)
        print(f"   {name}: {list(kwargs.keys())}")

    print("\nAll custom feature extractor tests passed!")
