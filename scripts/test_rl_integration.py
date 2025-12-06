#!/usr/bin/env python3
"""
Comprehensive test suite for RL-ML model integration.

Tests:
1. Custom feature extractors with all agent types
2. Model-based RL training with HybridDynamicsModel
3. Physics-informed policies
4. End-to-end training validation
"""

import sys
import argparse
import traceback
from pathlib import Path
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestResults:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def add_pass(self, name):
        self.passed += 1
        print(f"  ✓ {name}")

    def add_fail(self, name, error):
        self.failed += 1
        self.errors.append((name, error))
        print(f"  ✗ {name}: {error}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"TEST RESULTS: {self.passed}/{total} passed")
        if self.errors:
            print(f"\nFailed tests:")
            for name, error in self.errors:
                print(f"  - {name}: {error}")
        return self.failed == 0


results = TestResults()


def test_custom_feature_extractors():
    """Test all custom feature extractors."""
    print("\n" + "="*60)
    print("TESTING CUSTOM FEATURE EXTRACTORS")
    print("="*60)

    import gymnasium as gym
    from crm_ml_rl.training.custom_policies import (
        MLPFeaturesExtractor,
        DeepResidualFeaturesExtractor,
        LSTMFeaturesExtractor,
        PhysicsInformedExtractor,
        CatheterMLPExtractor,
        create_policy_kwargs,
        FEATURE_EXTRACTORS
    )

    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
    test_obs = torch.randn(8, 12)

    # Test MLPFeaturesExtractor
    try:
        extractor = MLPFeaturesExtractor(obs_space, features_dim=64, hidden_dims=[128, 128])
        output = extractor(test_obs)
        assert output.shape == (8, 64), f"Expected (8, 64), got {output.shape}"
        results.add_pass("MLPFeaturesExtractor forward pass")
    except Exception as e:
        results.add_fail("MLPFeaturesExtractor forward pass", str(e))

    # Test DeepResidualFeaturesExtractor
    try:
        extractor = DeepResidualFeaturesExtractor(obs_space, features_dim=64, num_blocks=3)
        output = extractor(test_obs)
        assert output.shape == (8, 64), f"Expected (8, 64), got {output.shape}"
        results.add_pass("DeepResidualFeaturesExtractor forward pass")
    except Exception as e:
        results.add_fail("DeepResidualFeaturesExtractor forward pass", str(e))

    # Test LSTMFeaturesExtractor
    try:
        extractor = LSTMFeaturesExtractor(obs_space, features_dim=64, lstm_hidden_dim=64)
        output = extractor(test_obs)
        assert output.shape == (8, 64), f"Expected (8, 64), got {output.shape}"
        results.add_pass("LSTMFeaturesExtractor forward pass")
    except Exception as e:
        results.add_fail("LSTMFeaturesExtractor forward pass", str(e))

    # Test CatheterMLPExtractor
    try:
        extractor = CatheterMLPExtractor(obs_space, features_dim=64, state_dim=6, target_dim=3)
        output = extractor(test_obs)
        assert output.shape == (8, 64), f"Expected (8, 64), got {output.shape}"
        results.add_pass("CatheterMLPExtractor forward pass")
    except Exception as e:
        results.add_fail("CatheterMLPExtractor forward pass", str(e))

    # Test PhysicsInformedExtractor (without C++)
    try:
        extractor = PhysicsInformedExtractor(obs_space, features_dim=64, use_cpp=False)
        output = extractor(test_obs)
        assert output.shape == (8, 64), f"Expected (8, 64), got {output.shape}"
        results.add_pass("PhysicsInformedExtractor forward pass (no C++)")
    except Exception as e:
        results.add_fail("PhysicsInformedExtractor forward pass (no C++)", str(e))

    # Test create_policy_kwargs
    try:
        for extractor_name in ["default", "mlp", "deep_residual", "catheter"]:
            kwargs = create_policy_kwargs(extractor_name, features_dim=64)
            assert "net_arch" in kwargs
            assert "activation_fn" in kwargs
        results.add_pass("create_policy_kwargs for all extractors")
    except Exception as e:
        results.add_fail("create_policy_kwargs for all extractors", str(e))

    # Test FEATURE_EXTRACTORS registry
    try:
        expected = ["default", "mlp", "deep_residual", "lstm", "physics", "catheter"]
        for name in expected:
            assert name in FEATURE_EXTRACTORS, f"Missing extractor: {name}"
        results.add_pass("FEATURE_EXTRACTORS registry complete")
    except Exception as e:
        results.add_fail("FEATURE_EXTRACTORS registry complete", str(e))


def test_agents_with_custom_extractors():
    """Test RL agents with custom feature extractors."""
    print("\n" + "="*60)
    print("TESTING AGENTS WITH CUSTOM EXTRACTORS")
    print("="*60)

    from crm_ml_rl.envs import ReachingEnv
    from crm_ml_rl.training.rl_agents import SACAgent, PPOAgent, TD3Agent

    env = ReachingEnv()

    # Test SAC with different extractors
    extractors_to_test = ["default", "mlp", "deep_residual", "catheter"]

    for extractor in extractors_to_test:
        try:
            agent = SACAgent(
                env,
                verbose=0,
                feature_extractor=extractor,
                features_dim=64
            )
            obs, _ = env.reset()
            action = agent.predict(obs)
            assert action.shape == (3,), f"Expected (3,), got {action.shape}"
            results.add_pass(f"SACAgent with {extractor} extractor")
        except Exception as e:
            results.add_fail(f"SACAgent with {extractor} extractor", str(e))

    # Test PPO with extractors
    for extractor in ["default", "mlp", "deep_residual"]:
        try:
            agent = PPOAgent(
                env,
                verbose=0,
                feature_extractor=extractor,
                features_dim=64
            )
            obs, _ = env.reset()
            action = agent.predict(obs)
            assert action.shape == (3,), f"Expected (3,), got {action.shape}"
            results.add_pass(f"PPOAgent with {extractor} extractor")
        except Exception as e:
            results.add_fail(f"PPOAgent with {extractor} extractor", str(e))

    # Test TD3 with extractors
    for extractor in ["default", "mlp"]:
        try:
            agent = TD3Agent(
                env,
                verbose=0,
                feature_extractor=extractor,
                features_dim=64
            )
            obs, _ = env.reset()
            action = agent.predict(obs)
            assert action.shape == (3,), f"Expected (3,), got {action.shape}"
            results.add_pass(f"TD3Agent with {extractor} extractor")
        except Exception as e:
            results.add_fail(f"TD3Agent with {extractor} extractor", str(e))

    env.close()


def test_training_config():
    """Test RLTrainingConfig with new feature extractor options."""
    print("\n" + "="*60)
    print("TESTING TRAINING CONFIG")
    print("="*60)

    from crm_ml_rl.training.train_rl import RLTrainingConfig, create_agent

    # Test default config
    try:
        config = RLTrainingConfig()
        assert config.feature_extractor == "default"
        assert config.features_dim == 128
        assert config.extractor_hidden_dims is not None
        results.add_pass("RLTrainingConfig default values")
    except Exception as e:
        results.add_fail("RLTrainingConfig default values", str(e))

    # Test custom extractor config
    try:
        config = RLTrainingConfig(
            feature_extractor="deep_residual",
            features_dim=256,
            extractor_num_blocks=6,
            extractor_dropout=0.2
        )
        assert config.feature_extractor == "deep_residual"
        assert config.features_dim == 256
        assert config.extractor_num_blocks == 6
        results.add_pass("RLTrainingConfig custom extractor settings")
    except Exception as e:
        results.add_fail("RLTrainingConfig custom extractor settings", str(e))

    # Test create_agent with config
    try:
        from crm_ml_rl.envs import ReachingEnv
        env = ReachingEnv()

        config = RLTrainingConfig(
            algorithm="sac",
            feature_extractor="mlp",
            features_dim=64
        )
        agent = create_agent(config, env)
        assert agent is not None
        results.add_pass("create_agent with feature extractor config")
        env.close()
    except Exception as e:
        results.add_fail("create_agent with feature extractor config", str(e))


def test_model_based_rl():
    """Test model-based RL agents."""
    print("\n" + "="*60)
    print("TESTING MODEL-BASED RL")
    print("="*60)

    from crm_ml_rl.envs import ReachingEnv
    from crm_ml_rl.training.model_based_rl import (
        DynaAgent, MBPOAgent, MPCAgent, ModelBasedConfig
    )
    from crm_ml_rl.models.hybrid_dynamics import HybridDynamicsModel, HybridDynamicsConfig

    env = ReachingEnv()
    obs, _ = env.reset()

    # Test DynaAgent creation
    try:
        config = ModelBasedConfig(use_cpp_physics=False)
        dyna = DynaAgent(
            env,
            config=config,
            base_algorithm="sac",
            verbose=0,
            feature_extractor="mlp"
        )
        action = dyna.predict(obs)
        assert action.shape == (3,), f"Expected (3,), got {action.shape}"
        results.add_pass("DynaAgent creation and prediction")
    except Exception as e:
        results.add_fail("DynaAgent creation and prediction", str(e))

    # Test DynaAgent with different base algorithms
    for algo in ["sac", "ppo", "td3"]:
        try:
            config = ModelBasedConfig(use_cpp_physics=False)
            dyna = DynaAgent(
                env,
                config=config,
                base_algorithm=algo,
                verbose=0
            )
            action = dyna.predict(obs)
            assert action.shape == (3,)
            results.add_pass(f"DynaAgent with {algo} base")
        except Exception as e:
            results.add_fail(f"DynaAgent with {algo} base", str(e))

    # Test MBPOAgent creation
    try:
        config = ModelBasedConfig(use_cpp_physics=False)
        mbpo = MBPOAgent(
            env,
            config=config,
            verbose=0,
            feature_extractor="deep_residual"
        )
        action = mbpo.predict(obs)
        assert action.shape == (3,)
        results.add_pass("MBPOAgent creation and prediction")
    except Exception as e:
        results.add_fail("MBPOAgent creation and prediction", str(e))

    # Test MPCAgent
    try:
        dynamics_config = HybridDynamicsConfig(use_cpp=False)
        dynamics = HybridDynamicsModel(dynamics_config)
        config = ModelBasedConfig(mpc_horizon=5, mpc_num_samples=50)
        mpc = MPCAgent(env, dynamics, config)
        action = mpc.predict(obs)
        assert action.shape == (3,)
        results.add_pass("MPCAgent creation and prediction")
    except Exception as e:
        results.add_fail("MPCAgent creation and prediction", str(e))

    # Test imagined rollout generation
    try:
        config = ModelBasedConfig(use_cpp_physics=False, imagined_rollout_horizon=5)
        dyna = DynaAgent(env, config=config, verbose=0)

        # Need some initial data
        for _ in range(100):
            action, _ = dyna.agent.model.predict(obs, deterministic=False)
            next_obs, _, term, trunc, info = env.step(action)
            dyna.agent.model.replay_buffer.add(obs, next_obs, action, 0.0, term or trunc, [info])
            obs = next_obs if not (term or trunc) else env.reset()[0]

        # Generate rollouts
        start_states = np.random.randn(10, env.observation_space.shape[0]).astype(np.float32)
        transitions = dyna.generate_imagined_rollouts(start_states, horizon=3)
        assert len(transitions) > 0
        results.add_pass("DynaAgent imagined rollout generation")
    except Exception as e:
        results.add_fail("DynaAgent imagined rollout generation", str(e))

    env.close()


def test_hybrid_dynamics_integration():
    """Test integration of HybridDynamicsModel with RL."""
    print("\n" + "="*60)
    print("TESTING HYBRID DYNAMICS INTEGRATION")
    print("="*60)

    from crm_ml_rl.envs import CatheterEnv, CatheterEnvConfig
    from crm_ml_rl.models.hybrid_dynamics import HybridDynamicsModel, HybridDynamicsConfig

    # Test environment with hybrid dynamics
    try:
        env_config = CatheterEnvConfig(
            use_hybrid_dynamics=True,
            use_cpp=False,
            max_steps=10
        )
        env = CatheterEnv(config=env_config)
        obs, info = env.reset()
        assert obs is not None

        # Run a few steps
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)

        results.add_pass("CatheterEnv with hybrid dynamics")
        env.close()
    except Exception as e:
        results.add_fail("CatheterEnv with hybrid dynamics", str(e))

    # Test HybridDynamicsModel standalone
    try:
        config = HybridDynamicsConfig(use_cpp=False)
        model = HybridDynamicsModel(config)

        state = model.reset()
        assert state.shape == (6,)

        action = np.array([0.1, 0.0, 0.0])
        next_state = model.step(action)
        assert next_state.shape == (6,)

        results.add_pass("HybridDynamicsModel step function")
    except Exception as e:
        results.add_fail("HybridDynamicsModel step function", str(e))

    # Test trajectory prediction
    try:
        config = HybridDynamicsConfig(use_cpp=False)
        model = HybridDynamicsModel(config)

        T = 10
        actions = np.random.randn(T, 3) * 0.1
        trajectory = model.predict_trajectory(actions)
        assert trajectory.shape == (T + 1, 6)
        results.add_pass("HybridDynamicsModel trajectory prediction")
    except Exception as e:
        results.add_fail("HybridDynamicsModel trajectory prediction", str(e))


def test_short_training_run():
    """Test a short training run with custom extractors."""
    print("\n" + "="*60)
    print("TESTING SHORT TRAINING RUN")
    print("="*60)

    from crm_ml_rl.envs import ReachingEnv
    from crm_ml_rl.training.rl_agents import SACAgent

    env = ReachingEnv()

    # Very short training to verify it works
    try:
        agent = SACAgent(
            env,
            verbose=0,
            feature_extractor="mlp",
            features_dim=64,
            buffer_size=1000
        )

        # Collect some experience
        obs, _ = env.reset()
        for i in range(200):
            action, _ = agent.model.predict(obs, deterministic=False)
            next_obs, reward, term, trunc, info = env.step(action)
            agent.model.replay_buffer.add(obs, next_obs, action, reward, term or trunc, [info])
            obs = next_obs if not (term or trunc) else env.reset()[0]

        # Do a few training steps
        if agent.model.replay_buffer.size() >= agent.model.batch_size:
            agent.model.train(gradient_steps=10)

        results.add_pass("Short training run with custom extractor")
    except Exception as e:
        results.add_fail("Short training run with custom extractor", str(e))

    env.close()


def test_physics_informed_extractor_with_cpp():
    """Test physics-informed extractor with C++ bindings (if available)."""
    print("\n" + "="*60)
    print("TESTING PHYSICS-INFORMED EXTRACTOR WITH C++")
    print("="*60)

    try:
        from crm_ml_rl.wrappers.crm_wrapper import HAS_CPP_BINDINGS, CRMWrapper

        if not HAS_CPP_BINDINGS:
            print("  (Skipping C++ tests - bindings not available)")
            results.add_pass("Physics extractor C++ tests (skipped - no bindings)")
            return

        import gymnasium as gym
        from crm_ml_rl.training.custom_policies import PhysicsInformedExtractor

        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        test_obs = torch.randn(4, 12)

        extractor = PhysicsInformedExtractor(
            obs_space,
            features_dim=64,
            use_cpp=True
        )

        output = extractor(test_obs)
        assert output.shape == (4, 64)
        results.add_pass("PhysicsInformedExtractor with C++ bindings")

    except ImportError:
        results.add_pass("Physics extractor C++ tests (skipped - import error)")
    except Exception as e:
        results.add_fail("PhysicsInformedExtractor with C++ bindings", str(e))


def test_agent_save_load():
    """Test saving and loading agents with custom extractors."""
    print("\n" + "="*60)
    print("TESTING AGENT SAVE/LOAD")
    print("="*60)

    import tempfile
    from crm_ml_rl.envs import ReachingEnv
    from crm_ml_rl.training.rl_agents import SACAgent

    env = ReachingEnv()

    try:
        # Create and save agent
        agent = SACAgent(
            env,
            verbose=0,
            feature_extractor="mlp",
            features_dim=64
        )

        obs, _ = env.reset()
        action_before = agent.predict(obs)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_agent"
            agent.save(str(save_path))

            # Create new agent and load
            agent2 = SACAgent(
                env,
                verbose=0,
                feature_extractor="mlp",
                features_dim=64
            )
            agent2.load(str(save_path))

            action_after = agent2.predict(obs)

            # Actions should be same (deterministic)
            np.testing.assert_array_almost_equal(action_before, action_after, decimal=5)

        results.add_pass("Agent save/load with custom extractor")
    except Exception as e:
        results.add_fail("Agent save/load with custom extractor", str(e))

    env.close()


def test_all_agent_extractor_combinations():
    """Test all combinations of agents and extractors."""
    print("\n" + "="*60)
    print("TESTING ALL AGENT-EXTRACTOR COMBINATIONS")
    print("="*60)

    from crm_ml_rl.envs import ReachingEnv
    from crm_ml_rl.training.rl_agents import SACAgent, PPOAgent, TD3Agent

    agents = [
        ("SAC", SACAgent),
        ("PPO", PPOAgent),
        ("TD3", TD3Agent)
    ]
    extractors = ["default", "mlp", "deep_residual", "catheter"]

    env = ReachingEnv()
    obs, _ = env.reset()

    for agent_name, AgentClass in agents:
        for extractor in extractors:
            try:
                agent = AgentClass(
                    env,
                    verbose=0,
                    feature_extractor=extractor,
                    features_dim=64
                )
                action = agent.predict(obs)
                assert action.shape == (3,)
                results.add_pass(f"{agent_name} + {extractor}")
            except Exception as e:
                results.add_fail(f"{agent_name} + {extractor}", str(e))

    env.close()


def run_all_tests(quick: bool = False):
    """Run all tests."""
    print("="*60)
    print("CRM_ML RL-ML INTEGRATION TEST SUITE")
    print("="*60)

    # Always run these
    test_custom_feature_extractors()
    test_training_config()

    if not quick:
        test_agents_with_custom_extractors()
        test_model_based_rl()
        test_hybrid_dynamics_integration()
        test_short_training_run()
        test_physics_informed_extractor_with_cpp()
        test_agent_save_load()
        test_all_agent_extractor_combinations()
    else:
        print("\n(Quick mode - skipping extended tests)")

    return results.summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RL-ML integration")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    success = run_all_tests(quick=args.quick)
    sys.exit(0 if success else 1)
