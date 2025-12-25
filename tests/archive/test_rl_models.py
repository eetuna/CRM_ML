from examples.rl_examples import (
    run_dyna_transformer_example,
    run_dyna_diffusion_example,
    run_mpc_transformer_example,
)
import numpy as np


def test_dyna_transformer_example_runs():
    transitions = run_dyna_transformer_example(num_start_states=2)
    assert transitions > 0


def test_dyna_diffusion_example_runs():
    transitions = run_dyna_diffusion_example(num_start_states=2)
    assert transitions > 0


def test_mpc_transformer_action_is_finite():
    action = run_mpc_transformer_example()
    assert action.shape == (3,)
    assert np.all(np.isfinite(action))
