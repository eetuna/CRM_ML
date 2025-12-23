import numpy as np
import pytest

from crm_ml_rl.wrappers.crm_wrapper import CRMWrapper, HAS_CPP_BINDINGS


@pytest.mark.skipif(not HAS_CPP_BINDINGS, reason="C++ bindings not available")
def test_adaptive_stepping_stable_case():
    """Stable case should converge and not report divergence under RK4."""
    wrapper = CRMWrapper(use_cpp=True, disable_cpp_fallback=True)
    if not wrapper.is_using_cpp:
        pytest.skip("C++ bindings not available")

    wrapper._cpp_dynamics.set_integrator("rk4")

    currents = np.zeros(3)
    insertion_length = 50.0

    init_success = wrapper.initialize_dynamics(currents, insertion_length=insertion_length)
    assert init_success

    result = wrapper.step_dynamics(currents, insertion_length=insertion_length)
    assert result.get("converged", False)
    assert result.get("diverged", False) is False
