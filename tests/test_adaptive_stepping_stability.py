import json
import time
from pathlib import Path

import numpy as np
import pytest

from crm_ml_rl.wrappers.crm_wrapper import CRMWrapper, HAS_CPP_BINDINGS


def load_failing_case():
    failing_case_path = Path(__file__).parent.parent / "docs" / "architecture" / "TASK_1_7_FAILING_CASE.json"
    with failing_case_path.open("r") as handle:
        return json.load(handle)


@pytest.mark.skipif(not HAS_CPP_BINDINGS, reason="C++ bindings not available")
def test_adaptive_stepping_handles_spikes():
    """Adaptive stepping should return promptly and report convergence/divergence."""
    wrapper = CRMWrapper(use_cpp=True, disable_cpp_fallback=True)
    if not wrapper.is_using_cpp:
        pytest.skip("C++ bindings not available")

    wrapper._cpp_dynamics.set_integrator("rk4")

    case = load_failing_case()
    seed = case["seed"]

    currents = np.asarray(case["currents"], dtype=np.float64)
    insertion = float(case["insertion"])

    start = time.time()
    result = wrapper._cpp_dynamics.step_from_seed(
        currents,
        insertion,
        np.asarray(seed["v"][0], dtype=np.float64),
        np.asarray(seed["w"][0], dtype=np.float64),
        np.asarray(seed["p"][0], dtype=np.float64),
        np.asarray(seed["R"][0], dtype=np.float64),
        np.asarray(seed["xf"], dtype=np.float64),
        np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
    )
    elapsed = time.time() - start

    assert elapsed < 10.0
    assert "converged" in result or "diverged" in result


@pytest.mark.skipif(not HAS_CPP_BINDINGS, reason="C++ bindings not available")
def test_adaptive_stepping_max_subdivision_limits_runtime():
    """Max subdivision should prevent long stalls when cases are unstable."""
    wrapper = CRMWrapper(use_cpp=True, disable_cpp_fallback=True)
    if not wrapper.is_using_cpp:
        pytest.skip("C++ bindings not available")

    wrapper._cpp_dynamics.set_integrator("rk4")

    case = load_failing_case()
    seed = case["seed"]

    currents = np.asarray(case["currents"], dtype=np.float64)
    insertion = float(case["insertion"])

    start = time.time()
    result = wrapper._cpp_dynamics.step_from_seed(
        currents,
        insertion,
        np.asarray(seed["v"][0], dtype=np.float64),
        np.asarray(seed["w"][0], dtype=np.float64),
        np.asarray(seed["p"][0], dtype=np.float64),
        np.asarray(seed["R"][0], dtype=np.float64),
        np.asarray(seed["xf"], dtype=np.float64),
        np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
    )
    elapsed = time.time() - start

    assert elapsed < 10.0
    if not result.get("converged", False):
        assert result.get("diverged", False) is True
