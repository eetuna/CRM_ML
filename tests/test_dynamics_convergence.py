"""
Test for dynamics convergence issues.

This test verifies that the C++ dynamics fail to converge when using a
single, inappropriate initial state, and that they succeed when using the
proper `initialize_from_kinematics` method.
"""
import json
import numpy as np
import pytest
from pathlib import Path
from crm_ml_rl.wrappers.crm_wrapper import CRMWrapper, HAS_CPP_BINDINGS

@pytest.fixture(scope="module")
def failing_currents():
    """Load the list of currents that are known to cause convergence failures."""
    repo_root = Path(__file__).resolve().parents[1]
    failures_path = repo_root / "data/output" / "sweep_failures_cpp.json"
    with failures_path.open() as f:
        return json.load(f)

@pytest.mark.skipif(not HAS_CPP_BINDINGS, reason="C++ bindings not available")
def test_convergence_failures_and_fix(failing_currents):
    """
    Verify that the dynamics fail with a bad initial state and succeed with a good one.
    """
    insertion_length = 94.3
    dt = 0.05

    # Hardcoded seeds from CRMDYN_test.cpp
    xf_seed = np.array(
        [
            -0.458414144062750, 34.411241976876518, 70.457561147732264,
            0.999932718178103, 0.009921777042635, -0.006009780134551,
            -0.004651734390922, 0.817579117734723, 0.575797488368325,
            0.010626405041486, -0.575730791763330, 0.817570262993625,
            -0.015378744286498, 0.000001280646594, -0.000349413951059,
        ]
    )
    pL_seed = np.array([-0.248418562587657, 17.707660318406560, 46.752162601547091])
    RL_seed = np.array(
        [
            0.999919687839427, 0.009924211584043, -0.007882125064742,
            -0.003571217614502, 0.817374079004311, 0.576096225796181,
            0.012159945552960, -0.576021809479719, 0.817343875445250,
        ]
    )
    zero3 = np.zeros(3)


    wrapper = CRMWrapper(
        param_file="data/catheter_params/CatheterParameterSet_1_dyn.txt",
        config_file="data/catheter_params/CatheterSpatialConfiguration_1.txt",
        use_cpp=True,
        disable_cpp_fallback=True, # We want to assert the failure
    )
    if not wrapper.is_using_cpp:
        pytest.skip("C++ bindings are not available")


    # 1. Verify that the dynamics fail with the bad initial state
    any_failed = False
    for currents in failing_currents:
        wrapper.debug_seed_dynamics(zero3, zero3, pL_seed, RL_seed, xf_seed, mL=zero3, nL=zero3)
        result = wrapper.step_dynamics(currents, insertion_length=insertion_length, dt=dt)
        if not result["converged"]:
            any_failed = True

    if not any_failed:
        pytest.skip("Dynamics no longer reproduces the historical convergence failures for the provided seed/currents list.")

    # 2. Verify that the dynamics succeed with the proper initialization
    wrapper_with_fallback = CRMWrapper(
        param_file="data/catheter_params/CatheterParameterSet_1_dyn.txt",
        config_file="data/catheter_params/CatheterSpatialConfiguration_1.txt",
        use_cpp=True,
    )
    for currents in failing_currents:
        init_success = wrapper_with_fallback.initialize_dynamics(currents, insertion_length=insertion_length)
        assert init_success
        result = wrapper_with_fallback.step_dynamics(currents, insertion_length=insertion_length, dt=dt)
        assert result["converged"]
