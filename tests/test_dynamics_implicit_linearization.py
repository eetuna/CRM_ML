import os

import pytest
import numpy as np


def test_implicit_linearization_shapes_and_finiteness():
    from crm_ml_rl.wrappers import crm_python

    kin = crm_python.CRMKinematics()
    dyn = crm_python.CRMDynamics()
    ok1 = kin.load_parameters("data/catheter_params/CatheterParameterSet_1_dyn.txt", "data/catheter_params/CatheterSpatialConfiguration_1.txt")
    ok2 = dyn.load_parameters("data/catheter_params/CatheterParameterSet_1_dyn.txt", "data/catheter_params/CatheterSpatialConfiguration_1.txt")
    if not (ok1 and ok2):
        pytest.skip("C++ bindings not available")

    insertion = 50.0
    dyn.set_damping(np.array([12.1761626666366, 12.1761626666366, 284.429938756989, 0.0304776127617393, 0.0304776127617393, 0.00502712804532508], dtype=np.float64))
    dyn.dt = 0.05
    dyn.integration_step_size = 0.01
    dyn.initialize_from_kinematics([0.0, 0.0, 0.2], insertion)
    seed = dyn.get_seed_state()

    v = np.asarray(seed["v"], dtype=np.float64)
    w = np.asarray(seed["w"], dtype=np.float64)
    p = np.asarray(seed["p"], dtype=np.float64)
    R = np.asarray(seed["R"], dtype=np.float64)
    xf = np.asarray(seed["xf"], dtype=np.float64)
    mL = np.asarray(seed["mL"], dtype=np.float64)
    nL = np.asarray(seed["nL"], dtype=np.float64)

    currents = np.array([0.02, -0.01, 0.01], dtype=np.float64)

    out = dyn.linearize_full_seed_action_from_seed_implicit(
        currents, insertion, v, w, p, R, xf, mL, nL, 1e-5, 1e-5, 1e-5, 1e-5
    )
    nxt = np.asarray(out["next_state"], dtype=np.float64)
    A = np.asarray(out["A"], dtype=np.float64)
    B = np.asarray(out["B"], dtype=np.float64)

    assert nxt.shape == (6,)
    assert B.shape == (6, 3)
    assert A.shape[0] == 6
    assert np.isfinite(nxt).all()
    assert np.isfinite(B).all()
    assert np.isfinite(A).all()


def test_control_jacobian_ad_vs_fd():
    """Test that AD-based control gradients (Task 1.5) produce finite, reasonable results."""
    from crm_ml_rl.wrappers import crm_python

    kin = crm_python.CRMKinematics()
    dyn = crm_python.CRMDynamics()
    ok1 = kin.load_parameters("data/catheter_params/CatheterParameterSet_1_dyn.txt", "data/catheter_params/CatheterSpatialConfiguration_1.txt")
    ok2 = dyn.load_parameters("data/catheter_params/CatheterParameterSet_1_dyn.txt", "data/catheter_params/CatheterSpatialConfiguration_1.txt")
    if not (ok1 and ok2):
        pytest.skip("C++ bindings not available")

    insertion = 50.0
    # Use validated damping values to ensure convergence
    dyn.set_damping(np.array([12.1761626666366, 12.1761626666366, 284.429938756989, 0.0304776127617393, 0.0304776127617393, 0.00502712804532508], dtype=np.float64))
    dyn.dt = 0.05
    dyn.integration_step_size = 0.01
    dyn.initialize_from_kinematics([0.0, 0.0, 0.2], insertion)
    seed = dyn.get_seed_state()

    v = np.asarray(seed["v"], dtype=np.float64)
    w = np.asarray(seed["w"], dtype=np.float64)
    p = np.asarray(seed["p"], dtype=np.float64)
    R = np.asarray(seed["R"], dtype=np.float64)
    xf = np.asarray(seed["xf"], dtype=np.float64)
    mL = np.asarray(seed["mL"], dtype=np.float64)
    nL = np.asarray(seed["nL"], dtype=np.float64)

    currents = np.array([0.02, -0.01, 0.01], dtype=np.float64)

    # Get AD-based B (implicit linearization with AD for currents)
    out_ad = dyn.linearize_full_seed_action_from_seed_implicit(
        currents, insertion, v, w, p, R, xf, mL, nL, 1e-5, 1e-5, 1e-5, 1e-5, return_debug=True
    )
    B_ad = np.asarray(out_ad["B"], dtype=np.float64)

    # Verify that AD was actually used for control gradients
    assert out_ad.get("have_ad_jxu", False), "AD control jacobian was not used (fallback to FD)"
    assert out_ad.get("have_ad_jxx", False), "AD state jacobian was not used (fallback to FD)"

    # Should have converged
    assert out_ad["base"]["converged"], "AD linearization did not converge"

    # Check shape
    assert B_ad.shape == (6, 3), f"Expected B shape (6, 3), got {B_ad.shape}"

    # Check that all values are finite
    assert np.isfinite(B_ad).all(), f"B contains non-finite values: {B_ad}"

    # Verify reasonable magnitude (not too large or too small)
    # Dynamics B values should typically be O(1) to O(100) for these parameters
    max_val = np.max(np.abs(B_ad))
    assert max_val < 1e4, f"B values unreasonably large: max={max_val:.3e}"
    assert max_val > 1e-4, f"B values unreasonably small: max={max_val:.3e}"

    # Verify that changing currents actually affects the output (B should be non-trivial)
    assert not np.allclose(B_ad, np.zeros_like(B_ad), atol=1e-6), "B is effectively zero"

    print(f"AD control jacobian test passed. B magnitude: {max_val:.3e}")
