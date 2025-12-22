import os

import pytest
import numpy as np


@pytest.mark.skipif(os.environ.get("CRM_RUN_IMPLICIT_LINEARIZATION_TESTS", "0") != "1", reason="slow")
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
