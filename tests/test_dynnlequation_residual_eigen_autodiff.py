import os

import numpy as np
import pytest


@pytest.mark.skipif(os.environ.get("CRM_RUN_DYNNLEQUATION_AD_TESTS", "0") != "1", reason="slow")
def test_dynnlequation_ad_jxx_close_to_fd():
    from crm_ml_rl.wrappers import crm_python

    dyn = crm_python.CRMDynamics()
    ok = dyn.load_parameters(
        "data/catheter_params/CatheterParameterSet_1_dyn.txt",
        "data/catheter_params/CatheterSpatialConfiguration_1.txt",
    )
    if not ok:
        pytest.skip("C++ bindings not available")

    insertion = 94.3
    dyn.set_damping(
        np.array(
            [
                12.1761626666366,
                12.1761626666366,
                284.429938756989,
                0.0304776127617393,
                0.0304776127617393,
                0.00502712804532508,
            ],
            dtype=np.float64,
        )
    )
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
        currents,
        insertion,
        v,
        w,
        p,
        R,
        xf,
        mL,
        nL,
        1e-4,
        1e-5,
        1e-5,
        1e-5,
        True,
    )

    assert out["have_ad_jxx"] is True
    J = np.asarray(out["Jxx"], dtype=np.float64)
    Jfd = np.asarray(out["Jxx_fd"], dtype=np.float64)
    assert J.shape == (6, 6)
    assert Jfd.shape == (6, 6)
    assert np.isfinite(J).all()
    assert np.isfinite(Jfd).all()

    rel = np.linalg.norm(J - Jfd) / (np.linalg.norm(Jfd) + 1e-12)
    assert rel < 0.3

