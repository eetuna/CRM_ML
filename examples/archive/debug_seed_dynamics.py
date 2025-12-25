"""
Debug script to seed C++ dynamics with hardcoded values from CRMDYN_test.cpp
and ensure the bindings stay on the C++ path.
"""

import numpy as np
from crm_ml_rl.wrappers.crm_wrapper import CRMWrapper


def run_debug_seed_step():
    # Hardcoded values from main/CRMDYN_test.cpp (single actuator set)
    damping = np.array([
        12.1761626666366, 12.1761626666366, 284.429938756989,
        0.0304776127617393, 0.0304776127617393, 0.00502712804532508
    ])
    dt = 0.05
    xf = np.array([
        -0.458414144062750,
        34.411241976876518,
        70.457561147732264,
        0.999932718178103,
        0.009921777042635,
        -0.006009780134551,
        -0.004651734390922,
        0.817579117734723,
        0.575797488368325,
        0.010626405041486,
        -0.575730791763330,
        0.817570262993625,
        -0.015378744286498,
        0.000001280646594,
        -0.000349413951059
    ])
    p_L = np.array([[-0.248418562587657, 17.707660318406560, 46.752162601547091]])
    R_L = np.array([[
        0.999919687839427, 0.009924211584043, -0.007882125064742,
        -0.003571217614502, 0.817374079004311, 0.576096225796181,
        0.012159945552960, -0.576021809479719, 0.817343875445250
    ]])

    crm = CRMWrapper(use_cpp=True)
    crm.set_timestep(dt)
    crm.set_damping(damping)

    crm.debug_seed_dynamics(
        v=np.zeros((1, 3)),
        w=np.zeros((1, 3)),
        p=p_L,
        R=R_L,
        xf=xf,
        mL=np.zeros((1, 3)),
        nL=np.zeros((1, 3)),
    )

    # Use insertion_length=94.3 to match the seed values from CRMDYN_test.cpp
    # Use currents [0.0, 0.0, 0.1] which is a known-good configuration
    res = crm.step_dynamics(np.array([0.0, 0.0, 0.1]), insertion_length=94.3, dt=dt)
    return {
        "using_cpp": crm.is_using_cpp,
        "converged": res.get("converged", False),
        "tip_position": res["tip_position"],
    }


if __name__ == "__main__":
    out = run_debug_seed_step()
    print(out)
