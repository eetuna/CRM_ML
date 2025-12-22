"""
Compare CRMDYNTest reference numbers against the Python bindings.

This mirrors the hard-coded seeds and parameters used in main/CRMDYN_test.cpp
so we can sanity-check that the pybind11 layer produces the same dynamics
outputs as the standalone C++ test executable.
"""

import os
import numpy as np

from crm_ml_rl.wrappers.crm_wrapper import CRMWrapper

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")


def main():
    # Params pulled from main/CRMDYN_test.cpp
    currents = np.array([0.0, 0.0, 0.1])
    insertion_length = 94.3
    dt = 0.05

    xf_seed = np.array([
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
        -0.000349413951059,
    ])
    pL_seed = np.array([-0.248418562587657, 17.707660318406560, 46.752162601547091])
    RL_seed = np.array(
        [
            0.999919687839427,
            0.009924211584043,
            -0.007882125064742,
            -0.003571217614502,
            0.817374079004311,
            0.576096225796181,
            0.012159945552960,
            -0.576021809479719,
            0.817343875445250,
        ]
    )
    zero3 = np.zeros(3)
    damping = np.array(
        [
            12.1761626666366,
            12.1761626666366,
            284.429938756989,
            0.0304776127617393,
            0.0304776127617393,
            0.00502712804532508,
        ]
    )

    wrapper = CRMWrapper(
        param_file="data/catheter_params/CatheterParameterSet_1_dyn.txt",
        config_file="data/catheter_params/CatheterSpatialConfiguration_1.txt",
        use_cpp=True,
        flip_third_current=False,
    )
    if not wrapper.is_using_cpp:
        raise RuntimeError("C++ bindings are not available")

    # Match the C++ test settings
    wrapper.set_damping(damping)
    wrapper._cpp_dynamics.integration_step_size = 0.2
    wrapper._cpp_dynamics.dt = dt
    wrapper.debug_seed_dynamics(zero3, zero3, pL_seed, RL_seed, xf_seed, mL=zero3, nL=zero3)

    result = wrapper.step_dynamics(currents, insertion_length=insertion_length, dt=dt)

    print("Converged:", result["converged"])
    print("Tip position (bindings):", result["tip_position"])
    print("Tip velocity (bindings):", result["tip_velocity"])
    print("Expected tip position from CRMDYNTest: [-0.60077064 44.16475476 80.67207825]")
    print("Position error (mm):", result["tip_position"] - np.array([-0.60077064, 44.16475476, 80.67207825]))


if __name__ == "__main__":
    main()
