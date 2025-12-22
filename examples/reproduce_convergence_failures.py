"""
Reproduce and analyze dynamics convergence failures.

This script loads the failing current combinations from `data/output/sweep_failures_cpp.json`
and attempts to run the C++ dynamics with the same hardcoded initial seed
used in `CRMDYN_grid_sweep.cpp`.

This is expected to fail and will demonstrate the problem of using a single
initial state for a wide range of currents.
"""
import json
import numpy as np
from pathlib import Path
from crm_ml_rl.wrappers.crm_wrapper import CRMWrapper, HAS_CPP_BINDINGS

def run_failure_reproduction():
    if not HAS_CPP_BINDINGS:
        print("C++ bindings not available, skipping.")
        return

    # Load failing currents
    repo_root = Path(__file__).resolve().parents[1]
    with (repo_root / "data/output" / "sweep_failures_cpp.json").open() as f:
        failing_currents = json.load(f)

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
    damping = np.array(
        [
            12.1761626666366, 12.1761626666366, 284.429938756989,
            0.0304776127617393, 0.0304776127617393, 0.00502712804532508,
        ]
    )
    zero3 = np.zeros(3)
    insertion_length = 50.0
    dt = 0.05

    wrapper = CRMWrapper(
        param_file="data/catheter_params/CatheterParameterSet_1_dyn.txt",
        config_file="data/catheter_params/CatheterSpatialConfiguration_1.txt",
        use_cpp=True,
        disable_cpp_fallback=True, # Don't fallback, we want to see the failures
    )
    if not wrapper.is_using_cpp:
        raise RuntimeError("C++ bindings are not available")

    # Match the C++ test settings
    wrapper.set_damping(damping)
    wrapper._cpp_dynamics.integration_step_size = 0.2
    wrapper._cpp_dynamics.dt = dt

    failures = 0
    for i, currents in enumerate(failing_currents):
        # Seed the dynamics with the same hardcoded values every time
        wrapper.debug_seed_dynamics(zero3, zero3, pL_seed, RL_seed, xf_seed, mL=zero3, nL=zero3)
        
        result = wrapper.step_dynamics(currents, insertion_length=insertion_length, dt=dt)
        
        if not result["converged"]:
            failures += 1
            print(f"Currents {currents} failed to converge as expected.")
        else:
            print(f"Currents {currents} unexpectedly converged.")

    print(f"\nReproduced {failures}/{len(failing_currents)} failures.")

if __name__ == "__main__":
    run_failure_reproduction()
