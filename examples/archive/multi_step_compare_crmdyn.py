"""
Run the CRMDYNTest executable and compare its multi-step tip positions with the
pybind11 dynamics in CRMWrapper.

This pulls the same seeds/currents from main/CRMDYN_test.cpp, runs 4 iterations,
and reports per-step position deltas. Helpful to ensure the binding tracks the
standalone C++ dynamics over several updates.
"""

import os
import re
import subprocess
from pathlib import Path

import numpy as np

from crm_ml_rl.wrappers.crm_wrapper import CRMWrapper, HAS_CPP_BINDINGS

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")


def run_crmdyn_test() -> np.ndarray:
    """Invoke ./build/CRMDYNTest and extract tip positions each iteration."""
    exe = Path(__file__).resolve().parent.parent / "build" / "CRMDYNTest"
    if not exe.exists():
        raise FileNotFoundError("CRMDYNTest binary not found; build it with `cmake .. && make` in ./build")
    out = subprocess.check_output([str(exe)], text=True, cwd=exe.parent)
    pattern = re.compile(r"Catheter Tip Position --\s*\n([0-9eE+\.\-\s]+)\n----")
    matches = pattern.findall(out)
    if not matches:
        raise RuntimeError(f"Could not parse tip positions from CRMDYNTest output. Sample:\n{out[:400]}")
    positions = []
    for m in matches:
        vals = [float(x) for x in m.strip().split()]
        if len(vals) >= 3:
            positions.append(vals[:3])
    return np.asarray(positions)


def run_binding_steps() -> np.ndarray:
    """Seed bindings like CRMDYNTest and step 4 times."""
    if not HAS_CPP_BINDINGS:
        raise RuntimeError("C++ bindings not built")

    currents = np.array([0.0, 0.0, 0.1])
    insertion_length = 94.3
    dt = 0.05
    xf_seed = np.array(
        [
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
        ]
    )
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
    wrapper.set_damping(damping)
    wrapper._cpp_dynamics.integration_step_size = 0.2
    wrapper._cpp_dynamics.dt = dt
    wrapper.debug_seed_dynamics(zero3, zero3, pL_seed, RL_seed, xf_seed, mL=zero3, nL=zero3)

    positions = []
    for _ in range(4):
        result = wrapper.step_dynamics(currents, insertion_length=insertion_length, dt=dt)
        if not result.get("converged", False):
            raise RuntimeError("Binding dynamics did not converge")
        positions.append(result["tip_position"])
    return np.vstack(positions)


def main():
    cpp_positions = run_crmdyn_test()
    binding_positions = run_binding_steps()

    print("CRMDYNTest tip positions:")
    print(cpp_positions)
    print("\\nBindings tip positions:")
    print(binding_positions)

    deltas = binding_positions - cpp_positions[: len(binding_positions)]
    print("\\nPer-step position deltas (bindings - CRMDYNTest) [mm]:")
    print(deltas)
    print("Max abs delta:", np.abs(deltas).max())


if __name__ == "__main__":
    main()
