"""
Grid sweep to probe convergence of the C++ dynamics bindings with tuned parameters.

Uses CatheterParameterSet_1_dyn and the damping/integration settings from CRMDYNTest.
You can adjust the grid ranges below. Prints a summary of failures and a few sample
positions so you can spot unstable regions quickly.
"""

import os
import itertools
from typing import List, Tuple

import numpy as np

from crm_ml_rl.wrappers.crm_wrapper import CRMWrapper, HAS_CPP_BINDINGS

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")


def make_wrapper(dt: float, step_size: float):
    wrapper = CRMWrapper(
        param_file="catheterdata/CatheterParameterSet_1_dyn.txt",
        config_file="catheterdata/CatheterSpatialConfiguration_1.txt",
        use_cpp=True,
        flip_third_current=False,
        disable_cpp_fallback=True,
    )
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
    wrapper.set_damping(damping)
    wrapper._cpp_dynamics.integration_step_size = step_size
    wrapper._cpp_dynamics.dt = dt
    return wrapper


def try_case(curr, ins, dt, step_sizes):
    """Try a single current/insert pair with progressively smaller step sizes."""
    curr_arr = np.array(curr, dtype=float)

    for step_size in step_sizes:
        w = make_wrapper(dt, step_size)
        init_ok = w.initialize_dynamics(
            curr_arr, ins
        )
        if not init_ok:
            continue
        result = w.step_dynamics(curr_arr, insertion_length=ins, dt=dt)
        if result.get("converged", False) and w.is_using_cpp:
            return True, result["tip_position"], step_size
    return False, None, None


def sweep(currents_grid: List[Tuple[float, float, float]], insertion_lengths: List[float], dt: float = 0.05):
    if not HAS_CPP_BINDINGS:
        raise RuntimeError("C++ bindings not built")

    failures = []
    samples = []
    step_sizes = [0.01, 0.05, 0.1, 0.2]  # smallest first for stability
    for ins in insertion_lengths:
        for curr in currents_grid:
            ok, pos, used_step = try_case(curr, ins, dt, step_sizes)
            if ok:
                samples.append((curr, ins, pos, used_step))
            else:
                failures.append((curr, ins, "all step sizes failed"))
    return failures, samples


def main():
    # Modest grid to keep runtime reasonable; adjust as needed
    c_vals = [-0.2, -0.1, 0.0, 0.1, 0.2]
    currents_grid = []
    for c1, c2, c3 in itertools.product(c_vals, repeat=3):
        # If channel 3 is zero, keep channel 1/2 at zero as well (no inc/dec when c3==0)
        if c3 == 0.0 and (c1 != 0.0 or c2 != 0.0):
            continue
        currents_grid.append((c1, c2, c3))
    # Use the insertion length from CRMDYNTest (no sweep over insertion)
    insertion_lengths = [94.3]
    failures, samples = sweep(currents_grid, insertion_lengths)

    print(f"Tested {len(currents_grid) * len(insertion_lengths)} cases")
    print(f"Failures: {len(failures)}")
    if failures:
        for c, ins, why in failures[:10]:
            print(f"  curr={c}, ins={ins}: {why}")
        if len(failures) > 10:
            print(f"  ... {len(failures) - 10} more")

    print("\nSample converged positions (first 5):")
    for c, ins, pos, step_used in samples[:5]:
        print(f"  curr={c}, ins={ins}, step={step_used} -> tip={pos}")


if __name__ == "__main__":
    main()
