"""
Demonstrate the fix for dynamics convergence failures.

This script loads the failing current combinations from `data/output/sweep_failures_cpp.json`
and runs the dynamics with a proper initialization strategy. For each set of
currents, it first calls `initialize_from_kinematics` to find a good
starting point for the solver.

This is expected to significantly reduce or eliminate the convergence failures.
"""
import json
import numpy as np
from pathlib import Path
from crm_ml_rl.wrappers.crm_wrapper import CRMWrapper, HAS_CPP_BINDINGS

def run_convergence_fix_demonstration():
    if not HAS_CPP_BINDINGS:
        print("C++ bindings not available, skipping.")
        return

    # Load failing currents
    repo_root = Path(__file__).resolve().parents[1]
    with (repo_root / "data/output" / "sweep_failures_cpp.json").open() as f:
        failing_currents = json.load(f)

    insertion_length = 50.0
    dt = 0.05

    wrapper = CRMWrapper(
        param_file="data/catheter_params/CatheterParameterSet_1_dyn.txt",
        config_file="data/catheter_params/CatheterSpatialConfiguration_1.txt",
        use_cpp=True,
    )
    if not wrapper.is_using_cpp:
        raise RuntimeError("C++ bindings are not available")

    successes = 0
    for i, currents in enumerate(failing_currents):
        # Initialize the dynamics from the FK solution for the given currents
        init_success = wrapper.initialize_dynamics(currents, insertion_length=insertion_length)

        if not init_success:
            print(f"Currents {currents} failed to initialize from FK.")
            continue

        result = wrapper.step_dynamics(currents, insertion_length=insertion_length, dt=dt)
        
        if result["converged"]:
            successes += 1
            print(f"Currents {currents} converged successfully with proper initialization.")
        else:
            print(f"Currents {currents} still failed to converge.")

    print(f"\nSuccessfully converged in {successes}/{len(failing_currents)} cases.")

if __name__ == "__main__":
    run_convergence_fix_demonstration()
