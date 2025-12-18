"""
Diagnose and find stable parameters for dynamics convergence failures.

This script takes the set of currents known to cause convergence failures
and systematically tests them against a grid of more robust simulation
parameters to find a stable configuration.

For each failing current, it tries:
1. A range of smaller integration step sizes.
2. A range of increased damping values.

The script reports which combination of parameters (if any) allows each
failing case to converge, providing a map of settings required to stabilize
the simulation in difficult regions of the state space.
"""
import time
import json
import numpy as np
from pathlib import Path
from crm_ml_rl.wrappers.crm_wrapper import CRMWrapper, HAS_CPP_BINDINGS

def run_diagnostic(wrapper: CRMWrapper, currents, insertion_length, dt):
    """
    Runs a single, statefully initialized step.
    """
    init_success = wrapper.initialize_dynamics(currents, insertion_length)
    if not init_success:
        return False, "FK Initialization Failed"
    
    result = wrapper.step_dynamics(currents, insertion_length, dt)
    if result.get("converged", False):
        return True, "Converged"
    else:
        return False, f"localmin={result.get('localmin', 'N/A')}"


def main():
    if not HAS_CPP_BINDINGS:
        print("C++ bindings not available, skipping diagnostics.")
        return

    # Load the currents known to cause issues
    repo_root = Path(__file__).resolve().parents[1]
    with (repo_root / "output_data" / "sweep_failures_cpp.json").open() as f:
        failing_currents = json.load(f)

    # --- Diagnostic Parameters ---
    step_sizes_to_test = [0.2, 0.1, 0.05, 0.01]
    # Default damping from the wrapper
    base_damping = np.array([
        12.1761626666366, 12.1761626666366, 284.429938756989,
        0.0304776127617393, 0.0304776127617393, 0.00502712804532508
    ])
    damping_multipliers_to_test = [1.0, 2.0, 5.0, 10.0]
    
    insertion_length = 94.3
    dt = 0.05

    print("="*70)
    print("Running Convergence Diagnostics")
    print(f"Testing {len(failing_currents)} failing current combinations...")
    print("="*70)

    fixed_cases = {}

    for i, currents in enumerate(failing_currents):
        currents_tuple = tuple(currents)
        print(f"\n[{i+1}/{len(failing_currents)}] Testing current: {currents_tuple}")
        
        found_fix = False
        for step_size in step_sizes_to_test:
            for damp_mult in damping_multipliers_to_test:
                if found_fix:
                    continue

                damping = base_damping * damp_mult
                
                wrapper = CRMWrapper(
                    use_cpp=True, 
                    disable_cpp_fallback=True,
                    damping=damping
                )
                wrapper._cpp_dynamics.integration_step_size = step_size

                converged, reason = run_diagnostic(wrapper, np.array(currents), insertion_length, dt)

                if converged:
                    print(f"  --> SUCCESS with step_size={step_size}, damping_mult={damp_mult}")
                    fixed_cases[currents_tuple] = {
                        "step_size": step_size,
                        "damping_multiplier": damp_mult,
                    }
                    found_fix = True
        
        if not found_fix:
            print(f"  --> FAILED to find a stable configuration for this current.")


    # --- Summary ---
    print("\n" + "="*70)
    print("Diagnostics Summary")
    print("="*70)
    
    num_fixed = len(fixed_cases)
    num_total = len(failing_currents)
    
    print(f"Successfully found stable parameters for {num_fixed}/{num_total} failing cases.")

    if fixed_cases:
        print("\n--- Successful Configurations ---")
        for currents, params in fixed_cases.items():
            print(f"Current: {currents}")
            print(f"  - Integration Step Size: {params['step_size']}")
            print(f"  - Damping Multiplier:    {params['damping_multiplier']}")
            print("-" * 30)
            
    unfixed_cases = [c for c in failing_currents if tuple(c) not in fixed_cases]
    if unfixed_cases:
        print("\n--- Unresolved Cases ---")
        for currents in unfixed_cases:
            print(f" - {tuple(currents)}")

if __name__ == "__main__":
    main()
