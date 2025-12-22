"""
Compare forward kinematics between the C++ bindings and a fresh wrapper instance,
verifying they produce identical tip positions when both use the C++ solver.

Usage:
  python3 scripts/compare_fk_cpp_vs_wrapper.py
"""

import numpy as np

from crm_ml_rl.wrappers.crm_wrapper import CRMWrapper, HAS_CPP_BINDINGS


def compare_fk(currents, insertion_length=94.3, tol=1e-6):
    if not HAS_CPP_BINDINGS:
        raise RuntimeError("C++ bindings not available")

    # Direct C++ binding via CRMWrapper with C++ enabled
    w_cpp = CRMWrapper(
        param_file="data/catheter_params/CatheterParameterSet_1_dyn.txt",
        config_file="data/catheter_params/CatheterSpatialConfiguration_1.txt",
        use_cpp=True,
        flip_third_current=False,
        disable_cpp_fallback=True,
        strict_cpp=True,
    )
    res_cpp = w_cpp.forward_kinematics(currents, insertion_length=insertion_length)
    tip_cpp = np.array(res_cpp["tip_position"])

    # Fresh wrapper, also using C++ (should match)
    w2 = CRMWrapper(
        param_file="data/catheter_params/CatheterParameterSet_1_dyn.txt",
        config_file="data/catheter_params/CatheterSpatialConfiguration_1.txt",
        use_cpp=True,
        flip_third_current=False,
        disable_cpp_fallback=True,
        strict_cpp=True,
    )
    res_w = w2.forward_kinematics(currents, insertion_length=insertion_length)
    tip_w = np.array(res_w["tip_position"])

    err = np.linalg.norm(tip_cpp - tip_w)
    print(f"currents {currents}, insertion {insertion_length} -> ||delta|| = {err}")
    if err > tol:
        print("Mismatch exceeds tolerance!")
    else:
        print("Match within tolerance.")


def main():
    test_currents = [
        np.array([0.0, 0.0, 0.1]),
        np.array([-0.1, 0.1, 0.2]),
        np.array([0.2, -0.1, -0.2]),
        np.array([0.0, 0.0, -0.3]),
    ]
    for curr in test_currents:
        compare_fk(curr)


if __name__ == "__main__":
    main()
