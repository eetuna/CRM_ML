#!/usr/bin/env python3
"""
Benchmark Task 1.4: AD vs FD performance for dynamics Jacobians.

This script measures:
  - Parameter Jacobian via AutoDiff (compute_parameter_jacobian)
  - Parameter Jacobian via finite differences (damping-only FD)
  - Linearization via implicit AD (linearize_full_seed_action_from_seed_implicit)
  - Linearization via full FD (linearize_full_seed_action_from_seed)
  - Linearization via currents-only FD (linearize_action_from_seed)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from crm_ml_rl.wrappers import crm_python


BASE_DAMPING = np.array(
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


def setup_dyn(insertion: float):
    dyn = crm_python.CRMDynamics()
    ok = dyn.load_parameters(
        "data/catheter_params/CatheterParameterSet_1_dyn.txt",
        "data/catheter_params/CatheterSpatialConfiguration_1.txt",
    )
    if not ok:
        raise RuntimeError("Failed to load parameters")
    dyn.set_damping(BASE_DAMPING)
    dyn.dt = 0.05
    dyn.integration_step_size = 0.1
    dyn.initialize_from_kinematics([0.0, 0.0, 0.2], insertion)
    return dyn


def get_seed(dyn):
    seed = dyn.get_seed_state()
    v = np.asarray(seed["v"], dtype=np.float64)
    w = np.asarray(seed["w"], dtype=np.float64)
    p = np.asarray(seed["p"], dtype=np.float64)
    R = np.asarray(seed["R"], dtype=np.float64)
    xf = np.asarray(seed["xf"], dtype=np.float64)
    mL = np.asarray(seed["mL"], dtype=np.float64)
    nL = np.asarray(seed["nL"], dtype=np.float64)

    # Ensure non-zero velocity to avoid zero damping gradients.
    if np.allclose(v, 0.0):
        v = v.copy()
        v[0] = 1e-3
    if np.allclose(w, 0.0):
        w = w.copy()
        w[0] = 1e-3

    return v, w, p, R, xf, mL, nL


def time_block(fn, repeats: int):
    times = []
    failures = 0
    for _ in range(repeats):
        t0 = time.perf_counter()
        try:
            fn()
        except Exception:
            failures += 1
            continue
        times.append(time.perf_counter() - t0)
    return times, failures


def ensure_converged(result, label: str):
    if not isinstance(result, dict):
        return
    if "converged" in result and not bool(result["converged"]):
        raise RuntimeError(f"{label}: unconverged")
    if "base" in result:
        base = result["base"]
        if isinstance(base, dict) and "converged" in base and not bool(base["converged"]):
            raise RuntimeError(f"{label}: base unconverged")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--insertion", type=float, default=94.3)
    ap.add_argument("--out", type=str, default="data/output/benchmark_task1_4.json")
    args = ap.parse_args()

    dyn = setup_dyn(args.insertion)
    v, w, p, R, xf, mL, nL = get_seed(dyn)
    currents = np.array([0.02, -0.01, 0.01], dtype=np.float64)

    # AD parameter Jacobian
    def run_param_ad():
        out = dyn.compute_parameter_jacobian(currents, args.insertion, v, w, p, R, xf, mL, nL)
        ensure_converged(out, "param_ad")

    # FD parameter Jacobian (damping-only via compute_residual_at_state)
    def run_param_fd():
        eps = 1e-5
        base = dyn.compute_residual_at_state(currents, args.insertion, v, w, p, R, xf, mL, nL)
        ensure_converged(base, "param_fd_base")
        _ = np.asarray(base["residual"], dtype=np.float64)
        for i in range(6):
            pert = BASE_DAMPING.copy()
            pert[i] += eps
            dyn.set_damping(pert)
            out = dyn.compute_residual_at_state(currents, args.insertion, v, w, p, R, xf, mL, nL)
            ensure_converged(out, "param_fd_pert")
        dyn.set_damping(BASE_DAMPING)

    # Linearization: implicit AD
    def run_lin_implicit():
        out = dyn.linearize_full_seed_action_from_seed_implicit(
            currents, args.insertion, v, w, p, R, xf, mL, nL, 1e-4, 1e-5, 1e-5, 1e-5
        )
        ensure_converged(out, "lin_implicit")

    # Linearization: full FD
    def run_lin_fd():
        out = dyn.linearize_full_seed_action_from_seed(
            currents, args.insertion, v, w, p, R, xf, mL, nL, 1e-4, 1e-4
        )
        ensure_converged(out, "lin_full_fd")

    # Linearization: currents-only FD
    def run_lin_currents_fd():
        out = dyn.linearize_action_from_seed(currents, args.insertion, v, w, p, R, xf, mL, nL, 1e-4)
        ensure_converged(out, "lin_currents_fd")

    param_ad_times, param_ad_fail = time_block(run_param_ad, args.repeats)
    param_fd_times, param_fd_fail = time_block(run_param_fd, args.repeats)
    lin_impl_times, lin_impl_fail = time_block(run_lin_implicit, args.repeats)
    lin_fd_times, lin_fd_fail = time_block(run_lin_fd, args.repeats)
    lin_curr_times, lin_curr_fail = time_block(run_lin_currents_fd, args.repeats)

    results = {
        "repeats": args.repeats,
        "insertion": args.insertion,
        "param_jacobian_ad_sec": param_ad_times,
        "param_jacobian_ad_failures": param_ad_fail,
        "param_jacobian_fd_damping_sec": param_fd_times,
        "param_jacobian_fd_damping_failures": param_fd_fail,
        "linearize_implicit_ad_sec": lin_impl_times,
        "linearize_implicit_ad_failures": lin_impl_fail,
        "linearize_full_fd_sec": lin_fd_times,
        "linearize_full_fd_failures": lin_fd_fail,
        "linearize_currents_fd_sec": lin_curr_times,
        "linearize_currents_fd_failures": lin_curr_fail,
        "notes": [
            "FD parameter benchmark is damping-only (set_damping is the only exposed setter).",
            "Implicit linearization uses AD Jxx; full_fd is the expensive baseline.",
        ],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    def summarize(label, data):
        arr = np.array(data, dtype=np.float64)
        if arr.size == 0:
            return {"label": label, "mean_s": None, "p50_s": None, "p95_s": None}
        return {
            "label": label,
            "mean_s": float(arr.mean()),
            "p50_s": float(np.median(arr)),
            "p95_s": float(np.quantile(arr, 0.95)),
        }

    summary = [
        summarize("param_jacobian_ad", results["param_jacobian_ad_sec"]),
        summarize("param_jacobian_fd_damping", results["param_jacobian_fd_damping_sec"]),
        summarize("linearize_implicit_ad", results["linearize_implicit_ad_sec"]),
        summarize("linearize_full_fd", results["linearize_full_fd_sec"]),
        summarize("linearize_currents_fd", results["linearize_currents_fd_sec"]),
    ]
    print(json.dumps(summary, indent=2))
    print(f"Saved raw results to {out_path}")


if __name__ == "__main__":
    main()
