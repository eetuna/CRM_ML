"""
Compare dynamics linearizations:

- Full-step finite differences: CRMDynamics.linearize_full_seed_action_from_seed
- Implicit scaffold: CRMDynamics.linearize_full_seed_action_from_seed_implicit

This script is intentionally "tooling" (not a unit test) because the implicit
scaffold can be slow: it performs many residual/IVP evaluations.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")


@dataclass(frozen=True)
class Paths:
    param_file: str = "catheterdata/CatheterParameterSet_1_dyn.txt"
    config_file: str = "catheterdata/CatheterSpatialConfiguration_1.txt"
    insertion_length: float = 94.3
    dt: float = 0.05
    integration_step_size: float = 0.01
    damping: Tuple[float, float, float, float, float, float] = (
        12.1761626666366,
        12.1761626666366,
        284.429938756989,
        0.0304776127617393,
        0.0304776127617393,
        0.00502712804532508,
    )


def _stats(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    diff = a - b
    max_abs = float(np.max(np.abs(diff)))
    denom = np.abs(b) + 1e-12
    max_rel = float(np.max(np.abs(diff) / denom))
    frob_abs = float(np.linalg.norm(diff))
    frob_ref = float(np.linalg.norm(b))
    return {"max_abs": max_abs, "max_rel": max_rel, "frob_abs": frob_abs, "frob_ref": frob_ref}


def _seed_dim(num_sets: int) -> int:
    return num_sets * 3 + num_sets * 3 + num_sets * 3 + num_sets * 9 + 15 + num_sets * 3 + num_sets * 3


def _seed_blocks(num_sets: int) -> Dict[str, slice]:
    dim_v = num_sets * 3
    dim_w = num_sets * 3
    dim_p = num_sets * 3
    dim_R = num_sets * 9
    dim_xf = 15
    dim_mL = num_sets * 3
    dim_nL = num_sets * 3
    i_v0 = 0
    i_w0 = i_v0 + dim_v
    i_p0 = i_w0 + dim_w
    i_R0 = i_p0 + dim_p
    i_xf0 = i_R0 + dim_R
    i_mL0 = i_xf0 + dim_xf
    i_nL0 = i_mL0 + dim_mL
    return {
        "v": slice(i_v0, i_w0),
        "w": slice(i_w0, i_p0),
        "p": slice(i_p0, i_R0),
        "R": slice(i_R0, i_xf0),
        "xf": slice(i_xf0, i_mL0),
        "mL": slice(i_mL0, i_nL0),
        "nL": slice(i_nL0, i_nL0 + dim_nL),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=int, default=1, help="Number of random points to test")
    ap.add_argument("--insertion", type=float, default=Paths.insertion_length)
    ap.add_argument("--eps-u", type=float, default=1e-4)
    ap.add_argument("--eps-seed", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from crm_ml_rl.wrappers import crm_python

    kin = crm_python.CRMKinematics()
    dyn = crm_python.CRMDynamics()
    ok1 = kin.load_parameters(Paths.param_file, Paths.config_file)
    ok2 = dyn.load_parameters(Paths.param_file, Paths.config_file)
    if not (ok1 and ok2):
        raise RuntimeError("Failed to load CRM parameters.")

    # Configure dynamics to match common stable settings used in scripts.
    dyn.set_damping(np.array(Paths.damping, dtype=np.float64))
    dyn.dt = float(Paths.dt)
    dyn.integration_step_size = float(Paths.integration_step_size)

    # Initialize internal state at a stable bias, then extract a seed.
    dyn.initialize_from_kinematics([0.0, 0.0, 0.2], float(args.insertion))
    seed = dyn.get_seed_state()
    num_sets = int(np.asarray(seed["v"]).shape[0])
    assert _seed_dim(num_sets) == (num_sets * 3 + num_sets * 3 + num_sets * 3 + num_sets * 9 + 15 + num_sets * 3 + num_sets * 3)

    rng = np.random.default_rng(args.seed)
    blocks = _seed_blocks(num_sets)

    # Note: mL/nL affect the solver initialization, not the equilibrium equations directly.
    # Expect large discrepancies between implicit and full-step FD in those blocks.
    compare_blocks = ["v", "w", "p", "R", "xf"]

    for k in range(args.points):
        v = np.asarray(seed["v"], dtype=np.float64)
        w = np.asarray(seed["w"], dtype=np.float64)
        p = np.asarray(seed["p"], dtype=np.float64)
        R = np.asarray(seed["R"], dtype=np.float64)
        xf = np.asarray(seed["xf"], dtype=np.float64)
        mL = np.asarray(seed["mL"], dtype=np.float64)
        nL = np.asarray(seed["nL"], dtype=np.float64)

        # Try to find a converged point; otherwise fall back to zero currents.
        currents = None
        fd = None
        for _ in range(30):
            cand = (rng.standard_normal(3) * 0.01).astype(np.float64)
            fd_cand = dyn.linearize_full_seed_action_from_seed(
                cand, float(args.insertion), v, w, p, R, xf, mL, nL, float(args.eps_u), float(args.eps_seed)
            )
            if bool(fd_cand.get("base", {}).get("converged", False)):
                currents = cand
                fd = fd_cand
                break
        if currents is None:
            currents = np.zeros(3, dtype=np.float64)
            fd = dyn.linearize_full_seed_action_from_seed(
                currents, float(args.insertion), v, w, p, R, xf, mL, nL, float(args.eps_u), float(args.eps_seed)
            )

        imp = dyn.linearize_full_seed_action_from_seed_implicit(
            currents,
            float(args.insertion),
            v,
            w,
            p,
            R,
            xf,
            mL,
            nL,
            float(args.eps_seed),
            float(args.eps_seed),
            float(args.eps_seed),
            float(args.eps_seed),
        )

        B_fd = np.asarray(fd["B"], dtype=np.float64)
        B_imp = np.asarray(imp["B"], dtype=np.float64)
        print(
            f"[{k}] currents={currents.tolist()} converged_fd={bool(fd['base']['converged'])} converged_imp={bool(imp['base']['converged'])}"
        )
        print(f"[{k}] B stats:", _stats(B_imp, B_fd))

        A_fd = np.asarray(fd["A"], dtype=np.float64)
        A_imp = np.asarray(imp["A"], dtype=np.float64)
        for name in compare_blocks:
            sl = blocks[name]
            print(f"[{k}] A[{name}] stats:", _stats(A_imp[:, sl], A_fd[:, sl]))

        print(f"[{k}] residual_norm={float(imp.get('residual_norm', np.nan)):.3e}")


if __name__ == "__main__":
    main()
