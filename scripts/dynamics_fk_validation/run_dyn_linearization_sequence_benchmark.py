import argparse
import csv
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def load_crm_python(so_path: str):
    # Extension init symbol is PyInit_crm_python, so the module name must be "crm_python".
    spec = importlib.util.spec_from_file_location("crm_python", so_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load extension spec from {so_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["crm_python"] = module
    spec.loader.exec_module(module)
    return module


def read_currents_csv(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"Missing header in CSV: {path}")
        fields = [c.strip() for c in reader.fieldnames]
        for req in ("c1", "c2", "c3"):
            if req not in fields:
                raise RuntimeError(f"CSV {path} must contain columns c1,c2,c3; got {fields}")
        rows = []
        for row in reader:
            rows.append([float(row["c1"]), float(row["c2"]), float(row["c3"])])
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise RuntimeError(f"Unexpected currents shape from CSV {path}: {arr.shape}")
    return arr


def tip_state6(step_out: Dict[str, Any]) -> np.ndarray:
    tip_pos = np.asarray(step_out["tip_position"], dtype=np.float64).reshape(-1)
    tip_vel = np.asarray(step_out.get("tip_velocity", np.zeros(3)), dtype=np.float64).reshape(-1)
    if tip_pos.size != 3 or tip_vel.size != 3:
        raise RuntimeError(f"Unexpected tip sizes: pos={tip_pos.size}, vel={tip_vel.size}")
    return np.concatenate([tip_pos, tip_vel], axis=0)


def seed_to_arrays(seed: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    v = np.asarray(seed["v"], dtype=np.float64)
    w = np.asarray(seed["w"], dtype=np.float64)
    p = np.asarray(seed["p"], dtype=np.float64)
    R = np.asarray(seed["R"], dtype=np.float64)
    xf = np.asarray(seed["xf"], dtype=np.float64)
    mL = np.asarray(seed.get("mL", np.zeros_like(v)), dtype=np.float64)
    nL = np.asarray(seed.get("nL", np.zeros_like(v)), dtype=np.float64)
    return v, w, p, R, xf, mL, nL


def perturb_seed(
    v: np.ndarray,
    w: np.ndarray,
    p: np.ndarray,
    R: np.ndarray,
    xf: np.ndarray,
    mL: np.ndarray,
    nL: np.ndarray,
    rng: np.random.Generator,
    sigma: float,
):
    return (
        v + rng.normal(scale=sigma, size=v.shape),
        w + rng.normal(scale=sigma, size=w.shape),
        p + rng.normal(scale=sigma, size=p.shape),
        R + rng.normal(scale=sigma, size=R.shape),
        xf + rng.normal(scale=sigma, size=xf.shape),
        mL + rng.normal(scale=sigma, size=mL.shape),
        nL + rng.normal(scale=sigma, size=nL.shape),
    )


def flatten_seed(v, w, p, R, xf, mL, nL) -> np.ndarray:
    return np.concatenate(
        [v.reshape(-1), w.reshape(-1), p.reshape(-1), R.reshape(-1), xf.reshape(-1), mL.reshape(-1), nL.reshape(-1)],
        axis=0,
    )


@dataclass
class LinearizationOutputs:
    y0: np.ndarray  # (6,1)
    B_fd: Optional[np.ndarray]  # (6,3)
    A_imp: np.ndarray  # (6,seed_dim)
    B_imp: np.ndarray  # (6,3)
    have_ad_jxx: Optional[bool] = None
    residual_norm: Optional[float] = None
    A_fd_full: Optional[np.ndarray] = None
    B_fd_full: Optional[np.ndarray] = None


def compute_linearizations(
    dyn,
    currents: np.ndarray,
    insertion: float,
    seed_arrays_tuple: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    eps_u: float,
    eps_seed: float,
    eps_residual_x: float,
    eps_residual_theta: float,
    eps_g_x: float,
    eps_g_theta: float,
    compute_full_fd: bool,
    compute_currents_fd: bool,
    compute_implicit: bool,
) -> LinearizationOutputs:
    v, w, p, R, xf, mL, nL = seed_arrays_tuple

    step0 = dyn.step_from_seed(currents, insertion, v, w, p, R, xf, mL, nL)
    if not bool(step0.get("converged", True)):
        raise RuntimeError("Base step_from_seed did not converge")
    y0 = tip_state6(step0).reshape(6, 1)

    B_fd = None
    if compute_currents_fd:
        out_b = dyn.linearize_action_from_seed(currents, insertion, v, w, p, R, xf, mL, nL, float(eps_u))
        B_fd = np.asarray(out_b["B"], dtype=np.float64).reshape(6, 3)

    A_imp = np.full((6, 0), np.nan, dtype=np.float64)
    B_imp = np.full((6, 3), np.nan, dtype=np.float64)
    have_ad_jxx = None
    residual_norm = None
    if compute_implicit:
        have_debug = False
        try:
            out_imp = dyn.linearize_full_seed_action_from_seed_implicit(
                currents,
                insertion,
                v,
                w,
                p,
                R,
                xf,
                mL,
                nL,
                float(eps_residual_x),
                float(eps_residual_theta),
                float(eps_g_x),
                float(eps_g_theta),
                True,
            )
            have_debug = True
        except TypeError:
            out_imp = dyn.linearize_full_seed_action_from_seed_implicit(
                currents,
                insertion,
                v,
                w,
                p,
                R,
                xf,
                mL,
                nL,
                float(eps_residual_x),
                float(eps_residual_theta),
                float(eps_g_x),
                float(eps_g_theta),
            )

        base = out_imp["base"]
        if not bool(base.get("converged", True)):
            raise RuntimeError("Implicit base did not converge")

        A_imp = np.asarray(out_imp["A"], dtype=np.float64)
        B_imp = np.asarray(out_imp["B"], dtype=np.float64).reshape(6, 3)

        if have_debug:
            have_ad_jxx = bool(out_imp.get("have_ad_jxx", False))
            residual_norm = float(out_imp.get("residual_norm", float("nan")))

    A_fd_full = None
    B_fd_full = None
    if compute_full_fd:
        out_full = dyn.linearize_full_seed_action_from_seed(
            currents,
            insertion,
            v,
            w,
            p,
            R,
            xf,
            mL,
            nL,
            float(eps_u),
            float(eps_seed),
        )
        A_fd_full = np.asarray(out_full["A"], dtype=np.float64)
        B_fd_full = np.asarray(out_full["B"], dtype=np.float64).reshape(6, 3)

    return LinearizationOutputs(
        y0=y0,
        B_fd=B_fd,
        A_imp=A_imp,
        B_imp=B_imp,
        have_ad_jxx=have_ad_jxx,
        residual_norm=residual_norm,
        A_fd_full=A_fd_full,
        B_fd_full=B_fd_full,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", required=True, help="Path to crm_python*.so")
    ap.add_argument("--currents-csv", required=True, help="CSV with columns c1,c2,c3")
    ap.add_argument("--out-jsonl", required=True, help="Output JSONL path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--insertion", type=float, default=94.3)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--integration-step-size", type=float, default=0.2)
    ap.add_argument("--max-steps", type=int, default=-1, help="If >0, limit processed steps")
    ap.add_argument("--start", type=int, default=0, help="Start index into the CSV sequence")
    ap.add_argument("--stride", type=int, default=1, help="Process every Kth step (subsampling)")
    ap.add_argument("--dirs", type=int, default=6, help="Directional checks per step")
    ap.add_argument("--du-sigma", type=float, default=1e-4)
    ap.add_argument("--seed-sigma", type=float, default=1e-4)
    ap.add_argument("--eps-u", type=float, default=1e-4)
    ap.add_argument("--eps-seed", type=float, default=1e-4)
    ap.add_argument("--eps-residual-x", type=float, default=1e-4)
    ap.add_argument("--eps-residual-theta", type=float, default=1e-5)
    ap.add_argument("--eps-g-x", type=float, default=1e-5)
    ap.add_argument("--eps-g-theta", type=float, default=1e-5)
    ap.add_argument("--full-fd-every", type=int, default=0, help="If >0, compute full-FD A,B every K steps")
    ap.add_argument(
        "--step-markers",
        action="store_true",
        help="Print STEP_BEGIN/STEP_END markers to help attribute 'Unbounded' logs to a timestep k",
    )
    ap.add_argument(
        "--skip-linearize-action-fd",
        action="store_true",
        help="Skip linearize_action_from_seed (currents-only FD baseline) to isolate unbounded warnings to implicit path.",
    )
    ap.add_argument(
        "--skip-implicit",
        action="store_true",
        help="Skip linearize_full_seed_action_from_seed_implicit to isolate unbounded warnings to FD-only path.",
    )
    ap.add_argument(
        "--skip-currents-only-validation",
        action="store_true",
        help="Skip currents-only validation that calls step_from_seed(currents+du, seed).",
    )
    ap.add_argument(
        "--skip-directional-validation",
        action="store_true",
        help="Skip full directional validation that calls step_from_seed at (u±du, seed±dseed).",
    )
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

    mod = load_crm_python(args.so)
    dyn = mod.CRMDynamics()
    ok = dyn.load_parameters(
        "catheterdata/CatheterParameterSet_1_dyn.txt",
        "catheterdata/CatheterSpatialConfiguration_1.txt",
    )
    if not ok:
        raise RuntimeError("Failed to load parameters")

    dyn.set_damping(
        np.array(
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
    )
    dyn.set_timestep(float(args.dt))
    dyn.integration_step_size = float(args.integration_step_size)

    currents_seq = read_currents_csv(args.currents_csv)
    start = int(max(0, args.start))
    stride = int(max(1, args.stride))
    currents_seq = currents_seq[start::stride]
    if args.max_steps and args.max_steps > 0:
        currents_seq = currents_seq[: args.max_steps]
    if len(currents_seq) == 0:
        raise RuntimeError("No rows in currents csv")

    # Critical: initialize dynamics seed from kinematics at first current.
    dyn.initialize_from_kinematics(currents_seq[0], float(args.insertion))

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        wrote = 0
        for k, currents in enumerate(currents_seq):
            if args.step_markers:
                print(f"STEP_BEGIN k={int(k)} u={currents.tolist()}", flush=True)
            seed_dict = dyn.get_seed_state()
            seed_arrays_tuple = seed_to_arrays(seed_dict)
            v, w, p, R, xf, mL, nL = seed_arrays_tuple
            seed0_flat = flatten_seed(v, w, p, R, xf, mL, nL)

            compute_full_fd = bool(args.full_fd_every and args.full_fd_every > 0 and (k % args.full_fd_every == 0))
            row: Dict[str, Any] = {
                "k": int(k),
                "currents": currents.tolist(),
                "seed_dim": int(seed0_flat.size),
                "compute_full_fd": bool(compute_full_fd),
            }
            try:
                out = compute_linearizations(
                    dyn=dyn,
                    currents=currents,
                    insertion=float(args.insertion),
                    seed_arrays_tuple=seed_arrays_tuple,
                    eps_u=float(args.eps_u),
                    eps_seed=float(args.eps_seed),
                    eps_residual_x=float(args.eps_residual_x),
                    eps_residual_theta=float(args.eps_residual_theta),
                    eps_g_x=float(args.eps_g_x),
                    eps_g_theta=float(args.eps_g_theta),
                    compute_full_fd=compute_full_fd,
                    compute_currents_fd=not bool(args.skip_linearize_action_fd),
                    compute_implicit=not bool(args.skip_implicit),
                )
            except Exception as e:
                row["ok"] = False
                row["error"] = repr(e)
                f.write(json.dumps(row))
                f.write("\n")
                # Attempt to advance; if step fails, reinitialize and retry once.
                step = dyn.step(currents, float(args.insertion))
                if not bool(step.get("converged", True)):
                    dyn.initialize_from_kinematics(currents, float(args.insertion))
                continue

            # Currents-only check for B_fd (no seed perturbation).
            if args.skip_currents_only_validation:
                row["curr_only_err_Bfd"] = float("nan")
            else:
                if out.B_fd is None:
                    raise RuntimeError("currents-only validation requested but B_fd was not computed")
                b_errs: List[float] = []
                for _ in range(args.dirs):
                    du = rng.normal(scale=float(args.du_sigma), size=(3,))
                    step = dyn.step_from_seed(currents + du, float(args.insertion), v, w, p, R, xf, mL, nL)
                    if not bool(step.get("converged", True)):
                        continue
                    y_true = tip_state6(step).reshape(6, 1)
                    y_pred = out.y0 + out.B_fd @ du.reshape(3, 1)
                    b_errs.append(float(np.linalg.norm(y_pred - y_true) / (np.linalg.norm(y_true) + 1e-9)))
                row["curr_only_err_Bfd"] = float(np.mean(b_errs)) if b_errs else float("nan")

            # Full directional checks: implicit vs directional FD baseline.
            if args.skip_directional_validation:
                row["dir_err_dirfd"] = float("nan")
                row["dir_err_imp"] = float("nan")
            else:
                if np.any(~np.isfinite(out.B_imp)):
                    raise RuntimeError("directional validation requested but implicit B_imp was not computed")
                dir_imp_errs: List[float] = []
                dir_fd_errs: List[float] = []
                for _ in range(args.dirs):
                    du = rng.normal(scale=float(args.du_sigma), size=(3,))
                    dv, dw, dp, dR, dxf, dmL, dnL = perturb_seed(v, w, p, R, xf, mL, nL, rng, float(args.seed_sigma))

                    step_p = dyn.step_from_seed(currents + du, float(args.insertion), dv, dw, dp, dR, dxf, dmL, dnL)
                    if not bool(step_p.get("converged", True)):
                        continue
                    y_plus = tip_state6(step_p).reshape(6, 1)

                    dv_m, dw_m, dp_m, dR_m, dxf_m, dmL_m, dnL_m = (
                        2 * v - dv,
                        2 * w - dw,
                        2 * p - dp,
                        2 * R - dR,
                        2 * xf - dxf,
                        2 * mL - dmL,
                        2 * nL - dnL,
                    )
                    step_m = dyn.step_from_seed(
                        currents - du, float(args.insertion), dv_m, dw_m, dp_m, dR_m, dxf_m, dmL_m, dnL_m
                    )
                    if not bool(step_m.get("converged", True)):
                        continue
                    y_minus = tip_state6(step_m).reshape(6, 1)

                    y_pred_dirfd = out.y0 + (y_plus - y_minus) * 0.5
                    dir_fd_errs.append(float(np.linalg.norm(y_pred_dirfd - y_plus) / (np.linalg.norm(y_plus) + 1e-9)))

                    seed_flat = flatten_seed(dv, dw, dp, dR, dxf, dmL, dnL)
                    dseed = (seed_flat - seed0_flat).reshape(-1, 1)
                    y_pred_imp = out.y0 + out.B_imp @ du.reshape(3, 1) + out.A_imp @ dseed
                    dir_imp_errs.append(float(np.linalg.norm(y_pred_imp - y_plus) / (np.linalg.norm(y_plus) + 1e-9)))

                row["dir_err_dirfd"] = float(np.mean(dir_fd_errs)) if dir_fd_errs else float("nan")
                row["dir_err_imp"] = float(np.mean(dir_imp_errs)) if dir_imp_errs else float("nan")

            row["ok"] = True
            row["have_ad_jxx"] = out.have_ad_jxx
            row["residual_norm"] = out.residual_norm
            row["B_fd"] = out.B_fd.tolist() if out.B_fd is not None else None
            row["A_imp"] = out.A_imp.tolist()
            row["B_imp"] = out.B_imp.tolist()
            if out.A_fd_full is not None:
                row["A_fd_full"] = out.A_fd_full.tolist()
            if out.B_fd_full is not None:
                row["B_fd_full"] = out.B_fd_full.tolist()

            f.write(json.dumps(row))
            f.write("\n")
            wrote += 1

            # Advance internal state for the next step (warm-start).
            step = dyn.step(currents, float(args.insertion))
            if not bool(step.get("converged", True)):
                dyn.initialize_from_kinematics(currents, float(args.insertion))
            if args.step_markers:
                print(f"STEP_END k={int(k)}", flush=True)

        print(f"Wrote {wrote} step rows to {args.out_jsonl}")


if __name__ == "__main__":
    main()
