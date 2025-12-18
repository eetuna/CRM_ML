import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def load_crm_python(so_path: str):
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


def flatten_seed(v, w, p, R, xf, mL, nL) -> np.ndarray:
    return np.concatenate(
        [v.reshape(-1), w.reshape(-1), p.reshape(-1), R.reshape(-1), xf.reshape(-1), mL.reshape(-1), nL.reshape(-1)], axis=0
    )


def parse_indices(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


@dataclass
class SampleResult:
    csv_index: int
    currents: List[float]
    converged_step: bool
    have_ad_jxx: Optional[bool]
    residual_norm: Optional[float]
    B_imp: Optional[List[List[float]]]
    dir_err_imp_mean: float
    dir_err_dirfd_mean: float
    n_dir_ok: int
    t_implicit_ms: float


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", required=True)
    ap.add_argument("--currents-csv", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--insertion", type=float, default=94.3)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--integration-step-size", type=float, default=0.2)
    ap.add_argument("--circle-start", type=int, required=True)
    ap.add_argument("--circle-len", type=int, required=True)
    ap.add_argument("--sample-indices", default="", help="Comma-separated CSV indices to evaluate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dirs", type=int, default=5)
    ap.add_argument("--du-sigma", type=float, default=1e-5)
    ap.add_argument("--seed-sigma", type=float, default=0.0)
    ap.add_argument("--eps-residual-x", type=float, default=1e-4)
    ap.add_argument("--eps-residual-theta", type=float, default=1e-5)
    ap.add_argument("--eps-g-x", type=float, default=1e-5)
    ap.add_argument("--eps-g-theta", type=float, default=1e-5)
    args = ap.parse_args()

    rng = np.random.default_rng(int(args.seed))
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

    currents_all = read_currents_csv(args.currents_csv)
    n = int(currents_all.shape[0])
    circle_start = int(args.circle_start)
    circle_len = int(args.circle_len)
    if circle_start < 0 or circle_len <= 0 or circle_start + circle_len > n:
        raise RuntimeError(f"Invalid circle slice: start={circle_start} len={circle_len} n={n}")

    sample_indices = parse_indices(args.sample_indices)
    if not sample_indices:
        raise RuntimeError("sample-indices must be non-empty")
    sample_set = set(sample_indices)

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

    # Initialize from FK at the first current of the whole sequence (ramp-inclusive).
    dyn.initialize_from_kinematics(currents_all[0], float(args.insertion))

    results: List[Dict[str, Any]] = []
    n_reinit = 0
    n_step_fail = 0

    for i in range(n):
        u = currents_all[i]

        # Evaluate only on sampled indices (which should be in the circle segment).
        if i in sample_set:
            seed_dict = dyn.get_seed_state()
            v, w, p, R, xf, mL, nL = seed_to_arrays(seed_dict)
            seed0_flat = flatten_seed(v, w, p, R, xf, mL, nL)

            # Nominal output at (seed,u) to compare against perturbations.
            step0 = dyn.step_from_seed(u, float(args.insertion), v, w, p, R, xf, mL, nL)
            conv0 = bool(step0.get("converged", True))
            y0 = tip_state6(step0).reshape(6, 1)

            t0 = time.perf_counter()
            have_debug = False
            try:
                out_imp = dyn.linearize_full_seed_action_from_seed_implicit(
                    u,
                    float(args.insertion),
                    v,
                    w,
                    p,
                    R,
                    xf,
                    mL,
                    nL,
                    float(args.eps_residual_x),
                    float(args.eps_residual_theta),
                    float(args.eps_g_x),
                    float(args.eps_g_theta),
                    True,
                )
                have_debug = True
            except TypeError:
                out_imp = dyn.linearize_full_seed_action_from_seed_implicit(
                    u,
                    float(args.insertion),
                    v,
                    w,
                    p,
                    R,
                    xf,
                    mL,
                    nL,
                    float(args.eps_residual_x),
                    float(args.eps_residual_theta),
                    float(args.eps_g_x),
                    float(args.eps_g_theta),
                )
            t_ms = (time.perf_counter() - t0) * 1000.0

            base = out_imp["base"]
            if not bool(base.get("converged", True)):
                # Still record; treat as unusable for directional checks.
                results.append(
                    SampleResult(
                        csv_index=int(i),
                        currents=u.tolist(),
                        converged_step=conv0,
                        have_ad_jxx=bool(out_imp.get("have_ad_jxx", False)) if have_debug else None,
                        residual_norm=float(out_imp.get("residual_norm", float("nan"))) if have_debug else None,
                        B_imp=None,
                        dir_err_imp_mean=float("nan"),
                        dir_err_dirfd_mean=float("nan"),
                        n_dir_ok=0,
                        t_implicit_ms=float(t_ms),
                    ).__dict__
                )
            else:
                A_imp = np.asarray(out_imp["A"], dtype=np.float64)
                B_imp = np.asarray(out_imp["B"], dtype=np.float64).reshape(6, 3)
                have_ad_jxx = bool(out_imp.get("have_ad_jxx", False)) if have_debug else None
                residual_norm = float(out_imp.get("residual_norm", float("nan"))) if have_debug else None

                dir_imp_errs: List[float] = []
                dir_fd_errs: List[float] = []
                n_dir_ok = 0
                for _ in range(int(args.dirs)):
                    du = rng.normal(scale=float(args.du_sigma), size=(3,))

                    if float(args.seed_sigma) > 0.0:
                        dv = v + rng.normal(scale=float(args.seed_sigma), size=v.shape)
                        dw = w + rng.normal(scale=float(args.seed_sigma), size=w.shape)
                        dp = p + rng.normal(scale=float(args.seed_sigma), size=p.shape)
                        dR = R + rng.normal(scale=float(args.seed_sigma), size=R.shape)
                        dxf = xf + rng.normal(scale=float(args.seed_sigma), size=xf.shape)
                        dmL = mL + rng.normal(scale=float(args.seed_sigma), size=mL.shape)
                        dnL = nL + rng.normal(scale=float(args.seed_sigma), size=nL.shape)
                    else:
                        dv, dw, dp, dR, dxf, dmL, dnL = v, w, p, R, xf, mL, nL

                    step_p = dyn.step_from_seed(u + du, float(args.insertion), dv, dw, dp, dR, dxf, dmL, dnL)
                    if not bool(step_p.get("converged", True)):
                        continue
                    y_plus = tip_state6(step_p).reshape(6, 1)

                    # Symmetric two-solve baseline; mirror both u and seed.
                    dv_m, dw_m, dp_m, dR_m, dxf_m, dmL_m, dnL_m = (
                        2 * v - dv,
                        2 * w - dw,
                        2 * p - dp,
                        2 * R - dR,
                        2 * xf - dxf,
                        2 * mL - dmL,
                        2 * nL - dnL,
                    )
                    step_m = dyn.step_from_seed(u - du, float(args.insertion), dv_m, dw_m, dp_m, dR_m, dxf_m, dmL_m, dnL_m)
                    if not bool(step_m.get("converged", True)):
                        continue
                    y_minus = tip_state6(step_m).reshape(6, 1)

                    denom = float(np.linalg.norm(y_plus) + 1e-9)
                    y_pred_dirfd = y0 + 0.5 * (y_plus - y_minus)
                    dir_fd_errs.append(float(np.linalg.norm(y_pred_dirfd - y_plus) / denom))

                    if float(args.seed_sigma) > 0.0:
                        seed_flat = flatten_seed(dv, dw, dp, dR, dxf, dmL, dnL)
                        dseed = (seed_flat - seed0_flat).reshape(-1, 1)
                        y_pred_imp = y0 + B_imp @ du.reshape(3, 1) + A_imp @ dseed
                    else:
                        y_pred_imp = y0 + B_imp @ du.reshape(3, 1)
                    dir_imp_errs.append(float(np.linalg.norm(y_pred_imp - y_plus) / denom))

                    n_dir_ok += 1

                results.append(
                    SampleResult(
                        csv_index=int(i),
                        currents=u.tolist(),
                        converged_step=conv0,
                        have_ad_jxx=have_ad_jxx,
                        residual_norm=residual_norm,
                        B_imp=B_imp.tolist(),
                        dir_err_imp_mean=float(np.mean(dir_imp_errs)) if dir_imp_errs else float("nan"),
                        dir_err_dirfd_mean=float(np.mean(dir_fd_errs)) if dir_fd_errs else float("nan"),
                        n_dir_ok=int(n_dir_ok),
                        t_implicit_ms=float(t_ms),
                    ).__dict__
                )

        # Advance the internal state through the whole ramp+circle sequence.
        out_step = dyn.step(u, float(args.insertion))
        if not bool(out_step.get("converged", True)):
            n_step_fail += 1
            # Reinit from FK at the same current and retry once.
            if dyn.initialize_from_kinematics(u, float(args.insertion)):
                n_reinit += 1
                out_step2 = dyn.step(u, float(args.insertion))
                if not bool(out_step2.get("converged", True)):
                    n_step_fail += 1

    report = {
        "module_so": str(args.so),
        "currents_csv": str(args.currents_csv),
        "insertion": float(args.insertion),
        "dt": float(args.dt),
        "integration_step_size": float(args.integration_step_size),
        "n_steps_total": int(n),
        "circle_start": int(circle_start),
        "circle_len": int(circle_len),
        "sample_indices": [int(x) for x in sample_indices],
        "dirs": int(args.dirs),
        "du_sigma": float(args.du_sigma),
        "seed_sigma": float(args.seed_sigma),
        "n_step_fail": int(n_step_fail),
        "n_reinit": int(n_reinit),
        "samples": results,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote report: {out_path}")


if __name__ == "__main__":
    main()

