import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def pick_sample_indices(circle_start: int, circle_len: int, k: int, *, seed: int) -> List[int]:
    rng = np.random.default_rng(seed)
    if k <= 0:
        raise ValueError("k must be > 0")
    k = min(int(k), int(circle_len))
    idxs = rng.choice(np.arange(circle_start, circle_start + circle_len), size=k, replace=False)
    idxs = sorted(int(x) for x in idxs.tolist())
    return idxs


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rel_frob(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(b) + 1e-12)
    return float(np.linalg.norm(a - b) / denom)


def summarize(vals: List[float]) -> Tuple[int, float, float]:
    v = [x for x in vals if np.isfinite(x)]
    if not v:
        return 0, float("nan"), float("nan")
    return len(v), float(np.mean(v)), float(np.max(v))


def run_one(
    so_path: Path,
    currents_csv: Path,
    out_json: Path,
    *,
    insertion: float,
    dt: float,
    integration_step_size: float,
    circle_start: int,
    circle_len: int,
    sample_indices: List[int],
    seed: int,
    dirs: int,
    du_sigma: float,
    seed_sigma: float,
) -> None:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    cmd = [
        "python3",
        "scripts/dynamics_fk_validation/run_circle_linearization_report_one_module.py",
        "--so",
        str(so_path),
        "--currents-csv",
        str(currents_csv),
        "--out-json",
        str(out_json),
        "--insertion",
        str(insertion),
        "--dt",
        str(dt),
        "--integration-step-size",
        str(integration_step_size),
        "--circle-start",
        str(int(circle_start)),
        "--circle-len",
        str(int(circle_len)),
        "--sample-indices",
        ",".join(str(i) for i in sample_indices),
        "--seed",
        str(int(seed)),
        "--dirs",
        str(int(dirs)),
        "--du-sigma",
        str(float(du_sigma)),
        "--seed-sigma",
        str(float(seed_sigma)),
    ]
    subprocess.check_call(cmd, env=env)


def infer_circle_slice_for_ramp_then_circle1(currents_csv: Path) -> Tuple[int, int]:
    # For these specific CSVs in output_data:
    # - hold1: ramp 120 rows + circle 200 rows => total 320
    # - hold2: ramp 240 rows + circle 400 rows => total 640
    # We infer from total length.
    import csv as _csv

    with open(currents_csv, "r", encoding="utf-8") as f:
        n = sum(1 for _ in f) - 1  # skip header
    if n == 320:
        return 120, 200
    if n == 640:
        return 240, 400
    raise RuntimeError(f"Unrecognized ramp_then_circle1 length {n} for {currents_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--currents-csv", required=True)
    ap.add_argument("--eigen-so", default="crm_ml_rl/wrappers/crm_python.cpython-310-x86_64-linux-gnu.so")
    ap.add_argument(
        "--template-so",
        default=".worktrees/autodiff_template/crm_ml_rl/wrappers/crm_python.cpython-310-x86_64-linux-gnu.so",
    )
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--k", type=int, default=25, help="Number of circle timesteps to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dirs", type=int, default=5)
    ap.add_argument("--du-sigma", type=float, default=1e-5)
    ap.add_argument("--seed-sigma", type=float, default=0.0)
    ap.add_argument("--insertion", type=float, default=94.3)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--integration-step-size", type=float, default=0.2)
    args = ap.parse_args()

    currents_csv = Path(args.currents_csv)
    eigen_so = Path(args.eigen_so)
    template_so = Path(args.template_so)
    if not currents_csv.exists():
        raise SystemExit(f"Missing currents csv: {currents_csv}")
    if not eigen_so.exists():
        raise SystemExit(f"Missing eigen .so: {eigen_so}")
    if not template_so.exists():
        raise SystemExit(f"Missing template .so: {template_so}")

    circle_start, circle_len = infer_circle_slice_for_ramp_then_circle1(currents_csv)
    sample_indices = pick_sample_indices(circle_start, circle_len, int(args.k), seed=int(args.seed))

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    eigen_out = out_json.with_suffix(".eigen.json")
    tpl_out = out_json.with_suffix(".template.json")

    run_one(
        eigen_so,
        currents_csv,
        eigen_out,
        insertion=float(args.insertion),
        dt=float(args.dt),
        integration_step_size=float(args.integration_step_size),
        circle_start=circle_start,
        circle_len=circle_len,
        sample_indices=sample_indices,
        seed=int(args.seed),
        dirs=int(args.dirs),
        du_sigma=float(args.du_sigma),
        seed_sigma=float(args.seed_sigma),
    )
    run_one(
        template_so,
        currents_csv,
        tpl_out,
        insertion=float(args.insertion),
        dt=float(args.dt),
        integration_step_size=float(args.integration_step_size),
        circle_start=circle_start,
        circle_len=circle_len,
        sample_indices=sample_indices,
        seed=int(args.seed),
        dirs=int(args.dirs),
        du_sigma=float(args.du_sigma),
        seed_sigma=float(args.seed_sigma),
    )

    eigen_rep = read_json(eigen_out)
    tpl_rep = read_json(tpl_out)

    # Align samples by csv_index
    eig_by_i = {int(s["csv_index"]): s for s in eigen_rep["samples"]}
    tpl_by_i = {int(s["csv_index"]): s for s in tpl_rep["samples"]}
    common = sorted(set(eig_by_i.keys()) & set(tpl_by_i.keys()))

    rel_B: List[float] = []
    eig_dir_imp: List[float] = []
    tpl_dir_imp: List[float] = []
    eig_dir_fd: List[float] = []
    tpl_dir_fd: List[float] = []
    eig_t: List[float] = []
    tpl_t: List[float] = []
    n_dir_ok_min: List[int] = []

    for i in common:
        e = eig_by_i[i]
        t = tpl_by_i[i]
        if e.get("B_imp") is not None and t.get("B_imp") is not None:
            Be = np.asarray(e["B_imp"], dtype=np.float64)
            Bt = np.asarray(t["B_imp"], dtype=np.float64)
            rel_B.append(rel_frob(Be, Bt))
        eig_dir_imp.append(float(e.get("dir_err_imp_mean", float("nan"))))
        tpl_dir_imp.append(float(t.get("dir_err_imp_mean", float("nan"))))
        eig_dir_fd.append(float(e.get("dir_err_dirfd_mean", float("nan"))))
        tpl_dir_fd.append(float(t.get("dir_err_dirfd_mean", float("nan"))))
        eig_t.append(float(e.get("t_implicit_ms", float("nan"))))
        tpl_t.append(float(t.get("t_implicit_ms", float("nan"))))
        n_dir_ok_min.append(int(min(int(e.get("n_dir_ok", 0)), int(t.get("n_dir_ok", 0)))))

    out = {
        "currents_csv": str(currents_csv),
        "circle_start": int(circle_start),
        "circle_len": int(circle_len),
        "sample_indices": sample_indices,
        "seed": int(args.seed),
        "dirs": int(args.dirs),
        "du_sigma": float(args.du_sigma),
        "seed_sigma": float(args.seed_sigma),
        "eigen_report": str(eigen_out),
        "template_report": str(tpl_out),
        "summary": {
            "n_common": int(len(common)),
            "rel_B_imp_eigen_vs_template": {
                "n": summarize(rel_B)[0],
                "mean": summarize(rel_B)[1],
                "max": summarize(rel_B)[2],
            },
            "dir_err_imp": {
                "eigen": {"n": summarize(eig_dir_imp)[0], "mean": summarize(eig_dir_imp)[1], "max": summarize(eig_dir_imp)[2]},
                "template": {"n": summarize(tpl_dir_imp)[0], "mean": summarize(tpl_dir_imp)[1], "max": summarize(tpl_dir_imp)[2]},
            },
            "dir_err_dirfd": {
                "eigen": {"n": summarize(eig_dir_fd)[0], "mean": summarize(eig_dir_fd)[1], "max": summarize(eig_dir_fd)[2]},
                "template": {"n": summarize(tpl_dir_fd)[0], "mean": summarize(tpl_dir_fd)[1], "max": summarize(tpl_dir_fd)[2]},
            },
            "implicit_ms": {
                "eigen": {"n": summarize(eig_t)[0], "mean": summarize(eig_t)[1], "max": summarize(eig_t)[2]},
                "template": {"n": summarize(tpl_t)[0], "mean": summarize(tpl_t)[1], "max": summarize(tpl_t)[2]},
            },
            "n_dir_ok_min": {"min": int(min(n_dir_ok_min)) if n_dir_ok_min else 0},
        },
    }

    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["summary"], indent=2))
    print(f"Wrote merged report: {out_json}")


if __name__ == "__main__":
    main()

