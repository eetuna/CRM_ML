import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def rel_frob(A: np.ndarray, B: np.ndarray) -> float:
    denom = float(np.linalg.norm(B) + 1e-12)
    return float(np.linalg.norm(A - B) / denom)


@dataclass
class Summary:
    n_total: int
    n_ok: int
    mean: float
    max: float


def summarize(vals: List[float]) -> Summary:
    v = [x for x in vals if np.isfinite(x)]
    if not v:
        return Summary(n_total=len(vals), n_ok=0, mean=float("nan"), max=float("nan"))
    return Summary(n_total=len(vals), n_ok=len(v), mean=float(np.mean(v)), max=float(np.max(v)))


def run_one(
    so_path: Path,
    currents_csv: Path,
    out_jsonl: Path,
    *,
    insertion: float,
    dt: float,
    integration_step_size: float,
    seed: int,
    dirs: int,
    du_sigma: float,
    seed_sigma: float,
    full_fd_every: int,
    start: int,
    stride: int,
    max_steps: int,
) -> None:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl")

    cmd = [
        "python3",
        "scripts/dynamics_fk_validation/run_dyn_linearization_sequence_benchmark.py",
        "--so",
        str(so_path),
        "--currents-csv",
        str(currents_csv),
        "--out-jsonl",
        str(out_jsonl),
        "--insertion",
        str(insertion),
        "--dt",
        str(dt),
        "--integration-step-size",
        str(integration_step_size),
        "--seed",
        str(seed),
        "--start",
        str(int(start)),
        "--stride",
        str(int(stride)),
        "--max-steps",
        str(int(max_steps)),
        "--dirs",
        str(dirs),
        "--du-sigma",
        str(du_sigma),
        "--seed-sigma",
        str(seed_sigma),
    ]
    if full_fd_every and full_fd_every > 0:
        cmd += ["--full-fd-every", str(full_fd_every)]

    subprocess.check_call(cmd, env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--currents-csv", required=True)
    ap.add_argument("--eigen-so", default="crm_ml_rl/wrappers/crm_python.cpython-310-x86_64-linux-gnu.so")
    ap.add_argument(
        "--template-so",
        default=".worktrees/autodiff_template/crm_ml_rl/wrappers/crm_python.cpython-310-x86_64-linux-gnu.so",
    )
    ap.add_argument("--out-prefix", default="output_data/seq_bench")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dirs", type=int, default=6)
    ap.add_argument("--du-sigma", type=float, default=1e-4)
    ap.add_argument("--seed-sigma", type=float, default=1e-4)
    ap.add_argument("--insertion", type=float, default=94.3)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--integration-step-size", type=float, default=0.2)
    ap.add_argument("--full-fd-every", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=1000000)
    args = ap.parse_args()

    currents_csv = Path(args.currents_csv)
    if not currents_csv.exists():
        raise SystemExit(f"Missing currents csv: {currents_csv}")

    eigen_so = Path(args.eigen_so)
    tpl_so = Path(args.template_so)
    if not eigen_so.exists():
        raise SystemExit(f"Missing eigen .so: {eigen_so}")
    if not tpl_so.exists():
        raise SystemExit(f"Missing template .so: {tpl_so}")

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    eigen_out = out_prefix.with_suffix("").as_posix() + "_eigen.jsonl"
    tpl_out = out_prefix.with_suffix("").as_posix() + "_template.jsonl"
    eigen_out_p = Path(eigen_out)
    tpl_out_p = Path(tpl_out)

    run_one(
        eigen_so,
        currents_csv,
        eigen_out_p,
        insertion=float(args.insertion),
        dt=float(args.dt),
        integration_step_size=float(args.integration_step_size),
        seed=int(args.seed),
        dirs=int(args.dirs),
        du_sigma=float(args.du_sigma),
        seed_sigma=float(args.seed_sigma),
        full_fd_every=int(args.full_fd_every),
        start=int(args.start),
        stride=int(args.stride),
        max_steps=int(args.max_steps),
    )
    run_one(
        tpl_so,
        currents_csv,
        tpl_out_p,
        insertion=float(args.insertion),
        dt=float(args.dt),
        integration_step_size=float(args.integration_step_size),
        seed=int(args.seed),
        dirs=int(args.dirs),
        du_sigma=float(args.du_sigma),
        seed_sigma=float(args.seed_sigma),
        full_fd_every=int(args.full_fd_every),
        start=int(args.start),
        stride=int(args.stride),
        max_steps=int(args.max_steps),
    )

    eigen_rows = read_jsonl(eigen_out_p)
    tpl_rows = read_jsonl(tpl_out_p)

    # Align on k where both are ok.
    tpl_by_k = {int(r["k"]): r for r in tpl_rows if "k" in r}
    common: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for r in eigen_rows:
        k = int(r.get("k", -1))
        if k in tpl_by_k:
            common.append((r, tpl_by_k[k]))

    rel_B_imp: List[float] = []
    rel_A_imp: List[float] = []
    rel_B_fd: List[float] = []
    eig_curr_err: List[float] = []
    tpl_curr_err: List[float] = []
    eig_dir_imp: List[float] = []
    tpl_dir_imp: List[float] = []
    eig_dir_fd: List[float] = []
    tpl_dir_fd: List[float] = []

    for e, t in common:
        if not (bool(e.get("ok", False)) and bool(t.get("ok", False))):
            continue
        Be = np.asarray(e["B_imp"], dtype=np.float64)
        Bt = np.asarray(t["B_imp"], dtype=np.float64)
        rel_B_imp.append(rel_frob(Be, Bt))

        Ae = np.asarray(e["A_imp"], dtype=np.float64)
        At = np.asarray(t["A_imp"], dtype=np.float64)
        rel_A_imp.append(rel_frob(Ae, At))

        Bfde = np.asarray(e["B_fd"], dtype=np.float64)
        Bfdt = np.asarray(t["B_fd"], dtype=np.float64)
        rel_B_fd.append(rel_frob(Bfde, Bfdt))

        eig_curr_err.append(float(e.get("curr_only_err_Bfd", float("nan"))))
        tpl_curr_err.append(float(t.get("curr_only_err_Bfd", float("nan"))))
        eig_dir_imp.append(float(e.get("dir_err_imp", float("nan"))))
        tpl_dir_imp.append(float(t.get("dir_err_imp", float("nan"))))
        eig_dir_fd.append(float(e.get("dir_err_dirfd", float("nan"))))
        tpl_dir_fd.append(float(t.get("dir_err_dirfd", float("nan"))))

    print(f"Currents CSV: {currents_csv}")
    print(f"Eigen JSONL: {eigen_out_p}")
    print(f"Template JSONL: {tpl_out_p}")
    print(f"Common steps: {len(common)}")
    sB = summarize(rel_B_imp)
    sA = summarize(rel_A_imp)
    sBfd = summarize(rel_B_fd)
    print(f"eigen vs template: rel||B_imp|| mean/max {sB.mean:.3e} / {sB.max:.3e} (n={sB.n_ok})")
    print(f"eigen vs template: rel||A_imp|| mean/max {sA.mean:.3e} / {sA.max:.3e} (n={sA.n_ok})")
    print(f"eigen vs template: rel||B_fd|| mean/max {sBfd.mean:.3e} / {sBfd.max:.3e} (n={sBfd.n_ok})")

    se = summarize(eig_curr_err)
    st = summarize(tpl_curr_err)
    print(f"curr-only pred err (B_fd) eigen mean/max {se.mean:.3e} / {se.max:.3e} (n={se.n_ok})")
    print(f"curr-only pred err (B_fd) tpl   mean/max {st.mean:.3e} / {st.max:.3e} (n={st.n_ok})")

    se = summarize(eig_dir_fd)
    st = summarize(tpl_dir_fd)
    print(f"directional FD baseline err eigen mean/max {se.mean:.3e} / {se.max:.3e} (n={se.n_ok})")
    print(f"directional FD baseline err tpl   mean/max {st.mean:.3e} / {st.max:.3e} (n={st.n_ok})")

    se = summarize(eig_dir_imp)
    st = summarize(tpl_dir_imp)
    print(f"directional implicit err eigen mean/max {se.mean:.3e} / {se.max:.3e} (n={se.n_ok})")
    print(f"directional implicit err tpl   mean/max {st.mean:.3e} / {st.max:.3e} (n={st.n_ok})")


if __name__ == "__main__":
    main()
