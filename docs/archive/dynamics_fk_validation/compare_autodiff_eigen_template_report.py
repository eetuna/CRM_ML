import argparse
import json
import os
import subprocess
import sys

import numpy as np


def frob_rel(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(b) + 1e-12
    return float(np.linalg.norm(a - b) / denom)


def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--current-so", default="crm_ml_rl/wrappers/crm_python.cpython-310-x86_64-linux-gnu.so")
    ap.add_argument(
        "--template-so",
        default=".worktrees/autodiff_template/crm_ml_rl/wrappers/crm_python.cpython-310-x86_64-linux-gnu.so",
    )
    ap.add_argument("--candidates", type=int, default=80)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--dirs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--insertion", type=float, default=50.0)
    args = ap.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

    # Get a reference seed using the *current* module (already importable in this workspace).
    from crm_ml_rl.wrappers import crm_python as cur_mod

    dyn = cur_mod.CRMDynamics()
    ok = dyn.load_parameters(
        "data/catheter_params/CatheterParameterSet_1_dyn.txt",
        "data/catheter_params/CatheterSpatialConfiguration_1.txt",
    )
    if not ok:
        raise RuntimeError("Failed to load parameters with current module")
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
    dyn.dt = 0.05
    dyn.integration_step_size = 0.01
    dyn.initialize_from_kinematics([0.0, 0.0, 0.2], args.insertion)
    seed = dyn.get_seed_state()

    seed_npz = "data/output/seed_for_compare_autodiff.npz"
    os.makedirs("data/output", exist_ok=True)
    np.savez(
        seed_npz,
        v=np.asarray(seed["v"], dtype=np.float64),
        w=np.asarray(seed["w"], dtype=np.float64),
        p=np.asarray(seed["p"], dtype=np.float64),
        R=np.asarray(seed["R"], dtype=np.float64),
        xf=np.asarray(seed["xf"], dtype=np.float64),
        mL=np.asarray(seed["mL"], dtype=np.float64),
        nL=np.asarray(seed["nL"], dtype=np.float64),
    )

    rng = np.random.default_rng(args.seed)
    # Keep currents small to avoid coil integration blow-ups (which also spam stdout).
    currents = rng.uniform(low=-0.01, high=0.01, size=(args.candidates, 3)).astype(np.float64)
    currents[:, 2] += 0.02
    currents_npz = "data/output/currents_for_compare_autodiff.npz"
    np.savez(currents_npz, currents=currents)

    eigen_out = "data/output/bench_eigen.jsonl"
    tpl_out = "data/output/bench_template.jsonl"

    cmd = [
        sys.executable,
        "scripts/run_dyn_linearization_benchmark.py",
        "--seed-npz",
        seed_npz,
        "--currents-npz",
        currents_npz,
        "--trials",
        str(args.trials),
        "--dirs",
        str(args.dirs),
        "--seed",
        str(args.seed),
        "--insertion",
        str(args.insertion),
    ]

    os.makedirs("data/output", exist_ok=True)
    os.makedirs("/tmp/mpl", exist_ok=True)
    env = dict(os.environ)
    env["MPLCONFIGDIR"] = "/tmp/mpl"
    env["PYTHONWARNINGS"] = "ignore"

    with open("data/output/bench_eigen.log", "w", encoding="utf-8") as log:
        subprocess.run(cmd + ["--so", args.current_so, "--out-jsonl", eigen_out], check=True, env=env, stdout=log, stderr=log)
    with open("data/output/bench_template.log", "w", encoding="utf-8") as log:
        subprocess.run(cmd + ["--so", args.template_so, "--out-jsonl", tpl_out], check=True, env=env, stdout=log, stderr=log)

    eigen_rows = {r["idx"]: r for r in load_jsonl(eigen_out)}
    tpl_rows = {r["idx"]: r for r in load_jsonl(tpl_out)}
    common = sorted(set(eigen_rows.keys()) & set(tpl_rows.keys()))

    if not common:
        raise RuntimeError("No common converged trials between eigen and template runs")

    # Aggregate comparisons on common indices.
    rel_B_eigen_vs_Bfd = []
    rel_B_tpl_vs_Bfd = []
    rel_B_eigen_vs_tpl = []
    err_Bfd = []
    err_dirfd = []
    err_eigen = []
    err_tpl = []
    have_ad_jxx = 0

    for i in common:
        er = eigen_rows[i]
        tr = tpl_rows[i]

        B_fd = np.asarray(er["B_fd"], dtype=np.float64)
        B_e = np.asarray(er["B_imp"], dtype=np.float64)
        B_t = np.asarray(tr["B_imp"], dtype=np.float64)

        rel_B_eigen_vs_Bfd.append(frob_rel(B_e, B_fd))
        rel_B_tpl_vs_Bfd.append(frob_rel(B_t, B_fd))
        rel_B_eigen_vs_tpl.append(frob_rel(B_e, B_t))

        err_Bfd.append(float(er["curr_only_err_Bfd"]))
        err_dirfd.append(float(er["dir_err_dirfd"]))
        err_eigen.append(float(er["dir_err_imp"]))
        err_tpl.append(float(tr["dir_err_imp"]))

        if bool(er.get("have_ad_jxx", False)):
            have_ad_jxx += 1

    def mm(x):
        x = np.asarray(x, dtype=np.float64)
        return float(np.nanmean(x)), float(np.nanmax(x))

    print(f"Candidates: {args.candidates}")
    print(f"Trials used (common converged): {len(common)}")
    print(f"Eigen implicit had AD Jxx in: {have_ad_jxx}/{len(common)}")
    print("")

    m, M = mm(rel_B_eigen_vs_Bfd)
    print(f"eigen_imp vs B_fd: rel||B|| mean/max {m:.3e}/{M:.3e}")
    m, M = mm(rel_B_tpl_vs_Bfd)
    print(f"tpl_imp vs B_fd:   rel||B|| mean/max {m:.3e}/{M:.3e}")
    m, M = mm(rel_B_eigen_vs_tpl)
    print(f"eigen_imp vs tpl:  rel||B|| mean/max {m:.3e}/{M:.3e}")
    print("")

    m, M = mm(err_Bfd)
    print(f"Currents-only check B_fd:    mean/max {m:.3e}/{M:.3e}")
    m, M = mm(err_dirfd)
    print(f"Directional FD baseline:     mean/max {m:.3e}/{M:.3e}")
    m, M = mm(err_eigen)
    print(f"Directional eigen_imp:       mean/max {m:.3e}/{M:.3e}")
    m, M = mm(err_tpl)
    print(f"Directional tpl_imp:         mean/max {m:.3e}/{M:.3e}")


if __name__ == "__main__":
    main()
