"""
Sample FK workspace on a 3D current grid and cache results.

Example (10mA steps on [-0.3,0.3]):
  MPLCONFIGDIR=/tmp/mpl python3 scripts/fk_workspace_grid.py --bound 0.3 --step 0.01 --ins 94.3 --integration-step 0.2
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib.pyplot as plt  # noqa: E402

from crm_ml_rl.wrappers import crm_python  # noqa: E402


def make_grid(bound: float, step: float) -> np.ndarray:
    bound = float(bound)
    step = float(step)
    if step <= 0:
        raise ValueError("step must be > 0")
    # Inclusive endpoint grid.
    n = int(round((2.0 * bound) / step)) + 1
    vals = np.linspace(-bound, bound, n, dtype=np.float64)
    return vals


def fk_workspace_grid(
    kin: "crm_python.CRMKinematics",
    vals: np.ndarray,
    insertion_length: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = int(vals.size)
    total = n**3
    U = np.zeros((total, 3), dtype=np.float64)
    P = np.full((total, 3), np.nan, dtype=np.float64)
    DU0 = np.full((total, 3), np.nan, dtype=np.float64)
    conv = np.zeros((total,), dtype=np.uint8)

    idx = 0
    du0_guess = None

    for iz, c3 in enumerate(vals):
        yvals = vals[::-1] if (iz % 2 == 1) else vals
        for iy, c2 in enumerate(yvals):
            xvals = vals[::-1] if ((iz + iy) % 2 == 1) else vals
            for c1 in xvals:
                u = np.array([c1, c2, c3], dtype=np.float64)
                U[idx] = u
                if du0_guess is None:
                    out = kin.forward_kinematics(u, insertion_length)
                else:
                    out = kin.forward_kinematics_with_guess(u, insertion_length, du0_guess)
                ok = bool(out["converged"])
                conv[idx] = 1 if ok else 0
                if ok:
                    P[idx] = np.asarray(out["tip_position"], dtype=np.float64).reshape(3)
                    du0_sol = np.asarray(out["delta_u0"], dtype=np.float64).reshape(3)
                    DU0[idx] = du0_sol
                    du0_guess = du0_sol

                idx += 1
                if idx % 5000 == 0:
                    print(f"FK workspace: {idx}/{total} ({idx/total:.1%})")

    return U, P, DU0, conv


def plot_workspace_xy(P: np.ndarray, conv: np.ndarray, out_png: Path, *, max_points: int = 200_000) -> None:
    ok = conv.astype(bool) & np.isfinite(P).all(axis=1)
    P_ok = P[ok]
    if P_ok.shape[0] > max_points:
        # Deterministic downsample for plotting only.
        idx = np.linspace(0, P_ok.shape[0] - 1, max_points, dtype=int)
        P_ok = P_ok[idx]

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    ax.scatter(P_ok[:, 0], P_ok[:, 1], s=2, alpha=0.25)
    ax.set_title("FK workspace (XY)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--param-file", default="data/catheter_params/CatheterParameterSet_1_dyn.txt")
    ap.add_argument("--config-file", default="data/catheter_params/CatheterSpatialConfiguration_1.txt")
    ap.add_argument("--ins", type=float, default=94.3, help="Insertion length (mm)")
    ap.add_argument("--bound", type=float, default=0.3, help="Current bound (A)")
    ap.add_argument("--step", type=float, default=0.01, help="Current grid step (A)")
    ap.add_argument("--integration-step", type=float, default=0.2, help="Along-rod integration step (mm)")
    ap.add_argument("--out", default="", help="Output npz path (optional)")
    args = ap.parse_args()

    vals = make_grid(args.bound, args.step)
    total = int(vals.size) ** 3
    print(f"Grid: n={vals.size} per axis => total={total} FK solves")

    out_dir = Path("data/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.out:
        out_npz = Path(args.out)
    else:
        out_npz = out_dir / f"workspace_fk_ins{args.ins}_b{args.bound}_step{args.step}_int{args.integration_step}.npz"

    if out_npz.exists():
        print(f"Cache hit: {out_npz}")
        data = np.load(out_npz)
        U = data["U"]
        P = data["P"]
        DU0 = data["DU0"] if "DU0" in data else None
        conv = data["conv"]
    else:
        kin = crm_python.CRMKinematics()
        if not kin.load_parameters(args.param_file, args.config_file):
            raise RuntimeError("Failed to load parameters into CRMKinematics.")
        kin.integration_step_size = float(args.integration_step)

        U, P, DU0, conv = fk_workspace_grid(kin, vals, float(args.ins))
        np.savez_compressed(out_npz, U=U, P=P, DU0=DU0, conv=conv, vals=vals)

    ok = conv.astype(bool) & np.isfinite(P).all(axis=1)
    P_ok = P[ok]
    print(f"Converged: {int(ok.sum())}/{len(ok)} ({ok.mean():.1%})")
    if len(P_ok):
        mins = P_ok.min(axis=0)
        maxs = P_ok.max(axis=0)
        print(f"Tip range (mm): x[{mins[0]:.1f},{maxs[0]:.1f}] y[{mins[1]:.1f},{maxs[1]:.1f}] z[{mins[2]:.1f},{maxs[2]:.1f}]")

    plots_dir = Path("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_png = plots_dir / f"workspace_fk_xy_ins{args.ins}_b{args.bound}_step{args.step}.png"
    plot_workspace_xy(P, conv, out_png)
    print(f"Saved: {out_npz}")
    print(f"Saved: {out_png}")
    if DU0 is None:
        print("Note: workspace cache is missing DU0; rerun with --out to regenerate including DU0.")


if __name__ == "__main__":
    main()
