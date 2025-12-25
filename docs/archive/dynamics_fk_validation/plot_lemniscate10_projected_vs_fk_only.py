"""
Fast plot: projected lemniscate (a=10) vs FK recomputed under the stored IK currents.

No workspace scatter, no dynamics.

Uses:
  - data/output/lemniscate_ypm40_a10_dp_ik.npz (lem*_proj, lem*_currents)

Outputs:
  - plots/lemniscate10_proj_vs_fk_xy.png
  - plots/lemniscate10_proj_vs_fk_multiview.png
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib.pyplot as plt  # noqa: E402

from crm_ml_rl.wrappers import crm_python  # noqa: E402


def fk_positions(kin: "crm_python.CRMKinematics", currents: np.ndarray, insertion_length: float) -> np.ndarray:
    p = np.zeros((currents.shape[0], 3), dtype=np.float64)
    du0_guess = None
    for i in range(currents.shape[0]):
        u = np.asarray(currents[i], dtype=np.float64).reshape(3)
        if du0_guess is None:
            out = kin.forward_kinematics(u, insertion_length)
        else:
            out = kin.forward_kinematics_with_guess(u, insertion_length, du0_guess)
        p[i] = np.asarray(out["tip_position"], dtype=np.float64).reshape(3)
        du0_guess = np.asarray(out["delta_u0"], dtype=np.float64).reshape(3)
    return p


def main() -> None:
    data = np.load("data/output/lemniscate_ypm40_a10_dp_ik.npz")
    l1_proj = np.asarray(data["lem1_proj"], dtype=np.float64)
    l2_proj = np.asarray(data["lem2_proj"], dtype=np.float64)
    u1 = np.asarray(data["lem1_currents"], dtype=np.float64)
    u2 = np.asarray(data["lem2_currents"], dtype=np.float64)

    insertion_length = 50.0
    kin = crm_python.CRMKinematics()
    assert kin.load_parameters("data/catheter_params/CatheterParameterSet_1_dyn.txt", "data/catheter_params/CatheterSpatialConfiguration_1.txt")
    kin.integration_step_size = 0.2

    l1_fk = fk_positions(kin, u1, insertion_length)
    l2_fk = fk_positions(kin, u2, insertion_length)

    e1 = np.linalg.norm(l1_proj - l1_fk, axis=1)
    e2 = np.linalg.norm(l2_proj - l2_fk, axis=1)

    plots = Path("plots")
    plots.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    ax.plot(l1_proj[:, 0], l1_proj[:, 1], lw=2.0, c="tab:orange", label="lem1 projected (+y)")
    ax.plot(l1_fk[:, 0], l1_fk[:, 1], lw=2.0, c="tab:green", label="lem1 FK(currents)")
    ax.plot(l2_proj[:, 0], l2_proj[:, 1], lw=2.0, c="tab:red", label="lem2 projected (-y)")
    ax.plot(l2_fk[:, 0], l2_fk[:, 1], lw=2.0, c="tab:blue", label="lem2 FK(currents)")
    ax.set_title("Lemniscate10: projected vs FK (XY)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plots / "lemniscate10_proj_vs_fk_xy.png", dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ax.plot(l1_proj[:, 0], l1_proj[:, 1], l1_proj[:, 2], lw=2.0, c="tab:orange", label="lem1 proj")
    ax.plot(l1_fk[:, 0], l1_fk[:, 1], l1_fk[:, 2], lw=2.0, c="tab:green", label="lem1 FK")
    ax.plot(l2_proj[:, 0], l2_proj[:, 1], l2_proj[:, 2], lw=2.0, c="tab:red", label="lem2 proj")
    ax.plot(l2_fk[:, 0], l2_fk[:, 1], l2_fk[:, 2], lw=2.0, c="tab:blue", label="lem2 FK")
    ax.set_title("3D")
    ax.legend(loc="best")

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(l1_proj[:, 0], l1_proj[:, 1], lw=2.0, c="tab:orange", label="lem1 proj")
    ax.plot(l1_fk[:, 0], l1_fk[:, 1], lw=2.0, c="tab:green", label="lem1 FK")
    ax.plot(l2_proj[:, 0], l2_proj[:, 1], lw=2.0, c="tab:red", label="lem2 proj")
    ax.plot(l2_fk[:, 0], l2_fk[:, 1], lw=2.0, c="tab:blue", label="lem2 FK")
    ax.set_title("XY")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax = fig.add_subplot(2, 2, 3)
    ax.plot(l1_proj[:, 0], l1_proj[:, 2], lw=2.0, c="tab:orange")
    ax.plot(l1_fk[:, 0], l1_fk[:, 2], lw=2.0, c="tab:green")
    ax.plot(l2_proj[:, 0], l2_proj[:, 2], lw=2.0, c="tab:red")
    ax.plot(l2_fk[:, 0], l2_fk[:, 2], lw=2.0, c="tab:blue")
    ax.set_title("XZ")
    ax.grid(True, alpha=0.25)

    ax = fig.add_subplot(2, 2, 4)
    ax.plot(l1_proj[:, 1], l1_proj[:, 2], lw=2.0, c="tab:orange")
    ax.plot(l1_fk[:, 1], l1_fk[:, 2], lw=2.0, c="tab:green")
    ax.plot(l2_proj[:, 1], l2_proj[:, 2], lw=2.0, c="tab:red")
    ax.plot(l2_fk[:, 1], l2_fk[:, 2], lw=2.0, c="tab:blue")
    ax.set_title("YZ")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(plots / "lemniscate10_proj_vs_fk_multiview.png", dpi=200)
    plt.close(fig)

    print(f"lem1 ||proj-FK||: mean={e1.mean():.3f}mm p95={np.percentile(e1,95):.3f}mm max={e1.max():.3f}mm")
    print(f"lem2 ||proj-FK||: mean={e2.mean():.3f}mm p95={np.percentile(e2,95):.3f}mm max={e2.max():.3f}mm")
    print("Saved: plots/lemniscate10_proj_vs_fk_xy.png")
    print("Saved: plots/lemniscate10_proj_vs_fk_multiview.png")


if __name__ == "__main__":
    main()

