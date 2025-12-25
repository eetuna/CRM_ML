"""
Superimpose dense FK workspace + projected circle (r=10) + projected lemniscate (a=10).

Uses:
  - data/output/workspace_fk_ins94.3_b0.3_step0.01_int0.2.npz (workspace)
  - data/output/fk_ik_test_dp_projection.npz (projected circles)
  - data/output/lemniscate_ypm40_a10_dp_ik.npz (projected lemniscates)

Outputs:
  - plots/superimpose_workspace_circle10_lemniscate10_xy.png
  - plots/superimpose_workspace_circle10_lemniscate10_multiview.png
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib.pyplot as plt  # noqa: E402


def downsample(P: np.ndarray, max_points: int) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    if P.shape[0] <= max_points:
        return P
    idx = np.linspace(0, P.shape[0] - 1, max_points, dtype=int)
    return P[idx]


def main() -> None:
    ws = np.load("data/output/workspace_fk_ins94.3_b0.3_step0.01_int0.2.npz")
    P = np.asarray(ws["P"], dtype=np.float64)
    conv = np.asarray(ws["conv"]).astype(bool)
    ok = conv & np.isfinite(P).all(axis=1)
    P_ok = P[ok]

    circle = np.load("data/output/fk_ik_test_dp_projection.npz")
    c1 = np.asarray(circle["circle1_proj"], dtype=np.float64)
    c2 = np.asarray(circle["circle2_proj"], dtype=np.float64)

    lem = np.load("data/output/lemniscate_ypm40_a10_dp_ik.npz")
    l1 = np.asarray(lem["lem1_proj"], dtype=np.float64)
    l2 = np.asarray(lem["lem2_proj"], dtype=np.float64)

    plots = Path("plots")
    plots.mkdir(parents=True, exist_ok=True)

    P_plot = downsample(P_ok, 250_000)

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    ax.scatter(P_plot[:, 0], P_plot[:, 1], s=2, alpha=0.15, c="0.6", label="workspace")
    ax.plot(c1[:, 0], c1[:, 1], lw=2.0, c="tab:blue", label="circle proj (y=+40,r=10)")
    ax.plot(l1[:, 0], l1[:, 1], lw=2.0, c="tab:orange", label="lemniscate proj (y=+40,a=10)")
    ax.plot(c2[:, 0], c2[:, 1], lw=2.0, c="tab:cyan", label="circle proj (y=-40,r=10)")
    ax.plot(l2[:, 0], l2[:, 1], lw=2.0, c="tab:red", label="lemniscate proj (y=-40,a=10)")
    ax.set_title("Workspace + projected trajectories (XY)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plots / "superimpose_workspace_circle10_lemniscate10_xy.png", dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ax.scatter(P_plot[:, 0], P_plot[:, 1], P_plot[:, 2], s=2, alpha=0.05, c="0.6", label="workspace")
    ax.plot(c1[:, 0], c1[:, 1], c1[:, 2], lw=2.0, c="tab:blue", label="circle +y")
    ax.plot(l1[:, 0], l1[:, 1], l1[:, 2], lw=2.0, c="tab:orange", label="lemniscate +y")
    ax.plot(c2[:, 0], c2[:, 1], c2[:, 2], lw=2.0, c="tab:cyan", label="circle -y")
    ax.plot(l2[:, 0], l2[:, 1], l2[:, 2], lw=2.0, c="tab:red", label="lemniscate -y")
    ax.set_title("3D")
    ax.legend(loc="best")

    ax = fig.add_subplot(2, 2, 2)
    ax.scatter(P_plot[:, 0], P_plot[:, 1], s=2, alpha=0.15, c="0.6", label="workspace")
    ax.plot(c1[:, 0], c1[:, 1], lw=2.0, c="tab:blue", label="circle +y")
    ax.plot(l1[:, 0], l1[:, 1], lw=2.0, c="tab:orange", label="lemniscate +y")
    ax.plot(c2[:, 0], c2[:, 1], lw=2.0, c="tab:cyan", label="circle -y")
    ax.plot(l2[:, 0], l2[:, 1], lw=2.0, c="tab:red", label="lemniscate -y")
    ax.set_title("XY")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax = fig.add_subplot(2, 2, 3)
    ax.scatter(P_plot[:, 0], P_plot[:, 2], s=2, alpha=0.15, c="0.6")
    ax.plot(c1[:, 0], c1[:, 2], lw=2.0, c="tab:blue")
    ax.plot(l1[:, 0], l1[:, 2], lw=2.0, c="tab:orange")
    ax.plot(c2[:, 0], c2[:, 2], lw=2.0, c="tab:cyan")
    ax.plot(l2[:, 0], l2[:, 2], lw=2.0, c="tab:red")
    ax.set_title("XZ")
    ax.grid(True, alpha=0.25)

    ax = fig.add_subplot(2, 2, 4)
    ax.scatter(P_plot[:, 1], P_plot[:, 2], s=2, alpha=0.15, c="0.6")
    ax.plot(c1[:, 1], c1[:, 2], lw=2.0, c="tab:blue")
    ax.plot(l1[:, 1], l1[:, 2], lw=2.0, c="tab:orange")
    ax.plot(c2[:, 1], c2[:, 2], lw=2.0, c="tab:cyan")
    ax.plot(l2[:, 1], l2[:, 2], lw=2.0, c="tab:red")
    ax.set_title("YZ")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(plots / "superimpose_workspace_circle10_lemniscate10_multiview.png", dpi=200)
    plt.close(fig)

    print("Saved: plots/superimpose_workspace_circle10_lemniscate10_xy.png")
    print("Saved: plots/superimpose_workspace_circle10_lemniscate10_multiview.png")


if __name__ == "__main__":
    main()

