"""
Plot dense FK workspace with lemniscate (a=10) overlays:
  - projected points (onto workspace)
  - FK tip positions under the IK currents (post-IK)

Uses:
  - data/output/workspace_fk_ins94.3_b0.3_step0.01_int0.2.npz
  - data/output/lemniscate_ypm40_a10_dp_ik.npz

Outputs:
  - plots/lemniscate10_workspace_proj_fk_xy.png
  - plots/lemniscate10_workspace_proj_fk_multiview.png
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib.pyplot as plt  # noqa: E402


def downsample(P: np.ndarray, max_points: int) -> np.ndarray:
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
    P_plot = downsample(P_ok, 250_000)

    lem = np.load("data/output/lemniscate_ypm40_a10_dp_ik.npz")
    l1_proj = np.asarray(lem["lem1_proj"], dtype=np.float64)
    l1_fk = np.asarray(lem["lem1_fk"], dtype=np.float64)
    l2_proj = np.asarray(lem["lem2_proj"], dtype=np.float64)
    l2_fk = np.asarray(lem["lem2_fk"], dtype=np.float64)

    plots = Path("plots")
    plots.mkdir(parents=True, exist_ok=True)

    # XY
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    ax.scatter(P_plot[:, 0], P_plot[:, 1], s=2, alpha=0.15, c="0.6", label="workspace")
    ax.plot(l1_proj[:, 0], l1_proj[:, 1], lw=2.0, c="tab:orange", label="lem1 projected (+y)")
    ax.plot(l1_fk[:, 0], l1_fk[:, 1], lw=2.0, c="tab:green", label="lem1 FK(IK currents)")
    ax.plot(l2_proj[:, 0], l2_proj[:, 1], lw=2.0, c="tab:red", label="lem2 projected (-y)")
    ax.plot(l2_fk[:, 0], l2_fk[:, 1], lw=2.0, c="tab:blue", label="lem2 FK(IK currents)")
    ax.set_title("Workspace + lemniscate10 projected vs FK")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plots / "lemniscate10_workspace_proj_fk_xy.png", dpi=200)
    plt.close(fig)

    # Multiview
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ax.scatter(P_plot[:, 0], P_plot[:, 1], P_plot[:, 2], s=2, alpha=0.05, c="0.6", label="workspace")
    ax.plot(l1_proj[:, 0], l1_proj[:, 1], l1_proj[:, 2], lw=2.0, c="tab:orange", label="lem1 proj")
    ax.plot(l1_fk[:, 0], l1_fk[:, 1], l1_fk[:, 2], lw=2.0, c="tab:green", label="lem1 FK")
    ax.plot(l2_proj[:, 0], l2_proj[:, 1], l2_proj[:, 2], lw=2.0, c="tab:red", label="lem2 proj")
    ax.plot(l2_fk[:, 0], l2_fk[:, 1], l2_fk[:, 2], lw=2.0, c="tab:blue", label="lem2 FK")
    ax.set_title("3D")
    ax.legend(loc="best")

    ax = fig.add_subplot(2, 2, 2)
    ax.scatter(P_plot[:, 0], P_plot[:, 1], s=2, alpha=0.15, c="0.6", label="workspace")
    ax.plot(l1_proj[:, 0], l1_proj[:, 1], lw=2.0, c="tab:orange", label="lem1 proj")
    ax.plot(l1_fk[:, 0], l1_fk[:, 1], lw=2.0, c="tab:green", label="lem1 FK")
    ax.plot(l2_proj[:, 0], l2_proj[:, 1], lw=2.0, c="tab:red", label="lem2 proj")
    ax.plot(l2_fk[:, 0], l2_fk[:, 1], lw=2.0, c="tab:blue", label="lem2 FK")
    ax.set_title("XY")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax = fig.add_subplot(2, 2, 3)
    ax.scatter(P_plot[:, 0], P_plot[:, 2], s=2, alpha=0.15, c="0.6")
    ax.plot(l1_proj[:, 0], l1_proj[:, 2], lw=2.0, c="tab:orange")
    ax.plot(l1_fk[:, 0], l1_fk[:, 2], lw=2.0, c="tab:green")
    ax.plot(l2_proj[:, 0], l2_proj[:, 2], lw=2.0, c="tab:red")
    ax.plot(l2_fk[:, 0], l2_fk[:, 2], lw=2.0, c="tab:blue")
    ax.set_title("XZ")
    ax.grid(True, alpha=0.25)

    ax = fig.add_subplot(2, 2, 4)
    ax.scatter(P_plot[:, 1], P_plot[:, 2], s=2, alpha=0.15, c="0.6")
    ax.plot(l1_proj[:, 1], l1_proj[:, 2], lw=2.0, c="tab:orange")
    ax.plot(l1_fk[:, 1], l1_fk[:, 2], lw=2.0, c="tab:green")
    ax.plot(l2_proj[:, 1], l2_proj[:, 2], lw=2.0, c="tab:red")
    ax.plot(l2_fk[:, 1], l2_fk[:, 2], lw=2.0, c="tab:blue")
    ax.set_title("YZ")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(plots / "lemniscate10_workspace_proj_fk_multiview.png", dpi=200)
    plt.close(fig)

    e1 = np.linalg.norm(l1_proj - l1_fk, axis=1)
    e2 = np.linalg.norm(l2_proj - l2_fk, axis=1)
    print(f"lem1 ||proj-FK||: mean={e1.mean():.3f}mm p95={np.percentile(e1,95):.3f}mm max={e1.max():.3f}mm")
    print(f"lem2 ||proj-FK||: mean={e2.mean():.3f}mm p95={np.percentile(e2,95):.3f}mm max={e2.max():.3f}mm")
    print("Saved: plots/lemniscate10_workspace_proj_fk_xy.png")
    print("Saved: plots/lemniscate10_workspace_proj_fk_multiview.png")


if __name__ == "__main__":
    main()

