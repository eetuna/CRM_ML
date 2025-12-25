"""
Superimpose projected circle and projected lemniscate (figure-8) in tip space.

Uses:
  - data/output/fk_ik_test_dp_projection.npz (projected circles)
  - data/output/lemniscate_y40_r5_dp_0p1mm_hold2.npz (projected lemniscate)

Outputs:
  - plots/superimpose_projected_circle_vs_lemniscate_xy.png
  - plots/superimpose_projected_circle_vs_lemniscate_multiview.png
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    circle = np.load("data/output/fk_ik_test_dp_projection.npz")
    lem = np.load("data/output/lemniscate_y40_r5_dp_0p1mm_hold2.npz")

    c1 = np.asarray(circle["circle1_proj"], dtype=np.float64)
    c2 = np.asarray(circle["circle2_proj"], dtype=np.float64)
    l = np.asarray(lem["tip_projected"], dtype=np.float64)

    plots = Path("plots")
    plots.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    ax.plot(c1[:, 0], c1[:, 1], lw=2.0, c="tab:blue", label="circle proj (y=+40,r=10)")
    ax.plot(c2[:, 0], c2[:, 1], lw=2.0, c="tab:cyan", label="circle proj (y=-40,r=10)")
    ax.plot(l[:, 0], l[:, 1], lw=2.0, c="tab:orange", label="lemniscate proj (y=+40,a=5)")
    ax.set_title("Projected trajectories (XY)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plots / "superimpose_projected_circle_vs_lemniscate_xy.png", dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ax.plot(c1[:, 0], c1[:, 1], c1[:, 2], lw=2.0, c="tab:blue", label="circle +y")
    ax.plot(c2[:, 0], c2[:, 1], c2[:, 2], lw=2.0, c="tab:cyan", label="circle -y")
    ax.plot(l[:, 0], l[:, 1], l[:, 2], lw=2.0, c="tab:orange", label="lemniscate +y")
    ax.set_title("3D")
    ax.legend(loc="best")

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(c1[:, 0], c1[:, 1], lw=2.0, c="tab:blue", label="circle +y")
    ax.plot(c2[:, 0], c2[:, 1], lw=2.0, c="tab:cyan", label="circle -y")
    ax.plot(l[:, 0], l[:, 1], lw=2.0, c="tab:orange", label="lemniscate +y")
    ax.set_title("XY")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax = fig.add_subplot(2, 2, 3)
    ax.plot(c1[:, 0], c1[:, 2], lw=2.0, c="tab:blue")
    ax.plot(c2[:, 0], c2[:, 2], lw=2.0, c="tab:cyan")
    ax.plot(l[:, 0], l[:, 2], lw=2.0, c="tab:orange")
    ax.set_title("XZ")
    ax.grid(True, alpha=0.25)

    ax = fig.add_subplot(2, 2, 4)
    ax.plot(c1[:, 1], c1[:, 2], lw=2.0, c="tab:blue")
    ax.plot(c2[:, 1], c2[:, 2], lw=2.0, c="tab:cyan")
    ax.plot(l[:, 1], l[:, 2], lw=2.0, c="tab:orange")
    ax.set_title("YZ")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(plots / "superimpose_projected_circle_vs_lemniscate_multiview.png", dpi=200)
    plt.close(fig)

    print("Saved: plots/superimpose_projected_circle_vs_lemniscate_xy.png")
    print("Saved: plots/superimpose_projected_circle_vs_lemniscate_multiview.png")


if __name__ == "__main__":
    main()

