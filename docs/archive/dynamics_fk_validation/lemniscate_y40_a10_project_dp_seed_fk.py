"""
Lemniscate (figure-8) centered at (x=0,y=40) with scale a=10mm:
  - project desired points onto dense FK workspace (DP in current space)
  - use the corresponding workspace seed currents directly
  - compute FK tip positions for those currents (warm-started)
  - plot workspace + desired + projected + FK

This avoids dynamics (no "Coil integration Unbounded!!" spam) and focuses on:
  - where the projected lemniscate lies in the workspace
  - FK-vs-projected consistency using the selected currents

Outputs:
  - data/output/lemniscate_y40_a10_dp_seed_fk.npz
  - data/output/lemniscate_currents_y40_a10_dp_seed_n200.csv
  - plots/lemniscate_y40_a10_workspace_des_proj_fk_xy.png
  - plots/lemniscate_y40_a10_workspace_des_proj_fk_multiview.png
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib.pyplot as plt  # noqa: E402

from scipy.spatial import cKDTree  # noqa: E402

from crm_ml_rl.wrappers import crm_python  # noqa: E402


@dataclass(frozen=True)
class Config:
    workspace_npz: str = "data/output/workspace_fk_ins94.3_b0.3_step0.01_int0.2.npz"
    insertion_length: float = 50.0
    integration_step_size: float = 0.2
    param_file: str = "data/catheter_params/CatheterParameterSet_1_dyn.txt"
    config_file: str = "data/catheter_params/CatheterSpatialConfiguration_1.txt"

    center_xy: Tuple[float, float] = (0.0, 40.0)
    a_mm: float = 10.0
    points: int = 200
    start_angle_rad: float = -np.pi / 2.0

    k_nearest: int = 300
    alpha_u: float = 0.4


def downsample(P: np.ndarray, max_points: int) -> np.ndarray:
    if P.shape[0] <= max_points:
        return P
    idx = np.linspace(0, P.shape[0] - 1, max_points, dtype=int)
    return P[idx]


def choose_z0_from_workspace(P_ok: np.ndarray, center_xy: Tuple[float, float]) -> float:
    cx, cy = float(center_xy[0]), float(center_xy[1])
    for r in (5.0, 10.0, 20.0, 40.0):
        m = (np.abs(P_ok[:, 0] - cx) <= r) & (np.abs(P_ok[:, 1] - cy) <= r)
        if np.any(m):
            return float(np.median(P_ok[m, 2]))
    return float(np.median(P_ok[:, 2]))


def lemniscate_points(center_xyz: np.ndarray, a: float, n: int, start_angle: float) -> np.ndarray:
    t = start_angle + np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False, dtype=np.float64)
    pts = np.repeat(center_xyz.reshape(1, 3), int(n), axis=0)
    pts[:, 0] = center_xyz[0] + a * np.sin(t)
    pts[:, 1] = center_xyz[1] + a * np.sin(2.0 * t)
    return pts


def project_dp_indices(
    p_des: np.ndarray,
    U_ok: np.ndarray,
    tree: "cKDTree",
    *,
    k_nearest: int,
    alpha_u: float,
) -> np.ndarray:
    n = int(p_des.shape[0])
    k = int(max(1, k_nearest))
    dists, idxs = tree.query(p_des, k=k, workers=-1)
    if k == 1:
        dists = dists.reshape(n, 1)
        idxs = idxs.reshape(n, 1)

    dp = np.full((n, k), np.inf, dtype=np.float64)
    bp = np.full((n, k), -1, dtype=np.int32)
    dp[0] = dists[0]

    for i in range(1, n):
        prev_u = U_ok[idxs[i - 1]]
        cur_u = U_ok[idxs[i]]
        trans = float(alpha_u) * np.linalg.norm(prev_u[:, None, :] - cur_u[None, :, :], axis=2)
        total = dp[i - 1][:, None] + trans
        argmin = np.argmin(total, axis=0)
        dp[i] = dists[i] + total[argmin, np.arange(k)]
        bp[i] = argmin.astype(np.int32)

    j = int(np.argmin(dp[-1]))
    sel = np.zeros((n,), dtype=np.int32)
    for i in range(n - 1, -1, -1):
        sel[i] = idxs[i, j]
        j = int(bp[i, j]) if i > 0 else j
    return sel


def fk_positions_for_currents(cfg: Config, kin: "crm_python.CRMKinematics", currents: np.ndarray) -> np.ndarray:
    p = np.zeros((currents.shape[0], 3), dtype=np.float64)
    du0_guess = None
    for i in range(currents.shape[0]):
        u = np.asarray(currents[i], dtype=np.float64).reshape(3)
        if du0_guess is None:
            out = kin.forward_kinematics(u, cfg.insertion_length)
        else:
            out = kin.forward_kinematics_with_guess(u, cfg.insertion_length, du0_guess)
        p[i] = np.asarray(out["tip_position"], dtype=np.float64).reshape(3)
        du0_guess = np.asarray(out["delta_u0"], dtype=np.float64).reshape(3)
    return p


def plot_xy(P_plot: np.ndarray, des: np.ndarray, proj: np.ndarray, fk: np.ndarray, out_png: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    ax.scatter(P_plot[:, 0], P_plot[:, 1], s=2, alpha=0.15, c="0.6", label="workspace")
    ax.plot(des[:, 0], des[:, 1], lw=2.0, c="tab:blue", label="desired")
    ax.plot(proj[:, 0], proj[:, 1], lw=2.0, c="tab:orange", label="projected(ws)")
    ax.plot(fk[:, 0], fk[:, 1], lw=2.0, c="tab:green", label="FK(seed currents)")
    ax.set_title("Workspace + lemniscate10 (desired/proj/FK)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_multiview(P_plot: np.ndarray, des: np.ndarray, proj: np.ndarray, fk: np.ndarray, out_png: Path) -> None:
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ax.scatter(P_plot[:, 0], P_plot[:, 1], P_plot[:, 2], s=2, alpha=0.05, c="0.6", label="workspace")
    ax.plot(des[:, 0], des[:, 1], des[:, 2], lw=2.0, c="tab:blue", label="desired")
    ax.plot(proj[:, 0], proj[:, 1], proj[:, 2], lw=2.0, c="tab:orange", label="projected")
    ax.plot(fk[:, 0], fk[:, 1], fk[:, 2], lw=2.0, c="tab:green", label="FK(seed)")
    ax.set_title("3D")
    ax.legend(loc="best")

    ax = fig.add_subplot(2, 2, 2)
    ax.scatter(P_plot[:, 0], P_plot[:, 1], s=2, alpha=0.15, c="0.6", label="workspace")
    ax.plot(des[:, 0], des[:, 1], lw=2.0, c="tab:blue", label="desired")
    ax.plot(proj[:, 0], proj[:, 1], lw=2.0, c="tab:orange", label="projected")
    ax.plot(fk[:, 0], fk[:, 1], lw=2.0, c="tab:green", label="FK(seed)")
    ax.set_title("XY")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax = fig.add_subplot(2, 2, 3)
    ax.scatter(P_plot[:, 0], P_plot[:, 2], s=2, alpha=0.15, c="0.6")
    ax.plot(des[:, 0], des[:, 2], lw=2.0, c="tab:blue")
    ax.plot(proj[:, 0], proj[:, 2], lw=2.0, c="tab:orange")
    ax.plot(fk[:, 0], fk[:, 2], lw=2.0, c="tab:green")
    ax.set_title("XZ")
    ax.grid(True, alpha=0.25)

    ax = fig.add_subplot(2, 2, 4)
    ax.scatter(P_plot[:, 1], P_plot[:, 2], s=2, alpha=0.15, c="0.6")
    ax.plot(des[:, 1], des[:, 2], lw=2.0, c="tab:blue")
    ax.plot(proj[:, 1], proj[:, 2], lw=2.0, c="tab:orange")
    ax.plot(fk[:, 1], fk[:, 2], lw=2.0, c="tab:green")
    ax.set_title("YZ")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    cfg = Config()

    ws = np.load(cfg.workspace_npz)
    P = np.asarray(ws["P"], dtype=np.float64)
    U = np.asarray(ws["U"], dtype=np.float64)
    conv = np.asarray(ws["conv"]).astype(bool)
    ok = conv & np.isfinite(P).all(axis=1)
    P_ok = P[ok]
    U_ok = U[ok]
    if P_ok.size == 0:
        raise RuntimeError("No converged workspace points.")

    z0 = choose_z0_from_workspace(P_ok, cfg.center_xy)
    center = np.array([cfg.center_xy[0], cfg.center_xy[1], z0], dtype=np.float64)
    des = lemniscate_points(center, cfg.a_mm, cfg.points, cfg.start_angle_rad)

    tree = cKDTree(P_ok)
    sel = project_dp_indices(des, U_ok, tree, k_nearest=cfg.k_nearest, alpha_u=cfg.alpha_u)
    proj = P_ok[sel]
    seed_u = U_ok[sel]

    kin = crm_python.CRMKinematics()
    if not kin.load_parameters(cfg.param_file, cfg.config_file):
        raise RuntimeError("Failed to load kinematics parameters.")
    kin.integration_step_size = float(cfg.integration_step_size)
    fk = fk_positions_for_currents(cfg, kin, seed_u)

    e = np.linalg.norm(proj - fk, axis=1)
    print(f"proj-FK (seed currents): mean={e.mean():.3f}mm p95={np.percentile(e,95):.3f}mm max={e.max():.3f}mm")

    out_dir = Path("data/output")
    plots_dir = Path("plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_dir / "lemniscate_y40_a10_dp_seed_fk.npz",
        tip_desired=des,
        tip_projected=proj,
        currents=seed_u,
        tip_fk=fk,
        center_xyz=center,
        k_nearest=cfg.k_nearest,
        alpha_u=cfg.alpha_u,
        points=cfg.points,
        a_mm=cfg.a_mm,
        insertion_length=cfg.insertion_length,
        integration_step_size=cfg.integration_step_size,
    )
    np.savetxt(out_dir / "lemniscate_currents_y40_a10_dp_seed_n200.csv", seed_u, delimiter=",", header="c1,c2,c3", comments="")

    P_plot = downsample(P_ok, 250_000)
    plot_xy(P_plot, des, proj, fk, plots_dir / "lemniscate_y40_a10_workspace_des_proj_fk_xy.png")
    plot_multiview(P_plot, des, proj, fk, plots_dir / "lemniscate_y40_a10_workspace_des_proj_fk_multiview.png")
    print("Saved: data/output/lemniscate_y40_a10_dp_seed_fk.npz")
    print("Saved: data/output/lemniscate_currents_y40_a10_dp_seed_n200.csv")
    print("Saved: plots/lemniscate_y40_a10_workspace_des_proj_fk_xy.png")
    print("Saved: plots/lemniscate_y40_a10_workspace_des_proj_fk_multiview.png")


if __name__ == "__main__":
    main()

