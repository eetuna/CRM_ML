"""
Generate a figure-8 (lemniscate) tip trajectory centered at (x=0,y=40) with scale ~5mm,
project it onto a dense FK workspace, generate IK currents (DP projection + normalized pinv),
then run FK and dynamics on the resulting current sequence and plot comparisons.

Outputs:
  - data/output/lemniscate_y40_r5_dp_0p1mm_hold2.npz
  - data/output/lemniscate_currents_y40_r5_dp_0p1mm_n200.csv
  - plots/lemniscate_y40_r5_multiview.png
  - plots/lemniscate_y40_r5_timeseries.png
  - plots/lemniscate_y40_r5_workspace_projection_xy.png
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
from crm_ml_rl.wrappers.crm_wrapper import CRMWrapper, HAS_CPP_BINDINGS  # noqa: E402


@dataclass(frozen=True)
class Config:
    # Workspace cache (dense)
    workspace_npz: str = "data/output/workspace_fk_ins94.3_b0.3_step0.01_int0.2.npz"

    # Model
    insertion_length: float = 50.0
    dt: float = 0.05
    integration_step_size: float = 0.2
    param_file: str = "data/catheter_params/CatheterParameterSet_1_dyn.txt"
    config_file: str = "data/catheter_params/CatheterSpatialConfiguration_1.txt"

    # Desired lemniscate in XY
    center_xy: Tuple[float, float] = (0.0, 40.0)
    scale_mm: float = 5.0
    points: int = 200  # one lap
    start_angle_rad: float = -np.pi / 2.0

    # DP projection params
    k_nearest: int = 300
    alpha_u: float = 0.4

    # IK params (tight)
    step_size: float = 1e-3
    threshold_start_mm: float = 0.1
    threshold_traj_mm: float = 0.1
    itrmax: int = 500

    # Dynamics timing
    c3_start: float = 0.01
    ramp_steps: int = 120
    hold_steps_ramp: int = 2
    hold_steps_per_point: int = 2
    nonzero_eps: float = 0.005


def hold_each(currents: np.ndarray, hold_steps: int) -> np.ndarray:
    hold_steps = int(max(1, hold_steps))
    return np.repeat(np.asarray(currents, dtype=np.float64), hold_steps, axis=0)


def interp_ramp(u0: np.ndarray, u1: np.ndarray, steps: int) -> np.ndarray:
    steps = int(max(1, steps))
    t = np.linspace(0.0, 1.0, steps, endpoint=False, dtype=np.float64)
    return (1.0 - t)[:, None] * u0[None, :] + t[:, None] * u1[None, :]


def ensure_nonzero(currents: np.ndarray, eps: float, *, prefer_axis: int = 2) -> np.ndarray:
    eps = float(abs(eps))
    if eps <= 0:
        eps = 1e-6
    out = np.asarray(currents, dtype=np.float64).copy()
    norms = np.linalg.norm(out, axis=1)
    bad = norms < eps
    if np.any(bad):
        sign = np.sign(out[bad, prefer_axis])
        sign[sign == 0.0] = 1.0
        out[bad, prefer_axis] = sign * eps
    return out


def choose_z0_from_workspace(P_ok: np.ndarray, center_xy: Tuple[float, float]) -> float:
    cx, cy = float(center_xy[0]), float(center_xy[1])
    for r in (5.0, 10.0, 20.0, 40.0):
        m = (np.abs(P_ok[:, 0] - cx) <= r) & (np.abs(P_ok[:, 1] - cy) <= r)
        if np.any(m):
            return float(np.median(P_ok[m, 2]))
    return float(np.median(P_ok[:, 2]))


def lemniscate_points(center_xyz: np.ndarray, a: float, n: int, start_angle: float) -> np.ndarray:
    """
    Simple figure-8 using a Lissajous-type curve:
      x = a * sin(t)
      y = a * sin(2t)
    This yields a sideways figure-8 centered at the origin with ~a amplitude in both x and y.
    """
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


def pinv_normalized_track(
    cfg: Config,
    kin: "crm_python.CRMKinematics",
    p_targets: np.ndarray,
    u_start: np.ndarray,
    du0_guess: np.ndarray | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    U_out = np.zeros((len(p_targets), 3), dtype=np.float64)
    P_fk = np.zeros((len(p_targets), 3), dtype=np.float64)
    err = np.zeros((len(p_targets),), dtype=np.float64)

    u = np.asarray(u_start, dtype=np.float64).reshape(3).copy()
    for i in range(len(p_targets)):
        thr = cfg.threshold_start_mm if i == 0 else cfg.threshold_traj_mm
        for _ in range(int(cfg.itrmax)):
            if du0_guess is None:
                out = kin.fk_and_jacobian(u, cfg.insertion_length)
            else:
                out = kin.fk_and_jacobian(u, cfg.insertion_length, du0_guess)
            p = np.asarray(out["tip_position"], dtype=np.float64).reshape(3)
            du0_guess = np.asarray(out["delta_u0"], dtype=np.float64).reshape(3)
            J = np.asarray(out["jacobian"], dtype=np.float64)[:3, :3]
            e = p_targets[i] - p
            e_norm = float(np.linalg.norm(e))
            if e_norm <= thr:
                break
            du = np.linalg.pinv(J) @ e
            du_norm = float(np.linalg.norm(du))
            if du_norm < 1e-12 or not np.isfinite(du_norm):
                break
            u = u + cfg.step_size * (du / du_norm)

        if du0_guess is None:
            out = kin.forward_kinematics(u, cfg.insertion_length)
        else:
            out = kin.forward_kinematics_with_guess(u, cfg.insertion_length, du0_guess)
        p = np.asarray(out["tip_position"], dtype=np.float64).reshape(3)
        du0_guess = np.asarray(out["delta_u0"], dtype=np.float64).reshape(3)

        U_out[i] = u
        P_fk[i] = p
        err[i] = float(np.linalg.norm(p_targets[i] - p))

    return U_out, P_fk, err, du0_guess


def fk_trajectory(cfg: Config, kin: "crm_python.CRMKinematics", currents: np.ndarray) -> np.ndarray:
    p_fk = np.zeros((currents.shape[0], 3), dtype=np.float64)
    du0_guess = None
    for i in range(currents.shape[0]):
        u = currents[i]
        if du0_guess is None:
            out = kin.forward_kinematics(u, cfg.insertion_length)
        else:
            out = kin.forward_kinematics_with_guess(u, cfg.insertion_length, du0_guess)
        p_fk[i] = np.asarray(out["tip_position"], dtype=np.float64).reshape(3)
        du0_guess = np.asarray(out["delta_u0"], dtype=np.float64).reshape(3)
    return p_fk


def run_dynamics(cfg: Config, currents: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    wrapper = CRMWrapper(
        param_file=cfg.param_file,
        config_file=cfg.config_file,
        use_cpp=True,
        disable_cpp_fallback=True,
    )
    if not (HAS_CPP_BINDINGS and wrapper.is_using_cpp):
        raise RuntimeError("C++ bindings not active for dynamics.")

    wrapper._cpp_dynamics.dt = float(cfg.dt)
    wrapper._cpp_dynamics.integration_step_size = float(cfg.integration_step_size)

    ok = wrapper.initialize_dynamics(currents[0], insertion_length=cfg.insertion_length)
    if not ok:
        raise RuntimeError("Dynamics initialize_dynamics failed at first current.")

    n = int(currents.shape[0])
    p_dyn = np.zeros((n, 3), dtype=np.float64)
    conv = np.zeros((n,), dtype=bool)
    for i in range(n):
        out = wrapper.step_dynamics(currents[i], insertion_length=cfg.insertion_length, dt=cfg.dt)
        p_dyn[i] = np.asarray(out["tip_position"], dtype=np.float64).reshape(3)
        conv[i] = bool(out.get("converged", False))
        if not conv[i]:
            ok2 = wrapper.initialize_dynamics(currents[i], insertion_length=cfg.insertion_length)
            if ok2:
                out2 = wrapper.step_dynamics(currents[i], insertion_length=cfg.insertion_length, dt=cfg.dt)
                p_dyn[i] = np.asarray(out2["tip_position"], dtype=np.float64).reshape(3)
                conv[i] = bool(out2.get("converged", False))
    t = np.arange(n, dtype=np.float64) * cfg.dt
    return t, p_dyn, conv


def plot_workspace_projection_xy(P_ok: np.ndarray, p_des: np.ndarray, p_proj: np.ndarray, out_png: Path) -> None:
    P_plot = P_ok
    if P_plot.shape[0] > 200_000:
        idx = np.linspace(0, P_plot.shape[0] - 1, 200_000, dtype=int)
        P_plot = P_plot[idx]
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    ax.scatter(P_plot[:, 0], P_plot[:, 1], s=2, alpha=0.18, c="0.6", label="workspace")
    ax.plot(p_des[:, 0], p_des[:, 1], lw=2.0, c="tab:blue", label="desired lemniscate")
    ax.scatter(p_proj[:, 0], p_proj[:, 1], s=10, c="tab:orange", label="projected(ws)")
    ax.set_title("Workspace projection (XY)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_multiview(p_des: np.ndarray, p_proj: np.ndarray, p_fk: np.ndarray, p_dyn: np.ndarray, conv_dyn: np.ndarray, out_png: Path) -> None:
    ok = conv_dyn
    fig = plt.figure(figsize=(12, 9))

    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ax.plot(p_des[:, 0], p_des[:, 1], p_des[:, 2], lw=2.0, c="tab:blue", label="desired")
    ax.scatter(p_proj[:, 0], p_proj[:, 1], p_proj[:, 2], s=10, c="tab:orange", label="projected")
    ax.plot(p_fk[:, 0], p_fk[:, 1], p_fk[:, 2], lw=2.0, c="tab:green", label="FK")
    ax.plot(p_dyn[ok, 0], p_dyn[ok, 1], p_dyn[ok, 2], lw=1.5, c="tab:red", label="Dyn(ok)")
    ax.set_title("3D")
    ax.legend(loc="best")

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(p_des[:, 0], p_des[:, 1], lw=2.0, c="tab:blue", label="desired")
    ax.scatter(p_proj[:, 0], p_proj[:, 1], s=10, c="tab:orange", label="projected")
    ax.plot(p_fk[:, 0], p_fk[:, 1], lw=2.0, c="tab:green", label="FK")
    ax.plot(p_dyn[ok, 0], p_dyn[ok, 1], lw=1.5, c="tab:red", label="Dyn(ok)")
    ax.set_title("XY")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax = fig.add_subplot(2, 2, 3)
    ax.plot(p_des[:, 0], p_des[:, 2], lw=2.0, c="tab:blue")
    ax.scatter(p_proj[:, 0], p_proj[:, 2], s=10, c="tab:orange")
    ax.plot(p_fk[:, 0], p_fk[:, 2], lw=2.0, c="tab:green")
    ax.plot(p_dyn[ok, 0], p_dyn[ok, 2], lw=1.5, c="tab:red")
    ax.set_title("XZ")
    ax.grid(True, alpha=0.25)

    ax = fig.add_subplot(2, 2, 4)
    ax.plot(p_des[:, 1], p_des[:, 2], lw=2.0, c="tab:blue")
    ax.scatter(p_proj[:, 1], p_proj[:, 2], s=10, c="tab:orange")
    ax.plot(p_fk[:, 1], p_fk[:, 2], lw=2.0, c="tab:green")
    ax.plot(p_dyn[ok, 1], p_dyn[ok, 2], lw=1.5, c="tab:red")
    ax.set_title("YZ")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_timeseries(t: np.ndarray, u: np.ndarray, p_proj: np.ndarray, p_fk: np.ndarray, p_dyn: np.ndarray, conv_dyn: np.ndarray, out_png: Path) -> None:
    e_ik = np.linalg.norm(p_proj - p_fk, axis=1)
    e_dyn = np.linalg.norm(p_fk - p_dyn, axis=1)
    fig, axs = plt.subplots(5, 1, figsize=(12, 12), sharex=True)

    axs[0].plot(t, u[:, 0], label="c1")
    axs[0].plot(t, u[:, 1], label="c2")
    axs[0].plot(t, u[:, 2], label="c3")
    axs[0].set_ylabel("currents (A)")
    axs[0].legend(ncols=3, loc="upper right")

    axs[1].plot(t, e_ik, c="tab:green", label="||proj-FK||")
    axs[1].plot(t, e_dyn, c="tab:red", label="||FK-Dyn||")
    axs[1].axhline(0.1, c="k", ls=":", lw=1.0, label="0.1mm")
    axs[1].set_ylabel("err (mm)")
    axs[1].grid(True, alpha=0.25)
    axs[1].legend(loc="upper right")

    axs[2].plot(t, conv_dyn.astype(float), c="k")
    axs[2].set_ylabel("dyn conv")
    axs[2].set_yticks([0, 1])
    axs[2].set_yticklabels(["fail", "ok"])

    axs[3].plot(t, p_fk[:, 0], c="tab:green", label="x FK")
    axs[3].plot(t, p_dyn[:, 0], c="tab:red", label="x Dyn", alpha=0.85)
    axs[3].set_ylabel("x (mm)")
    axs[3].legend(loc="upper right")

    axs[4].plot(t, p_fk[:, 1], c="tab:green", label="y FK")
    axs[4].plot(t, p_dyn[:, 1], c="tab:red", label="y Dyn", alpha=0.85)
    axs[4].set_ylabel("y (mm)")
    axs[4].set_xlabel("time (s)")
    axs[4].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    if not HAS_CPP_BINDINGS:
        raise RuntimeError("C++ bindings not available.")
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
    p_des = lemniscate_points(center, cfg.scale_mm, cfg.points, cfg.start_angle_rad)

    tree = cKDTree(P_ok)
    sel = project_dp_indices(p_des, U_ok, tree, k_nearest=cfg.k_nearest, alpha_u=cfg.alpha_u)
    p_proj = P_ok[sel]
    u_seed = U_ok[sel]

    kin = crm_python.CRMKinematics()
    assert kin.load_parameters(cfg.param_file, cfg.config_file)
    kin.integration_step_size = float(cfg.integration_step_size)

    U_ik, P_ik_fk, e_ik, _ = pinv_normalized_track(cfg, kin, p_proj, u_seed[0], du0_guess=None)

    # Build timeline (ramp + held lemniscate)
    u_init = np.array([0.0, 0.0, cfg.c3_start], dtype=np.float64)
    i_start = int(np.argmin(np.linalg.norm(U_ik - u_init[None, :], axis=1)))
    U_ik = np.roll(U_ik, -i_start, axis=0)
    p_des = np.roll(p_des, -i_start, axis=0)
    p_proj = np.roll(p_proj, -i_start, axis=0)
    P_ik_fk = np.roll(P_ik_fk, -i_start, axis=0)

    ramp = interp_ramp(u_init, U_ik[0], cfg.ramp_steps)
    ramp = hold_each(ramp, cfg.hold_steps_ramp)
    traj = hold_each(U_ik, cfg.hold_steps_per_point)
    currents = ensure_nonzero(np.concatenate([ramp, traj], axis=0), cfg.nonzero_eps, prefer_axis=2)

    # FK + Dyn
    p_fk = fk_trajectory(cfg, kin, currents)
    t, p_dyn, conv_dyn = run_dynamics(cfg, currents)

    # Expand projected target to match held timeline for plotting errors
    p_proj_t = np.vstack([np.repeat(p_proj[0:1], len(ramp), axis=0), np.repeat(p_proj, cfg.hold_steps_per_point, axis=0)])

    out_dir = Path("data/output")
    plots_dir = Path("plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    npz = out_dir / "lemniscate_y40_r5_dp_0p1mm_hold2.npz"
    np.savez_compressed(
        npz,
        t=t,
        currents=currents,
        tip_desired=p_des,
        tip_projected=p_proj,
        tip_fk=p_fk,
        tip_dyn=p_dyn,
        dyn_converged=conv_dyn.astype(np.uint8),
        ik_err_proj_fk=e_ik,
        center_xyz=center,
        k_nearest=cfg.k_nearest,
        alpha_u=cfg.alpha_u,
        points=cfg.points,
        hold_steps_per_point=cfg.hold_steps_per_point,
        hold_steps_ramp=cfg.hold_steps_ramp,
        dt=cfg.dt,
        insertion_length=cfg.insertion_length,
    )

    csv = out_dir / f"lemniscate_currents_y40_r5_dp_0p1mm_n{cfg.points}.csv"
    np.savetxt(csv, U_ik, delimiter=",", header="c1,c2,c3", comments="")

    plot_workspace_projection_xy(P_ok, p_des, p_proj, plots_dir / "lemniscate_y40_r5_workspace_projection_xy.png")
    plot_multiview(
        np.vstack([np.repeat(p_des[0:1], len(ramp), axis=0), np.repeat(p_des, cfg.hold_steps_per_point, axis=0)]),
        p_proj_t,
        p_fk,
        p_dyn,
        conv_dyn,
        plots_dir / "lemniscate_y40_r5_multiview.png",
    )
    plot_timeseries(t, currents, p_proj_t, p_fk, p_dyn, conv_dyn, plots_dir / "lemniscate_y40_r5_timeseries.png")

    e_dyn = np.linalg.norm(p_fk - p_dyn, axis=1)
    e_dyn_ok = e_dyn[conv_dyn]
    print(f"IK ||proj-FK||: mean={e_ik.mean():.3f}mm p95={np.percentile(e_ik,95):.3f}mm max={e_ik.max():.3f}mm")
    print(f"Dyn ||FK-Dyn|| (ok): mean={e_dyn_ok.mean():.3f}mm p95={np.percentile(e_dyn_ok,95):.3f}mm max={e_dyn_ok.max():.3f}mm fail={int((~conv_dyn).sum())}")
    print(f"Saved: {npz}")
    print(f"Saved: {csv}")
    print("Saved: plots/lemniscate_y40_r5_workspace_projection_xy.png")
    print("Saved: plots/lemniscate_y40_r5_multiview.png")
    print("Saved: plots/lemniscate_y40_r5_timeseries.png")


if __name__ == "__main__":
    main()
