"""
Lemniscate (figure-8) trajectory suite for y=±40 with scale a=10mm:
  1) generate desired lemniscates in tip space (200 points each)
  2) project to dense FK workspace via DP/Viterbi (current-smooth)
  3) solve IK currents to match projected targets to 0.1mm (normalized pinv)
  4) run FK vs dynamics for:
       - ramp + lemniscate1 (+y)     with hold=1 and hold=2
       - ramp + lemniscate1 + bridge + lemniscate2 (-y) with hold=1 and hold=2

Outputs:
  - data/output/lemniscate1_currents_y40_a10_dp_0p1mm_n200.csv
  - data/output/lemniscate2_currents_y-40_a10_dp_0p1mm_n200.csv
  - data/output/lemniscate_ypm40_a10_dp_ik.npz
  - data/output/dyn_fk_lem1_y40_a10_hold{1,2}.npz
  - data/output/dyn_fk_lem12_ypm40_a10_hold{1,2}.npz
  - plots/lemniscate_ypm40_a10_workspace_projection_xy.png
  - plots/dyn_fk_lem1_y40_a10_hold{1,2}_multiview.png
  - plots/dyn_fk_lem1_y40_a10_hold{1,2}_timeseries.png
  - plots/dyn_fk_lem12_ypm40_a10_hold{1,2}_multiview.png
  - plots/dyn_fk_lem12_ypm40_a10_hold{1,2}_timeseries.png
"""

from __future__ import annotations

import os
import argparse
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
    workspace_npz: str = "data/output/workspace_fk_ins94.3_b0.3_step0.01_int0.2_withDU0.npz"
    current_bound: float = 0.3  # must match workspace generation bounds

    param_file: str = "data/catheter_params/CatheterParameterSet_1_dyn.txt"
    config_file: str = "data/catheter_params/CatheterSpatialConfiguration_1.txt"
    insertion_length: float = 50.0
    integration_step_size: float = 0.2  # mm (FK + Dyn along-rod)
    dt: float = 0.05  # s

    # Desired lemniscates (XY)
    center1_xy: Tuple[float, float] = (0.0, 40.0)
    center2_xy: Tuple[float, float] = (0.0, -40.0)
    a_mm: float = 10.0
    points: int = 200
    start_angle_rad: float = -np.pi / 2.0

    # DP projection
    k_nearest: int = 300
    alpha_u: float = 0.4

    # IK (tight)
    step_size: float = 1e-3
    threshold_start_mm: float = 0.1
    threshold_traj_mm: float = 0.1
    itrmax: int = 500

    # Rollout timing
    c3_start: float = 0.01
    ramp_steps: int = 120
    bridge_steps: int = 480
    nonzero_eps: float = 0.005


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


def interp_ramp(u0: np.ndarray, u1: np.ndarray, steps: int) -> np.ndarray:
    steps = int(max(1, steps))
    t = np.linspace(0.0, 1.0, steps, endpoint=False, dtype=np.float64)
    return (1.0 - t)[:, None] * u0[None, :] + t[:, None] * u1[None, :]


def hold_each(currents: np.ndarray, hold_steps: int) -> np.ndarray:
    hold_steps = int(max(1, hold_steps))
    return np.repeat(np.asarray(currents, dtype=np.float64), hold_steps, axis=0)


def rotate_to_index(seq: np.ndarray, start_idx: int) -> np.ndarray:
    start_idx = int(start_idx) % int(seq.shape[0])
    return np.roll(seq, -start_idx, axis=0)


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


def pinv_normalized_track(
    cfg: Config,
    kin: "crm_python.CRMKinematics",
    p_targets: np.ndarray,
    u_start: np.ndarray,
    du0_guess: np.ndarray | None,
    *,
    du0_targets: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    U_out = np.zeros((len(p_targets), 3), dtype=np.float64)
    P_fk = np.zeros((len(p_targets), 3), dtype=np.float64)
    err = np.zeros((len(p_targets),), dtype=np.float64)

    u = np.asarray(u_start, dtype=np.float64).reshape(3).copy()
    for i in range(len(p_targets)):
        thr = cfg.threshold_start_mm if i == 0 else cfg.threshold_traj_mm
        if du0_targets is not None:
            du0_guess = np.asarray(du0_targets[i], dtype=np.float64).reshape(3)
        for _ in range(int(cfg.itrmax)):
            if du0_guess is None:
                out = kin.fk_and_jacobian(u, cfg.insertion_length)
            else:
                out = kin.fk_and_jacobian(u, cfg.insertion_length, du0_guess)
            p = np.asarray(out["tip_position"], dtype=np.float64).reshape(3)
            du0_guess = np.asarray(out["delta_u0"], dtype=np.float64).reshape(3)
            J = np.asarray(out["jacobian"], dtype=np.float64)[:3, :3]
            e = p_targets[i] - p
            if float(np.linalg.norm(e)) <= thr:
                break
            du = np.linalg.pinv(J) @ e
            du_norm = float(np.linalg.norm(du))
            if du_norm < 1e-12 or not np.isfinite(du_norm):
                break
            u = u + cfg.step_size * (du / du_norm)
            u = np.clip(u, -cfg.current_bound, cfg.current_bound)

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


def fk_positions_for_currents(cfg: Config, kin: "crm_python.CRMKinematics", currents: np.ndarray) -> np.ndarray:
    """FK tip positions for a current sequence, warm-starting delta_u0."""
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


def fk_positions_for_currents_per_step_du0(
    cfg: Config,
    kin: "crm_python.CRMKinematics",
    currents: np.ndarray,
    du0_per_step: np.ndarray,
) -> np.ndarray:
    """FK tip positions for currents, using a per-step delta_u0 seed (branch-stable)."""
    currents = np.asarray(currents, dtype=np.float64)
    du0_per_step = np.asarray(du0_per_step, dtype=np.float64)
    if currents.shape != du0_per_step.shape:
        raise ValueError(f"currents shape {currents.shape} must match du0_per_step {du0_per_step.shape}")
    p = np.zeros((currents.shape[0], 3), dtype=np.float64)
    for i in range(currents.shape[0]):
        out = kin.forward_kinematics_with_guess(currents[i], cfg.insertion_length, du0_per_step[i])
        p[i] = np.asarray(out["tip_position"], dtype=np.float64).reshape(3)
    return p


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


def plot_workspace_projection_xy(P_ok: np.ndarray, des1: np.ndarray, proj1: np.ndarray, des2: np.ndarray, proj2: np.ndarray, out_png: Path) -> None:
    P_plot = P_ok
    if P_plot.shape[0] > 200_000:
        idx = np.linspace(0, P_plot.shape[0] - 1, 200_000, dtype=int)
        P_plot = P_plot[idx]
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    ax.scatter(P_plot[:, 0], P_plot[:, 1], s=2, alpha=0.18, c="0.6", label="workspace")
    ax.scatter(proj1[:, 0], proj1[:, 1], s=10, c="tab:orange", label="projected lem1 (+y)")
    if len(proj2):
        ax.scatter(proj2[:, 0], proj2[:, 1], s=10, c="tab:red", label="projected lem2 (-y)")
    ax.set_title("Workspace projection (XY)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_multiview(
    title: str,
    p_des_t: np.ndarray,
    p_proj_t: np.ndarray,
    p_fk: np.ndarray,
    p_dyn: np.ndarray,
    conv_dyn: np.ndarray,
    out_png: Path,
) -> None:
    ok = conv_dyn
    mask_proj = np.isfinite(p_proj_t).all(axis=1)
    fig = plt.figure(figsize=(12, 9))

    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ax.plot(p_fk[:, 0], p_fk[:, 1], p_fk[:, 2], lw=2.0, c="tab:green", label="FK")
    ax.plot(p_dyn[ok, 0], p_dyn[ok, 1], p_dyn[ok, 2], lw=1.5, c="tab:red", label="Dyn(ok)")
    ax.scatter(p_proj_t[mask_proj, 0], p_proj_t[mask_proj, 1], p_proj_t[mask_proj, 2], s=8, c="tab:orange", label="projected")
    ax.set_title("3D")
    ax.legend(loc="best")

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(p_fk[:, 0], p_fk[:, 1], lw=2.0, c="tab:green", label="FK")
    ax.plot(p_dyn[ok, 0], p_dyn[ok, 1], lw=1.5, c="tab:red", label="Dyn(ok)")
    ax.scatter(p_proj_t[mask_proj, 0], p_proj_t[mask_proj, 1], s=8, c="tab:orange", label="projected")
    ax.set_title("XY")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax = fig.add_subplot(2, 2, 3)
    ax.plot(p_fk[:, 0], p_fk[:, 2], lw=2.0, c="tab:green")
    ax.plot(p_dyn[ok, 0], p_dyn[ok, 2], lw=1.5, c="tab:red")
    ax.scatter(p_proj_t[mask_proj, 0], p_proj_t[mask_proj, 2], s=8, c="tab:orange")
    ax.set_title("XZ")
    ax.grid(True, alpha=0.25)

    ax = fig.add_subplot(2, 2, 4)
    ax.plot(p_fk[:, 1], p_fk[:, 2], lw=2.0, c="tab:green")
    ax.plot(p_dyn[ok, 1], p_dyn[ok, 2], lw=1.5, c="tab:red")
    ax.scatter(p_proj_t[mask_proj, 1], p_proj_t[mask_proj, 2], s=8, c="tab:orange")
    ax.set_title("YZ")
    ax.grid(True, alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_timeseries(
    title: str,
    t: np.ndarray,
    u: np.ndarray,
    p_proj_t: np.ndarray,
    p_fk: np.ndarray,
    p_dyn: np.ndarray,
    conv_dyn: np.ndarray,
    out_png: Path,
) -> None:
    e_ik = np.linalg.norm(np.where(np.isfinite(p_proj_t), p_proj_t, p_fk) - p_fk, axis=1)
    e_dyn = np.linalg.norm(p_fk - p_dyn, axis=1)
    fig, axs = plt.subplots(5, 1, figsize=(12, 12), sharex=True)

    axs[0].plot(t, u[:, 0], label="c1")
    axs[0].plot(t, u[:, 1], label="c2")
    axs[0].plot(t, u[:, 2], label="c3")
    axs[0].set_ylabel("currents (A)")
    axs[0].legend(ncols=3, loc="upper right")

    axs[1].plot(t, e_ik, c="tab:green", label="||proj-FK|| (0 on ramps/bridge)")
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

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def rollout_case(
    cfg: Config,
    kin: "crm_python.CRMKinematics",
    name: str,
    u_seq: np.ndarray,
    p_des_seq: np.ndarray,
    p_proj_seq: np.ndarray,
    *,
    hold: int,
    include_bridge_and_lem2: bool,
    u_seq2: np.ndarray | None = None,
    p_des_seq2: np.ndarray | None = None,
    p_proj_seq2: np.ndarray | None = None,
) -> None:
    out_dir = Path("data/output")
    plots_dir = Path("plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    u_init = np.array([0.0, 0.0, cfg.c3_start], dtype=np.float64)
    i_start = int(np.argmin(np.linalg.norm(u_seq - u_init[None, :], axis=1)))
    u_seq = rotate_to_index(u_seq, i_start)
    p_des_seq = rotate_to_index(p_des_seq, i_start)
    p_proj_seq = rotate_to_index(p_proj_seq, i_start)

    ramp = hold_each(interp_ramp(u_init, u_seq[0], cfg.ramp_steps), hold)
    traj1 = hold_each(u_seq, hold)

    if include_bridge_and_lem2:
        assert u_seq2 is not None and p_des_seq2 is not None and p_proj_seq2 is not None
        i_start2 = int(np.argmin(np.linalg.norm(u_seq2 - (-u_init)[None, :], axis=1)))
        u_seq2 = rotate_to_index(u_seq2, i_start2)
        p_des_seq2 = rotate_to_index(p_des_seq2, i_start2)
        p_proj_seq2 = rotate_to_index(p_proj_seq2, i_start2)

        bridge = ensure_nonzero(interp_ramp(traj1[-1], u_seq2[0], cfg.bridge_steps), cfg.nonzero_eps)
        traj2 = hold_each(u_seq2, hold)
        currents = ensure_nonzero(np.concatenate([ramp, traj1, bridge, traj2], axis=0), cfg.nonzero_eps)

        # Build desired/projected timelines with NaNs during ramp/bridge
        nan3 = np.full((len(bridge), 3), np.nan, dtype=np.float64)
        nanr = np.full((len(ramp), 3), np.nan, dtype=np.float64)
        p_des_t = np.concatenate([nanr, hold_each(p_des_seq, hold), nan3, hold_each(p_des_seq2, hold)], axis=0)
        p_proj_t = np.concatenate([nanr, hold_each(p_proj_seq, hold), nan3, hold_each(p_proj_seq2, hold)], axis=0)
    else:
        currents = ensure_nonzero(np.concatenate([ramp, traj1], axis=0), cfg.nonzero_eps)
        nanr = np.full((len(ramp), 3), np.nan, dtype=np.float64)
        p_des_t = np.concatenate([nanr, hold_each(p_des_seq, hold)], axis=0)
        p_proj_t = np.concatenate([nanr, hold_each(p_proj_seq, hold)], axis=0)

    p_fk = fk_trajectory(cfg, kin, currents)
    t, p_dyn, conv_dyn = run_dynamics(cfg, currents)

    e_dyn = np.linalg.norm(p_fk - p_dyn, axis=1)
    e_ok = e_dyn[conv_dyn]

    npz = out_dir / f"{name}_hold{hold}.npz"
    np.savez_compressed(
        npz,
        t=t,
        currents=currents,
        tip_desired=p_des_t,
        tip_projected=p_proj_t,
        tip_fk=p_fk,
        tip_dyn=p_dyn,
        dyn_converged=conv_dyn.astype(np.uint8),
        fk_dyn_err=e_dyn,
        hold=hold,
        dt=cfg.dt,
        insertion_length=cfg.insertion_length,
        integration_step_size=cfg.integration_step_size,
    )

    mv = plots_dir / f"{name}_hold{hold}_multiview.png"
    ts = plots_dir / f"{name}_hold{hold}_timeseries.png"
    plot_multiview(f"{name} hold={hold}", p_des_t, p_proj_t, p_fk, p_dyn, conv_dyn, mv)
    plot_timeseries(f"{name} hold={hold}", t, currents, p_proj_t, p_fk, p_dyn, conv_dyn, ts)

    print(f"{name} hold={hold}: dyn_fail={int((~conv_dyn).sum())} mean={e_ok.mean():.3f}mm p95={np.percentile(e_ok,95):.3f}mm max={e_ok.max():.3f}mm -> {npz}")


def main() -> None:
    if not HAS_CPP_BINDINGS:
        raise RuntimeError("C++ bindings not available.")
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-lem1", action="store_true", help="Only run ramp+lemniscate1 (y=+40) for hold=1/2.")
    args = ap.parse_args()

    cfg = Config()

    ws = np.load(cfg.workspace_npz)
    P = np.asarray(ws["P"], dtype=np.float64)
    U = np.asarray(ws["U"], dtype=np.float64)
    DU0 = np.asarray(ws["DU0"], dtype=np.float64) if "DU0" in ws.files else None
    conv = np.asarray(ws["conv"]).astype(bool)
    ok = conv & np.isfinite(P).all(axis=1)
    P_ok = P[ok]
    U_ok = U[ok]
    DU0_ok = DU0[ok] if DU0 is not None else None
    if P_ok.size == 0:
        raise RuntimeError("No converged workspace points.")

    z1 = choose_z0_from_workspace(P_ok, cfg.center1_xy)
    des1 = lemniscate_points(np.array([cfg.center1_xy[0], cfg.center1_xy[1], z1], dtype=np.float64), cfg.a_mm, cfg.points, cfg.start_angle_rad)

    tree = cKDTree(P_ok)
    sel1 = project_dp_indices(des1, U_ok, tree, k_nearest=cfg.k_nearest, alpha_u=cfg.alpha_u)
    proj1 = P_ok[sel1]
    seed1 = U_ok[sel1]
    du0_seed1 = DU0_ok[sel1] if DU0_ok is not None else None

    kin = crm_python.CRMKinematics()
    assert kin.load_parameters(cfg.param_file, cfg.config_file)
    kin.integration_step_size = float(cfg.integration_step_size)

    # First try using the projected workspace seed currents directly. Since proj points come from FK sampling
    # on the same model config, FK(seed) should already match proj (up to solver tolerance).
    if du0_seed1 is not None and np.isfinite(du0_seed1).all():
        FK1_seed = fk_positions_for_currents_per_step_du0(cfg, kin, seed1, du0_seed1)
    else:
        FK1_seed = fk_positions_for_currents(cfg, kin, seed1)
    e1_seed = np.linalg.norm(proj1 - FK1_seed, axis=1)

    # If needed (rare), refine with clipped normalized-pinv IK starting from the first seed and tracking.
    if float(np.percentile(e1_seed, 95)) <= cfg.threshold_traj_mm and float(np.max(e1_seed)) <= 5.0 * cfg.threshold_traj_mm:
        U1, FK1, e1 = seed1, FK1_seed, e1_seed
    else:
        U1, FK1, e1, _ = pinv_normalized_track(cfg, kin, proj1, seed1[0], du0_guess=None, du0_targets=du0_seed1)

    if args.only_lem1:
        U2 = FK2 = e2 = None
        des2 = proj2 = None
        z2 = None
    else:
        z2 = choose_z0_from_workspace(P_ok, cfg.center2_xy)
        des2 = lemniscate_points(np.array([cfg.center2_xy[0], cfg.center2_xy[1], z2], dtype=np.float64), cfg.a_mm, cfg.points, cfg.start_angle_rad)
        sel2 = project_dp_indices(des2, U_ok, tree, k_nearest=cfg.k_nearest, alpha_u=cfg.alpha_u)
        proj2 = P_ok[sel2]
        seed2 = U_ok[sel2]
        du0_seed2 = DU0_ok[sel2] if DU0_ok is not None else None
        if du0_seed2 is not None and np.isfinite(du0_seed2).all():
            FK2_seed = fk_positions_for_currents_per_step_du0(cfg, kin, seed2, du0_seed2)
        else:
            FK2_seed = fk_positions_for_currents(cfg, kin, seed2)
        e2_seed = np.linalg.norm(proj2 - FK2_seed, axis=1)
        if float(np.percentile(e2_seed, 95)) <= cfg.threshold_traj_mm and float(np.max(e2_seed)) <= 5.0 * cfg.threshold_traj_mm:
            U2, FK2, e2 = seed2, FK2_seed, e2_seed
        else:
            U2, FK2, e2, _ = pinv_normalized_track(cfg, kin, proj2, seed2[0], du0_guess=None, du0_targets=du0_seed2)

    out_dir = Path("data/output")
    plots_dir = Path("plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_dir / "lemniscate_ypm40_a10_dp_ik.npz",
        lem1_des=des1,
        lem1_proj=proj1,
        lem1_fk=FK1,
        lem1_currents=U1,
        lem1_err_proj_fk=e1,
        k_nearest=cfg.k_nearest,
        alpha_u=cfg.alpha_u,
        a_mm=cfg.a_mm,
        points=cfg.points,
        z0_lem1=z1,
        z0_lem2=z2,
    )

    np.savetxt(out_dir / "lemniscate1_currents_y40_a10_dp_0p1mm_n200.csv", U1, delimiter=",", header="c1,c2,c3", comments="")
    if not args.only_lem1:
        np.savetxt(out_dir / "lemniscate2_currents_y-40_a10_dp_0p1mm_n200.csv", U2, delimiter=",", header="c1,c2,c3", comments="")

    if args.only_lem1:
        plot_workspace_projection_xy(P_ok, des1, proj1, des1[:0], proj1[:0], plots_dir / "lemniscate_y40_a10_workspace_projection_xy.png")
    else:
        plot_workspace_projection_xy(P_ok, des1, proj1, des2, proj2, plots_dir / "lemniscate_ypm40_a10_workspace_projection_xy.png")

    print(f"IK lem1 ||proj-FK||: mean={e1.mean():.3f}mm p95={np.percentile(e1,95):.3f}mm max={e1.max():.3f}mm")
    if not args.only_lem1:
        print(f"IK lem2 ||proj-FK||: mean={e2.mean():.3f}mm p95={np.percentile(e2,95):.3f}mm max={e2.max():.3f}mm")

    # Suite runs
    for hold in (1, 2):
        rollout_case(
            cfg,
            kin,
            "dyn_fk_lem1_y40_a10",
            U1,
            des1,
            proj1,
            hold=hold,
            include_bridge_and_lem2=False,
        )
        if not args.only_lem1:
            rollout_case(
                cfg,
                kin,
                "dyn_fk_lem12_ypm40_a10",
                U1,
                des1,
                proj1,
                hold=hold,
                include_bridge_and_lem2=True,
                u_seq2=U2,
                p_des_seq2=des2,
                p_proj_seq2=proj2,
            )


if __name__ == "__main__":
    main()
