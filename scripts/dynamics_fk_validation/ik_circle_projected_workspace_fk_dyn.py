"""
Workspace-projected IK current generation for desired tip-position circles,
with FK consistency and FK-vs-dynamics comparison.

Implements the MATLAB-style normalized pseudoinverse update (pinv + normalized step),
but first projects the desired circle points onto an FK-sampled workspace so the targets
are reachable.

Outputs:
  - data/output/ik_circle_projected_workspace.npz
  - plots/ik_circle_projected_workspace_multiview.png
  - plots/ik_circle_projected_workspace_timeseries.png
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib.pyplot as plt  # noqa: E402

from crm_ml_rl.wrappers import crm_python  # noqa: E402
from crm_ml_rl.wrappers.crm_wrapper import CRMWrapper, HAS_CPP_BINDINGS  # noqa: E402


@dataclass(frozen=True)
class Config:
    insertion_length: float = 94.3
    dt: float = 0.05
    # MATLAB/MEX uses 0.2mm along-rod integration step; it's much faster than 0.01mm.
    integration_step_size: float = 0.2

    param_file: str = "data/catheter_params/CatheterParameterSet_1_dyn.txt"
    config_file: str = "data/catheter_params/CatheterSpatialConfiguration_1.txt"

    # Current bounds and workspace sampling resolution
    current_bound: float = 0.2
    workspace_grid_n: int = 11  # 11^3 = 1331 FK evaluations (smoother workspace projection)

    # Desired circles (mm) before projection
    radius_mm: float = 10.0
    center1_xy: Tuple[float, float] = (0.0, 40.0)
    center2_xy: Tuple[float, float] = (0.0, -40.0)
    points_per_circle: int = 100
    start_angle_rad: float = -np.pi / 2.0

    # z-plane selection for the desired circles (computed from FK at a reference current)
    z0_probe_currents: Tuple[float, float, float] = (0.0, 0.0, 0.2)

    # MATLAB-like normalized pinv IK data/simulation_parameters
    step_size: float = 1e-2
    threshold_start_mm: float = 1.0
    threshold_traj_mm: float = 1.0
    itrmax: int = 100

    # Ramps (current space) for dynamics/fk timeline
    ramp_steps_to_first: int = 80
    ramp_steps_between_circles: int = 160


def linspace(lo: float, hi: float, n: int) -> np.ndarray:
    if n <= 1:
        return np.array([lo], dtype=np.float64)
    return np.linspace(lo, hi, n, dtype=np.float64)


def circle_points(center_xyz: np.ndarray, radius: float, n: int, start_angle: float) -> np.ndarray:
    angles = start_angle + np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False, dtype=np.float64)
    pts = np.repeat(center_xyz.reshape(1, 3), int(n), axis=0)
    pts[:, 0] = center_xyz[0] + radius * np.cos(angles)
    pts[:, 1] = center_xyz[1] + radius * np.sin(angles)
    return pts


def interp_ramp(u0: np.ndarray, u1: np.ndarray, steps: int) -> np.ndarray:
    steps = int(max(1, steps))
    t = np.linspace(0.0, 1.0, steps, endpoint=False, dtype=np.float64)
    return (1.0 - t)[:, None] * u0[None, :] + t[:, None] * u1[None, :]


def fk_workspace(cfg: Config, kin: "crm_python.CRMKinematics") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (U, P, conv) for FK-sampled workspace points.
    """
    vals = linspace(-cfg.current_bound, cfg.current_bound, cfg.workspace_grid_n)
    U = np.zeros((cfg.workspace_grid_n**3, 3), dtype=np.float64)
    P = np.full((cfg.workspace_grid_n**3, 3), np.nan, dtype=np.float64)
    conv = np.zeros((cfg.workspace_grid_n**3,), dtype=bool)

    idx = 0
    du0_guess = None
    # Snake ordering for warm-start stability
    for iz, c3 in enumerate(vals):
        rev2 = (iz % 2 == 1)
        yvals = vals[::-1] if rev2 else vals
        for iy, c2 in enumerate(yvals):
            rev1 = ((iz + iy) % 2 == 1)
            xvals = vals[::-1] if rev1 else vals
            for c1 in xvals:
                u = np.array([c1, c2, c3], dtype=np.float64)
                U[idx] = u
                if du0_guess is None:
                    out = kin.forward_kinematics(u, cfg.insertion_length)
                else:
                    out = kin.forward_kinematics_with_guess(u, cfg.insertion_length, du0_guess)
                conv[idx] = bool(out["converged"])
                if conv[idx]:
                    P[idx] = np.asarray(out["tip_position"], dtype=np.float64).reshape(3)
                    du0_guess = np.asarray(out["delta_u0"], dtype=np.float64).reshape(3)
                idx += 1
    return U, P, conv


def project_to_workspace(p_des: np.ndarray, U_ws: np.ndarray, P_ws: np.ndarray, conv_ws: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ok = conv_ws & np.isfinite(P_ws).all(axis=1)
    U_ok = U_ws[ok]
    P_ok = P_ws[ok]
    if len(P_ok) == 0:
        raise RuntimeError("Workspace FK sampling produced 0 converged points.")

    # Brute force NN (small workspace).
    p_proj = np.zeros_like(p_des)
    u_seed = np.zeros((len(p_des), 3), dtype=np.float64)
    for i in range(len(p_des)):
        d = np.linalg.norm(P_ok - p_des[i], axis=1)
        j = int(np.argmin(d))
        p_proj[i] = P_ok[j]
        u_seed[i] = U_ok[j]
    return p_proj, u_seed


def project_to_workspace_continuous(
    p_des: np.ndarray,
    U_ws: np.ndarray,
    P_ws: np.ndarray,
    conv_ws: np.ndarray,
    *,
    k_nearest: int = 50,
    continuity_weight: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project each desired point onto the FK workspace, but with a continuity preference so the projected
    trajectory doesn't jump between disconnected workspace branches (which can induce large current jumps).
    """
    ok = conv_ws & np.isfinite(P_ws).all(axis=1)
    U_ok = U_ws[ok]
    P_ok = P_ws[ok]
    if len(P_ok) == 0:
        raise RuntimeError("Workspace FK sampling produced 0 converged points.")

    k = int(max(1, min(k_nearest, len(P_ok))))
    p_proj = np.zeros_like(p_des)
    u_seed = np.zeros((len(p_des), 3), dtype=np.float64)

    prev_p = None
    for i in range(len(p_des)):
        d_des = np.linalg.norm(P_ok - p_des[i], axis=1)
        if prev_p is None:
            j = int(np.argmin(d_des))
        else:
            # Choose among the k nearest-to-desired points, preferring continuity with previous projection.
            cand = np.argpartition(d_des, k - 1)[:k]
            d_prev = np.linalg.norm(P_ok[cand] - prev_p, axis=1)
            score = d_des[cand] + float(continuity_weight) * d_prev
            j = int(cand[int(np.argmin(score))])
        p_proj[i] = P_ok[j]
        u_seed[i] = U_ok[j]
        prev_p = p_proj[i]

    return p_proj, u_seed


def pinv_normalized_track(
    cfg: Config,
    kin: "crm_python.CRMKinematics",
    p_targets: np.ndarray,
    u_start: np.ndarray,
    threshold_mm: float,
    du0_guess: np.ndarray | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Track targets sequentially with MATLAB-style normalized pinv steps.
    Returns (U_traj, P_fk, err).
    """
    U_out = np.zeros((len(p_targets), 3), dtype=np.float64)
    P_fk = np.zeros((len(p_targets), 3), dtype=np.float64)
    err = np.zeros((len(p_targets),), dtype=np.float64)

    u = np.asarray(u_start, dtype=np.float64).reshape(3).copy()
    for i in range(len(p_targets)):
        thr = cfg.threshold_start_mm if i == 0 else threshold_mm
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
            u = np.clip(u, -cfg.current_bound, cfg.current_bound)

        # record final
        if du0_guess is None:
            out = kin.forward_kinematics(u, cfg.insertion_length)
        else:
            out = kin.forward_kinematics_with_guess(u, cfg.insertion_length, du0_guess)
        p = np.asarray(out["tip_position"], dtype=np.float64).reshape(3)
        du0_guess = np.asarray(out["delta_u0"], dtype=np.float64).reshape(3)
        U_out[i] = u
        P_fk[i] = p
        err[i] = float(np.linalg.norm(p_targets[i] - p))

        # warm-start next target from this solution
    return U_out, P_fk, err, du0_guess


def run_dynamics(cfg: Config, currents: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    wrapper = CRMWrapper(
        param_file=cfg.param_file,
        config_file=cfg.config_file,
        use_cpp=True,
        disable_cpp_fallback=True,
    )
    if not wrapper.is_using_cpp:
        raise RuntimeError("C++ bindings not active for dynamics.")

    wrapper._cpp_dynamics.dt = float(cfg.dt)
    wrapper._cpp_dynamics.integration_step_size = float(cfg.integration_step_size)

    ok = wrapper.initialize_dynamics(currents[0], insertion_length=cfg.insertion_length)
    if not ok:
        raise RuntimeError("Dynamics initialize_dynamics failed at first current.")

    n = currents.shape[0]
    p_dyn = np.zeros((n, 3), dtype=np.float64)
    conv = np.zeros((n,), dtype=bool)
    for i in range(n):
        out = wrapper.step_dynamics(currents[i], insertion_length=cfg.insertion_length, dt=cfg.dt)
        p_dyn[i] = np.asarray(out["tip_position"], dtype=np.float64).reshape(3)
        conv[i] = bool(out.get("converged", False))
        if not conv[i]:
            # Try recover once.
            ok2 = wrapper.initialize_dynamics(currents[i], insertion_length=cfg.insertion_length)
            if ok2:
                out2 = wrapper.step_dynamics(currents[i], insertion_length=cfg.insertion_length, dt=cfg.dt)
                p_dyn[i] = np.asarray(out2["tip_position"], dtype=np.float64).reshape(3)
                conv[i] = bool(out2.get("converged", False))
    t = np.arange(n, dtype=np.float64) * cfg.dt
    return t, p_dyn, conv


def fk_trajectory(cfg: Config, kin: "crm_python.CRMKinematics", currents: np.ndarray) -> np.ndarray:
    """Compute FK tip positions for a current time series, warm-starting the BVP."""
    p_fk = np.zeros((currents.shape[0], 3), dtype=np.float64)
    du0_guess = None
    for i in range(currents.shape[0]):
        u = np.asarray(currents[i], dtype=np.float64).reshape(3)
        if du0_guess is None:
            out = kin.forward_kinematics(u, cfg.insertion_length)
        else:
            out = kin.forward_kinematics_with_guess(u, cfg.insertion_length, du0_guess)
        p_fk[i] = np.asarray(out["tip_position"], dtype=np.float64).reshape(3)
        du0_guess = np.asarray(out["delta_u0"], dtype=np.float64).reshape(3)
    return p_fk


def plot_all(
    t: np.ndarray,
    u: np.ndarray,
    p_des: np.ndarray,
    p_proj: np.ndarray,
    p_fk: np.ndarray,
    p_dyn: np.ndarray,
    conv_dyn: np.ndarray,
    *,
    threshold_mm: float,
) -> None:
    plots_dir = Path("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    ok = conv_dyn
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ax.plot(p_des[:, 0], p_des[:, 1], p_des[:, 2], lw=1.0, label="desired")
    ax.plot(p_proj[:, 0], p_proj[:, 1], p_proj[:, 2], lw=1.0, label="proj(ws)")
    ax.plot(p_fk[:, 0], p_fk[:, 1], p_fk[:, 2], lw=1.0, ls="--", label="fk")
    ax.plot(p_dyn[ok, 0], p_dyn[ok, 1], p_dyn[ok, 2], lw=1.0, alpha=0.9, label="dyn(ok)")
    if np.any(~ok):
        ax.scatter(p_dyn[~ok, 0], p_dyn[~ok, 1], p_dyn[~ok, 2], s=10, c="r", label="dyn fail")
    ax.set_title("3D")
    ax.legend(loc="best")

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(p_des[:, 0], p_des[:, 1], lw=1.0, label="desired")
    ax.plot(p_proj[:, 0], p_proj[:, 1], lw=1.0, label="proj(ws)")
    ax.plot(p_fk[:, 0], p_fk[:, 1], lw=1.0, ls="--", label="fk")
    ax.plot(p_dyn[ok, 0], p_dyn[ok, 1], lw=1.0, alpha=0.9, label="dyn")
    ax.set_title("XY")
    ax.axis("equal")
    ax.legend(loc="best")

    ax = fig.add_subplot(2, 2, 3)
    ax.plot(p_des[:, 0], p_des[:, 2], lw=1.0)
    ax.plot(p_proj[:, 0], p_proj[:, 2], lw=1.0)
    ax.plot(p_fk[:, 0], p_fk[:, 2], lw=1.0, ls="--")
    ax.plot(p_dyn[ok, 0], p_dyn[ok, 2], lw=1.0, alpha=0.9)
    ax.set_title("XZ")

    ax = fig.add_subplot(2, 2, 4)
    ax.plot(p_des[:, 1], p_des[:, 2], lw=1.0)
    ax.plot(p_proj[:, 1], p_proj[:, 2], lw=1.0)
    ax.plot(p_fk[:, 1], p_fk[:, 2], lw=1.0, ls="--")
    ax.plot(p_dyn[ok, 1], p_dyn[ok, 2], lw=1.0, alpha=0.9)
    ax.set_title("YZ")

    fig.tight_layout()
    fig.savefig(plots_dir / "ik_circle_projected_workspace_multiview.png", dpi=180)
    plt.close(fig)

    fig, axs = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    axs[0].plot(t, u[:, 0], label="c1")
    axs[0].plot(t, u[:, 1], label="c2")
    axs[0].plot(t, u[:, 2], label="c3")
    axs[0].set_ylabel("currents (A)")
    axs[0].legend(ncols=3, loc="upper right")

    e_proj_fk = np.linalg.norm(p_proj - p_fk, axis=1)
    e_des_proj = np.linalg.norm(p_des - p_proj, axis=1)
    e_des_fk = np.linalg.norm(p_des - p_fk, axis=1)
    e_fk_dyn = np.linalg.norm(p_fk - p_dyn, axis=1)
    axs[1].plot(t, e_proj_fk, label="||proj-fk|| (IK err)")
    axs[1].plot(t, e_des_proj, label="||des-proj|| (projection)")
    axs[1].plot(t, e_des_fk, label="||des-fk|| (des track)")
    axs[1].plot(t, e_fk_dyn, label="||fk-dyn|| (dyn err)")
    axs[1].axhline(float(threshold_mm), color="k", lw=1.0, ls=":", label=f"threshold={threshold_mm:.1f}mm")
    axs[1].set_ylabel("err (mm)")
    axs[1].legend(loc="upper right")

    axs[2].plot(t, conv_dyn.astype(float))
    axs[2].set_ylabel("dyn conv")
    axs[2].set_yticks([0, 1])
    axs[2].set_yticklabels(["fail", "ok"])

    axs[3].plot(t, p_fk[:, 0], label="x fk")
    axs[3].plot(t, p_dyn[:, 0], label="x dyn", alpha=0.8)
    axs[3].set_ylabel("x (mm)")
    axs[3].legend(loc="upper right")

    axs[4].plot(t, p_fk[:, 1], label="y fk")
    axs[4].plot(t, p_dyn[:, 1], label="y dyn", alpha=0.8)
    axs[4].set_ylabel("y (mm)")
    axs[4].set_xlabel("time (s)")
    axs[4].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(plots_dir / "ik_circle_projected_workspace_timeseries.png", dpi=180)
    plt.close(fig)


def plot_workspace_projection(P_ws: np.ndarray, conv_ws: np.ndarray, p_des_1: np.ndarray, p_des_2: np.ndarray, p_proj_1: np.ndarray, p_proj_2: np.ndarray) -> None:
    plots_dir = Path("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    ok = conv_ws & np.isfinite(P_ws).all(axis=1)
    P_ok = P_ws[ok]

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    ax.scatter(P_ok[:, 0], P_ok[:, 1], s=6, alpha=0.35, label="FK workspace (sampled)")
    ax.plot(p_des_1[:, 0], p_des_1[:, 1], lw=2.0, label="desired circle (+y)")
    ax.plot(p_des_2[:, 0], p_des_2[:, 1], lw=2.0, label="desired circle (-y)")
    ax.scatter(p_proj_1[:, 0], p_proj_1[:, 1], s=14, label="projected (+y)")
    ax.scatter(p_proj_2[:, 0], p_proj_2[:, 1], s=14, label="projected (-y)")
    ax.set_title("Desired circles and workspace projection (XY)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plots_dir / "ik_circle_workspace_projection_xy.png", dpi=200)
    plt.close(fig)


def main() -> None:
    if not HAS_CPP_BINDINGS:
        raise RuntimeError("C++ bindings not available (crm_python import failed).")

    cfg = Config()

    kin = crm_python.CRMKinematics()
    assert kin.load_parameters(cfg.param_file, cfg.config_file)
    kin.integration_step_size = float(cfg.integration_step_size)

    # Workspace sampling (cached)
    cache = Path("data/output") / f"workspace_fk_ins{cfg.insertion_length}_b{cfg.current_bound}_n{cfg.workspace_grid_n}_step{cfg.integration_step_size}.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        data = np.load(cache)
        U_ws = data["U"]
        P_ws = data["P"]
        conv_ws = data["conv"].astype(bool)
    else:
        U_ws, P_ws, conv_ws = fk_workspace(cfg, kin)
        np.savez_compressed(cache, U=U_ws, P=P_ws, conv=conv_ws.astype(np.uint8))

    # Choose circle plane z0 from FK at a reference current
    out0 = kin.forward_kinematics(np.asarray(cfg.z0_probe_currents, dtype=np.float64), cfg.insertion_length)
    z0 = float(np.asarray(out0["tip_position"], dtype=np.float64).reshape(3)[2])

    c1 = np.array([cfg.center1_xy[0], cfg.center1_xy[1], z0], dtype=np.float64)
    c2 = np.array([cfg.center2_xy[0], cfg.center2_xy[1], z0], dtype=np.float64)
    p_des_1 = circle_points(c1, cfg.radius_mm, cfg.points_per_circle, cfg.start_angle_rad)
    p_des_2 = circle_points(c2, cfg.radius_mm, cfg.points_per_circle, cfg.start_angle_rad)

    # Project desired points onto workspace and use the corresponding workspace currents as seeds.
    p_proj_1, u_seed_1 = project_to_workspace_continuous(p_des_1, U_ws, P_ws, conv_ws)
    p_proj_2, u_seed_2 = project_to_workspace_continuous(p_des_2, U_ws, P_ws, conv_ws)

    plot_workspace_projection(P_ws, conv_ws, p_des_1, p_des_2, p_proj_1, p_proj_2)

    # Track projected targets with normalized pinv
    du0_guess = None
    U1, P1_fk, e1, du0_guess = pinv_normalized_track(cfg, kin, p_proj_1, u_seed_1[0], cfg.threshold_traj_mm, du0_guess)
    U2, P2_fk, e2, du0_guess = pinv_normalized_track(cfg, kin, p_proj_2, u_seed_2[0], cfg.threshold_traj_mm, du0_guess)

    # Build timeline currents and targets for plotting
    ramp0 = interp_ramp(U1[0], U1[0], cfg.ramp_steps_to_first)  # no-op but keeps shape
    ramp12 = interp_ramp(U1[-1], U2[0], cfg.ramp_steps_between_circles)
    currents = np.concatenate([ramp0, U1, ramp12, U2], axis=0)
    p_des = np.concatenate(
        [
            np.repeat(p_des_1[0].reshape(1, 3), len(ramp0), axis=0),
            p_des_1,
            np.repeat(p_des_2[0].reshape(1, 3), len(ramp12), axis=0),
            p_des_2,
        ],
        axis=0,
    )
    p_proj = np.concatenate(
        [
            np.repeat(p_proj_1[0].reshape(1, 3), len(ramp0), axis=0),
            p_proj_1,
            np.repeat(p_proj_2[0].reshape(1, 3), len(ramp12), axis=0),
            p_proj_2,
        ],
        axis=0,
    )

    # Dynamics rollout
    t, p_dyn, conv_dyn = run_dynamics(cfg, currents)

    # FK for the full timeline (so ramps are consistent with the applied currents).
    p_fk = fk_trajectory(cfg, kin, currents)

    out_npz = Path("data/output") / "ik_circle_projected_workspace.npz"
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        t=t,
        currents=currents,
        tip_desired=p_des,
        tip_projected=p_proj,
        tip_fk=p_fk,
        tip_dyn=p_dyn,
        dyn_converged=conv_dyn.astype(np.uint8),
        circle1_err=e1,
        circle2_err=e2,
        z0=z0,
        workspace_cache=str(cache),
    )

    plot_all(t, currents, p_des, p_proj, p_fk, p_dyn, conv_dyn, threshold_mm=cfg.threshold_traj_mm)

    print(f"Workspace: converged={int(conv_ws.sum())}/{len(conv_ws)} cache={cache}")
    print(f"Circle1 proj FK err: mean={e1.mean():.3f}mm p95={np.percentile(e1,95):.3f}mm max={e1.max():.3f}mm ok={(e1<=cfg.threshold_traj_mm).mean():.2%}")
    print(f"Circle2 proj FK err: mean={e2.mean():.3f}mm p95={np.percentile(e2,95):.3f}mm max={e2.max():.3f}mm ok={(e2<=cfg.threshold_traj_mm).mean():.2%}")
    ed = np.linalg.norm(p_fk - p_dyn, axis=1)
    # Circle-only windows (exclude ramps).
    i0 = len(ramp0)
    i1 = i0 + len(U1)
    j0 = i1 + len(ramp12)
    j1 = j0 + len(U2)
    ed_circles = np.concatenate([ed[i0:i1], ed[j0:j1]])
    print(f"Dyn err vs FK: mean={ed.mean():.3f}mm p95={np.percentile(ed,95):.3f}mm max={ed.max():.3f}mm dyn_fail={int((~conv_dyn).sum())}")
    print(f"Dyn err vs FK (circles only): mean={ed_circles.mean():.3f}mm p95={np.percentile(ed_circles,95):.3f}mm max={ed_circles.max():.3f}mm")
    print(f"Saved: {out_npz}")
    print("Saved: plots/ik_circle_projected_workspace_multiview.png")
    print("Saved: plots/ik_circle_projected_workspace_timeseries.png")


if __name__ == "__main__":
    main()
