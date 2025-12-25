"""
Generate a slow, slew-limited current trajectory that biases c3 to one half-plane
and runs (c1,c2) circles, then repeats at other c3 biases, and plots results.

Outputs:
  - data/output/trajectory_c3_halfplane_circle.npz
  - plots/trajectory_c3_halfplane_circle_multiview.png
  - plots/trajectory_c3_halfplane_circle_timeseries.png
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

# Matplotlib cache path in this environment is not writable by default.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib.pyplot as plt  # noqa: E402

from crm_ml_rl.wrappers.crm_wrapper import CRMWrapper, HAS_CPP_BINDINGS  # noqa: E402


@dataclass(frozen=True)
class TrajectoryConfig:
    insertion_length: float = 94.3
    dt: float = 0.05
    integration_step_size: float = 0.01
    a_xy: float = 0.08
    f_hz: float = 0.05
    laps_each: int = 1
    c3_start: float = 0.01  # start away from 0 to avoid (0,0,0) init
    include_midplane: bool = False  # if True, also do a circle at c3_mid
    ramp1_s: float = 10.0  # c3_start -> +0.2
    ramp_mid_s: float = 5.0  # +0.2 -> +0.1 (only if include_midplane)
    ramp_cross_s: float = 20.0  # +0.2 -> -0.2 (slow sign change)
    dwell_s: float = 2.0
    c3_hi: float = 0.2
    c3_mid: float = 0.1
    c3_lo: float = -0.2
    fk_stride: int = 5  # compute FK every N steps (1 = every step)
    # C++ data/simulation_parameters
    param_file: str = "data/catheter_params/CatheterParameterSet_1_dyn.txt"
    config_file: str = "data/catheter_params/CatheterSpatialConfiguration_1.txt"
    damping: Tuple[float, float, float, float, float, float] = (
        12.1761626666366,
        12.1761626666366,
        284.429938756989,
        0.0304776127617393,
        0.0304776127617393,
        0.00502712804532508,
    )


def _n_steps(duration_s: float, dt: float) -> int:
    return int(np.round(duration_s / dt))


def ramp_c3(c3_a: float, c3_b: float, duration_s: float, dt: float) -> np.ndarray:
    n = max(1, _n_steps(duration_s, dt))
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    c3 = c3_a + (c3_b - c3_a) * t
    u = np.zeros((n, 3), dtype=np.float64)
    u[:, 2] = c3
    return u


def dwell(u_const: np.ndarray, duration_s: float, dt: float) -> np.ndarray:
    n = max(1, _n_steps(duration_s, dt))
    u = np.repeat(u_const.reshape(1, 3), n, axis=0).astype(np.float64)
    return u


def circle_xy(c3_const: float, a_xy: float, f_hz: float, laps: int, dt: float) -> np.ndarray:
    if laps <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    period = 1.0 / max(1e-9, f_hz)
    duration_s = laps * period
    n = max(1, _n_steps(duration_s, dt))
    t = np.arange(n, dtype=np.float64) * dt
    w = 2.0 * np.pi * f_hz
    u = np.zeros((n, 3), dtype=np.float64)
    u[:, 0] = a_xy * np.cos(w * t)
    u[:, 1] = a_xy * np.sin(w * t)
    u[:, 2] = c3_const
    return u


def build_schedule(cfg: TrajectoryConfig) -> np.ndarray:
    segs = []
    # Start with a small positive c3 to avoid initializing at (0,0,0).
    segs.append(dwell(np.array([0.0, 0.0, cfg.c3_start]), cfg.dt, cfg.dt))
    segs.append(ramp_c3(cfg.c3_start, cfg.c3_hi, cfg.ramp1_s, cfg.dt))
    segs.append(circle_xy(cfg.c3_hi, cfg.a_xy, cfg.f_hz, cfg.laps_each, cfg.dt))
    segs.append(dwell(np.array([0.0, 0.0, cfg.c3_hi]), cfg.dwell_s, cfg.dt))

    if cfg.include_midplane:
        segs.append(ramp_c3(cfg.c3_hi, cfg.c3_mid, cfg.ramp_mid_s, cfg.dt))
        segs.append(circle_xy(cfg.c3_mid, cfg.a_xy, cfg.f_hz, cfg.laps_each, cfg.dt))
        segs.append(dwell(np.array([0.0, 0.0, cfg.c3_mid]), cfg.dwell_s, cfg.dt))
        segs.append(ramp_c3(cfg.c3_mid, cfg.c3_lo, cfg.ramp_cross_s, cfg.dt))
    else:
        segs.append(ramp_c3(cfg.c3_hi, cfg.c3_lo, cfg.ramp_cross_s, cfg.dt))
    segs.append(circle_xy(cfg.c3_lo, cfg.a_xy, cfg.f_hz, cfg.laps_each, cfg.dt))
    segs.append(dwell(np.array([0.0, 0.0, cfg.c3_lo]), cfg.dwell_s, cfg.dt))

    return np.concatenate(segs, axis=0)


def run_dynamics(cfg: TrajectoryConfig, currents: np.ndarray):
    if not HAS_CPP_BINDINGS:
        raise RuntimeError("C++ bindings not available (crm_python import failed).")

    wrapper = CRMWrapper(
        param_file=cfg.param_file,
        config_file=cfg.config_file,
        use_cpp=True,
        damping=np.array(cfg.damping, dtype=np.float64),
        disable_cpp_fallback=True,
    )
    if not wrapper.is_using_cpp:
        raise RuntimeError("C++ bindings not active; cannot run convergence test.")

    # Configure numerics
    wrapper._cpp_dynamics.dt = float(cfg.dt)
    wrapper._cpp_dynamics.integration_step_size = float(cfg.integration_step_size)

    # Initialize from FK at the first current
    ok = wrapper.initialize_dynamics(currents[0], insertion_length=cfg.insertion_length)
    if not ok:
        raise RuntimeError("Initial FK initialization failed for the first current.")

    n = currents.shape[0]
    tip_pos = np.zeros((n, 3), dtype=np.float64)
    tip_vel = np.zeros((n, 3), dtype=np.float64)
    converged = np.zeros((n,), dtype=bool)
    localmin = np.zeros((n,), dtype=np.int32)

    for i in range(n):
        out = wrapper.step_dynamics(currents[i], insertion_length=cfg.insertion_length, dt=cfg.dt)
        tip_pos[i] = np.asarray(out["tip_position"], dtype=np.float64).reshape(3)
        tip_vel[i] = np.asarray(out.get("tip_velocity", np.zeros(3)), dtype=np.float64).reshape(3)
        converged[i] = bool(out.get("converged", False))
        localmin[i] = int(out.get("localmin", -999))

        # Optional recovery: if a step fails, attempt FK re-init and retry once (matches wrapper logic)
        if not converged[i]:
            ok2 = wrapper.initialize_dynamics(currents[i], insertion_length=cfg.insertion_length)
            if ok2:
                out2 = wrapper.step_dynamics(currents[i], insertion_length=cfg.insertion_length, dt=cfg.dt)
                tip_pos[i] = np.asarray(out2["tip_position"], dtype=np.float64).reshape(3)
                tip_vel[i] = np.asarray(out2.get("tip_velocity", np.zeros(3)), dtype=np.float64).reshape(3)
                converged[i] = bool(out2.get("converged", False))
                localmin[i] = int(out2.get("localmin", -999))

    t = np.arange(n, dtype=np.float64) * cfg.dt
    return t, tip_pos, tip_vel, converged, localmin


def run_fk(cfg: TrajectoryConfig, currents: np.ndarray):
    if not HAS_CPP_BINDINGS:
        raise RuntimeError("C++ bindings not available (crm_python import failed).")

    wrapper = CRMWrapper(
        param_file=cfg.param_file,
        config_file=cfg.config_file,
        use_cpp=True,
        damping=np.array(cfg.damping, dtype=np.float64),
        disable_cpp_fallback=True,
    )
    if not wrapper.is_using_cpp:
        raise RuntimeError("C++ bindings not active; cannot run FK.")

    wrapper._cpp_kinematics.integration_step_size = float(cfg.integration_step_size)

    n = currents.shape[0]
    p_fk = np.full((n, 3), np.nan, dtype=np.float64)
    conv_fk = np.zeros((n,), dtype=bool)

    stride = max(1, int(cfg.fk_stride))
    for i in range(0, n, stride):
        out = wrapper.forward_kinematics(currents[i], insertion_length=cfg.insertion_length)
        p_fk[i] = np.asarray(out["tip_position"], dtype=np.float64).reshape(3)
        conv_fk[i] = bool(out.get("converged", True))

    return p_fk, conv_fk


def plot_multiview(
    t: np.ndarray,
    u: np.ndarray,
    p_dyn: np.ndarray,
    conv_dyn: np.ndarray,
    p_fk: np.ndarray,
    conv_fk: np.ndarray,
    out_dir: Path,
) -> Path:
    ok = conv_dyn
    fig = plt.figure(figsize=(12, 9))

    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax3d.plot(p_dyn[ok, 0], p_dyn[ok, 1], p_dyn[ok, 2], lw=1.0, label="dyn (ok)")
    fk_ok = np.isfinite(p_fk).all(axis=1) & conv_fk
    if np.any(fk_ok):
        ax3d.plot(p_fk[fk_ok, 0], p_fk[fk_ok, 1], p_fk[fk_ok, 2], lw=1.0, ls="--", alpha=0.7, label="fk")
    if np.any(~ok):
        ax3d.scatter(p_dyn[~ok, 0], p_dyn[~ok, 1], p_dyn[~ok, 2], s=10, c="r", label="dyn fail")
    ax3d.set_title("Tip trajectory (3D)")
    ax3d.set_xlabel("x (mm)")
    ax3d.set_ylabel("y (mm)")
    ax3d.set_zlabel("z (mm)")
    ax3d.legend(loc="best")

    ax_xy = fig.add_subplot(2, 2, 2)
    ax_xy.plot(p_dyn[ok, 0], p_dyn[ok, 1], lw=1.0)
    if np.any(fk_ok):
        ax_xy.plot(p_fk[fk_ok, 0], p_fk[fk_ok, 1], lw=1.0, ls="--", alpha=0.7)
    if np.any(~ok):
        ax_xy.scatter(p_dyn[~ok, 0], p_dyn[~ok, 1], s=10, c="r")
    ax_xy.set_title("XY")
    ax_xy.set_xlabel("x (mm)")
    ax_xy.set_ylabel("y (mm)")
    ax_xy.axis("equal")

    ax_xz = fig.add_subplot(2, 2, 3)
    ax_xz.plot(p_dyn[ok, 0], p_dyn[ok, 2], lw=1.0)
    if np.any(fk_ok):
        ax_xz.plot(p_fk[fk_ok, 0], p_fk[fk_ok, 2], lw=1.0, ls="--", alpha=0.7)
    if np.any(~ok):
        ax_xz.scatter(p_dyn[~ok, 0], p_dyn[~ok, 2], s=10, c="r")
    ax_xz.set_title("XZ")
    ax_xz.set_xlabel("x (mm)")
    ax_xz.set_ylabel("z (mm)")

    ax_yz = fig.add_subplot(2, 2, 4)
    ax_yz.plot(p_dyn[ok, 1], p_dyn[ok, 2], lw=1.0)
    if np.any(fk_ok):
        ax_yz.plot(p_fk[fk_ok, 1], p_fk[fk_ok, 2], lw=1.0, ls="--", alpha=0.7)
    if np.any(~ok):
        ax_yz.scatter(p_dyn[~ok, 1], p_dyn[~ok, 2], s=10, c="r")
    ax_yz.set_title("YZ")
    ax_yz.set_xlabel("y (mm)")
    ax_yz.set_ylabel("z (mm)")

    fig.suptitle("CRM dynamics: half-plane c3 bias + XY circles")
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "trajectory_c3_halfplane_circle_multiview.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_timeseries(
    t: np.ndarray,
    u: np.ndarray,
    p_dyn: np.ndarray,
    conv_dyn: np.ndarray,
    localmin_dyn: np.ndarray,
    p_fk: np.ndarray,
    conv_fk: np.ndarray,
    out_dir: Path,
) -> Path:
    fig, axs = plt.subplots(5, 1, figsize=(12, 12), sharex=True)

    axs[0].plot(t, u[:, 0], label="c1")
    axs[0].plot(t, u[:, 1], label="c2")
    axs[0].plot(t, u[:, 2], label="c3")
    axs[0].set_ylabel("currents (A)")
    axs[0].legend(ncols=3, loc="upper right")

    axs[1].plot(t, p_dyn[:, 0], label="x (dyn)")
    axs[1].plot(t, p_dyn[:, 1], label="y (dyn)")
    axs[1].plot(t, p_dyn[:, 2], label="z (dyn)")
    fk_ok = np.isfinite(p_fk).all(axis=1) & conv_fk
    if np.any(fk_ok):
        axs[1].plot(t[fk_ok], p_fk[fk_ok, 0], ls="--", alpha=0.6, label="x (fk)")
        axs[1].plot(t[fk_ok], p_fk[fk_ok, 1], ls="--", alpha=0.6, label="y (fk)")
        axs[1].plot(t[fk_ok], p_fk[fk_ok, 2], ls="--", alpha=0.6, label="z (fk)")
    axs[1].set_ylabel("tip pos (mm)")
    axs[1].legend(ncols=3, loc="upper right")

    axs[2].plot(t, conv_dyn.astype(float), lw=1.0, label="dyn")
    if np.any(fk_ok):
        axs[2].plot(t[fk_ok], conv_fk[fk_ok].astype(float), lw=1.0, ls="--", alpha=0.6, label="fk")
    axs[2].set_ylabel("converged")
    axs[2].set_yticks([0, 1])
    axs[2].set_yticklabels(["fail", "ok"])
    axs[2].legend(loc="upper right")

    axs[3].plot(t, localmin_dyn, lw=1.0)
    axs[3].set_ylabel("localmin")

    err = np.full((len(t),), np.nan, dtype=np.float64)
    if np.any(fk_ok):
        err[fk_ok] = np.linalg.norm(p_dyn[fk_ok] - p_fk[fk_ok], axis=1)
    axs[4].plot(t, err, lw=1.0)
    axs[4].set_ylabel("||dyn-fk|| (mm)")
    axs[4].set_xlabel("time (s)")

    fig.suptitle("Dynamics time series")
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "trajectory_c3_halfplane_circle_timeseries.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> None:
    cfg = TrajectoryConfig()
    out_npz = Path("data/output") / "trajectory_c3_halfplane_circle.npz"
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    recompute = os.environ.get("CRM_TRAJ_RECOMPUTE", "0").strip() in ("1", "true", "TRUE", "yes", "YES")

    cfg_vec = np.array(
        [
            cfg.insertion_length,
            cfg.dt,
            cfg.integration_step_size,
            cfg.a_xy,
            cfg.f_hz,
            float(cfg.laps_each),
            cfg.c3_start,
            float(cfg.include_midplane),
            float(cfg.fk_stride),
        ],
        dtype=np.float64,
    )

    if out_npz.exists() and not recompute:
        data = np.load(out_npz, allow_pickle=False)
        prev_cfg = data.get("cfg", None)
        if prev_cfg is None or prev_cfg.shape != cfg_vec.shape or not np.allclose(prev_cfg.astype(np.float64), cfg_vec):
            recompute = True
        else:
            t = data["t"]
            u = data["currents"]
            p_dyn = data["tip_position"]
            v_dyn = data["tip_velocity"]
            conv_dyn = data["converged"].astype(bool)
            localmin_dyn = data["localmin"].astype(np.int32)
            p_fk = data["fk_tip_position"]
            conv_fk = data["fk_converged"].astype(bool)
    if (not out_npz.exists()) or recompute:
        u = build_schedule(cfg)
        t, p_dyn, v_dyn, conv_dyn, localmin_dyn = run_dynamics(cfg, u)
        p_fk, conv_fk = run_fk(cfg, u)
        np.savez_compressed(
            out_npz,
            t=t,
            currents=u,
            tip_position=p_dyn,
            tip_velocity=v_dyn,
            converged=conv_dyn.astype(np.uint8),
            localmin=localmin_dyn,
            fk_tip_position=p_fk,
            fk_converged=conv_fk.astype(np.uint8),
            cfg=cfg_vec,
        )

    plot_dir = Path("plots")
    p1 = plot_multiview(t, u, p_dyn, conv_dyn, p_fk, conv_fk, plot_dir)
    p2 = plot_timeseries(t, u, p_dyn, conv_dyn, localmin_dyn, p_fk, conv_fk, plot_dir)

    fail = int((~conv_dyn).sum())
    fk_ok = np.isfinite(p_fk).all(axis=1) & conv_fk
    err_stats = "n/a"
    if np.any(fk_ok):
        err = np.linalg.norm(p_dyn[fk_ok] - p_fk[fk_ok], axis=1)
        err_stats = f"mean={err.mean():.3f}mm p95={np.percentile(err, 95):.3f}mm max={err.max():.3f}mm (fk samples={fk_ok.sum()})"
    print(f"Steps: {len(t)}  failures: {fail} ({fail/len(t):.2%})")
    print(f"FK-vs-DYN error: {err_stats}")
    print(f"Saved: {out_npz}")
    print(f"Saved: {p1}")
    print(f"Saved: {p2}")


if __name__ == '__main__':
    main()
