"""
Run FK and dynamics on a ramped circle current sequence and compare tip trajectories.

Sequence:
  1) Start at u=[0,0,c3_start] (avoid exact 0,0,0).
  2) Ramp to circle1 start current.
  3) Traverse circle1 currents (full loop).
  4) Ramp c1,c2->0 (keep c3), then change sign of c3 with c1=c2=0 (skip c3=0),
     then ramp to circle2 start current.
  5) Traverse circle2 currents (full loop).

Inputs are the exported IK currents from `data/output/*.csv`.
Outputs:
  - data/output/dyn_fk_ramped_circles.npz
  - plots/dyn_fk_ramped_circles_multiview.png
  - plots/dyn_fk_ramped_circles_timeseries.png
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
    # Currents (exported)
    circle1_csv: str = "data/output/circle1_currents_y40_r10_dp_0p1mm_n200.csv"
    circle2_csv: str = "data/output/circle2_currents_y-40_r10_dp_0p1mm_n200.csv"

    # Model
    insertion_length: float = 50.0
    dt: float = 0.05
    integration_step_size: float = 0.2
    param_file: str = "data/catheter_params/CatheterParameterSet_1_dyn.txt"
    config_file: str = "data/catheter_params/CatheterSpatialConfiguration_1.txt"

    # Ramps
    c3_start: float = 0.01
    ramp_to_circle_steps: int = 120
    bridge_steps: int = 480  # transition from circle1 end -> circle2 start
    nonzero_eps: float = 0.005  # avoid exactly (0,0,0)
    # Circle traversal (one lap)
    circle_points: int = 200  # resample currents to this many points per circle
    hold_steps_per_point: int = 2  # dt=0.05 -> 0.1s per point
    hold_steps_ramp: int = 2  # dt=0.05 -> 0.1s per ramp point


def load_currents_csv(path: str) -> np.ndarray:
    arr = np.genfromtxt(path, delimiter=",", skip_header=1)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] != 3:
        raise ValueError(f"Expected 3 columns in {path}, got shape {arr.shape}")
    return arr.astype(np.float64)


def rotate_to_index(seq: np.ndarray, start_idx: int) -> np.ndarray:
    start_idx = int(start_idx) % int(seq.shape[0])
    return np.roll(seq, -start_idx, axis=0)


def interp_ramp(u0: np.ndarray, u1: np.ndarray, steps: int) -> np.ndarray:
    steps = int(max(1, steps))
    t = np.linspace(0.0, 1.0, steps, endpoint=False, dtype=np.float64)
    return (1.0 - t)[:, None] * u0[None, :] + t[:, None] * u1[None, :]


def c3_flip_no_zero(c3_from: float, c3_to: float, steps: int, eps: float) -> np.ndarray:
    steps = int(max(2, steps))
    c3_from = float(c3_from)
    c3_to = float(c3_to)
    eps = float(abs(eps))
    if eps <= 0:
        eps = 1e-6

    crosses_zero = (c3_from > 0 and c3_to < 0) or (c3_from < 0 and c3_to > 0)
    if not crosses_zero:
        return np.linspace(c3_from, c3_to, steps, endpoint=False, dtype=np.float64)

    half = steps // 2
    a = np.linspace(c3_from, np.sign(c3_from) * eps, half, endpoint=False, dtype=np.float64)
    b = np.linspace(-np.sign(c3_from) * eps, c3_to, steps - half, endpoint=False, dtype=np.float64)
    return np.concatenate([a, b], axis=0)


def ensure_nonzero(currents: np.ndarray, eps: float, *, prefer_axis: int = 2) -> np.ndarray:
    """
    Ensure no row is exactly the 0-vector (or extremely close).
    If ||u|| < eps, nudge `prefer_axis` to +/-eps while keeping other channels unchanged.
    """
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


def resample_periodic(currents: np.ndarray, n_new: int) -> np.ndarray:
    """
    Periodically resample a closed current sequence to n_new points using linear interpolation.
    Assumes currents represent one lap and should wrap from end back to start.
    """
    cur = np.asarray(currents, dtype=np.float64)
    n = int(cur.shape[0])
    n_new = int(n_new)
    if n_new <= 0:
        raise ValueError("n_new must be > 0")
    if n_new == n:
        return cur.copy()

    cur_ext = np.vstack([cur, cur[0:1]])
    s_old = np.linspace(0.0, 1.0, n + 1, endpoint=True, dtype=np.float64)
    s_new = np.linspace(0.0, 1.0, n_new, endpoint=False, dtype=np.float64)
    out = np.zeros((n_new, cur.shape[1]), dtype=np.float64)
    for j in range(cur.shape[1]):
        out[:, j] = np.interp(s_new, s_old, cur_ext[:, j])
    return out


def hold_each(currents: np.ndarray, hold_steps: int) -> np.ndarray:
    hold_steps = int(max(1, hold_steps))
    return np.repeat(np.asarray(currents, dtype=np.float64), hold_steps, axis=0)


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


def plot_multiview(p_fk: np.ndarray, p_dyn: np.ndarray, conv_dyn: np.ndarray, out_png: Path) -> None:
    ok = conv_dyn
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ax.plot(p_fk[:, 0], p_fk[:, 1], p_fk[:, 2], lw=2.0, c="tab:blue", label="FK")
    ax.plot(p_dyn[ok, 0], p_dyn[ok, 1], p_dyn[ok, 2], lw=1.5, c="tab:orange", label="Dyn(ok)")
    if np.any(~ok):
        ax.scatter(p_dyn[~ok, 0], p_dyn[~ok, 1], p_dyn[~ok, 2], s=10, c="tab:red", label="Dyn fail")
    ax.set_title("3D")
    ax.legend(loc="best")

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(p_fk[:, 0], p_fk[:, 1], lw=2.0, c="tab:blue", label="FK")
    ax.plot(p_dyn[ok, 0], p_dyn[ok, 1], lw=1.5, c="tab:orange", label="Dyn(ok)")
    if np.any(~ok):
        ax.scatter(p_dyn[~ok, 0], p_dyn[~ok, 1], s=10, c="tab:red", label="Dyn fail")
    ax.set_title("XY")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax = fig.add_subplot(2, 2, 3)
    ax.plot(p_fk[:, 0], p_fk[:, 2], lw=2.0, c="tab:blue", label="FK")
    ax.plot(p_dyn[ok, 0], p_dyn[ok, 2], lw=1.5, c="tab:orange", label="Dyn(ok)")
    if np.any(~ok):
        ax.scatter(p_dyn[~ok, 0], p_dyn[~ok, 2], s=10, c="tab:red", label="Dyn fail")
    ax.set_title("XZ")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax = fig.add_subplot(2, 2, 4)
    ax.plot(p_fk[:, 1], p_fk[:, 2], lw=2.0, c="tab:blue", label="FK")
    ax.plot(p_dyn[ok, 1], p_dyn[ok, 2], lw=1.5, c="tab:orange", label="Dyn(ok)")
    if np.any(~ok):
        ax.scatter(p_dyn[~ok, 1], p_dyn[~ok, 2], s=10, c="tab:red", label="Dyn fail")
    ax.set_title("YZ")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_timeseries(t: np.ndarray, u: np.ndarray, p_fk: np.ndarray, p_dyn: np.ndarray, conv_dyn: np.ndarray, out_png: Path) -> None:
    e = np.linalg.norm(p_fk - p_dyn, axis=1)
    fig, axs = plt.subplots(5, 1, figsize=(12, 12), sharex=True)

    axs[0].plot(t, u[:, 0], label="c1")
    axs[0].plot(t, u[:, 1], label="c2")
    axs[0].plot(t, u[:, 2], label="c3")
    axs[0].set_ylabel("currents (A)")
    axs[0].legend(ncols=3, loc="upper right")

    axs[1].plot(t, e, c="tab:purple", label="||FK-Dyn||")
    axs[1].set_ylabel("err (mm)")
    axs[1].grid(True, alpha=0.25)
    axs[1].legend(loc="upper right")

    axs[2].plot(t, conv_dyn.astype(float), c="k")
    axs[2].set_ylabel("dyn conv")
    axs[2].set_yticks([0, 1])
    axs[2].set_yticklabels(["fail", "ok"])

    axs[3].plot(t, p_fk[:, 0], c="tab:blue", label="x FK")
    axs[3].plot(t, p_dyn[:, 0], c="tab:orange", label="x Dyn", alpha=0.85)
    axs[3].set_ylabel("x (mm)")
    axs[3].legend(loc="upper right")

    axs[4].plot(t, p_fk[:, 1], c="tab:blue", label="y FK")
    axs[4].plot(t, p_dyn[:, 1], c="tab:orange", label="y Dyn", alpha=0.85)
    axs[4].set_ylabel("y (mm)")
    axs[4].set_xlabel("time (s)")
    axs[4].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def segment_slices(seg_lengths: np.ndarray) -> list[tuple[str, slice]]:
    names = ["ramp_to_circle1", "circle1", "bridge_to_circle2", "circle2"]
    seg_lengths = np.asarray(seg_lengths, dtype=int).reshape(-1)
    if len(seg_lengths) != len(names):
        raise ValueError(f"Expected {len(names)} segment lengths, got {len(seg_lengths)}")
    out: list[tuple[str, slice]] = []
    start = 0
    for name, n in zip(names, seg_lengths):
        out.append((name, slice(start, start + int(n))))
        start += int(n)
    return out


def print_error_breakdown(t: np.ndarray, u: np.ndarray, e: np.ndarray, conv: np.ndarray, seg_lengths: np.ndarray) -> None:
    conv = conv.astype(bool)
    segs = segment_slices(seg_lengths)
    print("FK-Dyn error breakdown (mm):")
    for name, sl in segs:
        ee = e[sl]
        cc = conv[sl]
        if np.any(cc):
            ee_ok = ee[cc]
            mean = float(ee_ok.mean())
            p95 = float(np.percentile(ee_ok, 95))
            mx = float(ee_ok.max())
            fails = int((~cc).sum())
        else:
            mean = p95 = mx = float("nan")
            fails = int((~cc).sum())
        print(f"  {name:14s}: mean={mean:.3f} p95={p95:.3f} max={mx:.3f} dyn_fail={fails}/{len(ee)}")

    k = 10
    idx = np.argsort(e)[-k:][::-1]
    print(f"Top {k} worst timesteps:")
    for i in idx:
        print(f"  i={int(i):4d} t={t[i]:7.3f}s err={e[i]:7.3f}mm conv={bool(conv[i])} u={u[i]}")


def main() -> None:
    if not HAS_CPP_BINDINGS:
        raise RuntimeError("C++ bindings not available.")
    cfg = Config()

    c1 = load_currents_csv(cfg.circle1_csv)
    c2 = load_currents_csv(cfg.circle2_csv)
    c1 = resample_periodic(c1, cfg.circle_points)
    c2 = resample_periodic(c2, cfg.circle_points)

    kin = crm_python.CRMKinematics()
    assert kin.load_parameters(cfg.param_file, cfg.config_file)
    kin.integration_step_size = float(cfg.integration_step_size)

    # Choose start points on circles closest to near-zero currents (to avoid "jumping to the opposite side").
    # For circle1 (typically +c3), pick the point closest to u_init.
    # For circle2 (typically -c3), pick the point closest to u_init mirrored in c3.
    u_init = np.array([0.0, 0.0, cfg.c3_start], dtype=np.float64)
    u_init_2 = np.array([0.0, 0.0, -cfg.c3_start], dtype=np.float64)
    i1 = int(np.argmin(np.linalg.norm(c1 - u_init[None, :], axis=1)))
    i2 = int(np.argmin(np.linalg.norm(c2 - u_init_2[None, :], axis=1)))
    c1r = rotate_to_index(c1, i1)
    c2r = rotate_to_index(c2, i2)

    u1_start = c1r[0].copy()
    u1_end = c1r[-1].copy()
    u2_start = c2r[0].copy()

    ramp0 = interp_ramp(u_init, u1_start, cfg.ramp_to_circle_steps)
    ramp0 = hold_each(ramp0, cfg.hold_steps_ramp)
    circle1 = hold_each(c1r, cfg.hold_steps_per_point)

    # Transition between circles allowing c1/c2 to vary, but never hit exactly (0,0,0).
    bridge = interp_ramp(u1_end, u2_start, cfg.bridge_steps)
    bridge = ensure_nonzero(bridge, cfg.nonzero_eps, prefer_axis=2)
    circle2 = hold_each(c2r, cfg.hold_steps_per_point)

    currents = np.concatenate([ramp0, circle1, bridge, circle2], axis=0)
    currents = ensure_nonzero(currents, cfg.nonzero_eps, prefer_axis=2)

    # FK along full timeline (warm-started)
    p_fk = fk_trajectory(cfg, kin, currents)

    # Dynamics
    t, p_dyn, conv_dyn = run_dynamics(cfg, currents)

    e = np.linalg.norm(p_fk - p_dyn, axis=1)
    ok = conv_dyn
    e_ok = e[ok]

    out_dir = Path("data/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    out_npz = out_dir / "dyn_fk_ramped_circles.npz"
    np.savez_compressed(
        out_npz,
        t=t,
        currents=currents,
        tip_fk=p_fk,
        tip_dyn=p_dyn,
        dyn_converged=conv_dyn.astype(np.uint8),
        fk_dyn_err=e,
        circle1_start_idx=i1,
        circle2_start_idx=i2,
        segments=np.array(
            [
                len(ramp0),
                len(circle1),
                len(bridge),
                len(circle2),
            ],
            dtype=np.int32,
        ),
    )

    mv = plots_dir / "dyn_fk_ramped_circles_multiview.png"
    ts = plots_dir / "dyn_fk_ramped_circles_timeseries.png"
    plot_multiview(p_fk, p_dyn, conv_dyn, mv)
    plot_timeseries(t, currents, p_fk, p_dyn, conv_dyn, ts)

    seg_lengths = np.array([len(ramp0), len(circle1), len(bridge), len(circle2)], dtype=np.int32)
    print_error_breakdown(t, currents, e, conv_dyn, seg_lengths)

    print(f"N={len(t)} dyn_fail={int((~conv_dyn).sum())}")
    if len(e_ok):
        print(f"FK-Dyn err (ok): mean={e_ok.mean():.3f}mm p95={np.percentile(e_ok,95):.3f}mm max={e_ok.max():.3f}mm")
    print(f"FK-Dyn err (all): mean={e.mean():.3f}mm p95={np.percentile(e,95):.3f}mm max={e.max():.3f}mm")
    print(f"Saved: {out_npz}")
    print(f"Saved: {mv}")
    print(f"Saved: {ts}")


if __name__ == "__main__":
    main()
