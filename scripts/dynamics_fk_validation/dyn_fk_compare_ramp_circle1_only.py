"""
FK vs Dynamics comparison for *only*:
  - ramp_to_circle1
  - circle1

This excludes bridge_to_circle2 and circle2.

Runs one lap of circle1 currents (default 200 points) and supports hold=1 or hold=2
by repeating each ramp/circle current row.

Outputs:
  - data/output/dyn_fk_ramp_circle1_hold{H}.npz
  - plots/dyn_fk_ramp_circle1_hold{H}_multiview.png
  - plots/dyn_fk_ramp_circle1_hold{H}_timeseries.png
"""

from __future__ import annotations

import argparse
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
    circle1_csv: str = "data/output/circle1_currents_y40_r10_dp_0p1mm_n200.csv"
    insertion_length: float = 94.3
    dt: float = 0.05
    integration_step_size: float = 0.2
    param_file: str = "data/catheter_params/CatheterParameterSet_1_dyn.txt"
    config_file: str = "data/catheter_params/CatheterSpatialConfiguration_1.txt"

    c3_start: float = 0.01
    ramp_to_circle_steps: int = 120
    nonzero_eps: float = 0.005
    circle_points: int = 200


def load_currents_csv(path: str) -> np.ndarray:
    arr = np.genfromtxt(path, delimiter=",", skip_header=1)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] != 3:
        raise ValueError(f"Expected 3 columns in {path}, got shape {arr.shape}")
    return arr.astype(np.float64)


def resample_periodic(currents: np.ndarray, n_new: int) -> np.ndarray:
    cur = np.asarray(currents, dtype=np.float64)
    n = int(cur.shape[0])
    n_new = int(n_new)
    if n_new == n:
        return cur.copy()
    cur_ext = np.vstack([cur, cur[0:1]])
    s_old = np.linspace(0.0, 1.0, n + 1, endpoint=True, dtype=np.float64)
    s_new = np.linspace(0.0, 1.0, n_new, endpoint=False, dtype=np.float64)
    out = np.zeros((n_new, 3), dtype=np.float64)
    for j in range(3):
        out[:, j] = np.interp(s_new, s_old, cur_ext[:, j])
    return out


def rotate_to_index(seq: np.ndarray, start_idx: int) -> np.ndarray:
    start_idx = int(start_idx) % int(seq.shape[0])
    return np.roll(seq, -start_idx, axis=0)


def interp_ramp(u0: np.ndarray, u1: np.ndarray, steps: int) -> np.ndarray:
    steps = int(max(1, steps))
    t = np.linspace(0.0, 1.0, steps, endpoint=False, dtype=np.float64)
    return (1.0 - t)[:, None] * u0[None, :] + t[:, None] * u1[None, :]


def hold_each(currents: np.ndarray, hold_steps: int) -> np.ndarray:
    hold_steps = int(max(1, hold_steps))
    return np.repeat(np.asarray(currents, dtype=np.float64), hold_steps, axis=0)


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
    ax.set_title("XY")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax = fig.add_subplot(2, 2, 3)
    ax.plot(p_fk[:, 0], p_fk[:, 2], lw=2.0, c="tab:blue", label="FK")
    ax.plot(p_dyn[ok, 0], p_dyn[ok, 2], lw=1.5, c="tab:orange", label="Dyn(ok)")
    ax.set_title("XZ")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax = fig.add_subplot(2, 2, 4)
    ax.plot(p_fk[:, 1], p_fk[:, 2], lw=2.0, c="tab:blue", label="FK")
    ax.plot(p_dyn[ok, 1], p_dyn[ok, 2], lw=1.5, c="tab:orange", label="Dyn(ok)")
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", type=int, default=1, choices=[1, 2])
    args = ap.parse_args()

    if not HAS_CPP_BINDINGS:
        raise RuntimeError("C++ bindings not available.")

    cfg = Config()
    hold = int(args.hold)

    c1 = load_currents_csv(cfg.circle1_csv)
    c1 = resample_periodic(c1, cfg.circle_points)

    u_init = np.array([0.0, 0.0, cfg.c3_start], dtype=np.float64)
    i1 = int(np.argmin(np.linalg.norm(c1 - u_init[None, :], axis=1)))
    c1r = rotate_to_index(c1, i1)

    u1_start = c1r[0].copy()

    ramp0 = interp_ramp(u_init, u1_start, cfg.ramp_to_circle_steps)
    ramp0 = hold_each(ramp0, hold)
    circle1 = hold_each(c1r, hold)

    currents = np.concatenate([ramp0, circle1], axis=0)
    currents = ensure_nonzero(currents, cfg.nonzero_eps, prefer_axis=2)

    kin = crm_python.CRMKinematics()
    assert kin.load_parameters(cfg.param_file, cfg.config_file)
    kin.integration_step_size = float(cfg.integration_step_size)

    p_fk = fk_trajectory(cfg, kin, currents)
    t, p_dyn, conv_dyn = run_dynamics(cfg, currents)

    e = np.linalg.norm(p_fk - p_dyn, axis=1)
    ok = conv_dyn
    e_ok = e[ok]

    n_ramp = len(ramp0)
    n_circle = len(circle1)
    e_ramp = e_ok[: int(np.sum(ok[:n_ramp]))] if np.any(ok[:n_ramp]) else np.array([])
    e_circle = e_ok[int(np.sum(ok[:n_ramp])) :] if np.any(ok[n_ramp:]) else np.array([])

    out_dir = Path("data/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    out_npz = out_dir / f"dyn_fk_ramp_circle1_hold{hold}.npz"
    np.savez_compressed(
        out_npz,
        t=t,
        currents=currents,
        tip_fk=p_fk,
        tip_dyn=p_dyn,
        dyn_converged=conv_dyn.astype(np.uint8),
        fk_dyn_err=e,
        segments=np.array([n_ramp, n_circle], dtype=np.int32),
        hold=hold,
        dt=cfg.dt,
        insertion_length=cfg.insertion_length,
    )

    mv = plots_dir / f"dyn_fk_ramp_circle1_hold{hold}_multiview.png"
    ts = plots_dir / f"dyn_fk_ramp_circle1_hold{hold}_timeseries.png"
    plot_multiview(p_fk, p_dyn, conv_dyn, mv)
    plot_timeseries(t, currents, p_fk, p_dyn, conv_dyn, ts)

    def _summ(name: str, arr: np.ndarray) -> str:
        if arr.size == 0:
            return f"{name}: n=0"
        return f"{name}: mean={arr.mean():.3f} p95={np.percentile(arr,95):.3f} max={arr.max():.3f} n={arr.size}"

    print(f"hold={hold} N={len(t)} dyn_fail={int((~conv_dyn).sum())}")
    print(_summ("all(ok)", e_ok))
    # Segment stats on full arrays (use ok mask per segment)
    er = e[:n_ramp][ok[:n_ramp]]
    ec = e[n_ramp:][ok[n_ramp:]]
    print(_summ("ramp(ok)", er))
    print(_summ("circle1(ok)", ec))
    print(f"Saved: {out_npz}")
    print(f"Saved: {mv}")
    print(f"Saved: {ts}")


if __name__ == "__main__":
    main()
