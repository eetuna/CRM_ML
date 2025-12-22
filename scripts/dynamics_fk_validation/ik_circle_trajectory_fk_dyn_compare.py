"""
Inverse-kinematics (IK) current generation for desired tip-position circles and
FK/Dynamics comparison plots.

What this script does:
1) Define two XY-plane circles in tip-position space centered at (±60, 0, z0) with radius 15mm.
2) Solve IK sequentially (warm-started) using CRM FK + analytical Jacobian to produce currents.
3) FK consistency check: compare achieved FK tip positions to desired targets.
4) Run dynamics rollout using the IK currents and compare FK vs dynamics vs desired.
5) Save data + plots.

Outputs:
  - data/output/ik_circle_fk_dyn.npz
  - plots/ik_circle_fk_dyn_multiview.png
  - plots/ik_circle_fk_dyn_timeseries.png
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib.pyplot as plt  # noqa: E402

from crm_ml_rl.wrappers.crm_wrapper import CRMWrapper, HAS_CPP_BINDINGS  # noqa: E402


@dataclass(frozen=True)
class Config:
    insertion_length: float = 94.3
    dt: float = 0.05
    integration_step_size: float = 0.01
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

    # Desired circles in tip position space (mm)
    radius_mm: float = 10.0
    # Note: with the default current bounds, reachable X is ~[-35,35]mm while Y is ~[-70,70]mm.
    # Use Y-shifted circles by default.
    center1_xy: Tuple[float, float] = (0.0, 50.0)
    center2_xy: Tuple[float, float] = (0.0, -50.0)

    # Circle traversal
    # Keep this moderate by default; pseudoinverse tracking can be slow if each waypoint needs many iterations.
    points_per_circle: int = 40
    start_angle_rad: float = -np.pi / 2.0  # "bottom" of circle

    # IK settings
    ik_method: Literal["pinv", "lm"] = "pinv"
    ik_tol_mm: float = 1.0  # used by LM; pinv uses thresholds below
    ik_max_iters: int = 20  # LM inner iterations
    ik_lm_lambda: float = 1e-2
    ik_step_clip: float = 0.05  # max delta current per iter (A)
    current_bounds: Tuple[float, float] = (-0.2, 0.2)

    # Regularize against big current changes between waypoints
    smooth_weight: float = 0.1
    # Bias currents (especially c3) toward a preferred half-plane.
    bias_weight: float = 1.0
    # Keep c3 fixed per half-plane (much more stable IK).
    fix_c3: bool = True

    # MATLAB-style normalized pseudoinverse tracking data/simulation_parameters
    pinv_step_size: float = 2e-2
    pinv_threshold_start_mm: float = 5.0
    pinv_threshold_traj_mm: float = 10.0
    pinv_itrmax: int = 60

    # Ramps in current space (for dynamics) to avoid big jumps between segments
    ramp_steps_to_first: int = 80
    ramp_steps_between_circles: int = 160

    # FK used to set z0 (plane height) at a stable bias current
    bias_currents_pos: Tuple[float, float, float] = (0.0, 0.0, 0.2)
    bias_currents_neg: Tuple[float, float, float] = (0.0, 0.0, -0.2)


def _clamp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


def _interp_ramp(u0: np.ndarray, u1: np.ndarray, steps: int) -> np.ndarray:
    steps = int(max(1, steps))
    t = np.linspace(0.0, 1.0, steps, endpoint=False, dtype=np.float64)
    return (1.0 - t)[:, None] * u0[None, :] + t[:, None] * u1[None, :]


def circle_points(center_xyz: np.ndarray, radius: float, n: int, start_angle: float) -> np.ndarray:
    n = int(n)
    angles = start_angle + np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=np.float64)
    pts = np.repeat(center_xyz.reshape(1, 3), n, axis=0)
    pts[:, 0] = center_xyz[0] + radius * np.cos(angles)
    pts[:, 1] = center_xyz[1] + radius * np.sin(angles)
    return pts


class IKSolver:
    def __init__(self, wrapper: CRMWrapper, cfg: Config):
        self.wrapper = wrapper
        self.cfg = cfg
        if not self.wrapper.is_using_cpp:
            raise RuntimeError("C++ bindings not active; IK requires C++ FK + Jacobian.")
        self._deltau0_guess: Optional[np.ndarray] = None

    def fk_and_jac(self, u: np.ndarray, update_guess: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        out = self.wrapper.fk_and_jacobian(
            u, insertion_length=self.cfg.insertion_length, deltau0_initialguess=self._deltau0_guess
        )
        p = np.asarray(out["tip_position"], dtype=np.float64).reshape(3)
        du0 = np.asarray(out["delta_u0"], dtype=np.float64).reshape(3)
        if update_guess:
            self._deltau0_guess = du0

        J = np.asarray(out["jacobian"], dtype=np.float64)
        if J.ndim != 2 or J.shape[0] < 3 or J.shape[1] < 3:
            raise RuntimeError(f"Unexpected Jacobian shape: {J.shape}")
        Jp = J[:3, :3]
        return p, Jp, du0

    def fk_only(self, u: np.ndarray, update_guess: bool = True) -> np.ndarray:
        out = self.wrapper.forward_kinematics(
            u, insertion_length=self.cfg.insertion_length, deltau0_initialguess=self._deltau0_guess
        )
        p = np.asarray(out["tip_position"], dtype=np.float64).reshape(3)
        du0 = np.asarray(out["delta_u0"], dtype=np.float64).reshape(3)
        if update_guess:
            self._deltau0_guess = du0
        return p

    def solve_lm(self, target_p: np.ndarray, u_init: np.ndarray, u_prev: Optional[np.ndarray], u_bias: np.ndarray) -> Dict:
        u = np.asarray(u_init, dtype=np.float64).reshape(3).copy()
        lo, hi = self.cfg.current_bounds
        u = _clamp(u, lo, hi)
        if self.cfg.fix_c3:
            u[2] = float(u_bias[2])

        best_u = u.copy()
        best_err = np.inf
        lm = float(self.cfg.ik_lm_lambda)
        no_improve = 0

        for it in range(int(self.cfg.ik_max_iters)):
            p, Jfull, _ = self.fk_and_jac(u)
            J = Jfull[:, :2] if self.cfg.fix_c3 else Jfull
            e = target_p - p
            err = float(np.linalg.norm(e))
            if err < best_err:
                best_err = err
                best_u = u.copy()
                no_improve = 0
            else:
                no_improve += 1
            if err <= self.cfg.ik_tol_mm:
                return {"ok": True, "u": u, "p": p, "err": err, "iters": it + 1}
            # If we haven't improved in a while, this target is likely unreachable with current bounds.
            if no_improve >= 5 and best_err > 25.0:
                break

            # Levenberg-Marquardt step on the free variables.
            # If fix_c3: variables are [c1,c2]; else variables are [c1,c2,c3].
            JTJ = J.T @ J
            rhs = J.T @ e
            if self.cfg.fix_c3:
                if u_prev is not None and self.cfg.smooth_weight > 0:
                    rhs = rhs - float(self.cfg.smooth_weight) * (u[:2] - u_prev[:2])
                if self.cfg.bias_weight > 0:
                    rhs = rhs - float(self.cfg.bias_weight) * (u[:2] - u_bias[:2])
                A = JTJ + (lm + float(self.cfg.smooth_weight if u_prev is not None else 0.0) + float(self.cfg.bias_weight)) * np.eye(2)
            else:
                if u_prev is not None and self.cfg.smooth_weight > 0:
                    rhs = rhs - float(self.cfg.smooth_weight) * (u - u_prev)
                if self.cfg.bias_weight > 0:
                    rhs = rhs - float(self.cfg.bias_weight) * (u - u_bias)
                A = JTJ + (lm + float(self.cfg.smooth_weight if u_prev is not None else 0.0) + float(self.cfg.bias_weight)) * np.eye(3)

            try:
                du = np.linalg.solve(A, rhs)
            except np.linalg.LinAlgError:
                lm *= 10.0
                continue

            # Clip per-iteration step to keep changes gentle.
            du_norm = float(np.linalg.norm(du))
            if du_norm > self.cfg.ik_step_clip:
                du = du * (self.cfg.ik_step_clip / (du_norm + 1e-12))

            # Ultra-cheap acceptance: single trial step. If it doesn't improve, increase LM and retry next iter.
            u_try = u.copy()
            if self.cfg.fix_c3:
                u_try[:2] = _clamp(u[:2] + du, lo, hi)
                u_try[2] = float(u_bias[2])
            else:
                u_try = _clamp(u + du, lo, hi)
            p_try, _, _ = self.fk_and_jac(u_try, update_guess=False)
            err_try = float(np.linalg.norm(target_p - p_try))
            if err_try < err:
                u = u_try
                lm = max(lm * 0.5, 1e-6)
            else:
                lm *= 10.0

        # Return best seen if we didn't reach tol.
        p_best, _, _ = self.fk_and_jac(best_u)
        return {"ok": best_err <= self.cfg.ik_tol_mm, "u": best_u, "p": p_best, "err": best_err, "iters": self.cfg.ik_max_iters}

    def step_pinv_normalized(self, target_p: np.ndarray, u: np.ndarray, u_bias: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        One MATLAB-style step:
          controlIncrement = pinv(J_dp) * (target_p - p)
          u <- u + stepSize * controlIncrement / ||controlIncrement||
        """
        lo, hi = self.cfg.current_bounds
        u = np.asarray(u, dtype=np.float64).reshape(3).copy()
        if self.cfg.fix_c3:
            u[2] = float(u_bias[2])
        u = _clamp(u, lo, hi)

        p, Jp, _ = self.fk_and_jac(u, update_guess=True)
        e = target_p - p
        err = float(np.linalg.norm(e))

        if self.cfg.fix_c3:
            # Only solve for c1,c2 (keep c3 fixed)
            J_use = Jp[:, :2]
        else:
            J_use = Jp

        # Pseudoinverse (SVD) and normalized step
        Jinv = np.linalg.pinv(J_use)
        du = Jinv @ e
        du_norm = float(np.linalg.norm(du))
        if du_norm < 1e-12 or not np.isfinite(du_norm):
            return u, p, err

        step_dir = du / du_norm
        if self.cfg.fix_c3:
            u[:2] = _clamp(u[:2] + self.cfg.pinv_step_size * step_dir, lo, hi)
            u[2] = float(u_bias[2])
        else:
            u = _clamp(u + self.cfg.pinv_step_size * step_dir, lo, hi)

        # Recompute p at new u (and warm-start guess)
        p2 = self.fk_only(u, update_guess=True)
        err2 = float(np.linalg.norm(target_p - p2))
        return u, p2, err2

    def track_pinv_normalized(
        self,
        target_p: np.ndarray,
        u_start: np.ndarray,
        u_bias: np.ndarray,
        threshold_mm: float,
    ) -> Dict:
        """
        MATLAB-like inner loop: iterate normalized pseudoinverse updates until close enough.
        """
        u = np.asarray(u_start, dtype=np.float64).reshape(3).copy()
        # Reset warm-start state for this solve to match MATLAB's pattern of using current FKsolution.
        # (We still warm-start within the loop as FKsolution changes.)
        for it in range(int(self.cfg.pinv_itrmax)):
            u, p, err = self.step_pinv_normalized(target_p, u, u_bias=u_bias)
            if err <= threshold_mm:
                return {"ok": True, "u": u, "p": p, "err": err, "iters": it + 1}
        return {"ok": False, "u": u, "p": p, "err": err, "iters": int(self.cfg.pinv_itrmax)}

    def solve(self, target_p: np.ndarray, u_init: np.ndarray, u_prev: Optional[np.ndarray], u_bias: np.ndarray, threshold_mm: float) -> Dict:
        if self.cfg.ik_method == "pinv":
            return self.track_pinv_normalized(target_p, u_init, u_bias=u_bias, threshold_mm=threshold_mm)
        return self.solve_lm(target_p, u_init, u_prev=u_prev, u_bias=u_bias)


def run_dynamics(wrapper: CRMWrapper, cfg: Config, currents: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    wrapper._cpp_dynamics.dt = float(cfg.dt)
    wrapper._cpp_dynamics.integration_step_size = float(cfg.integration_step_size)

    ok = wrapper.initialize_dynamics(currents[0], insertion_length=cfg.insertion_length)
    if not ok:
        raise RuntimeError("Dynamics initialize_dynamics failed at first current.")

    n = currents.shape[0]
    p = np.zeros((n, 3), dtype=np.float64)
    conv = np.zeros((n,), dtype=bool)
    localmin = np.zeros((n,), dtype=np.int32)

    for i in range(n):
        out = wrapper.step_dynamics(currents[i], insertion_length=cfg.insertion_length, dt=cfg.dt)
        p[i] = np.asarray(out["tip_position"], dtype=np.float64).reshape(3)
        conv[i] = bool(out.get("converged", False))
        localmin[i] = int(out.get("localmin", -999))
        if not conv[i]:
            # attempt FK re-init and retry once
            ok2 = wrapper.initialize_dynamics(currents[i], insertion_length=cfg.insertion_length)
            if ok2:
                out2 = wrapper.step_dynamics(currents[i], insertion_length=cfg.insertion_length, dt=cfg.dt)
                p[i] = np.asarray(out2["tip_position"], dtype=np.float64).reshape(3)
                conv[i] = bool(out2.get("converged", False))
                localmin[i] = int(out2.get("localmin", -999))

    t = np.arange(n, dtype=np.float64) * cfg.dt
    return t, p, conv, localmin


def plot_all(
    t: np.ndarray,
    u: np.ndarray,
    p_des: np.ndarray,
    p_fk: np.ndarray,
    p_dyn: np.ndarray,
    conv_dyn: np.ndarray,
    out_dir: Path,
) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Multiview spatial
    fig = plt.figure(figsize=(12, 9))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax3d.plot(p_des[:, 0], p_des[:, 1], p_des[:, 2], lw=1.0, label="desired")
    ax3d.plot(p_fk[:, 0], p_fk[:, 1], p_fk[:, 2], lw=1.0, ls="--", alpha=0.8, label="fk")
    ok = conv_dyn
    ax3d.plot(p_dyn[ok, 0], p_dyn[ok, 1], p_dyn[ok, 2], lw=1.0, alpha=0.9, label="dyn (ok)")
    if np.any(~ok):
        ax3d.scatter(p_dyn[~ok, 0], p_dyn[~ok, 1], p_dyn[~ok, 2], s=10, c="r", label="dyn fail")
    ax3d.set_title("Tip trajectory (3D)")
    ax3d.set_xlabel("x (mm)")
    ax3d.set_ylabel("y (mm)")
    ax3d.set_zlabel("z (mm)")
    ax3d.legend(loc="best")

    ax_xy = fig.add_subplot(2, 2, 2)
    ax_xy.plot(p_des[:, 0], p_des[:, 1], lw=1.0, label="desired")
    ax_xy.plot(p_fk[:, 0], p_fk[:, 1], lw=1.0, ls="--", alpha=0.8, label="fk")
    ax_xy.plot(p_dyn[ok, 0], p_dyn[ok, 1], lw=1.0, alpha=0.9, label="dyn")
    ax_xy.set_title("XY")
    ax_xy.set_xlabel("x (mm)")
    ax_xy.set_ylabel("y (mm)")
    ax_xy.axis("equal")
    ax_xy.legend(loc="best")

    ax_xz = fig.add_subplot(2, 2, 3)
    ax_xz.plot(p_des[:, 0], p_des[:, 2], lw=1.0)
    ax_xz.plot(p_fk[:, 0], p_fk[:, 2], lw=1.0, ls="--", alpha=0.8)
    ax_xz.plot(p_dyn[ok, 0], p_dyn[ok, 2], lw=1.0, alpha=0.9)
    ax_xz.set_title("XZ")
    ax_xz.set_xlabel("x (mm)")
    ax_xz.set_ylabel("z (mm)")

    ax_yz = fig.add_subplot(2, 2, 4)
    ax_yz.plot(p_des[:, 1], p_des[:, 2], lw=1.0)
    ax_yz.plot(p_fk[:, 1], p_fk[:, 2], lw=1.0, ls="--", alpha=0.8)
    ax_yz.plot(p_dyn[ok, 1], p_dyn[ok, 2], lw=1.0, alpha=0.9)
    ax_yz.set_title("YZ")
    ax_yz.set_xlabel("y (mm)")
    ax_yz.set_ylabel("z (mm)")

    fig.tight_layout()
    p_multi = out_dir / "ik_circle_fk_dyn_multiview.png"
    fig.savefig(p_multi, dpi=180)
    plt.close(fig)

    # Timeseries
    fig, axs = plt.subplots(6, 1, figsize=(12, 13), sharex=True)
    axs[0].plot(t, u[:, 0], label="c1")
    axs[0].plot(t, u[:, 1], label="c2")
    axs[0].plot(t, u[:, 2], label="c3")
    axs[0].set_ylabel("currents (A)")
    axs[0].legend(ncols=3, loc="upper right")

    axs[1].plot(t, p_des[:, 0], label="x des")
    axs[1].plot(t, p_fk[:, 0], ls="--", label="x fk")
    axs[1].plot(t, p_dyn[:, 0], alpha=0.9, label="x dyn")
    axs[1].set_ylabel("x (mm)")
    axs[1].legend(loc="upper right")

    axs[2].plot(t, p_des[:, 1], label="y des")
    axs[2].plot(t, p_fk[:, 1], ls="--", label="y fk")
    axs[2].plot(t, p_dyn[:, 1], alpha=0.9, label="y dyn")
    axs[2].set_ylabel("y (mm)")
    axs[2].legend(loc="upper right")

    axs[3].plot(t, p_des[:, 2], label="z des")
    axs[3].plot(t, p_fk[:, 2], ls="--", label="z fk")
    axs[3].plot(t, p_dyn[:, 2], alpha=0.9, label="z dyn")
    axs[3].set_ylabel("z (mm)")
    axs[3].legend(loc="upper right")

    e_fk = np.linalg.norm(p_des - p_fk, axis=1)
    e_dyn = np.linalg.norm(p_des - p_dyn, axis=1)
    axs[4].plot(t, e_fk, label="||des-fk||")
    axs[4].plot(t, e_dyn, label="||des-dyn||")
    axs[4].set_ylabel("pos err (mm)")
    axs[4].legend(loc="upper right")

    axs[5].plot(t, conv_dyn.astype(float))
    axs[5].set_ylabel("dyn conv")
    axs[5].set_yticks([0, 1])
    axs[5].set_yticklabels(["fail", "ok"])
    axs[5].set_xlabel("time (s)")

    fig.tight_layout()
    p_ts = out_dir / "ik_circle_fk_dyn_timeseries.png"
    fig.savefig(p_ts, dpi=180)
    plt.close(fig)

    return p_multi, p_ts


def main() -> None:
    if not HAS_CPP_BINDINGS:
        raise RuntimeError("C++ bindings not available (crm_python import failed).")

    cfg = Config()
    # Allow quick experimentation without editing code.
    method_env = os.environ.get("CRM_IK_METHOD", "").strip().lower()
    if method_env in ("pinv", "lm"):
        object.__setattr__(cfg, "ik_method", method_env)  # type: ignore[arg-type]
    wrapper = CRMWrapper(
        param_file=cfg.param_file,
        config_file=cfg.config_file,
        use_cpp=True,
        damping=np.array(cfg.damping, dtype=np.float64),
        disable_cpp_fallback=True,
    )
    if not wrapper.is_using_cpp:
        raise RuntimeError("C++ bindings not active; cannot run IK/FK/DYN comparison.")

    wrapper._cpp_kinematics.integration_step_size = float(cfg.integration_step_size)

    # Choose plane heights from stable FK probes at the two half-plane biases.
    u_bias_pos = np.array(cfg.bias_currents_pos, dtype=np.float64)
    u_bias_neg = np.array(cfg.bias_currents_neg, dtype=np.float64)
    p0_pos = wrapper.forward_kinematics(u_bias_pos, cfg.insertion_length)["tip_position"]
    p0_neg = wrapper.forward_kinematics(u_bias_neg, cfg.insertion_length)["tip_position"]
    z0_pos = float(np.asarray(p0_pos, dtype=np.float64).reshape(3)[2])
    z0_neg = float(np.asarray(p0_neg, dtype=np.float64).reshape(3)[2])

    c1 = np.array([cfg.center1_xy[0], cfg.center1_xy[1], z0_pos], dtype=np.float64)
    c2 = np.array([cfg.center2_xy[0], cfg.center2_xy[1], z0_neg], dtype=np.float64)
    p_des_1 = circle_points(c1, cfg.radius_mm, cfg.points_per_circle, cfg.start_angle_rad)
    p_des_2 = circle_points(c2, cfg.radius_mm, cfg.points_per_circle, cfg.start_angle_rad)

    solver = IKSolver(wrapper, cfg)

    # IK for circle 1, sequential warm-start.
    u_solutions_1 = np.zeros((cfg.points_per_circle, 3), dtype=np.float64)
    p_fk_1 = np.zeros((cfg.points_per_circle, 3), dtype=np.float64)
    errs_1 = np.zeros((cfg.points_per_circle,), dtype=np.float64)
    ok_1 = np.zeros((cfg.points_per_circle,), dtype=bool)

    u_prev = None
    u_guess = u_bias_pos.copy()
    solver._deltau0_guess = None
    for i in range(cfg.points_per_circle):
        if i % 10 == 0:
            print(f"IK circle1: {i}/{cfg.points_per_circle} ...")
        thr = cfg.pinv_threshold_start_mm if (cfg.ik_method == "pinv" and i == 0) else cfg.pinv_threshold_traj_mm
        res = solver.solve(p_des_1[i], u_guess, u_prev, u_bias=u_bias_pos, threshold_mm=thr)
        u = res["u"]
        u_solutions_1[i] = u
        p_fk_1[i] = res["p"]
        errs_1[i] = float(res["err"])
        ok_1[i] = bool(res["ok"])
        u_prev = u
        u_guess = u

    # IK for circle 2, start from last solution of circle 1.
    u_solutions_2 = np.zeros((cfg.points_per_circle, 3), dtype=np.float64)
    p_fk_2 = np.zeros((cfg.points_per_circle, 3), dtype=np.float64)
    errs_2 = np.zeros((cfg.points_per_circle,), dtype=np.float64)
    ok_2 = np.zeros((cfg.points_per_circle,), dtype=bool)

    u_prev = None
    u_guess = u_bias_neg.copy()
    solver._deltau0_guess = None
    for i in range(cfg.points_per_circle):
        if i % 10 == 0:
            print(f"IK circle2: {i}/{cfg.points_per_circle} ...")
        thr = cfg.pinv_threshold_start_mm if (cfg.ik_method == "pinv" and i == 0) else cfg.pinv_threshold_traj_mm
        res = solver.solve(p_des_2[i], u_guess, u_prev, u_bias=u_bias_neg, threshold_mm=thr)
        u = res["u"]
        u_solutions_2[i] = u
        p_fk_2[i] = res["p"]
        errs_2[i] = float(res["err"])
        ok_2[i] = bool(res["ok"])
        u_prev = u
        u_guess = u

    # Build a current timeline: ramp to circle1 start, circle1, ramp to circle2 start, circle2.
    u_start = u_solutions_1[0]
    ramp0 = _interp_ramp(u_bias_pos, u_start, cfg.ramp_steps_to_first)
    ramp12 = _interp_ramp(u_solutions_1[-1], u_solutions_2[0], cfg.ramp_steps_between_circles)
    currents = np.concatenate([ramp0, u_solutions_1, ramp12, u_solutions_2], axis=0)

    p_des = np.concatenate(
        [
            np.repeat(p_des_1[0].reshape(1, 3), len(ramp0), axis=0),
            p_des_1,
            np.repeat(p_des_2[0].reshape(1, 3), len(ramp12), axis=0),
            p_des_2,
        ],
        axis=0,
    )

    # FK positions for the timeline (IK consistency check).
    p_fk = np.zeros_like(p_des)
    for i in range(currents.shape[0]):
        p_fk[i] = solver.fk_only(currents[i], update_guess=True)

    # Dynamics rollout for the same currents.
    wrapper_dyn = CRMWrapper(
        param_file=cfg.param_file,
        config_file=cfg.config_file,
        use_cpp=True,
        damping=np.array(cfg.damping, dtype=np.float64),
        disable_cpp_fallback=True,
    )
    t, p_dyn, conv_dyn, localmin_dyn = run_dynamics(wrapper_dyn, cfg, currents)

    out_npz = Path("data/output") / "ik_circle_fk_dyn.npz"
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        t=t,
        currents=currents,
        tip_desired=p_des,
        tip_fk=p_fk,
        tip_dyn=p_dyn,
        dyn_converged=conv_dyn.astype(np.uint8),
        dyn_localmin=localmin_dyn,
        circle1_ok=ok_1.astype(np.uint8),
        circle2_ok=ok_2.astype(np.uint8),
        circle1_err=errs_1,
        circle2_err=errs_2,
        z0_pos=z0_pos,
        z0_neg=z0_neg,
    )

    p_multi, p_ts = plot_all(t, currents, p_des, p_fk, p_dyn, conv_dyn, Path("plots"))

    e_fk = np.linalg.norm(p_des - p_fk, axis=1)
    e_dyn = np.linalg.norm(p_des - p_dyn, axis=1)
    print(
        f"z0_pos={z0_pos:.3f}mm z0_neg={z0_neg:.3f}mm  steps={len(t)}  "
        f"dyn_fail={int((~conv_dyn).sum())} ({((~conv_dyn).mean()):.2%})"
    )
    print(f"IK circle1 ok={int(ok_1.sum())}/{len(ok_1)} mean_err={errs_1.mean():.2f}mm max_err={errs_1.max():.2f}mm")
    print(f"IK circle2 ok={int(ok_2.sum())}/{len(ok_2)} mean_err={errs_2.mean():.2f}mm max_err={errs_2.max():.2f}mm")
    print(f"Timeline FK err: mean={e_fk.mean():.2f}mm p95={np.percentile(e_fk,95):.2f}mm max={e_fk.max():.2f}mm")
    print(f"Timeline DYN err: mean={e_dyn.mean():.2f}mm p95={np.percentile(e_dyn,95):.2f}mm max={e_dyn.max():.2f}mm")
    print(f"Saved: {out_npz}")
    print(f"Saved: {p_multi}")
    print(f"Saved: {p_ts}")


if __name__ == "__main__":
    main()
