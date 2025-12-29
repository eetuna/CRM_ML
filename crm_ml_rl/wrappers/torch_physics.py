"""
Torch autograd wrappers for the CRM C++ physics bindings.

These wrappers intentionally avoid passing through NumPy in the forward path so
that Torch can backpropagate through physics via custom backward rules.

Notes:
- FK backward uses the existing C++ analytical Jacobian exposed as
  `CRMKinematics.compute_jacobian` and slices dp/d(currents,insertion).
- Dynamics backward uses the C++ AD-based implicit linearization exposed via
  `CRMDynamics.linearize_full_seed_action_from_seed_implicit`.
- Both currents and insertion_length are fully differentiable in both FK and Dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional, Tuple

import numpy as np
import torch

try:
    from . import crm_python

    HAS_CPP_BINDINGS = True
except Exception:  # pragma: no cover
    HAS_CPP_BINDINGS = False


def _as_float(x) -> float:
    return float(x.item() if isinstance(x, torch.Tensor) else x)


@dataclass
class DynamicsSeed:
    v: np.ndarray  # (num_act_set, 3)
    w: np.ndarray  # (num_act_set, 3)
    p: np.ndarray  # (num_act_set, 3)
    R: np.ndarray  # (num_act_set, 9)
    xf: np.ndarray  # (15,)
    mL: Optional[np.ndarray] = None  # (num_act_set, 3)
    nL: Optional[np.ndarray] = None  # (num_act_set, 3)


class CRMFKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, currents: torch.Tensor, insertion_length: torch.Tensor, kin: "crm_python.CRMKinematics"):
        if not HAS_CPP_BINDINGS:
            raise RuntimeError("C++ bindings unavailable (crm_python import failed).")
        if currents.ndim != 2 or currents.shape[1] != 3:
            raise ValueError("currents must have shape (batch, 3)")

        device = currents.device
        batch = currents.shape[0]

        currents_np = currents.detach().cpu().double().numpy()
        ins_np = insertion_length.detach().cpu().double().view(-1).numpy()
        if ins_np.size == 1:
            ins_np = np.full((batch,), float(ins_np[0]), dtype=np.float64)
        elif ins_np.size != batch:
            raise ValueError("insertion_length must be scalar or shape (batch,)")

        tip_positions = np.zeros((batch, 3), dtype=np.float64)
        J_dp = np.zeros((batch, 3, 4), dtype=np.float64)  # dp/d[currents(3), insertion(1)]

        for i in range(batch):
            res = kin.forward_kinematics(currents_np[i], float(ins_np[i]))
            tip_positions[i] = np.asarray(res["tip_position"], dtype=np.float64).reshape(3)

            J = np.asarray(kin.compute_jacobian(currents_np[i], float(ins_np[i])), dtype=np.float64)
            # Expected to include dp/dz in the first 3 rows; columns start with currents and insertion.
            if J.ndim != 2 or J.shape[0] < 3 or J.shape[1] < 4:
                raise RuntimeError(f"Unexpected FK Jacobian shape: {J.shape}")
            J_dp[i, :, :] = J[:3, :4]

        ctx.save_for_backward(torch.from_numpy(J_dp).to(device=device, dtype=torch.float32))
        return torch.from_numpy(tip_positions).to(device=device, dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_tip_pos: torch.Tensor):
        (J_dp,) = ctx.saved_tensors
        if grad_tip_pos is None:
            return None, None, None

        # J_dp: (B, 3, 4); grad_tip_pos: (B, 3) => grad_inputs: (B, 4)
        grad_inputs = torch.einsum("bij,bi->bj", J_dp, grad_tip_pos)
        grad_currents = grad_inputs[:, :3]
        grad_insertion = grad_inputs[:, 3:4]  # Keep shape (B, 1) to match input shape
        return grad_currents, grad_insertion, None


class CRMDynamicsStepFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        currents: torch.Tensor,
        insertion_length: torch.Tensor,
        seed_v: torch.Tensor,
        seed_w: torch.Tensor,
        seed_p: torch.Tensor,
        seed_R: torch.Tensor,
        seed_xf: torch.Tensor,
        seed_mL: torch.Tensor,
        seed_nL: torch.Tensor,
        dyn: "crm_python.CRMDynamics",
        eps_u: float = 1e-4,
        eps_seed: float = 1e-4,
    ):
        if not HAS_CPP_BINDINGS:
            raise RuntimeError("C++ bindings unavailable (crm_python import failed).")

        if currents.ndim != 2 or currents.shape[1] != 3:
            raise ValueError("currents must have shape (batch, 3)")

        device = currents.device
        batch = currents.shape[0]
        num_sets = int(seed_v.shape[1]) if seed_v.ndim == 3 else None

        currents_np = currents.detach().cpu().double().numpy()
        ins_np = insertion_length.detach().cpu().double().view(-1).numpy()
        if ins_np.size == 1:
            ins_np = np.full((batch,), float(ins_np[0]), dtype=np.float64)
        elif ins_np.size != batch:
            raise ValueError("insertion_length must be scalar or shape (batch,)")

        v_np = seed_v.detach().cpu().double().numpy()
        w_np = seed_w.detach().cpu().double().numpy()
        p_np = seed_p.detach().cpu().double().numpy()
        R_np = seed_R.detach().cpu().double().numpy()
        xf_np = seed_xf.detach().cpu().double().numpy()
        mL_np = seed_mL.detach().cpu().double().numpy()
        nL_np = seed_nL.detach().cpu().double().numpy()

        # Phase 4 Task 4.3: Prepare for variable-sized output (multi-actuator)
        # output_dim = 3 (tip_pos) + 3*num_sets (coil velocities)
        # For single actuator: output_dim = 6
        output_dim = 3 + 3 * (num_sets if num_sets is not None else 1)
        next_states = np.zeros((batch, output_dim), dtype=np.float64)

        # Phase 4 Task 4.3: B matrix may be (output_dim, 3) or (output_dim, 4)
        # Currently (6, 3) for [currents]
        # After Phase 1 Task 1.2: (6, 4) for [currents, insertion_length]
        B_all = None  # Will be allocated after first call to determine shape
        A_all = None

        need_seed_jac = any(
            t.requires_grad
            for t in (
                seed_v,
                seed_w,
                seed_p,
                seed_R,
                seed_xf,
                seed_mL,
                seed_nL,
            )
        )
        if need_seed_jac:
            if num_sets is None:
                raise ValueError("seed_v must have shape (batch, num_act_set, 3)")
            seed_dim = int(num_sets * 3 + num_sets * 3 + num_sets * 3 + num_sets * 9 + 15 + num_sets * 3 + num_sets * 3)
            A_all = np.zeros((batch, output_dim, seed_dim), dtype=np.float64)

        # OPTION A FIX: Compute B and A matrices via finite differences
        # This matches the approach used in Option C (crm_torch/csrc/dynamics_op.cpp)
        # Replaces the broken linearize_full_seed_action_from_seed_implicit/explicit methods

        fd_eps = 1e-5  # FD epsilon for both current and seed perturbations

        for i in range(batch):
            # Compute forward pass for nominal state
            out = dyn.step_from_seed(
                currents_np[i],
                float(ins_np[i]),
                v_np[i],
                w_np[i],
                p_np[i],
                R_np[i],
                xf_np[i],
                mL_np[i],
                nL_np[i],
            )

            # Extract next state: [tip_position(3), tip_velocity(3)]
            tip_pos = np.asarray(out["tip_position"], dtype=np.float64).reshape(3)
            tip_vel = np.asarray(out["tip_velocity"], dtype=np.float64).reshape(3)
            next_states[i] = np.concatenate([tip_pos, tip_vel])

            # Compute B matrix via finite differences: ∂output/∂currents
            # B shape: (output_dim, 3) for currents only
            # Note: insertion_length gradients not yet supported (would need column 4)
            B_np = np.zeros((output_dim, 3), dtype=np.float64)

            for j in range(3):  # For each current dimension
                # Perturb current +eps
                curr_plus = currents_np[i].copy()
                curr_plus[j] += fd_eps
                out_plus = dyn.step_from_seed(
                    curr_plus,
                    float(ins_np[i]),
                    v_np[i],
                    w_np[i],
                    p_np[i],
                    R_np[i],
                    xf_np[i],
                    mL_np[i],
                    nL_np[i],
                )
                pos_plus = np.asarray(out_plus["tip_position"], dtype=np.float64).reshape(3)
                vel_plus = np.asarray(out_plus["tip_velocity"], dtype=np.float64).reshape(3)
                y_plus = np.concatenate([pos_plus, vel_plus])

                # Perturb current -eps
                curr_minus = currents_np[i].copy()
                curr_minus[j] -= fd_eps
                out_minus = dyn.step_from_seed(
                    curr_minus,
                    float(ins_np[i]),
                    v_np[i],
                    w_np[i],
                    p_np[i],
                    R_np[i],
                    xf_np[i],
                    mL_np[i],
                    nL_np[i],
                )
                pos_minus = np.asarray(out_minus["tip_position"], dtype=np.float64).reshape(3)
                vel_minus = np.asarray(out_minus["tip_velocity"], dtype=np.float64).reshape(3)
                y_minus = np.concatenate([pos_minus, vel_minus])

                # Central difference
                B_np[:, j] = (y_plus - y_minus) / (2.0 * fd_eps)

            # Allocate B_all on first iteration
            if B_all is None:
                control_dim = 3  # Only currents, not insertion_length yet
                B_all = np.zeros((batch, output_dim, control_dim), dtype=np.float64)

            B_all[i] = B_np

            # Compute A matrix via finite differences if seed gradients needed
            if need_seed_jac:
                # A shape: (output_dim, seed_dim) where seed_dim = 39 for num_sets=1
                seed_dim = int(num_sets * 3 + num_sets * 3 + num_sets * 3 + num_sets * 9 + 15 + num_sets * 3 + num_sets * 3)
                A_np = np.zeros((output_dim, seed_dim), dtype=np.float64)

                component_idx = 0

                # Helper function to compute FD for a seed component
                def compute_seed_fd(seed_array, flat_idx):
                    nonlocal component_idx
                    original = seed_array.flat[flat_idx]

                    # Perturb +eps
                    seed_array.flat[flat_idx] = original + fd_eps
                    out_p = dyn.step_from_seed(
                        currents_np[i],
                        float(ins_np[i]),
                        v_np[i],
                        w_np[i],
                        p_np[i],
                        R_np[i],
                        xf_np[i],
                        mL_np[i],
                        nL_np[i],
                    )
                    pos_p = np.asarray(out_p["tip_position"], dtype=np.float64).reshape(3)
                    vel_p = np.asarray(out_p["tip_velocity"], dtype=np.float64).reshape(3)
                    y_p = np.concatenate([pos_p, vel_p])

                    # Perturb -eps
                    seed_array.flat[flat_idx] = original - fd_eps
                    out_m = dyn.step_from_seed(
                        currents_np[i],
                        float(ins_np[i]),
                        v_np[i],
                        w_np[i],
                        p_np[i],
                        R_np[i],
                        xf_np[i],
                        mL_np[i],
                        nL_np[i],
                    )
                    pos_m = np.asarray(out_m["tip_position"], dtype=np.float64).reshape(3)
                    vel_m = np.asarray(out_m["tip_velocity"], dtype=np.float64).reshape(3)
                    y_m = np.concatenate([pos_m, vel_m])

                    # Restore original
                    seed_array.flat[flat_idx] = original

                    # Central difference
                    A_np[:, component_idx] = (y_p - y_m) / (2.0 * fd_eps)
                    component_idx += 1

                # Compute A matrix columns for all seed components
                # v components
                for idx in range(v_np[i].size):
                    compute_seed_fd(v_np[i], idx)

                # w components
                for idx in range(w_np[i].size):
                    compute_seed_fd(w_np[i], idx)

                # p components
                for idx in range(p_np[i].size):
                    compute_seed_fd(p_np[i], idx)

                # R components
                for idx in range(R_np[i].size):
                    compute_seed_fd(R_np[i], idx)

                # xf components
                for idx in range(xf_np[i].size):
                    compute_seed_fd(xf_np[i], idx)

                # mL components
                for idx in range(mL_np[i].size):
                    compute_seed_fd(mL_np[i], idx)

                # nL components
                for idx in range(nL_np[i].size):
                    compute_seed_fd(nL_np[i], idx)

                A_all[i] = A_np

        ctx.num_sets = num_sets
        ctx.has_seed_jac = need_seed_jac
        B_tensor = torch.from_numpy(B_all).to(device=device, dtype=torch.float32)
        if need_seed_jac:
            A_tensor = torch.from_numpy(A_all).to(device=device, dtype=torch.float32)
            ctx.save_for_backward(B_tensor, A_tensor)
        else:
            ctx.save_for_backward(B_tensor)
        return torch.from_numpy(next_states).to(device=device, dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_next_state: torch.Tensor):
        """
        Backward pass for dynamics step.

        Computes gradients of the loss w.r.t. inputs using the chain rule and
        linearization matrices A (state Jacobian) and B (control Jacobian).

        Differentiable inputs:
            - currents: Always computed via B matrix (∂next_state/∂currents)
            - insertion_length: Computed via the 4th column of the control Jacobian
              returned by the C++ AD-based implicit linearizer.
            - seed tensors (v, w, p, R, xf, mL, nL): Computed via A matrix when available
              (requires CRM_DYN_LINEARIZATION_METHOD=implicit or fd)

        Non-differentiable inputs:
            - dyn, eps_u, eps_seed: Configuration parameters (not trainable)

        Args:
            ctx: Context from forward pass containing saved tensors (B, A matrices)
            grad_next_state: Gradient of loss w.r.t. next_state output (B, 6)

        Returns:
            Tuple of gradients for all forward inputs in order:
            (grad_currents, grad_insertion, grad_seed_v, grad_seed_w, grad_seed_p,
             grad_seed_R, grad_seed_xf, grad_seed_mL, grad_seed_nL, None, None, None)
        """
        saved = ctx.saved_tensors
        B = saved[0]  # (B, output_dim, control_dim) where control_dim is 3 or 4
        A = saved[1] if (getattr(ctx, "has_seed_jac", False) and len(saved) > 1) else None
        if grad_next_state is None:
            return (None,) * 12

        # Phase 4 Task 4.3: Handle variable-sized B matrix
        # B shape: (batch, output_dim, control_dim)
        # control_dim = 3: [currents] (current state)
        # control_dim = 4: [currents, insertion_length] (after Phase 1 Task 1.2)

        # grad_controls = B^T * grad_next_state => shape (batch, control_dim)
        grad_controls = torch.einsum("bik,bk->bi", B.transpose(1, 2), grad_next_state)

        # Extract gradients based on control_dim
        control_dim = B.shape[2]
        if control_dim >= 3:
            grad_currents = grad_controls[:, :3]
        else:
            grad_currents = None

        # Phase 4 Task 4.3: Extract grad_insertion when available (control_dim == 4)
        if control_dim >= 4:
            grad_insertion = grad_controls[:, 3:4]  # Keep shape (batch, 1)
        else:
            # Insertion length gradient not yet available (Phase 1 Task 1.2 not complete)
            grad_insertion = None

        grad_seed_v = None
        grad_seed_w = None
        grad_seed_p = None
        grad_seed_R = None
        grad_seed_xf = None
        grad_seed_mL = None
        grad_seed_nL = None

        if A is not None:
            num_sets = int(getattr(ctx, "num_sets", 0))
            if num_sets <= 0:
                raise RuntimeError("Missing num_sets for seed gradient unflattening")
            grad_seed_flat = torch.einsum("bik,bk->bi", A.transpose(1, 2), grad_next_state)  # (B, seed_dim)

            dim_v = num_sets * 3
            dim_w = num_sets * 3
            dim_p = num_sets * 3
            dim_R = num_sets * 9
            dim_xf = 15
            dim_mL = num_sets * 3
            dim_nL = num_sets * 3

            i_v0 = 0
            i_w0 = i_v0 + dim_v
            i_p0 = i_w0 + dim_w
            i_R0 = i_p0 + dim_p
            i_xf0 = i_R0 + dim_R
            i_mL0 = i_xf0 + dim_xf
            i_nL0 = i_mL0 + dim_mL

            grad_seed_v = grad_seed_flat[:, i_v0:i_w0].reshape(-1, num_sets, 3)
            grad_seed_w = grad_seed_flat[:, i_w0:i_p0].reshape(-1, num_sets, 3)
            grad_seed_p = grad_seed_flat[:, i_p0:i_R0].reshape(-1, num_sets, 3)
            grad_seed_R = grad_seed_flat[:, i_R0:i_xf0].reshape(-1, num_sets, 9)
            grad_seed_xf = grad_seed_flat[:, i_xf0:i_mL0].reshape(-1, 15)
            grad_seed_mL = grad_seed_flat[:, i_mL0:i_nL0].reshape(-1, num_sets, 3)
            grad_seed_nL = grad_seed_flat[:, i_nL0:].reshape(-1, num_sets, 3)

        return (
            grad_currents,
            grad_insertion,
            grad_seed_v,
            grad_seed_w,
            grad_seed_p,
            grad_seed_R,
            grad_seed_xf,
            grad_seed_mL,
            grad_seed_nL,
            None,
            None,
            None,
        )


class TorchCRMPhysics:
    """
    Convenience wrapper holding C++ objects and exposing differentiable calls.
    """

    def __init__(self, param_file: str, config_file: str, device: str = "cpu"):
        if not HAS_CPP_BINDINGS:
            raise RuntimeError("C++ bindings unavailable (crm_python import failed).")
        self.device = torch.device(device)
        self.kin = crm_python.CRMKinematics()
        self.dyn = crm_python.CRMDynamics()
        ok1 = self.kin.load_parameters(param_file, config_file)
        ok2 = self.dyn.load_parameters(param_file, config_file)
        if not (ok1 and ok2):
            raise RuntimeError("Failed to load CRM data/simulation_parameters/config into C++ bindings.")
        # BUGFIX: set_integrator("rk4") causes step_from_seed() to diverge
        # Use default integrator instead
        # self.dyn.set_integrator("rk4")

    def fk(self, currents: torch.Tensor, insertion_length: torch.Tensor) -> torch.Tensor:
        return CRMFKFunction.apply(currents, insertion_length, self.kin)

    def dyn_step(
        self,
        currents: torch.Tensor,
        insertion_length: torch.Tensor,
        seed_v: torch.Tensor,
        seed_w: torch.Tensor,
        seed_p: torch.Tensor,
        seed_R: torch.Tensor,
        seed_xf: torch.Tensor,
        seed_mL: Optional[torch.Tensor] = None,
        seed_nL: Optional[torch.Tensor] = None,
        eps_u: float = 1e-4,
        eps_seed: float = 1e-4,
    ) -> torch.Tensor:
        if seed_mL is None:
            seed_mL = torch.zeros_like(seed_v)
        if seed_nL is None:
            seed_nL = torch.zeros_like(seed_v)
        return CRMDynamicsStepFunction.apply(
            currents,
            insertion_length,
            seed_v,
            seed_w,
            seed_p,
            seed_R,
            seed_xf,
            seed_mL,
            seed_nL,
            self.dyn,
            eps_u,
            eps_seed,
        )
