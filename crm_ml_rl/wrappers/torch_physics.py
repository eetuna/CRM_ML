"""
Torch autograd wrappers for the CRM C++ physics bindings.

These wrappers intentionally avoid passing through NumPy in the forward path so
that Torch can backpropagate through physics via custom backward rules.

Notes:
- FK backward uses the existing C++ analytical Jacobian exposed as
  `CRMKinematics.compute_jacobian` and slices dp/d(currents,insertion).
- Dynamics backward currently uses finite-difference B = d(next_state)/d(currents)
  computed in C++ via `CRMDynamics.linearize_action_from_seed`.
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
        grad_insertion = grad_inputs[:, 3]
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

        next_states = np.zeros((batch, 6), dtype=np.float64)
        B_all = np.zeros((batch, 6, 3), dtype=np.float64)
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
            A_all = np.zeros((batch, 6, seed_dim), dtype=np.float64)

        for i in range(batch):
            if need_seed_jac:
                method = os.environ.get("CRM_DYN_LINEARIZATION_METHOD", "").strip().lower()
                if method == "implicit":
                    out = dyn.linearize_full_seed_action_from_seed_implicit(
                        currents_np[i],
                        float(ins_np[i]),
                        v_np[i],
                        w_np[i],
                        p_np[i],
                        R_np[i],
                        xf_np[i],
                        mL_np[i],
                        nL_np[i],
                        float(eps_seed),  # eps_residual_x
                        float(eps_seed),  # eps_residual_theta
                        float(eps_seed),  # eps_g_x
                        float(eps_seed),  # eps_g_theta
                    )
                else:
                    out = dyn.linearize_full_seed_action_from_seed(
                        currents_np[i],
                        float(ins_np[i]),
                        v_np[i],
                        w_np[i],
                        p_np[i],
                        R_np[i],
                        xf_np[i],
                        mL_np[i],
                        nL_np[i],
                        float(eps_u),
                        float(eps_seed),
                    )
                next_states[i] = np.asarray(out["next_state"], dtype=np.float64).reshape(6)
                B_all[i] = np.asarray(out["B"], dtype=np.float64).reshape(6, 3)
                A_all[i] = np.asarray(out["A"], dtype=np.float64).reshape(6, -1)
            else:
                out = dyn.linearize_action_from_seed(
                    currents_np[i],
                    float(ins_np[i]),
                    v_np[i],
                    w_np[i],
                    p_np[i],
                    R_np[i],
                    xf_np[i],
                    mL_np[i],
                    nL_np[i],
                    float(eps_u),
                )
                next_states[i] = np.asarray(out["next_state"], dtype=np.float64).reshape(6)
                B_all[i] = np.asarray(out["B"], dtype=np.float64).reshape(6, 3)

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
        saved = ctx.saved_tensors
        B = saved[0]  # (B, 6, 3)
        A = saved[1] if (getattr(ctx, "has_seed_jac", False) and len(saved) > 1) else None
        if grad_next_state is None:
            return (None,) * 12

        # grad_currents = B^T * grad_next_state
        grad_currents = torch.einsum("bik,bk->bi", B.transpose(1, 2), grad_next_state)
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
