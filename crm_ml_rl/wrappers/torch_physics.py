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

try:
    import crm_torch_ext
    HAS_TORCH_EXT = True
except ImportError:
    HAS_TORCH_EXT = False


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
                next_states[i] = np.asarray(out["next_state"], dtype=np.float64).reshape(output_dim)
                B_np = np.asarray(out["B"], dtype=np.float64)

                # Phase 4 Task 4.3: Allocate B_all on first iteration based on actual shape
                if B_all is None:
                    control_dim = B_np.shape[1] if B_np.ndim == 2 else B_np.size // output_dim
                    B_all = np.zeros((batch, output_dim, control_dim), dtype=np.float64)

                B_all[i] = B_np.reshape(output_dim, -1)
                A_all[i] = np.asarray(out["A"], dtype=np.float64).reshape(output_dim, -1)
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
                next_states[i] = np.asarray(out["next_state"], dtype=np.float64).reshape(output_dim)
                B_np = np.asarray(out["B"], dtype=np.float64)

                # Phase 4 Task 4.3: Allocate B_all on first iteration based on actual shape
                if B_all is None:
                    control_dim = B_np.shape[1] if B_np.ndim == 2 else B_np.size // output_dim
                    B_all = np.zeros((batch, output_dim, control_dim), dtype=np.float64)

                B_all[i] = B_np.reshape(output_dim, -1)

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

    def __init__(self, param_file: str, config_file: str, device: str = "cpu", use_cpp_extension: bool = False):
        if not HAS_CPP_BINDINGS:
            raise RuntimeError("C++ bindings unavailable (crm_python import failed).")
        
        self.device = torch.device(device)
        self.use_cpp_extension = use_cpp_extension

        # Initialize existing Python bindings
        self.kin = crm_python.CRMKinematics()
        self.dyn = crm_python.CRMDynamics()
        ok1 = self.kin.load_parameters(param_file, config_file)
        ok2 = self.dyn.load_parameters(param_file, config_file)
        if not (ok1 and ok2):
            raise RuntimeError("Failed to load CRM data/simulation_parameters/config into C++ bindings.")
        self.dyn.set_integrator("rk4")

        # Initialize C++ extension if requested
        if self.use_cpp_extension:
            if not HAS_TORCH_EXT:
                import warnings
                warnings.warn("use_cpp_extension=True but crm_torch_ext not installed. Falling back to Python wrapper.")
                self.use_cpp_extension = False
            else:
                # Initialize static parameters in extension singleton
                crm_torch_ext.initialize_params(param_file, config_file)
                crm_torch_ext.set_integrator("rk4")
                crm_torch_ext.set_integration_step_size(0.1)

    def set_timestep(self, dt: float):
        """Set simulation timestep."""
        self.dyn.set_timestep(dt)
        if self.use_cpp_extension:
            import crm_torch_ext
            crm_torch_ext.set_timestep(dt)

    def fk(self, currents: torch.Tensor, insertion_length: torch.Tensor) -> torch.Tensor:
        return CRMFKFunction.apply(currents, insertion_length, self.kin)

    def initialize_from_fk(self, currents: torch.Tensor, insertion_length: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Compute initial physical state from Forward Kinematics.
        
        Robustness Strategy:
        Always initializes from ZERO currents first to ensure a valid base state.
        Returns the state corresponding to ZERO currents.
        The transition to the requested 'currents' will happen during the first
        dynamics step via the C++ solver's internal homotopy (current ramp).
        
        Args:
            currents: Target starting currents [3] (Used only for shape validation)
            insertion_length: Initial insertion length [1]
            
        Returns:
            Tuple of [v, w, p, R, xf, mL, nL] tensors corresponding to ZERO currents.
        """
        if self.use_cpp_extension:
            import crm_torch_ext
            
            # 1. Validate inputs
            c = currents.detach().cpu().double().flatten()
            if c.shape[0] != 3:
                raise ValueError("Initial currents must have 3 elements")
            
            ins = insertion_length.detach().cpu().double().flatten()
            if ins.shape[0] != 1:
                raise ValueError("Insertion length must have 1 element")

            # 2. Robust Initialization: Solve for ZERO currents
            # This guarantees a valid physical seed (straight rod).
            zero_c = torch.zeros_like(c)
            
            # This call should always succeed with localmin=0
            return tuple(crm_torch_ext.initialize_from_fk(zero_c, ins))
        else:
            # Fallback using Python kin wrapper
            # Use zero currents here too for consistency
            c_np = np.zeros(3, dtype=np.float64)
            ins_val = float(insertion_length.item())
            
            res = self.kin.forward_kinematics(c_np, ins_val)
            if not res['converged']:
                raise RuntimeError("FK did not converge even for zero currents")
                
            # Manually pack tensors to match extension return format
            num_sets = self.kin.num_act_set
            options = torch.TensorOptions().dtype(torch.kFloat64)
            
            v = torch.zeros((num_sets, 3), **options)
            w = torch.zeros((num_sets, 3), **options)
            p = torch.from_numpy(res['tip_position']).reshape(1, 3).to(**options)
            R = torch.from_numpy(res['tip_rotation']).reshape(1, 9).to(**options)
            xf = torch.zeros(15, **options)
            xf[:3] = torch.from_numpy(res['tip_position'])
            xf[3:12] = torch.from_numpy(res['tip_rotation'])
            xf[12:15] = torch.from_numpy(res['delta_u0'])
            mL = torch.zeros((num_sets, 3), **options)
            nL = torch.zeros((num_sets, 3), **options)
            
            return (v, w, p, R, xf, mL, nL)

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
        return_full_state: bool = False,
    ) -> torch.Tensor | Tuple[torch::Tensor, ...]:
        if seed_mL is None:
            seed_mL = torch.zeros_like(seed_v)
        if seed_nL is None:
            seed_nL = torch.zeros_like(seed_v)

        if self.use_cpp_extension:
            # C++ Extension Path (Fast)
            batch_dim = currents.shape[0]
            if batch_dim == 1:
                # Squeeze to match unbatched extension call
                c_sq = currents.squeeze(0)
                
                # Ensure insertion is [1]
                if insertion_length.ndim == 1 and insertion_length.shape[0] == 1:
                    ins_arg = insertion_length
                elif insertion_length.ndim == 0:
                    ins_arg = insertion_length.unsqueeze(0)
                else:
                    ins_arg = insertion_length.reshape(1)

                sv_sq = seed_v.squeeze(0)
                sw_sq = seed_w.squeeze(0)
                sp_sq = seed_p.squeeze(0)
                sR_sq = seed_R.squeeze(0)
                sxf_sq = seed_xf.squeeze(0)
                smL_sq = seed_mL.squeeze(0)
                snL_sq = seed_nL.squeeze(0)

                # Call returns [next_state, next_v, next_w, next_p, next_R, next_xf, next_mL, next_nL]
                res_list = crm_torch_ext.crm_step(
                    c_sq, ins_arg, sv_sq, sw_sq, sp_sq, sR_sq, sxf_sq, smL_sq, snL_sq
                )
                
                # Reshape all outputs back to batch dimensions
                res_list_batched = [t.unsqueeze(0) for t in res_list]
                
                if return_full_state:
                    return tuple(res_list_batched)
                return res_list_batched[0]
            else:
                # Manual loop for batch > 1
                all_results = []
                for i in range(batch_dim):
                    ins_i = insertion_length[i] if insertion_length.ndim > 0 else insertion_length
                    res_i = crm_torch_ext.crm_step(
                        currents[i], ins_i,
                        seed_v[i], seed_w[i], seed_p[i], seed_R[i], seed_xf[i], seed_mL[i], seed_nL[i]
                    )
                    all_results.append(res_i)
                
                # Transpose list of lists to list of stacks
                num_outputs = len(all_results[0])
                final_outputs = []
                for j in range(num_outputs):
                    final_outputs.append(torch.stack([res[j] for res in all_results]))
                
                if return_full_state:
                    return tuple(final_outputs)
                return final_outputs[0]

        # Python Wrapper Path (Legacy/Fallback)
        # Note: legacy path doesn't easily support return_full_state yet in same format
        if return_full_state:
             raise NotImplementedError("return_full_state=True only supported with use_cpp_extension=True")

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
