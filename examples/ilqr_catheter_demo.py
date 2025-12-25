#!/usr/bin/env python3
"""
iLQR (Iterative Linear Quadratic Regulator) Demo for Catheter Control

This script demonstrates trajectory optimization using iLQR with:
1. Point reaching: Navigate catheter tip to a target position
2. Trajectory tracking: Follow a reference trajectory over time

Compares performance of:
- Implicit AD linearization (fast, accurate gradients)
- Full FD linearization (slow, baseline)
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from crm_ml_rl.wrappers import crm_python


# Stable damping values (from integrator stability analysis)
BASE_DAMPING = np.array([
    12.1761626666366,      # Linear damping X
    12.1761626666366,      # Linear damping Y
    284.429938756989,      # Linear damping Z
    0.0304776127617393,    # Angular damping X
    0.0304776127617393,    # Angular damping Y
    0.00502712804532508    # Angular damping Z
], dtype=np.float64)


class iLQRController:
    """Iterative Linear Quadratic Regulator for catheter control."""

    def __init__(
        self,
        dyn: crm_python.CRMDynamics,
        insertion_length: float,
        horizon: int = 50,
        max_iters: int = 20,
        tol: float = 1e-3,
        alpha_min: float = 1e-4,
        use_implicit: bool = True
    ):
        """
        Initialize iLQR controller.

        Args:
            dyn: CRMDynamics instance (must be initialized)
            insertion_length: Fixed insertion length (mm)
            horizon: Planning horizon (number of timesteps)
            max_iters: Maximum iterations for iLQR
            tol: Convergence tolerance for cost reduction
            alpha_min: Minimum line search step size
            use_implicit: Use implicit AD linearization (vs full FD)
        """
        self.dyn = dyn
        self.insertion_length = insertion_length
        self.horizon = horizon
        self.max_iters = max_iters
        self.tol = tol
        self.alpha_min = alpha_min
        self.use_implicit = use_implicit

        # State: [px, py, pz, vx, vy, vz] (6D, in mm and mm/s)
        # Action: [Ix, Iy, Iz] (3D)
        self.state_dim = 6
        self.action_dim = 3

        # Control input scaling: normalized [-1, 1] maps to physical currents
        # Using step() API which is more stable than step_from_seed()
        self.current_scale = 0.1  # Maximum current amplitude in Amps

    def _linearize(self, currents_normalized: np.ndarray, seed_state: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Linearize dynamics around current state.

        Args:
            currents_normalized: Normalized currents in [-1, 1]

        Returns:
            next_state: (6,) next state
            A: (6, seed_dim) state Jacobian
            B: (6, 3) control Jacobian (scaled for normalized inputs)
        """
        # Convert normalized currents to physical currents
        currents_physical = currents_normalized * self.current_scale

        v = seed_state['v']
        w = seed_state['w']
        p = seed_state['p']
        R = seed_state['R']
        xf = seed_state['xf']
        mL = seed_state['mL']
        nL = seed_state['nL']

        if self.use_implicit:
            result = self.dyn.linearize_full_seed_action_from_seed_implicit(
                currents_physical, self.insertion_length, v, w, p, R, xf, mL, nL
            )
        else:
            result = self.dyn.linearize_full_seed_action_from_seed(
                currents_physical, self.insertion_length, v, w, p, R, xf, mL, nL
            )

        next_state = np.array(result['next_state'], dtype=np.float64)  # (6,)
        A = np.array(result['A'], dtype=np.float64)  # (6, seed_dim)
        B_physical = np.array(result['B'], dtype=np.float64)  # (6, 3)

        # Scale B matrix for normalized inputs: B_normalized = B_physical * current_scale
        B = B_physical * self.current_scale

        return next_state, A, B

    def _step_forward(self, currents_normalized: np.ndarray, seed_state: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Take a forward dynamics step using step() API for stability.

        Args:
            currents_normalized: Normalized currents in [-1, 1]
            seed_state: Current seed state (used to set internal state)

        Returns:
            state: (6,) tip position + velocity
            new_seed: Updated seed state dict
        """
        # Convert normalized to physical currents
        currents_physical = currents_normalized * self.current_scale

        # Set internal seed state before stepping
        self.dyn.set_seed_state(
            seed_state['v'],
            seed_state['w'],
            seed_state['p'],
            seed_state['R'],
            seed_state['xf'],
            seed_state['mL'],
            seed_state['nL']
        )

        # Use step() instead of step_from_seed() for BVP stability
        result = self.dyn.step(currents_physical, self.insertion_length)

        tip_pos = np.array(result['tip_position'], dtype=np.float64)
        tip_vel = np.array(result['tip_velocity'], dtype=np.float64)
        state = np.concatenate([tip_pos, tip_vel])

        # Get new seed state from internal state
        new_seed = self.dyn.get_seed_state()

        return state, new_seed

    def _quadratize_cost(
        self,
        x: np.ndarray,
        u: np.ndarray,
        x_target: np.ndarray,
        is_final: bool = False
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Quadratic approximation of cost function.

        Cost: l(x, u) = 0.5 * ||x[:3] - x_target[:3]||^2 + 0.5 * R * ||u||^2

        Returns:
            l: scalar cost
            l_x: (state_dim,) cost gradient w.r.t. state
            l_u: (action_dim,) cost gradient w.r.t. action
            l_xx: (state_dim, state_dim) cost Hessian w.r.t. state
            l_uu: (action_dim, action_dim) cost Hessian w.r.t. action
            l_ux: (action_dim, state_dim) cost Hessian w.r.t. action and state
        """
        # Position tracking cost (only on position, not velocity)
        Q_pos = 100.0 if is_final else 10.0  # Higher weight on final state
        Q_vel = 1.0 if is_final else 0.1
        R_action = 0.1  # Action regularization - increased from 0.01 to penalize aggressive currents

        # Cost
        pos_error = x[:3] - x_target[:3]
        vel_error = x[3:6] - x_target[3:6] if x_target.shape[0] == 6 else x[3:6]

        l = 0.5 * Q_pos * np.dot(pos_error, pos_error) + \
            0.5 * Q_vel * np.dot(vel_error, vel_error) + \
            0.5 * R_action * np.dot(u, u)

        # Gradients
        l_x = np.zeros(self.state_dim)
        l_x[:3] = Q_pos * pos_error
        l_x[3:6] = Q_vel * vel_error
        l_u = R_action * u

        # Hessians
        l_xx = np.diag(np.concatenate([
            Q_pos * np.ones(3),
            Q_vel * np.ones(3)
        ]))
        l_uu = R_action * np.eye(self.action_dim)
        l_ux = np.zeros((self.action_dim, self.state_dim))

        return l, l_x, l_u, l_xx, l_uu, l_ux

    def _backward_pass(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        A_list: list,
        B_list: list,
        target_traj: np.ndarray
    ) -> Tuple[list, list, float]:
        """
        Backward pass: compute optimal gains and feedforward terms.

        Args:
            states: (T+1, state_dim) state trajectory
            actions: (T, action_dim) action sequence
            A_list: List of T state Jacobians
            B_list: List of T control Jacobians
            target_traj: (T+1, state_dim) or (state_dim,) target

        Returns:
            K_list: List of (action_dim, state_dim) feedback gains
            k_list: List of (action_dim,) feedforward terms
            expected_cost_reduction: Expected reduction in cost
        """
        T = self.horizon
        K_list = []
        k_list = []

        # Terminal cost-to-go
        x_final = states[-1]
        x_target_final = target_traj[-1] if target_traj.ndim == 2 else target_traj

        _, l_x, _, l_xx, _, _ = self._quadratize_cost(
            x_final, np.zeros(self.action_dim), x_target_final, is_final=True
        )

        V_x = l_x
        V_xx = l_xx
        expected_cost_reduction = 0.0

        # Backward recursion
        for t in range(T - 1, -1, -1):
            x_t = states[t]
            u_t = actions[t]
            x_target_t = target_traj[t] if target_traj.ndim == 2 else target_traj

            # A_list[t] is (6, seed_dim) where seed_dim >> 6
            # The state Jacobian A maps changes in seed state to changes in tip state
            # For iLQR, we approximate the state-to-state dynamics using only the B matrix
            # since the seed state includes many internal DOFs (v, w, p, R, xf, mL, nL)
            #
            # The seed state vector layout is approximately:
            # v (3), w (3), p (N*3), R (N*9), xf (N), mL (N*3), nL (N*3)
            # where N is the number of spatial segments
            #
            # For tip state [px, py, pz, vx, vy, vz], we could extract columns corresponding
            # to v (velocity seed) from A_list[t], but this is still an approximation
            #
            # A simpler and more stable approach: treat dynamics as primarily control-driven
            # and use A_t ≈ I (identity) to represent state persistence
            A_t = np.eye(self.state_dim)  # Simplified: assume state persists + control input
            B_t = B_list[t]  # (6, 3)

            # Quadratize cost
            l, l_x, l_u, l_xx, l_uu, l_ux = self._quadratize_cost(x_t, u_t, x_target_t, is_final=False)

            # Q-function approximation
            # Q(x,u) ≈ l(x,u) + V(f(x,u)) where f(x,u) = A*x + B*u
            Q_x = l_x + A_t.T @ V_x
            Q_u = l_u + B_t.T @ V_x
            Q_xx = l_xx + A_t.T @ V_xx @ A_t
            Q_uu = l_uu + B_t.T @ V_xx @ B_t
            Q_ux = l_ux + B_t.T @ V_xx @ A_t

            # Regularization for numerical stability
            Q_uu_reg = Q_uu + 1e-4 * np.eye(self.action_dim)

            # Solve for gains
            try:
                Q_uu_inv = np.linalg.inv(Q_uu_reg)
            except np.linalg.LinAlgError:
                # Fallback: use pseudoinverse
                Q_uu_inv = np.linalg.pinv(Q_uu_reg)

            k_t = -Q_uu_inv @ Q_u
            K_t = -Q_uu_inv @ Q_ux

            # Update value function
            V_x = Q_x - K_t.T @ Q_uu @ k_t
            V_xx = Q_xx - K_t.T @ Q_uu @ K_t

            # Store gains (in forward order, will reverse later)
            K_list.insert(0, K_t)
            k_list.insert(0, k_t)

            # Track expected cost reduction
            expected_cost_reduction += 0.5 * k_t.T @ Q_uu @ k_t

        return K_list, k_list, expected_cost_reduction

    def _forward_pass(
        self,
        states_nom: np.ndarray,
        actions_nom: np.ndarray,
        K_list: list,
        k_list: list,
        seeds_nom: list,
        alpha: float
    ) -> Tuple[np.ndarray, np.ndarray, list, float, bool]:
        """
        Forward pass with line search.

        Returns:
            states_new: (T+1, state_dim) updated trajectory
            actions_new: (T, action_dim) updated actions
            seeds_new: List of T+1 seed states
            cost_new: Total cost of new trajectory
            success: Whether forward pass succeeded
        """
        T = self.horizon
        states_new = np.zeros((T + 1, self.state_dim))
        actions_new = np.zeros((T, self.action_dim))
        seeds_new = [None] * (T + 1)

        states_new[0] = states_nom[0]
        seeds_new[0] = seeds_nom[0]
        cost_new = 0.0

        for t in range(T):
            # Compute state deviation (simplified - just use tip state)
            dx = states_new[t] - states_nom[t]

            # Update control with gains (in normalized space)
            u_new = actions_nom[t] + alpha * k_list[t] + K_list[t] @ dx

            # Clip to normalized bounds [-1, 1], then scale to physical currents
            u_normalized = np.clip(u_new, -1.0, 1.0)
            u_physical = u_normalized * self.current_scale

            actions_new[t] = u_normalized  # Store normalized action

            # Forward step using step() API for BVP stability
            try:
                # Set internal seed state
                self.dyn.set_seed_state(
                    seeds_new[t]['v'],
                    seeds_new[t]['w'],
                    seeds_new[t]['p'],
                    seeds_new[t]['R'],
                    seeds_new[t]['xf'],
                    seeds_new[t]['mL'],
                    seeds_new[t]['nL']
                )

                # Use step() instead of step_from_seed()
                currents_physical = u_normalized * self.current_scale
                result = self.dyn.step(currents_physical, self.insertion_length)

                # Check for divergence
                if result.get('diverged', False):
                    print(f"      Diverged at timestep {t}")
                    return states_nom, actions_nom, seeds_nom, float('inf'), False

                # Extract state
                tip_pos = np.array(result['tip_position'], dtype=np.float64)
                tip_vel = np.array(result['tip_velocity'], dtype=np.float64)
                states_new[t + 1] = np.concatenate([tip_pos, tip_vel])

                # Get updated seed state from internal state
                seeds_new[t + 1] = self.dyn.get_seed_state()

            except Exception as e:
                # Forward pass failed
                print(f"      Exception at timestep {t}: {type(e).__name__}: {e}")
                return states_nom, actions_nom, seeds_nom, float('inf'), False

        # Compute total cost (simplified - just use final cost for now)
        # In full implementation, would sum over all timesteps
        return states_new, actions_new, seeds_new, cost_new, True

    def solve(
        self,
        x0: np.ndarray,
        x_target: np.ndarray,
        u_init: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Solve for optimal trajectory to reach target.

        Args:
            x0: (6,) initial state [px, py, pz, vx, vy, vz]
            x_target: (6,) or (3,) target state (position or position+velocity)
            u_init: (T, 3) initial action sequence (default: zeros)

        Returns:
            states: (T+1, 6) optimal state trajectory
            actions: (T, 3) optimal action sequence
            info: Dict with convergence info
        """
        T = self.horizon

        # Expand target to 6D if needed
        if x_target.shape[0] == 3:
            x_target_full = np.concatenate([x_target, np.zeros(3)])
        else:
            x_target_full = x_target

        # Initialize trajectory with small random actions to break symmetry
        if u_init is None:
            actions = 0.01 * np.random.randn(T, self.action_dim)
        else:
            actions = u_init.copy()

        states = np.zeros((T + 1, self.state_dim))
        states[0] = x0
        seeds = [self.dyn.get_seed_state()]

        # Settling steps: run zero-current steps to reach equilibrium
        # TEMPORARILY DISABLED FOR DEBUGGING
        # print("Running settling steps (10 zero-current steps)...")
        # settling_seed = seeds[0]
        # for i in range(10):
        #     settling_state, settling_seed = self._step_forward(np.zeros(self.action_dim), settling_seed)
        # # Update initial seed and state after settling
        # seeds[0] = settling_seed
        # states[0] = settling_state

        # Initial forward rollout
        print("Initial forward rollout...")
        for t in range(T):
            states[t + 1], new_seed = self._step_forward(actions[t], seeds[t])
            seeds.append(new_seed)

        initial_cost = np.linalg.norm(states[-1, :3] - x_target_full[:3])
        print(f"Initial cost (final pos error): {initial_cost:.6f}")

        # iLQR iterations
        costs = [initial_cost]
        times_linearize = []
        times_backward = []
        times_forward = []

        for iteration in range(self.max_iters):
            iter_start = time.time()

            # Linearize along trajectory
            t_lin_start = time.time()
            A_list = []
            B_list = []
            for t in range(T):
                _, A_t, B_t = self._linearize(actions[t], seeds[t])
                A_list.append(A_t)
                B_list.append(B_t)
            times_linearize.append(time.time() - t_lin_start)

            # Backward pass
            t_back_start = time.time()
            K_list, k_list, expected_reduction = self._backward_pass(
                states, actions, A_list, B_list, x_target_full
            )
            times_backward.append(time.time() - t_back_start)

            # Forward pass with line search
            t_fwd_start = time.time()
            alphas = [1.0, 0.5, 0.25, 0.1]
            best_cost = float('inf')
            best_states, best_actions, best_seeds = None, None, None

            for alpha in alphas:
                states_new, actions_new, seeds_new, cost_new, success = self._forward_pass(
                    states, actions, K_list, k_list, seeds, alpha
                )

                if success:
                    actual_cost = np.linalg.norm(states_new[-1, :3] - x_target_full[:3])
                    print(f"    alpha={alpha:.2f}: cost={actual_cost:.6f}, success={success}")
                    if actual_cost < best_cost:
                        best_cost = actual_cost
                        best_states = states_new
                        best_actions = actions_new
                        best_seeds = seeds_new
                else:
                    print(f"    alpha={alpha:.2f}: FAILED (divergence or exception)")

            times_forward.append(time.time() - t_fwd_start)

            if best_states is None:
                print(f"Iteration {iteration + 1}: Line search failed")
                break

            # Update trajectory
            states = best_states
            actions = best_actions
            seeds = best_seeds
            costs.append(best_cost)

            # Check convergence
            cost_reduction = costs[-2] - costs[-1]
            iter_time = time.time() - iter_start

            print(f"Iteration {iteration + 1}: cost={best_cost:.6f}, "
                  f"reduction={cost_reduction:.6f}, time={iter_time:.3f}s "
                  f"(lin={times_linearize[-1]:.3f}s, back={times_backward[-1]:.3f}s, fwd={times_forward[-1]:.3f}s)")

            if cost_reduction < self.tol and cost_reduction >= 0:
                print(f"Converged! (reduction={cost_reduction:.6f} < tol={self.tol})")
                break

        info = {
            'costs': costs,
            'times_linearize': times_linearize,
            'times_backward': times_backward,
            'times_forward': times_forward,
            'converged': costs[-1] < initial_cost * 0.1,  # Within 10% of initial
            'final_error': costs[-1]
        }

        return states, actions, info


def setup_dyn(insertion: float = 94.3) -> crm_python.CRMDynamics:
    """Initialize dynamics with stable parameters."""
    dyn = crm_python.CRMDynamics()
    ok = dyn.load_parameters(
        "data/catheter_params/CatheterParameterSet_1_dyn.txt",
        "data/catheter_params/CatheterSpatialConfiguration_1.txt",
    )
    if not ok:
        raise RuntimeError("Failed to load parameters")

    dyn.set_damping(BASE_DAMPING)
    dyn.dt = 0.02  # 20ms timestep
    dyn.integration_step_size = 0.1
    dyn.initialize_from_kinematics([0.0, 0.0, 0.2], insertion)

    return dyn


def reaching_demo(use_implicit: bool = True) -> Dict:
    """Point reaching demo: move tip to target position."""
    print("\n" + "="*60)
    print("REACHING DEMO" + (" (Implicit AD)" if use_implicit else " (Full FD)"))
    print("="*60)

    # Setup
    insertion = 94.3
    dyn = setup_dyn(insertion=insertion)
    controller = iLQRController(dyn, insertion, horizon=30, max_iters=15, use_implicit=use_implicit)

    # Initial state (tip position from initialization, in mm)
    tip_pos = np.array(dyn.get_tip_position(), dtype=np.float64)
    x0 = np.concatenate([
        tip_pos,
        np.zeros(3)  # Zero initial velocity
    ])

    # Target: 50mm forward, 20mm right, 80mm up (in mm, like tip_pos)
    target = np.array([50.0, 20.0, 80.0])

    print(f"Initial position: {x0[:3]} mm")
    print(f"Target position: {target} mm")
    print(f"Initial distance: {np.linalg.norm(x0[:3] - target):.2f} mm")

    # Solve
    start_time = time.time()
    states, actions, info = controller.solve(x0, target)
    total_time = time.time() - start_time

    # Results
    final_pos = states[-1, :3]
    final_error = np.linalg.norm(final_pos - target)

    print(f"\nFinal position: {final_pos}")
    print(f"Final error: {final_error*1000:.2f} mm")
    print(f"Total time: {total_time:.2f}s")
    print(f"Converged: {info['converged']}")

    info['total_time'] = total_time
    info['final_position'] = final_pos
    info['target'] = target
    info['states'] = states
    info['actions'] = actions

    return info


def tracking_demo(use_implicit: bool = True) -> Dict:
    """Trajectory tracking demo: follow a sinusoidal path."""
    print("\n" + "="*60)
    print("TRACKING DEMO" + (" (Implicit AD)" if use_implicit else " (Full FD)"))
    print("="*60)

    # Setup
    dyn = setup_dyn(insertion=94.3)

    # Generate reference trajectory (sinusoidal, in mm)
    T = 50
    t = np.linspace(0, 2*np.pi, T)
    ref_traj = np.zeros((T, 6))
    ref_traj[:, 0] = 50.0 * np.sin(t)  # x: sinusoid (mm)
    ref_traj[:, 1] = 20.0 * np.cos(t)  # y: cosine (mm)
    ref_traj[:, 2] = 80.0 + 10.0 * np.sin(2*t)  # z: small oscillation (mm)
    # Velocities are zero (simplified)

    print(f"Tracking {T} waypoints")
    print("NOTE: Full tracking implementation would use receding horizon MPC")
    print("For now, just targeting final waypoint...")

    # For simplicity, just target the final waypoint (full tracking needs receding horizon)
    target_final = ref_traj[-1, :3]

    insertion = 94.3
    controller = iLQRController(dyn, insertion, horizon=30, max_iters=10, use_implicit=use_implicit)

    # Initial state
    tip_pos = np.array(dyn.get_tip_position(), dtype=np.float64)
    x0 = np.concatenate([tip_pos, np.zeros(3)])

    # Solve
    start_time = time.time()
    states, actions, info = controller.solve(x0, target_final)
    total_time = time.time() - start_time

    final_error = np.linalg.norm(states[-1, :3] - target_final)
    print(f"\nFinal tracking error: {final_error*1000:.2f} mm")
    print(f"Total time: {total_time:.2f}s")

    info['total_time'] = total_time
    info['reference'] = ref_traj
    info['states'] = states
    info['actions'] = actions

    return info


def comparison_experiment():
    """Compare Implicit AD vs Full FD."""
    print("\n" + "="*80)
    print("COMPARISON EXPERIMENT: Implicit AD vs Full FD")
    print("="*80)

    # Run with implicit AD
    result_implicit = reaching_demo(use_implicit=True)

    # Run with full FD
    result_fd = reaching_demo(use_implicit=False)

    # Compare
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)

    print(f"\nImplicit AD:")
    print(f"  Total time: {result_implicit['total_time']:.2f}s")
    print(f"  Final error: {result_implicit['final_error']*1000:.2f} mm")
    print(f"  Avg linearization time: {np.mean(result_implicit['times_linearize']):.4f}s")
    print(f"  Converged: {result_implicit['converged']}")

    print(f"\nFull FD:")
    print(f"  Total time: {result_fd['total_time']:.2f}s")
    print(f"  Final error: {result_fd['final_error']*1000:.2f} mm")
    print(f"  Avg linearization time: {np.mean(result_fd['times_linearize']):.4f}s")
    print(f"  Converged: {result_fd['converged']}")

    speedup = result_fd['total_time'] / result_implicit['total_time']
    lin_speedup = np.mean(result_fd['times_linearize']) / np.mean(result_implicit['times_linearize'])

    print(f"\nSpeedup (Implicit vs FD):")
    print(f"  Total: {speedup:.2f}x")
    print(f"  Linearization only: {lin_speedup:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="iLQR catheter control demo")
    parser.add_argument("--mode", choices=["reaching", "tracking", "compare"],
                        default="reaching", help="Demo mode")
    parser.add_argument("--method", choices=["implicit", "fd"],
                        default="implicit", help="Linearization method")
    args = parser.parse_args()

    use_implicit = (args.method == "implicit")

    if args.mode == "reaching":
        reaching_demo(use_implicit=use_implicit)
    elif args.mode == "tracking":
        tracking_demo(use_implicit=use_implicit)
    elif args.mode == "compare":
        comparison_experiment()


if __name__ == "__main__":
    main()
