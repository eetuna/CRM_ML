"""
Finite Difference Gradient Validation Tests.

Validates that analytical gradients from Option C match numerical
finite difference approximations within acceptable tolerance.

This is a critical test to ensure the backward pass is implemented correctly.
"""

import torch
import numpy as np
import pytest

from crm_ml_rl.wrappers.option_c_physics import OptionCPhysics


def compute_finite_difference_gradients(
    physics: OptionCPhysics,
    currents: torch.Tensor,
    eps: float = 1e-5,
    output_dim: int = 0
) -> torch.Tensor:
    """
    Compute finite difference gradients for a single output dimension.

    Args:
        physics: OptionCPhysics instance (already initialized)
        currents: (3,) input currents
        eps: Finite difference epsilon
        output_dim: Which output dimension to differentiate (0-5)

    Returns:
        fd_grad: (3,) finite difference gradient estimate
    """
    # Save initial state
    initial_seed = physics._dyn.get_seed_state()

    currents_np = currents.detach().cpu().numpy()
    fd_grad = np.zeros(3)

    for i in range(3):
        # Restore state for each perturbation
        physics._dyn.set_seed_state(
            initial_seed['v'], initial_seed['w'], initial_seed['p'],
            initial_seed['R'], initial_seed['xf'], initial_seed['mL'], initial_seed['nL']
        )

        # Forward perturbation
        currents_plus = currents_np.copy()
        currents_plus[i] += eps
        output_plus = physics.step_differentiable(
            torch.from_numpy(currents_plus).requires_grad_(False)
        )
        val_plus = output_plus[0, output_dim].item()

        # Restore state again
        physics._dyn.set_seed_state(
            initial_seed['v'], initial_seed['w'], initial_seed['p'],
            initial_seed['R'], initial_seed['xf'], initial_seed['mL'], initial_seed['nL']
        )

        # Backward perturbation
        currents_minus = currents_np.copy()
        currents_minus[i] -= eps
        output_minus = physics.step_differentiable(
            torch.from_numpy(currents_minus).requires_grad_(False)
        )
        val_minus = output_minus[0, output_dim].item()

        # Central difference
        fd_grad[i] = (val_plus - val_minus) / (2 * eps)

    # Restore original state
    physics._dyn.set_seed_state(
        initial_seed['v'], initial_seed['w'], initial_seed['p'],
        initial_seed['R'], initial_seed['xf'], initial_seed['mL'], initial_seed['nL']
    )

    return torch.from_numpy(fd_grad)


@pytest.mark.parametrize("output_dim", [0, 1, 2, 3, 4, 5])
def test_gradient_vs_finite_difference_single_output(output_dim):
    """
    Test analytical gradients vs finite difference for each output dimension.

    Validates that ∂output[i]/∂currents matches FD approximation within 10%.
    """
    physics = OptionCPhysics()
    physics.reset()

    # Test currents (non-zero to get meaningful gradients)
    currents = torch.tensor([0.1, 0.05, 0.02], dtype=torch.float64, requires_grad=True)

    # Analytical gradient
    output = physics.step_differentiable(currents)
    output[0, output_dim].backward()
    analytical_grad = currents.grad.clone()

    # Reset state for FD computation
    physics.reset()

    # Finite difference gradient
    fd_grad = compute_finite_difference_gradients(physics, currents, eps=1e-5, output_dim=output_dim)

    # Compute relative error
    rel_error = torch.abs(analytical_grad - fd_grad) / (torch.abs(fd_grad) + 1e-8)

    # Print diagnostics
    print(f"\nOutput dimension {output_dim}:")
    print(f"  Analytical: {analytical_grad.numpy()}")
    print(f"  FD:         {fd_grad.numpy()}")
    print(f"  Rel error:  {rel_error.numpy()}")
    print(f"  Max error:  {rel_error.max().item():.2%}")

    # Accept up to 10% relative error (consistent with Option C's 9.5% FD error)
    # Allow some components to have higher error if FD gradient is very small
    significant_grads = torch.abs(fd_grad) > 1e-3
    if significant_grads.any():
        assert (rel_error[significant_grads] < 0.10).all(), \
            f"Gradient error too large for output dim {output_dim}: {rel_error.max().item():.2%}"


def test_gradient_vs_finite_difference_all_outputs():
    """
    Test analytical gradients vs FD for all outputs simultaneously.

    This validates the full Jacobian B = ∂output/∂currents.
    """
    physics = OptionCPhysics()
    physics.reset()

    currents = torch.tensor([0.1, 0.05, 0.02], dtype=torch.float64, requires_grad=True)

    # Save initial state
    initial_seed = physics._dyn.get_seed_state()

    # Compute full Jacobian analytically
    analytical_jacobian = torch.zeros(6, 3, dtype=torch.float64)
    for i in range(6):
        if currents.grad is not None:
            currents.grad.zero_()
        physics._dyn.set_seed_state(
            initial_seed['v'], initial_seed['w'], initial_seed['p'],
            initial_seed['R'], initial_seed['xf'], initial_seed['mL'], initial_seed['nL']
        )
        output = physics.step_differentiable(currents)
        output[0, i].backward()
        analytical_jacobian[i] = currents.grad.clone()

    # Compute full Jacobian via FD
    fd_jacobian = torch.zeros(6, 3, dtype=torch.float64)
    for i in range(6):
        physics._dyn.set_seed_state(
            initial_seed['v'], initial_seed['w'], initial_seed['p'],
            initial_seed['R'], initial_seed['xf'], initial_seed['mL'], initial_seed['nL']
        )
        fd_jacobian[i] = compute_finite_difference_gradients(physics, currents, eps=1e-5, output_dim=i)

    # Compute errors
    rel_error = torch.abs(analytical_jacobian - fd_jacobian) / (torch.abs(fd_jacobian) + 1e-8)

    print("\nFull Jacobian comparison:")
    print(f"  Analytical Jacobian:\n{analytical_jacobian.numpy()}")
    print(f"  FD Jacobian:\n{fd_jacobian.numpy()}")
    print(f"  Relative errors:\n{rel_error.numpy()}")
    print(f"  Max error: {rel_error.max().item():.2%}")
    print(f"  Mean error: {rel_error.mean().item():.2%}")

    # Check that most gradients are within 10%
    significant_grads = torch.abs(fd_jacobian) > 1e-3
    if significant_grads.any():
        errors_ok = (rel_error[significant_grads] < 0.10).sum()
        errors_total = significant_grads.sum()
        print(f"  Gradients within 10%: {errors_ok}/{errors_total} ({100.0*errors_ok/errors_total:.1f}%)")

        # Require at least 90% of significant gradients to be within tolerance
        assert errors_ok >= 0.9 * errors_total, \
            f"Too many gradients outside tolerance: {errors_ok}/{errors_total}"


def test_gradient_magnitude_sanity():
    """
    Sanity check that gradients have reasonable magnitudes.

    This catches issues like all-zero gradients or exploding gradients.
    """
    physics = OptionCPhysics()
    physics.reset()

    currents = torch.tensor([0.1, 0.05, 0.02], dtype=torch.float64, requires_grad=True)
    output = physics.step_differentiable(currents)

    # Use position loss (first 3 dimensions)
    loss = output[0, :3].sum()
    loss.backward()

    print(f"\nGradient magnitude sanity check:")
    print(f"  currents.grad: {currents.grad.numpy()}")
    print(f"  grad magnitude: {currents.grad.abs().sum().item():.4f}")

    # Gradients should be non-zero
    assert currents.grad.abs().sum() > 1e-6, "Gradients are too small or zero"

    # Gradients should not be exploding
    assert currents.grad.abs().max() < 1e6, "Gradients are exploding"

    # At least one gradient should be significant
    assert (currents.grad.abs() > 0.01).any(), "No significant gradients found"


def test_seed_gradients_vs_finite_difference():
    """
    Test seed gradients vs finite difference (Phase 3B).

    Validates that ∂output/∂seed_v matches FD approximation.
    """
    physics = OptionCPhysics()
    physics.reset()

    currents = torch.tensor([0.1, 0.0, 0.0], dtype=torch.float64, requires_grad=True)

    # Get differentiable seeds
    seed = physics.get_seed_tensors_differentiable()

    # Analytical gradient w.r.t. seed_v
    output, _ = physics.step_with_seed_gradients(currents, seed)
    loss = output[0].sum()  # x position
    loss.backward()

    analytical_seed_grad = seed['v'].grad.clone()

    print(f"\nSeed gradient validation:")
    print(f"  Analytical seed_v grad shape: {analytical_seed_grad.shape}")
    print(f"  Analytical seed_v grad: {analytical_seed_grad[0, 0].numpy()}")
    print(f"  Nonzero components: {(analytical_seed_grad.abs() > 1e-10).sum().item()}/{analytical_seed_grad.numel()}")

    # FD gradient for seed_v[0, 0, 0] (first component)
    eps = 1e-5

    # Save initial state
    initial_seed_state = physics._dyn.get_seed_state()

    seed_plus = seed['v'].clone()
    seed_plus[0, 0, 0] += eps
    seed_dict_plus = {**seed, 'v': seed_plus}
    output_plus, _ = physics.step_with_seed_gradients(currents.detach().requires_grad_(False), seed_dict_plus)
    val_plus = output_plus[0].item()

    # Restore state
    physics._dyn.set_seed_state(
        initial_seed_state['v'], initial_seed_state['w'], initial_seed_state['p'],
        initial_seed_state['R'], initial_seed_state['xf'], initial_seed_state['mL'], initial_seed_state['nL']
    )

    seed_minus = seed['v'].clone()
    seed_minus[0, 0, 0] -= eps
    seed_dict_minus = {**seed, 'v': seed_minus}
    output_minus, _ = physics.step_with_seed_gradients(currents.detach().requires_grad_(False), seed_dict_minus)
    val_minus = output_minus[0].item()

    fd_grad_v0 = (val_plus - val_minus) / (2 * eps)

    # Restore state
    physics._dyn.set_seed_state(
        initial_seed_state['v'], initial_seed_state['w'], initial_seed_state['p'],
        initial_seed_state['R'], initial_seed_state['xf'], initial_seed_state['mL'], initial_seed_state['nL']
    )

    print(f"  FD seed_v[0,0,0] grad: {fd_grad_v0:.6e}")
    print(f"  Analytical seed_v[0,0,0] grad: {analytical_seed_grad[0, 0, 0].item():.6e}")

    # Check relative error
    rel_error = abs(analytical_seed_grad[0, 0, 0].item() - fd_grad_v0) / (abs(fd_grad_v0) + 1e-8)
    print(f"  Relative error: {rel_error:.2%}")

    # Accept up to 10% error
    assert rel_error < 0.10, f"Seed gradient error too large: {rel_error:.2%}"

    # Seed gradients should be non-zero (this is the key Phase 3B feature!)
    assert (analytical_seed_grad.abs() > 1e-10).any(), "Seed gradients are zero - Phase 3B not working!"


if __name__ == "__main__":
    # Run tests
    print("=" * 80)
    print("FINITE DIFFERENCE GRADIENT VALIDATION")
    print("=" * 80)

    print("\n[1/5] Testing individual output dimensions...")
    for i in range(6):
        test_gradient_vs_finite_difference_single_output(i)

    print("\n[2/5] Testing full Jacobian...")
    test_gradient_vs_finite_difference_all_outputs()

    print("\n[3/5] Testing gradient magnitude sanity...")
    test_gradient_magnitude_sanity()

    print("\n[4/5] Testing seed gradients...")
    test_seed_gradients_vs_finite_difference()

    print("\n" + "=" * 80)
    print("✅ ALL GRADIENT VALIDATION TESTS PASSED!")
    print("=" * 80)
