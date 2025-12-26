"""
Test backward pass infrastructure for Option C extension.

This test verifies that the implicit differentiation implementation
produces non-zero gradients for a simple test case.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import crm_torch_ext

# Initialize parameters
param_file = "data/catheter_params/CatheterParameterSet_1_dyn.txt"
config_file = "data/catheter_params/CatheterSpatialConfiguration_1.txt"

crm_torch_ext.initialize_params(param_file, config_file)
crm_torch_ext.set_timestep(0.001)
crm_torch_ext.set_integrator("abm4")
crm_torch_ext.set_integration_step_size(0.001)
crm_torch_ext.set_damping([0.1, 0.1, 0.1, 0.001, 0.001, 0.001])

print("Testing backward pass infrastructure...")

# Create simple test inputs
currents = torch.tensor([0.1, 0.05, 0.0], dtype=torch.float64, requires_grad=True)
insertion_length = torch.tensor([0.05], dtype=torch.float64, requires_grad=True)

# Simple seed state (near equilibrium)
num_sets = 1
seed_v = torch.zeros((num_sets, 3), dtype=torch.float64)
seed_w = torch.zeros((num_sets, 3), dtype=torch.float64)
seed_p = torch.tensor([[0.0, 0.0, 0.05]], dtype=torch.float64)
seed_R = torch.eye(3, dtype=torch.float64).reshape(1, 9)
seed_xf = torch.tensor([0.0, 0.0, 0.05] + [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] + [0.0, 0.0, 0.0], dtype=torch.float64)
seed_mL = torch.zeros((num_sets, 3), dtype=torch.float64)
seed_nL = torch.zeros((num_sets, 3), dtype=torch.float64)

print("Running forward pass...")
try:
    output = crm_torch_ext.crm_step(
        currents, insertion_length,
        seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL
    )
    print(f"Forward pass succeeded. Output shape: {output.shape}")
    print(f"Output: {output}")

    # Compute gradient by backpropagating through a simple loss
    print("\nTesting backward pass...")
    loss = output.sum()  # Simple loss: sum of all outputs
    loss.backward()

    print(f"\nGradient w.r.t. currents: {currents.grad}")
    print(f"Gradient w.r.t. insertion_length: {insertion_length.grad}")

    # Check if gradients are non-zero
    currents_grad_nonzero = torch.any(torch.abs(currents.grad) > 1e-10).item()
    insertion_grad_nonzero = torch.any(torch.abs(insertion_length.grad) > 1e-10).item()

    print(f"\nCurrents gradient non-zero: {currents_grad_nonzero}")
    print(f"Insertion gradient non-zero: {insertion_grad_nonzero}")

    if currents_grad_nonzero or insertion_grad_nonzero:
        print("\n✓ SUCCESS: Gradients are non-zero!")
        print("The implicit differentiation implementation is working.")
    else:
        print("\n✗ WARNING: All gradients are zero.")
        print("This may indicate an issue with the backward pass implementation.")

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
