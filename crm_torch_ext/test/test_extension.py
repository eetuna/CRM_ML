"""
Test suite for CRM Torch C++ Extension
Placeholder tests for Task 1.1 - will be expanded in Phase 3
"""

import pytest
import torch
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_import():
    """Test that extension can be imported (will fail until built)"""
    try:
        import crm_torch_ext
        print(f"Extension imported successfully, version: {crm_torch_ext.__version__}")
    except ImportError as e:
        pytest.skip(f"Extension not built yet: {e}")


def test_crm_step_callable():
    """Test that crm_step function exists and is callable"""
    try:
        from crm_torch_ext import crm_step
        assert callable(crm_step), "crm_step should be callable"
    except ImportError:
        pytest.skip("Extension not built yet")


def test_forward_placeholder():
    """Test forward pass with placeholder implementation"""
    try:
        from crm_torch_ext import crm_step

        # Create dummy inputs (NUM_ACT_SET=1)
        currents = torch.tensor([0.0, 0.0, 0.0])
        insertion_length = torch.tensor([94.3])
        seed_v = torch.zeros(1, 3)
        seed_w = torch.zeros(1, 3)
        seed_p = torch.zeros(1, 3)
        seed_R = torch.eye(3).reshape(1, 9)
        seed_xf = torch.zeros(15)
        seed_mL = torch.zeros(1, 3)
        seed_nL = torch.zeros(1, 3)

        # Call forward (placeholder returns zeros)
        result = crm_step(
            currents, insertion_length,
            seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL
        )

        # Check output shape
        expected_dim = 3 + 3 * 1  # 6 for NUM_ACT_SET=1
        assert result.shape == (expected_dim,), f"Expected shape ({expected_dim},), got {result.shape}"

        print("✓ Forward placeholder test passed")

    except ImportError:
        pytest.skip("Extension not built yet")


if __name__ == "__main__":
    print("CRM Torch Extension Test Suite")
    print("=" * 60)

    test_import()
    test_crm_step_callable()
    test_forward_placeholder()

    print("=" * 60)
    print("Placeholder tests complete")
