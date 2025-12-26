"""
Forward Pass Test (CP-C03)
Tests that the forward pass correctly converts tensors and returns expected output
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_forward_tensor_conversion():
    """Test that forward pass correctly converts torch tensors"""
    try:
        from crm_torch_ext import crm_step

        # Create test inputs
        currents = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        insertion_length = torch.tensor([94.3], dtype=torch.float64)

        # Seed state (NUM_ACT_SET=1)
        seed_v = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        seed_w = torch.zeros(1, 3, dtype=torch.float64)
        seed_p = torch.zeros(1, 3, dtype=torch.float64)
        seed_R = torch.eye(3, dtype=torch.float64).reshape(1, 9)
        seed_xf = torch.tensor([10.0, 20.0, 30.0] + [0.0] * 12, dtype=torch.float64)
        seed_mL = torch.zeros(1, 3, dtype=torch.float64)
        seed_nL = torch.zeros(1, 3, dtype=torch.float64)

        # Call forward
        result = crm_step(
            currents, insertion_length,
            seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL
        )

        # Check output shape
        expected_dim = 6  # 3 (tip_pos) + 3 (coil_vel for num_sets=1)
        assert result.shape == (expected_dim,), \
            f"Expected shape ({expected_dim},), got {result.shape}"

        # Check output type
        assert result.dtype == torch.float64, \
            f"Expected dtype float64, got {result.dtype}"

        # Check test pattern values (stub returns tip pos from xf, velocities from seed_v)
        tip_pos = result[:3].numpy()
        coil_vel = result[3:6].numpy()

        # Tip position should match first 3 elements of seed_xf
        expected_tip = np.array([10.0, 20.0, 30.0])
        assert np.allclose(tip_pos, expected_tip, atol=1e-10), \
            f"Expected tip_pos {expected_tip}, got {tip_pos}"

        # Coil velocities should match seed_v
        expected_vel = np.array([1.0, 2.0, 3.0])
        assert np.allclose(coil_vel, expected_vel, atol=1e-10), \
            f"Expected coil_vel {expected_vel}, got {coil_vel}"

        print("✓ Forward tensor conversion works correctly")
        print(f"  - Input shapes validated")
        print(f"  - Output shape: {result.shape}")
        print(f"  - Test pattern verified (tip_pos and coil_vel match expected)")
        return True

    except Exception as e:
        print(f"✗ Forward tensor conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_input_validation():
    """Test that forward pass validates input shapes"""
    try:
        from crm_torch_ext import crm_step

        # Test invalid currents shape
        try:
            crm_step(
                torch.zeros(4, dtype=torch.float64),  # Wrong: should be [3]
                torch.tensor([94.3], dtype=torch.float64),
                torch.zeros(1, 3, dtype=torch.float64),
                torch.zeros(1, 3, dtype=torch.float64),
                torch.zeros(1, 3, dtype=torch.float64),
                torch.eye(3, dtype=torch.float64).reshape(1, 9),
                torch.zeros(15, dtype=torch.float64),
                torch.zeros(1, 3, dtype=torch.float64),
                torch.zeros(1, 3, dtype=torch.float64),
            )
            print("✗ Should have raised error for invalid currents shape")
            return False
        except RuntimeError as e:
            if "currents must be shape [3]" in str(e):
                print("✓ Correctly validates currents shape")
            else:
                print(f"✗ Unexpected error: {e}")
                return False

        # Test invalid seed_xf shape
        try:
            crm_step(
                torch.zeros(3, dtype=torch.float64),
                torch.tensor([94.3], dtype=torch.float64),
                torch.zeros(1, 3, dtype=torch.float64),
                torch.zeros(1, 3, dtype=torch.float64),
                torch.zeros(1, 3, dtype=torch.float64),
                torch.eye(3, dtype=torch.float64).reshape(1, 9),
                torch.zeros(10, dtype=torch.float64),  # Wrong: should be [15]
                torch.zeros(1, 3, dtype=torch.float64),
                torch.zeros(1, 3, dtype=torch.float64),
            )
            print("✗ Should have raised error for invalid seed_xf shape")
            return False
        except RuntimeError as e:
            if "seed_xf must be shape [15]" in str(e):
                print("✓ Correctly validates seed_xf shape")
            else:
                print(f"✗ Unexpected error: {e}")
                return False

        return True

    except Exception as e:
        print(f"✗ Input validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_contiguity():
    """Test that forward pass handles non-contiguous tensors"""
    try:
        from crm_torch_ext import crm_step

        # Create non-contiguous tensor
        seed_v_2d = torch.zeros(2, 3, dtype=torch.float64)
        seed_v_non_contig = seed_v_2d[::2, :]  # Strided view

        result = crm_step(
            torch.zeros(3, dtype=torch.float64),
            torch.tensor([94.3], dtype=torch.float64),
            seed_v_non_contig,
            torch.zeros(1, 3, dtype=torch.float64),
            torch.zeros(1, 3, dtype=torch.float64),
            torch.eye(3, dtype=torch.float64).reshape(1, 9),
            torch.zeros(15, dtype=torch.float64),
            torch.zeros(1, 3, dtype=torch.float64),
            torch.zeros(1, 3, dtype=torch.float64),
        )

        assert result.shape == (6,), f"Expected shape (6,), got {result.shape}"
        print("✓ Correctly handles non-contiguous tensors (converts to contiguous)")
        return True

    except Exception as e:
        print(f"✗ Contiguity test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 70)
    print("CP-C03: Forward Pass Test")
    print("=" * 70)
    print()

    tests = [
        ("Tensor Conversion", test_forward_tensor_conversion),
        ("Input Validation", test_forward_input_validation),
        ("Contiguity Handling", test_forward_contiguity),
    ]

    results = []
    for name, test_func in tests:
        print(f"\nTest: {name}")
        print("-" * 70)
        results.append(test_func())
        print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n✓ CP-C03 ACCEPTANCE CRITERIA MET")
        print("  - Forward pass compiles")
        print("  - Output matches expected shape")
        print("  - Test pattern verified (stub implementation)")
        print("  - Input validation works")
        print("  - Tensor conversion correct")
        return 0
    else:
        print(f"\n✗ CP-C03 INCOMPLETE ({total - passed} tests failed)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
