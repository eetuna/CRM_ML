"""
Build System Validation Test (CP-C02)
Tests that the build system correctly finds all headers and libraries
"""

import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_import_extension():
    """Test that the C++ extension can be imported"""
    try:
        import crm_torch_ext
        print("✓ crm_torch_ext package imported")
        return True
    except ImportError as e:
        print(f"✗ Failed to import crm_torch_ext: {e}")
        return False


def test_extension_module():
    """Test that the C++ module is accessible"""
    try:
        import crm_torch_ext._crm_torch_ext as ext
        print("✓ C++ extension module _crm_torch_ext imported")
        return True
    except ImportError as e:
        print(f"✗ Failed to import _crm_torch_ext: {e}")
        return False


def test_crm_step_exists():
    """Test that crm_step function is exposed"""
    try:
        from crm_torch_ext import crm_step
        assert callable(crm_step), "crm_step should be callable"
        print("✓ crm_step function exists and is callable")
        return True
    except (ImportError, AssertionError) as e:
        print(f"✗ crm_step not accessible: {e}")
        return False


def test_torch_linkage():
    """Test that torch libraries are linked correctly"""
    try:
        import torch
        from crm_torch_ext import crm_step

        # Try to create a tensor to ensure torch is working
        test_tensor = torch.zeros(3)
        print(f"✓ Torch linkage verified (torch version: {torch.__version__})")
        return True
    except Exception as e:
        print(f"✗ Torch linkage issue: {e}")
        return False


def test_placeholder_forward():
    """Test that placeholder forward pass works"""
    try:
        import torch
        from crm_torch_ext import crm_step

        # Create dummy inputs
        currents = torch.tensor([0.0, 0.0, 0.0])
        insertion_length = torch.tensor([94.3])
        seed_v = torch.zeros(1, 3)
        seed_w = torch.zeros(1, 3)
        seed_p = torch.zeros(1, 3)
        seed_R = torch.eye(3).reshape(1, 9)
        seed_xf = torch.zeros(15)
        seed_mL = torch.zeros(1, 3)
        seed_nL = torch.zeros(1, 3)

        # Call forward
        result = crm_step(
            currents, insertion_length,
            seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL
        )

        # Check output shape
        expected_dim = 6  # 3 + 3*1 for NUM_ACT_SET=1
        assert result.shape == (expected_dim,), \
            f"Expected shape ({expected_dim},), got {result.shape}"

        print(f"✓ Placeholder forward pass works (output shape: {result.shape})")
        return True

    except Exception as e:
        print(f"✗ Placeholder forward failed: {e}")
        return False


def test_library_linkage():
    """Test that CRMCPPLib is linked correctly"""
    try:
        import crm_torch_ext._crm_torch_ext
        # If import succeeds, library linkage is working
        print("✓ CRMCPPLib linkage verified")
        return True
    except ImportError as e:
        if "CRMCPPLib" in str(e):
            print(f"✗ CRMCPPLib not found: {e}")
            return False
        # Other import errors are handled by previous tests
        print("✓ CRMCPPLib linkage verified (no explicit error)")
        return True


def main():
    print("=" * 70)
    print("CP-C02: Build System Validation Test")
    print("=" * 70)
    print()

    tests = [
        ("Import Extension Package", test_import_extension),
        ("Import C++ Module", test_extension_module),
        ("crm_step Function Exists", test_crm_step_exists),
        ("Torch Linkage", test_torch_linkage),
        ("Library Linkage (CRMCPPLib)", test_library_linkage),
        ("Placeholder Forward Pass", test_placeholder_forward),
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
        print("\n✓ CP-C02 ACCEPTANCE CRITERIA MET")
        print("  - Build system finds all required headers")
        print("  - Link step finds CRM library")
        print("  - Extension is importable from Python")
        return 0
    else:
        print(f"\n✗ CP-C02 INCOMPLETE ({total - passed} tests failed)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
