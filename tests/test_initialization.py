"""Tests for sign-aware initialization logic."""

import numpy as np
import pytest
from crm_ml_rl.wrappers.initialization_utils import (
    get_sign_aware_init_current,
    get_init_current_from_c3,
    get_init_current_from_position,
    INIT_CURRENT_MAGNITUDE
)


class TestSignAwareInitialization:
    """Test sign-aware initialization functions."""

    def test_positive_target_y(self):
        """Target with y > 0 should use positive init."""
        target = np.array([1.0, 5.0, 90.0])  # y = 5 > 0
        init = get_sign_aware_init_current(target_position=target)
        assert init[2] == INIT_CURRENT_MAGNITUDE  # +0.01
        assert init[0] == 0.0
        assert init[1] == 0.0

    def test_negative_target_y(self):
        """Target with y < 0 should use negative init."""
        target = np.array([1.0, -5.0, 90.0])  # y = -5 < 0
        init = get_sign_aware_init_current(target_position=target)
        assert init[2] == -INIT_CURRENT_MAGNITUDE  # -0.01
        assert init[0] == 0.0
        assert init[1] == 0.0

    def test_zero_target_y(self):
        """Target with y = 0 should use positive init (>= 0 rule)."""
        target = np.array([1.0, 0.0, 90.0])  # y = 0
        init = get_sign_aware_init_current(target_position=target)
        assert init[2] == INIT_CURRENT_MAGNITUDE  # +0.01

    def test_positive_c3_current(self):
        """Positive c3 should use positive init."""
        currents = np.array([0.1, 0.05, 0.02])  # c3 = 0.02 > 0
        init = get_sign_aware_init_current(target_currents=currents)
        assert init[2] == INIT_CURRENT_MAGNITUDE

    def test_negative_c3_current(self):
        """Negative c3 should use negative init."""
        currents = np.array([0.1, 0.05, -0.02])  # c3 = -0.02 < 0
        init = get_sign_aware_init_current(target_currents=currents)
        assert init[2] == -INIT_CURRENT_MAGNITUDE

    def test_target_position_priority_over_currents(self):
        """Target position should take priority over currents."""
        target = np.array([0, -5.0, 90.0])  # y < 0 → negative
        currents = np.array([0, 0, 0.02])   # c3 > 0 → would be positive
        init = get_sign_aware_init_current(
            target_position=target,
            target_currents=currents
        )
        assert init[2] == -INIT_CURRENT_MAGNITUDE  # Position wins

    def test_default_zero_no_target(self):
        """No info should use zero currents (natural rest)."""
        init = get_sign_aware_init_current()
        assert init[2] == 0.0
        assert np.allclose(init, [0, 0, 0])

    def test_default_positive(self):
        """Can specify positive default."""
        init = get_sign_aware_init_current(default_sign=1.0)
        assert init[2] == INIT_CURRENT_MAGNITUDE

    def test_default_negative(self):
        """Can specify negative default."""
        init = get_sign_aware_init_current(default_sign=-1.0)
        assert init[2] == -INIT_CURRENT_MAGNITUDE

    def test_standardized_magnitude(self):
        """All cases should use standardized 0.01 magnitude."""
        cases = [
            get_sign_aware_init_current(target_position=np.array([0, 5, 90])),
            get_sign_aware_init_current(target_position=np.array([0, -5, 90])),
            get_sign_aware_init_current(target_currents=np.array([0, 0, 0.1])),
            get_sign_aware_init_current(target_currents=np.array([0, 0, -0.1])),
        ]
        for init in cases:
            assert np.abs(init[2]) == INIT_CURRENT_MAGNITUDE
            assert init[0] == 0.0
            assert init[1] == 0.0

    def test_trajectory_uses_first_point(self):
        """Trajectory should use first point for sign detection."""
        trajectory = np.array([
            [0, 0, 0.02],   # First c3 > 0
            [0, 0, -0.05],  # Later c3 < 0 (ignored)
        ])
        init = get_sign_aware_init_current(target_currents=trajectory)
        assert init[2] == INIT_CURRENT_MAGNITUDE  # Uses first point

    def test_batch_position_uses_first(self):
        """Batch positions should use first for sign detection."""
        positions = np.array([
            [0, -5, 90],  # First y < 0
            [0, 10, 90],  # Later y > 0 (ignored)
        ])
        init = get_sign_aware_init_current(target_position=positions)
        assert init[2] == -INIT_CURRENT_MAGNITUDE  # Uses first point


class TestHelperFunctions:
    """Test helper functions."""

    def test_get_init_current_from_c3_positive(self):
        """Test c3 helper with positive value."""
        init = get_init_current_from_c3(0.05)
        assert init[2] == INIT_CURRENT_MAGNITUDE
        assert np.allclose(init, [0, 0, 0.01])

    def test_get_init_current_from_c3_negative(self):
        """Test c3 helper with negative value."""
        init = get_init_current_from_c3(-0.05)
        assert init[2] == -INIT_CURRENT_MAGNITUDE
        assert np.allclose(init, [0, 0, -0.01])

    def test_get_init_current_from_c3_zero(self):
        """Test c3 helper with zero (should be positive due to >= 0)."""
        init = get_init_current_from_c3(0.0)
        assert init[2] == INIT_CURRENT_MAGNITUDE

    def test_get_init_current_from_position_positive(self):
        """Test position helper with positive y."""
        init = get_init_current_from_position(np.array([1, 5, 90]))
        assert init[2] == INIT_CURRENT_MAGNITUDE
        assert np.allclose(init, [0, 0, 0.01])

    def test_get_init_current_from_position_negative(self):
        """Test position helper with negative y."""
        init = get_init_current_from_position(np.array([1, -5, 90]))
        assert init[2] == -INIT_CURRENT_MAGNITUDE
        assert np.allclose(init, [0, 0, -0.01])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
