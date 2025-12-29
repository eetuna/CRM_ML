"""
Centralized initialization utilities for Option C physics.

Provides sign-aware initialization currents that match the target half-plane
to improve gradient accuracy and convergence.
"""

import numpy as np
from typing import Optional, Union

# Standardized magnitude for initialization currents
INIT_CURRENT_MAGNITUDE = 0.01


def get_sign_aware_init_current(
    target_position: Optional[np.ndarray] = None,
    target_currents: Optional[np.ndarray] = None,
    default_sign: float = 0.0
) -> np.ndarray:
    """
    Get initialization currents with sign matching target half-plane.

    Priority (first available determines sign):
    1. If target_position provided: use sign(target_y)
    2. If target_currents provided: use sign(currents[..., 2]) (c3)
    3. Otherwise: use default_sign (0.0 = zero currents, natural rest)

    Args:
        target_position: Target tip position [x, y, z] or batch [N, 3]
        target_currents: Target currents [c1, c2, c3] or trajectory [N, 3]
        default_sign: Default sign if no target info (0.0 = zero currents)
                     Can be +1.0 or -1.0 to bias toward a half-plane

    Returns:
        init_currents: [0, 0, ±0.01] matching target half-plane
                      or [0, 0, 0] if default_sign=0.0

    Examples:
        >>> # Positive target position
        >>> get_sign_aware_init_current(target_position=np.array([0, 5, 90]))
        array([0.  , 0.  , 0.01])

        >>> # Negative target position
        >>> get_sign_aware_init_current(target_position=np.array([0, -5, 90]))
        array([0.  , 0.  , -0.01])

        >>> # Positive currents
        >>> get_sign_aware_init_current(target_currents=np.array([0.1, 0.05, 0.02]))
        array([0.  , 0.  , 0.01])

        >>> # No target info - use natural rest (zero currents)
        >>> get_sign_aware_init_current()
        array([0., 0., 0.])
    """
    sign = default_sign

    # Priority 1: Target position (y component)
    if target_position is not None:
        target_position = np.atleast_2d(target_position)
        y_value = target_position[0, 1]  # First point, y component
        sign = 1.0 if y_value >= 0 else -1.0

    # Priority 2: Target currents (c3 component)
    elif target_currents is not None:
        target_currents = np.atleast_2d(target_currents)
        c3_value = target_currents[0, 2]  # First point, c3 component
        sign = 1.0 if c3_value >= 0 else -1.0

    return np.array([0.0, 0.0, sign * INIT_CURRENT_MAGNITUDE], dtype=np.float64)


def get_init_current_from_c3(c3: float) -> np.ndarray:
    """
    Simple helper: get init current matching c3 sign.

    Args:
        c3: Third current component (determines direction)

    Returns:
        [0, 0, ±0.01] matching c3 sign

    Examples:
        >>> get_init_current_from_c3(0.05)
        array([0.  , 0.  , 0.01])

        >>> get_init_current_from_c3(-0.05)
        array([0.  , 0.  , -0.01])
    """
    sign = 1.0 if c3 >= 0 else -1.0
    return np.array([0.0, 0.0, sign * INIT_CURRENT_MAGNITUDE], dtype=np.float64)


def get_init_current_from_position(position: np.ndarray) -> np.ndarray:
    """
    Simple helper: get init current matching position y-sign.

    Args:
        position: Target position [x, y, z]

    Returns:
        [0, 0, ±0.01] matching sign of y

    Examples:
        >>> get_init_current_from_position(np.array([0, 5, 90]))
        array([0.  , 0.  , 0.01])

        >>> get_init_current_from_position(np.array([0, -5, 90]))
        array([0.  , 0.  , -0.01])
    """
    position = np.atleast_1d(position)
    y_value = position[1] if len(position) >= 2 else 0.0
    sign = 1.0 if y_value >= 0 else -1.0
    return np.array([0.0, 0.0, sign * INIT_CURRENT_MAGNITUDE], dtype=np.float64)
