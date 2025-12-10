"""
Validation utilities for comparing CRM predictions to experimental data.
"""

from typing import Dict
import numpy as np


def position_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute RMSE between two position trajectories (T, 3)."""
    pred = np.asarray(pred)
    target = np.asarray(target)
    T = min(len(pred), len(target))
    err = pred[:T] - target[:T]
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))


def position_mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute MAE between two position trajectories (T, 3)."""
    pred = np.asarray(pred)
    target = np.asarray(target)
    T = min(len(pred), len(target))
    err = pred[:T] - target[:T]
    return float(np.mean(np.linalg.norm(err, axis=1)))


def summarize_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Return a dict of basic metrics."""
    return {
        "rmse": position_rmse(pred, target),
        "mae": position_mae(pred, target),
    }
