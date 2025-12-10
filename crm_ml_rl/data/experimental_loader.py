"""
Experimental data loader for catheter trajectories.

Loads tracked trajectories and commanded currents from
`3D_dynamic_response_data_0124/` and prepares them for CRM validation or
ML model training. Includes an optional sign flip on the third current
channel to match the convention used in the historical dataset.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import scipy.io


@dataclass
class ExperimentalSample:
    currents: np.ndarray  # (T, 3)
    coil_positions: np.ndarray  # (T, 3)
    tip_positions: np.ndarray  # (T, 3)
    dt: float
    metadata: Dict


def _load_mat_currents(path: Path) -> np.ndarray:
    """Load expectedCurrents_traj from a .mat file."""
    mat = scipy.io.loadmat(path)
    if "expectedCurrents_traj" not in mat:
        raise ValueError(f"expectedCurrents_traj not found in {path}")
    curr = np.asarray(mat["expectedCurrents_traj"], dtype=float)
    # expectedCurrents_traj typically shape (3, N); transpose if needed
    if curr.shape[0] == 3:
        curr = curr.T
    return curr  # shape (T, 3)


def _load_mat_trajectory(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load coil/tip positions from a .mat trajectory file.
    Supports:
      - coil_position_mat/tip_position_mat (tracked)
      - catheterTipPosition (tip only)
      - circleTrajectory/lemniscateTrajectory (desired input path)
    """
    mat = scipy.io.loadmat(path)
    coil = tip = None

    # Tracked positions
    for coil_key, tip_key in [
        ("coil_position_mat", "tip_position_mat"),
        ("coil_positions", "tip_positions"),
        ("coilPos", "tipPos"),
    ]:
        if coil_key in mat and tip_key in mat:
            coil = np.asarray(mat[coil_key], dtype=float)
            tip = np.asarray(mat[tip_key], dtype=float)
            break

    # Tip-only files
    if tip is None and "catheterTipPosition" in mat:
        tip = np.asarray(mat["catheterTipPosition"], dtype=float)
        coil = tip.copy()

    # Desired trajectory (no measured coil): treat as tip reference
    if tip is None:
        for desired_key in ["circleTrajectory", "lemniscateTrajectory"]:
            if desired_key in mat:
                tip = np.asarray(mat[desired_key], dtype=float)
                coil = tip.copy()
                break

    if tip is None or coil is None:
        raise ValueError(f"Trajectory fields not found in {path}")

    # Ensure shape (T, 3)
    if coil.shape[0] == 3:
        coil = coil.T
    if tip.shape[0] == 3:
        tip = tip.T
    return coil, tip


def _load_txt_trajectory(path: Path, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load trajectory from the tracking .txt format used in output_trajectories.
    Columns: time, tick, <base>, <coil>, <tip> (see matlab/read_input_output.m).
    """
    data = np.loadtxt(path, delimiter=",", skiprows=2)
    # base = cols 3:5, coil = 8:10, tip = 17:19 (1-based in MATLAB)
    coil = (data[:, 8:11] - data[:, 3:6]) * 1000.0  # m -> mm, base frame
    tip = (data[:, 17:20] - data[:, 3:6]) * 1000.0
    # Downsample/interpolate to dt if needed: simple stride here for smoke usage
    if dt is not None and dt > 0:
        t = data[:, 0]
        target_times = np.arange(t[0], t[-1], dt)
        coil = np.vstack([np.interp(target_times, t, coil[:, i]) for i in range(3)]).T
        tip = np.vstack([np.interp(target_times, t, tip[:, i]) for i in range(3)]).T
    return coil, tip


def load_experimental_sample(
    currents_mat_path: Path,
    trajectory_path: Path,
    dt: float,
    flip_third_current: bool = True,
    trajectory_type: str = "circle",
    frequency_hz: Optional[float] = None,
) -> ExperimentalSample:
    """
    Load one experimental dataset (currents + trajectory).

    Args:
        currents_mat_path: path to circleCurrents.mat or lemniscateCurrents.mat
        trajectory_path: path to trajectory .mat or .txt file
        dt: desired timestep (s) for alignment
        flip_third_current: apply sign flip to third current channel (dataset-specific)
        trajectory_type: metadata tag
        frequency_hz: metadata tag
    """
    currents = _load_mat_currents(currents_mat_path)
    if flip_third_current:
        currents[:, 2] *= -1.0

    if trajectory_path.suffix.lower() == ".mat":
        coil, tip = _load_mat_trajectory(trajectory_path)
    else:
        coil, tip = _load_txt_trajectory(trajectory_path, dt)

    # Align lengths: truncate to min length
    T = min(len(currents), len(coil), len(tip))
    currents = currents[:T]
    coil = coil[:T]
    tip = tip[:T]

    return ExperimentalSample(
        currents=currents,
        coil_positions=coil,
        tip_positions=tip,
        dt=dt,
        metadata={
            "trajectory_type": trajectory_type,
            "frequency_hz": frequency_hz,
            "flip_third_current": flip_third_current,
            "currents_mat": str(currents_mat_path),
            "trajectory_path": str(trajectory_path),
        },
    )


def list_available_trajectories(root: Path) -> List[Path]:
    """List trajectory files under output_trajectories."""
    return sorted((root / "output_trajectories").glob("*"))


def filter_trajectories_by_frequency(paths: List[Path], min_hz: float = 20.0) -> List[Path]:
    """
    Filter trajectory paths by inferred frequency in the filename.

    Assumes filenames contain the frequency just before the extension
    (e.g., circle20.mat -> 20 Hz, lemniscate50.mat -> 50 Hz).
    """
    filtered = []
    for p in paths:
        name = p.stem
        digits = "".join([c for c in name[::-1] if c.isdigit()])
        freq = None
        if digits:
            try:
                freq = float(digits[::-1])
            except ValueError:
                freq = None
        if freq is not None and freq >= min_hz:
            filtered.append(p)
    return sorted(filtered)


def load_parameter_mat(path: Path) -> Dict:
    """
    Load MATLAB parameter identification results (model_*.mat).

    Returns a dictionary of parameter arrays if present.
    """
    mat = scipy.io.loadmat(path, squeeze_me=True)
    out = {}
    # nlgr_model from System Identification Toolbox may be opaque; skip if not usable.
    for key in ["damping", "E_", "Coil_align", "Coil_turnarea", "mass_", "SegmentLengths", "nlgr_model"]:
        if key in mat:
            out[key] = mat[key]
    # Some files only contain an opaque nlgr_model; return empty if nothing useful
    return out
