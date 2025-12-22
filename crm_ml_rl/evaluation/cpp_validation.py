"""
Validation harness to compare CRM FK/dynamics predictions against experimental data.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from crm_ml_rl.data.experimental_loader import load_experimental_sample
from crm_ml_rl.evaluation.validation_metrics import summarize_metrics
from crm_ml_rl.wrappers.crm_wrapper import CRMWrapper, CRMSimulator


@dataclass
class ValidationResult:
    fk_metrics: Dict[str, float]
    dyn_metrics: Optional[Dict[str, float]] = None
    metadata: Optional[Dict] = None


def validate_fk_against_experiment(
    currents_mat_path: Path,
    trajectory_path: Path,
    dt: float = 0.05,
    flip_third_current: bool = True,
    insertion_length: float = 94.3,
    use_cpp: bool = True,
    tracked_tip_key: str = "catheterTipPosition",
    param_file: Optional[Path] = None,
    config_file: Optional[Path] = None,
    damping_override: Optional[np.ndarray] = None,
    integration_step: Optional[float] = None,
) -> ValidationResult:
    """
    Run FK using CRMWrapper for each timestep and compare to tracked tip positions.
    """
    sample = load_experimental_sample(
        currents_mat_path,
        trajectory_path,
        dt=dt,
        flip_third_current=flip_third_current,
    )

    crm = CRMWrapper(
        param_file=str(param_file) if param_file else None,
        config_file=str(config_file) if config_file else None,
        use_cpp=use_cpp,
        damping=damping_override,
        flip_third_current=flip_third_current
    )
    if integration_step is not None and crm.use_cpp and crm.initialized:
        crm._cpp_kinematics.integration_step_size = integration_step

    preds = []
    for i in range(len(sample.currents)):
        res = crm.forward_kinematics(sample.currents[i], insertion_length)
        preds.append(res["tip_position"])
    preds = np.asarray(preds)

    target = sample.tip_positions
    fk_metrics = summarize_metrics(preds, target)
    return ValidationResult(
        fk_metrics=fk_metrics,
        metadata={**sample.metadata, "used_cpp": crm.is_using_cpp},
    )


def validate_dynamics_against_experiment(
    currents_mat_path: Path,
    trajectory_path: Path,
    dt: float = 0.05,
    flip_third_current: bool = True,
    insertion_length: float = 94.3,
    use_cpp: bool = True,
    tracked_tip_key: str = "catheterTipPosition",
    param_file: Optional[Path] = None,
    config_file: Optional[Path] = None,
    damping_override: Optional[np.ndarray] = None,
    integration_step: Optional[float] = None,
) -> ValidationResult:
    """
    Run one-step dynamics with CRMSimulator and compare predicted next tip position
    to the tracked data.
    """
    sample = load_experimental_sample(
        currents_mat_path,
        trajectory_path,
        dt=dt,
        flip_third_current=flip_third_current,
    )

    sim = CRMSimulator(
        param_file=str(param_file) if param_file else None,
        config_file=str(config_file) if config_file else None,
        dt=dt,
        use_cpp=use_cpp,
        damping=damping_override,
        flip_third_current=flip_third_current
    )
    if integration_step is not None and sim.wrapper.use_cpp and sim.wrapper.initialized:
        sim.wrapper._cpp_dynamics.integration_step_size = integration_step
    sim.reset(insertion_length=insertion_length)

    preds = []
    for i in range(len(sample.currents)):
        state = sim.wrapper.step_dynamics(sample.currents[i], insertion_length)
        preds.append(state["tip_position"])
    preds = np.asarray(preds)

    dyn_metrics = summarize_metrics(preds, sample.tip_positions)
    return ValidationResult(
        fk_metrics={},
        dyn_metrics=dyn_metrics,
        metadata={**sample.metadata, "used_cpp": sim.is_using_cpp},
    )
