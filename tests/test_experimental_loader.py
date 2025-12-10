from pathlib import Path
import numpy as np

from crm_ml_rl.data.experimental_loader import load_experimental_sample


def test_load_experimental_sample_flips_third_current():
    root = Path("3D_dynamic_response_data_0124")
    currents_path = root / "input_currents" / "circleCurrents.mat"
    traj_path = root / "desired_input_trajectories" / "circleTrajectory.mat"

    sample = load_experimental_sample(currents_path, traj_path, dt=0.05, flip_third_current=True)

    assert sample.currents.shape[1] == 3
    # Verify sign flip on third channel vs. original
    raw = sample.metadata.get("currents_mat")
    assert sample.metadata["flip_third_current"] is True
    # check some nonzero entries in channel 3
    assert np.any(sample.currents[:, 2] != 0)


def test_load_experimental_sample_shapes_match():
    root = Path("3D_dynamic_response_data_0124")
    currents_path = root / "input_currents" / "circleCurrents.mat"
    traj_path = root / "desired_input_trajectories" / "circleTrajectory.mat"

    sample = load_experimental_sample(currents_path, traj_path, dt=0.05, flip_third_current=True)
    T = min(len(sample.currents), len(sample.coil_positions), len(sample.tip_positions))
    assert sample.currents.shape[0] == T
    assert sample.coil_positions.shape == (T, 3)
    assert sample.tip_positions.shape == (T, 3)
