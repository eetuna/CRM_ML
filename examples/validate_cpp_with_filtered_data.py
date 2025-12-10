"""
Validate CRM FK/dynamics against experimental data using only trajectories
with frequency >= 20 Hz to avoid undersampling issues.
"""

from pathlib import Path
import numpy as np

from crm_ml_rl.data.experimental_loader import list_available_trajectories, filter_trajectories_by_frequency
from crm_ml_rl.evaluation.cpp_validation import validate_fk_against_experiment, validate_dynamics_against_experiment
from crm_ml_rl.evaluation.cpp_validation import np
from crm_ml_rl.wrappers.crm_wrapper import CRMWrapper


def run_filtered_validation(min_hz: float = 20.0):
    root = Path("3D_dynamic_response_data_0124")
    currents_mat = root / "input_currents" / "circleCurrents.mat"
    trajs = list_available_trajectories(root)
    # Only use .mat trajectories to avoid malformed txt files
    trajs = [p for p in trajs if p.suffix.lower() == ".mat"]
    trajs = filter_trajectories_by_frequency(trajs, min_hz=min_hz)

    # Tuned parameters from model3 (provided)
    damping = np.array([37.2693880683655, 252.376869843559, 0.0486226271142724, 0.0107902109976624])
    radii = np.array([1.5875, 0.9906])
    moduli = np.array([8.32474685242726, 1.74919735136304])
    coil_align = np.array([4.18570147066614, -2.20459240906065])
    coil_turnarea = np.array([2.06860702321253, 2.31412889108194, 1.62113824345259])
    mass = 4.77949554564165e-06
    seg_lengths = np.array([28.0476186100817, 28.3792193023515, 51.5353707263672])

    initial_state = {
        "v_L": np.array([[1.13615509095482, -0.806335234216135, 0.138736434128927]]),
        "w_L": np.array([[0.0332737247292234, 0.0235211527859674, 0.0643003216871210]]),
        "mL": np.array([[0.275808988716087, -0.00253745343250149, -0.0653461674634346]]),
        "nL": np.array([[-0.000682635425107078, 0.000533834186353920, -0.000132716114391372]]),
        "p_L": np.array([[-6.72122171998065, -31.8779578813764, 54.6141756344580]]),
        "R_L": np.array([[0.981814968794866, 0.0687246602214729, -0.176962556217286,
                          -0.183847391417432, 0.576560449021401, -0.796101794785338,
                          0.0473177688843032, 0.814158954020195, 0.578710366208475]]),
        "xf": np.array([
            -9.73493853631097, -54.1347218358986, 70.6300595577793,
            0.997277561522727, 0.0687198842924879, 0.0267301507906236,
            -0.0178311137896604, 0.576517034107645, -0.816890471931174,
            -0.0715469891430878, 0.814190100385923, 0.576172691472541,
            8.66048121702941e-06, 0.0110879780861997, 5.36920427714806e-09
        ])
    }

    results = []
    for traj in trajs:
        # Infer dt from frequency in filename
        freq = float("".join([c for c in traj.stem if c.isdigit()]) or "0")
        dt = 1.0 / freq if freq > 0 else 0.05

        # Build a wrapper with tuned params and seed state
        wrapper = CRMWrapper(
            use_cpp=True,
            flip_third_current=True,
            damping=damping
        )
        wrapper.set_timestep(0.1)
        # Seed dynamics state (diagnostic)
        wrapper.debug_seed_dynamics(
            v=initial_state["v_L"],
            w=initial_state["w_L"],
            p=initial_state["p_L"],
            R=initial_state["R_L"],
            xf=initial_state["xf"],
            mL=initial_state["mL"],
            nL=initial_state["nL"]
        )

        fk_res = validate_fk_against_experiment(
            currents_mat,
            traj,
            dt=dt,
            flip_third_current=True,
            damping_override=damping,
            integration_step=0.1
        )
        dyn_res = validate_dynamics_against_experiment(
            currents_mat,
            traj,
            dt=dt,
            flip_third_current=True,
            damping_override=damping,
            integration_step=0.1
        )
        results.append({
            "trajectory": traj.name,
            "freq_hz": freq,
            "fk_rmse": fk_res.fk_metrics["rmse"],
            "fk_mae": fk_res.fk_metrics["mae"],
            "fk_used_cpp": fk_res.metadata.get("used_cpp"),
            "dyn_rmse": dyn_res.dyn_metrics["rmse"],
            "dyn_mae": dyn_res.dyn_metrics["mae"],
            "dyn_used_cpp": dyn_res.metadata.get("used_cpp"),
        })
    return results


if __name__ == "__main__":
    out = run_filtered_validation()
    for r in out:
        print(r)
