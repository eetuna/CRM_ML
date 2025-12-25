"""
Export a single metadata bundle describing the current Y±40, r=10 setup:
  - workspace generation settings + cache file
  - DP projection + IK settings (0.1mm)
  - dynamics/FK ramped rollout settings (hold=2 baseline)

Writes:
  - data/output/run_metadata_y40_r10_dp_0p1mm_hold2.json
  - data/output/run_metadata_y40_r10_dp_0p1mm_hold2.txt
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def try_get_pkg_version(name: str) -> str | None:
    try:
        import importlib.metadata as md

        return md.version(name)
    except Exception:
        return None


def main() -> None:
    out_dir = Path("data/output")
    out_dir.mkdir(parents=True, exist_ok=True)

    workspace_npz = out_dir / "workspace_fk_ins94.3_b0.3_step0.01_int0.2.npz"
    ik_npz = out_dir / "fk_ik_test_dp_projection.npz"
    dyn_hold2_npz = out_dir / "dyn_fk_ramped_circles_hold2.npz"
    dyn_latest_npz = out_dir / "dyn_fk_ramped_circles.npz"

    meta: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": {
            "numpy": try_get_pkg_version("numpy"),
            "scipy": try_get_pkg_version("scipy"),
            "matplotlib": try_get_pkg_version("matplotlib"),
        },
        "files": {
            "workspace_npz": str(workspace_npz),
            "ik_npz": str(ik_npz),
            "dyn_hold2_npz": str(dyn_hold2_npz),
            "dyn_latest_npz": str(dyn_latest_npz) if dyn_latest_npz.exists() else None,
        },
        # These reflect the scripts as currently used for this setup.
        "workspace_generation": {
            "script": "scripts/fk_workspace_grid.py",
            "insertion_length_mm": 94.3,
            "current_bound_A": 0.3,
            "current_step_A": 0.01,
            "integration_step_size_mm": 0.2,
            "param_file": "data/catheter_params/CatheterParameterSet_1_dyn.txt",
            "config_file": "data/catheter_params/CatheterSpatialConfiguration_1.txt",
        },
        "dp_projection_and_ik": {
            "script": "scripts/fk_ik_test_projected_workspace_dp.py",
            "circle_centers_xy_mm": [[0.0, 40.0], [0.0, -40.0]],
            "radius_mm": 10.0,
            "points_per_circle": None,
            "start_angle_rad": float(-np.pi / 2.0),
            "projection_method": "dp",
            "k_nearest": None,
            "alpha_u": None,
            "ik_method": "normalized_pinv",
            "step_size": None,
            "threshold_start_mm": None,
            "threshold_traj_mm": None,
            "itrmax": None,
            "insertion_length_mm": 94.3,
            "integration_step_size_mm": 0.2,
            "param_file": "data/catheter_params/CatheterParameterSet_1_dyn.txt",
            "config_file": "data/catheter_params/CatheterSpatialConfiguration_1.txt",
        },
        "dyn_fk_ramped_rollout": {
            "script": "scripts/dyn_fk_compare_ramped_circles.py",
            "dt_s": 0.05,
            "integration_step_size_mm": 0.2,
            "c3_start_A": 0.01,
            "ramp_to_circle_steps": None,
            "bridge_steps": None,
            "nonzero_eps": None,
            "hold_steps_ramp": None,
            "hold_steps_per_point": None,
            "circle_points": None,
            "circle1_currents_csv": str(out_dir / "circle1_currents_y40_r10_dp_0p1mm_n200.csv"),
            "circle2_currents_csv": str(out_dir / "circle2_currents_y-40_r10_dp_0p1mm_n200.csv"),
        },
    }

    # Populate from ik npz
    ik = np.load(ik_npz)
    meta["dp_projection_and_ik"]["points_per_circle"] = int(ik["circle1_des"].shape[0])
    meta["dp_projection_and_ik"]["k_nearest"] = int(ik["k_nearest"]) if "k_nearest" in ik else None
    meta["dp_projection_and_ik"]["alpha_u"] = float(ik["alpha_u"]) if "alpha_u" in ik else None
    # The IK loop params are not stored in the npz; encode what we used in the script for this run.
    meta["dp_projection_and_ik"]["step_size"] = 1e-3
    meta["dp_projection_and_ik"]["threshold_start_mm"] = 0.1
    meta["dp_projection_and_ik"]["threshold_traj_mm"] = 0.1
    meta["dp_projection_and_ik"]["itrmax"] = 500

    # Populate from dyn hold2 npz
    dyn = np.load(dyn_hold2_npz)
    seg = np.asarray(dyn["segments"], dtype=int).reshape(-1).tolist()
    meta["dyn_fk_ramped_rollout"]["segments"] = {
        "order": ["ramp_to_circle1", "circle1", "bridge_to_circle2", "circle2"],
        "lengths": seg,
    }
    meta["dyn_fk_ramped_rollout"]["ramp_to_circle_steps"] = int(seg[0] // 2)  # hold2 -> doubled
    meta["dyn_fk_ramped_rollout"]["bridge_steps"] = int(seg[2])
    meta["dyn_fk_ramped_rollout"]["hold_steps_ramp"] = 2
    meta["dyn_fk_ramped_rollout"]["hold_steps_per_point"] = 2
    meta["dyn_fk_ramped_rollout"]["circle_points"] = int(seg[1] // 2)  # hold2 -> doubled
    meta["dyn_fk_ramped_rollout"]["nonzero_eps"] = 0.005
    meta["dyn_fk_ramped_rollout"]["circle1_start_idx_in_csv"] = int(dyn.get("circle1_start_idx", -1))
    meta["dyn_fk_ramped_rollout"]["circle2_start_idx_in_csv"] = int(dyn.get("circle2_start_idx", -1))

    # Error summary for hold2
    p_fk = np.asarray(dyn["tip_fk"], dtype=np.float64)
    p_dyn = np.asarray(dyn["tip_dyn"], dtype=np.float64)
    conv = np.asarray(dyn["dyn_converged"]).astype(bool)
    e = np.linalg.norm(p_fk - p_dyn, axis=1)
    e_ok = e[conv]
    meta["dyn_fk_ramped_rollout"]["fk_dyn_error_mm"] = {
        "mean_ok": float(e_ok.mean()) if len(e_ok) else None,
        "p95_ok": float(np.percentile(e_ok, 95)) if len(e_ok) else None,
        "max_ok": float(e_ok.max()) if len(e_ok) else None,
        "dyn_fail_count": int((~conv).sum()),
        "n_steps": int(len(e)),
    }

    out_json = out_dir / "run_metadata_y40_r10_dp_0p1mm_hold2.json"
    out_txt = out_dir / "run_metadata_y40_r10_dp_0p1mm_hold2.txt"

    out_json.write_text(json.dumps(meta, indent=2, sort_keys=False))

    lines = []
    lines.append(f"Run metadata: y=±40, r=10, DP projection, IK 0.1mm, hold=2")
    lines.append(f"timestamp_utc: {meta['timestamp_utc']}")
    lines.append("")
    lines.append("Files:")
    for k, v in meta["files"].items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Workspace:")
    for k, v in meta["workspace_generation"].items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("DP projection + IK:")
    for k, v in meta["dp_projection_and_ik"].items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Dyn/FK ramped rollout:")
    for k, v in meta["dyn_fk_ramped_rollout"].items():
        lines.append(f"  {k}: {v}")

    out_txt.write_text("\n".join(lines) + "\n")

    print(f"Saved: {out_json}")
    print(f"Saved: {out_txt}")


if __name__ == "__main__":
    main()

