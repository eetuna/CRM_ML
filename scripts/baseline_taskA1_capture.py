#!/usr/bin/env python3
"""
Capture baseline tip positions and residual norms for Task A1 refactor.

This script derives 1/2/3-segment parameter files from the 3-segment
CatheterParameterSet_1_dyn.txt while preserving NUM_ACT_SET=1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_param_file(path: Path) -> Dict[str, List[float] | List[str] | int]:
    data: Dict[str, List[float] | List[str] | int] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            key = parts[0]
            vals = parts[1:]
            if key == "CatheterConfig":
                data[key] = vals
            elif key == "NumLocalizationMarkers":
                data[key] = int(vals[0]) if vals else 0
            else:
                data[key] = [float(v) for v in vals]
    return data


def write_param_file(path: Path, data: Dict[str, List[float] | List[str] | int]) -> None:
    def fmt_floats(vals: List[float]) -> str:
        return " ".join(f"{v:.12g}" for v in vals)

    lines: List[str] = []
    config = data["CatheterConfig"]
    lines.append("CatheterConfig " + " ".join(config))
    lines.append(f"NumLocalizationMarkers {data['NumLocalizationMarkers']}")

    for key in (
        "oRlist",
        "iRlist",
        "YoungModlist",
        "ShearModlist",
        "CoilAlignmentAngles",
        "CoilTurnAreaMat",
        "SegmentLengths",
        "ActMass",
        "MarkerLoc",
        "rho",
        "ustarlist",
    ):
        vals = data.get(key, [])
        if isinstance(vals, list) and vals:
            lines.append(f"{key} {fmt_floats(vals)}")
        else:
            lines.append(f"{key}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_segment_mapping(config: List[str]) -> Tuple[List[int | None], List[int | None]]:
    flex_idx = 0
    act_idx = 0
    flex_map: List[int | None] = []
    act_map: List[int | None] = []
    for seg in config:
        if seg == "F":
            flex_map.append(flex_idx)
            act_map.append(None)
            flex_idx += 1
        elif seg == "A":
            flex_map.append(None)
            act_map.append(act_idx)
            act_idx += 1
        else:
            raise ValueError(f"Unsupported segment type: {seg}")
    return flex_map, act_map


def slice_params(
    base: Dict[str, List[float] | List[str] | int],
    segment_indices: List[int],
    segment_length_overrides: Dict[int, float] | None = None,
) -> Dict[str, List[float] | List[str] | int]:
    base_config = base["CatheterConfig"]
    if not isinstance(base_config, list):
        raise ValueError("Invalid base config")

    flex_map, act_map = build_segment_mapping(base_config)

    new_config = [base_config[i] for i in segment_indices]
    new_data: Dict[str, List[float] | List[str] | int] = {}
    new_data["CatheterConfig"] = new_config

    # Segment-level arrays
    seg_lengths = [base["SegmentLengths"][i] for i in segment_indices]  # type: ignore[index]
    if segment_length_overrides:
        for local_idx, value in segment_length_overrides.items():
            if 0 <= local_idx < len(seg_lengths):
                seg_lengths[local_idx] = value
    rho = [base["rho"][i] for i in segment_indices]  # type: ignore[index]

    # Flex-level arrays
    flex_indices = [flex_map[i] for i in segment_indices if flex_map[i] is not None]
    flex_indices = [i for i in flex_indices if i is not None]

    oRlist = [base["oRlist"][i] for i in flex_indices]  # type: ignore[index]
    iRlist = [base["iRlist"][i] for i in flex_indices]  # type: ignore[index]
    YoungModlist = [base["YoungModlist"][i] for i in flex_indices]  # type: ignore[index]
    ShearModlist = [base["ShearModlist"][i] for i in flex_indices]  # type: ignore[index]

    ustar = base["ustarlist"]  # type: ignore[assignment]
    ustarlist: List[float] = []
    for fi in flex_indices:
        ustarlist.extend(ustar[fi * 3 : fi * 3 + 3])

    # Actuator-level arrays (NUM_ACT_SET is 1 in this build)
    act_indices = [act_map[i] for i in segment_indices if act_map[i] is not None]
    act_indices = [i for i in act_indices if i is not None]

    CoilAlignmentAngles = base["CoilAlignmentAngles"]  # type: ignore[assignment]
    CoilTurnAreaMat = base["CoilTurnAreaMat"]  # type: ignore[assignment]
    ActMass = base["ActMass"]  # type: ignore[assignment]

    coil_angles: List[float] = []
    coil_mats: List[float] = []
    act_mass: List[float] = []
    for ai in act_indices:
        coil_angles.extend(CoilAlignmentAngles[ai * 2 : ai * 2 + 2])
        coil_mats.extend(CoilTurnAreaMat[ai * 9 : ai * 9 + 9])
        act_mass.append(ActMass[ai])

    total_len = sum(seg_lengths)
    markers = [m for m in base.get("MarkerLoc", []) if m <= total_len + 1e-9]  # type: ignore[list-item]

    new_data.update(
        {
            "NumLocalizationMarkers": len(markers),
            "oRlist": oRlist,
            "iRlist": iRlist,
            "YoungModlist": YoungModlist,
            "ShearModlist": ShearModlist,
            "CoilAlignmentAngles": coil_angles,
            "CoilTurnAreaMat": coil_mats,
            "SegmentLengths": seg_lengths,
            "ActMass": act_mass,
            "MarkerLoc": markers,
            "rho": rho,
            "ustarlist": ustarlist,
        }
    )

    return new_data


def run_baseline(param_path: Path, config_label: str, insertion_length: float) -> Dict[str, object]:
    from crm_ml_rl.wrappers import crm_python

    dyn = crm_python.CRMDynamics()

    cfg_path = Path("data/catheter_params/CatheterSpatialConfiguration_1.txt")
    ok = dyn.load_parameters(str(param_path), str(cfg_path))
    if not ok:
        raise RuntimeError(f"Failed to load params for {config_label}")

    damping = np.array(
        [
            12.1761626666366,
            12.1761626666366,
            284.429938756989,
            0.0304776127617393,
            0.0304776127617393,
            0.00502712804532508,
        ],
        dtype=np.float64,
    )
    dyn.set_damping(damping)
    dyn.dt = 0.05
    dyn.integration_step_size = 0.01

    currents = np.array([0.0, 0.0, 0.1], dtype=np.float64)

    init_ok = dyn.initialize_from_kinematics(currents, insertion_length)
    if not init_ok:
        raise RuntimeError(f"FK init failed for {config_label}")

    seed = dyn.get_seed_state()
    v = np.asarray(seed["v"], dtype=np.float64)
    w = np.asarray(seed["w"], dtype=np.float64)
    p = np.asarray(seed["p"], dtype=np.float64)
    R = np.asarray(seed["R"], dtype=np.float64)
    xf = np.asarray(seed["xf"], dtype=np.float64)
    mL = np.asarray(seed["mL"], dtype=np.float64)
    nL = np.asarray(seed["nL"], dtype=np.float64)

    out = dyn.linearize_full_seed_action_from_seed_implicit(
        currents,
        insertion_length,
        v,
        w,
        p,
        R,
        xf,
        mL,
        nL,
        1e-5,
        1e-5,
        1e-5,
        1e-5,
    )

    base = out["base"]
    tip_pos = np.asarray(base["tip_position"], dtype=np.float64).tolist()
    residual_norm = float(out["residual_norm"])

    return {
        "config": config_label,
        "param_file": str(param_path),
        "insertion_length": insertion_length,
        "currents": currents.tolist(),
        "tip_position": tip_pos,
        "residual_norm": residual_norm,
        "converged": bool(base["converged"]),
    }


def main() -> None:
    base_param = Path("data/catheter_params/CatheterParameterSet_1_dyn.txt")
    base = parse_param_file(base_param)

    out_dir = Path("data/baselines")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Segment indices from base F A F
    # 1 segment (flex-only): use A F with zero-length actuator to satisfy NUM_ACT_SET=1.
    # 2 segments: A F (proximal flexible)
    # 3 segments: F A F (full)
    segment_sets = {
        "1seg_AF0": [1, 2],
        "2seg_AF": [1, 2],
        "3seg_FAF": [0, 1, 2],
    }

    param_paths: Dict[str, Path] = {}
    for label, seg_idx in segment_sets.items():
        overrides = None
        if label == "1seg_AF0":
            overrides = {0: 0.0}
        derived = slice_params(base, seg_idx, overrides)
        out_path = out_dir / f"CatheterParameterSet_{label}_dyn.txt"
        write_param_file(out_path, derived)
        param_paths[label] = out_path

    baselines = []
    for label, path in param_paths.items():
        seg_lengths = parse_param_file(path)["SegmentLengths"]  # type: ignore[index]
        total_len = float(sum(seg_lengths))
        insertion = min(94.3, total_len)
        baselines.append(run_baseline(path, label, insertion))

    out_json = out_dir / "taskA1_baseline_autodiff_eigen.json"
    out_json.write_text(json.dumps(baselines, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
