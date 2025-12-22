"""
Export (1) desired circle points, (2) workspace-projected circle points, and (3) FK tip positions
for the ramp/bridge segments used in the dynamics rollout.

Uses existing cached outputs:
  - data/output/fk_ik_test_dp_projection.npz  (circle*_des, circle*_proj)
  - data/output/dyn_fk_ramped_circles_hold2.npz (tip_fk + segments for ramps/bridge)

Outputs CSVs (VS Code friendly) in data/output/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def save_xyz_csv(path: Path, xyz: np.ndarray) -> None:
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    np.savetxt(path, xyz, delimiter=",", header="x,y,z", comments="")


def save_labeled_xyz_csv(path: Path, rows: np.ndarray) -> None:
    """
    rows: Nx5 => [set_id, part_id, x, y, z]
      set_id: 1 for +y circle, 2 for -y circle
      part_id: 0 desired, 1 projected, 2 ramp_to_circle1, 3 bridge_to_circle2
    """
    rows = np.asarray(rows, dtype=np.float64).reshape(-1, 5)
    np.savetxt(path, rows, delimiter=",", header="set_id,part_id,x,y,z", comments="")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ik-npz", default="data/output/fk_ik_test_dp_projection.npz")
    ap.add_argument("--dyn-npz", default="data/output/dyn_fk_ramped_circles_hold2.npz")
    ap.add_argument("--out-dir", default="data/output")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ik = np.load(args.ik_npz)
    c1_des = np.asarray(ik["circle1_des"], dtype=np.float64)
    c2_des = np.asarray(ik["circle2_des"], dtype=np.float64)
    c1_proj = np.asarray(ik["circle1_proj"], dtype=np.float64)
    c2_proj = np.asarray(ik["circle2_proj"], dtype=np.float64)

    # Per-circle exports
    save_xyz_csv(out_dir / "circle1_desired_y40_r10.csv", c1_des)
    save_xyz_csv(out_dir / "circle2_desired_y-40_r10.csv", c2_des)
    save_xyz_csv(out_dir / "circle1_projected_ws_y40_r10.csv", c1_proj)
    save_xyz_csv(out_dir / "circle2_projected_ws_y-40_r10.csv", c2_proj)

    # Combined circle exports
    save_xyz_csv(out_dir / "circle_desired_ypm40_r10.csv", np.vstack([c1_des, c2_des]))
    save_xyz_csv(out_dir / "circle_projected_ws_ypm40_r10.csv", np.vstack([c1_proj, c2_proj]))

    # Ramps/bridge from dynamics timeline (FK tip positions)
    dyn = np.load(args.dyn_npz)
    tip_fk = np.asarray(dyn["tip_fk"], dtype=np.float64)
    seg = np.asarray(dyn["segments"], dtype=int).reshape(-1)
    if len(seg) != 4:
        raise RuntimeError(f"Expected 4 segments in dyn npz (ramp,circle1,bridge,circle2); got {len(seg)}")
    n_ramp, n_circle1, n_bridge, n_circle2 = map(int, seg.tolist())
    s0 = 0
    s1 = s0 + n_ramp
    s2 = s1 + n_circle1
    s3 = s2 + n_bridge
    s4 = s3 + n_circle2
    ramp_fk = tip_fk[s0:s1]
    bridge_fk = tip_fk[s2:s3]
    save_xyz_csv(out_dir / "ramp_to_circle1_fk_positions_hold2.csv", ramp_fk)
    save_xyz_csv(out_dir / "bridge_to_circle2_fk_positions_hold2.csv", bridge_fk)
    save_xyz_csv(out_dir / "ramps_fk_positions_hold2.csv", np.vstack([ramp_fk, bridge_fk]))

    # One combined labeled export for convenience
    rows = []
    # desired/projected circles
    rows.append(np.column_stack([np.ones(len(c1_des)), np.zeros(len(c1_des)), c1_des]))
    rows.append(np.column_stack([2 * np.ones(len(c2_des)), np.zeros(len(c2_des)), c2_des]))
    rows.append(np.column_stack([np.ones(len(c1_proj)), np.ones(len(c1_proj)), c1_proj]))
    rows.append(np.column_stack([2 * np.ones(len(c2_proj)), np.ones(len(c2_proj)), c2_proj]))
    # ramp/bridge (not specific to a circle set, label set_id=0)
    rows.append(np.column_stack([np.zeros(len(ramp_fk)), 2 * np.ones(len(ramp_fk)), ramp_fk]))
    rows.append(np.column_stack([np.zeros(len(bridge_fk)), 3 * np.ones(len(bridge_fk)), bridge_fk]))
    save_labeled_xyz_csv(out_dir / "circle_des_proj_and_ramps_hold2_labeled.csv", np.vstack(rows))

    print("Saved desired circles:")
    print(f"  {out_dir/'circle1_desired_y40_r10.csv'}")
    print(f"  {out_dir/'circle2_desired_y-40_r10.csv'}")
    print(f"  {out_dir/'circle_desired_ypm40_r10.csv'}")
    print("Saved projected circles:")
    print(f"  {out_dir/'circle1_projected_ws_y40_r10.csv'}")
    print(f"  {out_dir/'circle2_projected_ws_y-40_r10.csv'}")
    print(f"  {out_dir/'circle_projected_ws_ypm40_r10.csv'}")
    print("Saved ramp/bridge FK positions (hold2 timeline):")
    print(f"  {out_dir/'ramp_to_circle1_fk_positions_hold2.csv'}")
    print(f"  {out_dir/'bridge_to_circle2_fk_positions_hold2.csv'}")
    print(f"  {out_dir/'ramps_fk_positions_hold2.csv'}")
    print("Saved combined labeled export:")
    print(f"  {out_dir/'circle_des_proj_and_ramps_hold2_labeled.csv'}")


if __name__ == "__main__":
    main()

