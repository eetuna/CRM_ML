"""Quick script to inspect NPZ file structure."""
import numpy as np

files = [
    'data/output/dyn_fk_ramp_circle1_hold1.npz',
    'data/output/dyn_fk_ramp_circle1_hold2.npz',
    'data/output/dyn_fk_lem1_y40_a10_hold1.npz',
    'data/output/dyn_fk_lem1_y40_a10_hold2.npz',
]

for fpath in files:
    print(f"\n{'='*70}")
    print(f"File: {fpath}")
    print('='*70)

    try:
        data = np.load(fpath)
        print(f"Keys: {list(data.keys())}")

        for key in data.keys():
            arr = data[key]
            print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")

    except Exception as e:
        print(f"  ERROR: {e}")
