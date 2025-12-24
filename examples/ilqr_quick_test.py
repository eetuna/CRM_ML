#!/usr/bin/env python3
"""Quick iLQR test with minimal horizon for fast verification."""

import sys
import time
import numpy as np
from crm_ml_rl.wrappers import crm_python

# Import the controller
sys.path.insert(0, '/workspaces/catheter/CRM_ML/examples')
from ilqr_catheter_demo import iLQRController, BASE_DAMPING, setup_dyn

def quick_test():
    """Test iLQR with very short horizon (5 steps, 3 iterations)."""
    print("="*60)
    print("QUICK iLQR TEST (5-step horizon, 3 iterations)")
    print("="*60)

    # Setup
    insertion = 94.3
    dyn = setup_dyn(insertion=insertion)

    # Very short horizon for fast testing
    controller = iLQRController(
        dyn,
        insertion,
        horizon=5,          # Only 5 steps
        max_iters=3,        # Only 3 iterations
        use_implicit=True
    )

    # Initial state
    tip_pos = np.array(dyn.get_tip_position(), dtype=np.float64)
    x0 = np.concatenate([tip_pos, np.zeros(3)])

    # Target: small displacement
    target = tip_pos + np.array([10.0, 5.0, 5.0])  # 10mm x, 5mm y, 5mm z

    print(f"\nInitial position: {x0[:3]} mm")
    print(f"Target position:  {target} mm")
    print(f"Distance: {np.linalg.norm(x0[:3] - target):.2f} mm")

    # Solve
    print("\nRunning iLQR...")
    start_time = time.time()

    try:
        states, actions, info = controller.solve(x0, target)
        total_time = time.time() - start_time

        # Results
        final_pos = states[-1, :3]
        final_error = np.linalg.norm(final_pos - target)

        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"Final position: {final_pos} mm")
        print(f"Final error: {final_error:.2f} mm")
        print(f"Total time: {total_time:.2f}s")
        print(f"Converged: {info['converged']}")

        print(f"\nCost history: {[f'{c:.3f}' for c in info['costs']]}")
        print(f"Avg linearization time: {np.mean(info['times_linearize']):.3f}s")
        print(f"Avg backward pass time: {np.mean(info['times_backward']):.3f}s")
        print(f"Avg forward pass time: {np.mean(info['times_forward']):.3f}s")

        # Check if it improved
        improvement = info['costs'][0] - info['costs'][-1]
        print(f"\nCost reduction: {improvement:.3f} ({improvement/info['costs'][0]*100:.1f}%)")

        if improvement > 0:
            print("\n✓ SUCCESS: iLQR reduced cost!")
        else:
            print("\n⚠ Warning: Cost did not decrease")

        return True

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)
