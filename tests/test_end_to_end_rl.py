
import sys
from pathlib import Path
import pytest

# Add project root to path to allow importing from scripts
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.example_train_rl import quick_training_demo

def test_end_to_end_rl_demo():
    """
    Runs the quick_training_demo from scripts/example_train_rl.py
    as an end-to-end integration test.
    """
    print("Starting end-to-end RL integration test...")
    model = quick_training_demo()
    assert model is not None, "The demo function should return a trained model."
    print("End-to-end RL integration test completed successfully.")

