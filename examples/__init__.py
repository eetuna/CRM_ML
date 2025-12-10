"""
Example scripts demonstrating how to use CRM ML/RL models.

Run the example modules directly, e.g.:

    python -m examples.ml_examples
    python -m examples.rl_examples
"""

from .ml_examples import run_ml_model_examples
from .rl_examples import run_all_rl_examples

__all__ = [
    'run_ml_model_examples',
    'run_all_rl_examples',
]
