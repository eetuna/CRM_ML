import math

from examples.ml_examples import run_ml_model_examples


def test_ml_examples_smoke():
    """Ensure all ML model examples run and produce finite losses."""
    results = run_ml_model_examples()
    assert isinstance(results, dict)
    assert results, "expected at least one result"

    for name, value in results.items():
        assert math.isfinite(value), f"{name} returned non-finite loss"
        # Synthetic problems should converge quickly; allow moderate tolerance
        assert value < 50.0, f"{name} loss seems too large: {value}"
