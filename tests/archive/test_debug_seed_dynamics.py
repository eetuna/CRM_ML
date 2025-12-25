from examples.debug_seed_dynamics import run_debug_seed_step


def test_debug_seed_keeps_cpp_active():
    out = run_debug_seed_step()
    assert out["using_cpp"] is True
    assert out["converged"] is True
    # Tip position should be finite
    tip = out["tip_position"]
    assert tip.shape == (3,)
    assert (tip == tip).all()
