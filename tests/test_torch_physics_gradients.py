import pytest
import torch


def test_fk_backward_produces_gradients():
    try:
        from crm_ml_rl.wrappers.torch_physics import TorchCRMPhysics, HAS_CPP_BINDINGS
    except Exception:
        pytest.skip("torch_physics module unavailable")

    if not HAS_CPP_BINDINGS:
        pytest.skip("C++ bindings not available")

    physics = TorchCRMPhysics(
        param_file="catheterdata/CatheterParameterSet_1_dyn.txt",
        config_file="catheterdata/CatheterSpatialConfiguration_1.txt",
        device="cpu",
    )

    currents = torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float32, requires_grad=True)
    insertion = torch.tensor([50.0], dtype=torch.float32, requires_grad=True)

    tip = physics.fk(currents, insertion)
    loss = tip.sum()
    loss.backward()

    assert currents.grad is not None
    assert torch.isfinite(currents.grad).all()
    assert insertion.grad is not None
    assert torch.isfinite(insertion.grad).all()


def test_dyn_backward_produces_gradients():
    try:
        from crm_ml_rl.wrappers.torch_physics import TorchCRMPhysics, HAS_CPP_BINDINGS
    except Exception:
        pytest.skip("torch_physics module unavailable")

    if not HAS_CPP_BINDINGS:
        pytest.skip("C++ bindings not available")

    physics = TorchCRMPhysics(
        param_file="catheterdata/CatheterParameterSet_1_dyn.txt",
        config_file="catheterdata/CatheterSpatialConfiguration_1.txt",
        device="cpu",
    )

    # Seed C++ dynamics state from FK (nondifferentiable setup step, ok for this test).
    currents0_np = [0.0, 0.0, 0.0]
    physics.dyn.initialize_from_kinematics(currents0_np, 50.0)
    seed = physics.dyn.get_seed_state()

    seed_v = torch.tensor(seed["v"], dtype=torch.float32).unsqueeze(0)
    seed_w = torch.tensor(seed["w"], dtype=torch.float32).unsqueeze(0)
    seed_p = torch.tensor(seed["p"], dtype=torch.float32).unsqueeze(0)
    seed_R = torch.tensor(seed["R"], dtype=torch.float32).unsqueeze(0)
    seed_xf = torch.tensor(seed["xf"], dtype=torch.float32).unsqueeze(0)

    currents = torch.tensor([[0.05, -0.02, 0.01]], dtype=torch.float32, requires_grad=True)
    insertion = torch.tensor([50.0], dtype=torch.float32)

    nxt = physics.dyn_step(currents, insertion, seed_v, seed_w, seed_p, seed_R, seed_xf, eps=1e-4)
    loss = nxt.sum()
    loss.backward()

    assert currents.grad is not None
    assert torch.isfinite(currents.grad).all()
