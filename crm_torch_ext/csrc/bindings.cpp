// CRM Torch Extension - PyBind11 Bindings
// Option C: Module Registration

#include <torch/extension.h>
#include "crm_step_op.h"

namespace py = pybind11;

/**
 * @brief Custom autograd Function class for CRM step
 *
 * This class integrates crm_step_forward and crm_step_backward
 * into PyTorch's autograd system.
 */
class CRMStepFunction : public torch::autograd::Function<CRMStepFunction> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor currents,
        torch::Tensor insertion_length,
        torch::Tensor seed_v,
        torch::Tensor seed_w,
        torch::Tensor seed_p,
        torch::Tensor seed_R,
        torch::Tensor seed_xf,
        torch::Tensor seed_mL,
        torch::Tensor seed_nL
    ) {
        // Save tensors for backward pass
        ctx->save_for_backward({
            currents, insertion_length,
            seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL
        });

        // Call forward implementation
        return crm_torch::crm_step_forward(
            currents, insertion_length,
            seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL
        );
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        std::cerr << "[BINDINGS] CRMStepFunction::backward() called!" << std::endl;
        std::cerr.flush();

        // Retrieve saved tensors
        auto saved = ctx->get_saved_variables();
        auto currents = saved[0];
        auto insertion_length = saved[1];
        auto seed_v = saved[2];
        auto seed_w = saved[3];
        auto seed_p = saved[4];
        auto seed_R = saved[5];
        auto seed_xf = saved[6];
        auto seed_mL = saved[7];
        auto seed_nL = saved[8];

        std::cerr << "[BINDINGS] About to call crm_torch::crm_step_backward()" << std::endl;
        std::cerr.flush();

        // Call backward implementation
        auto grads = crm_torch::crm_step_backward(
            grad_outputs[0],
            currents, insertion_length,
            seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL
        );

        std::cerr << "[BINDINGS] crm_torch::crm_step_backward() returned" << std::endl;
        std::cerr.flush();

        // Convert to tensor_list (autograd expects this type)
        return torch::autograd::tensor_list(grads);
    }
};

/**
 * @brief Python-facing wrapper for crm_step
 *
 * This is the function exposed to Python users.
 */
torch::Tensor crm_step(
    torch::Tensor currents,
    torch::Tensor insertion_length,
    torch::Tensor seed_v,
    torch::Tensor seed_w,
    torch::Tensor seed_p,
    torch::Tensor seed_R,
    torch::Tensor seed_xf,
    torch::Tensor seed_mL,
    torch::Tensor seed_nL
) {
    return CRMStepFunction::apply(
        currents, insertion_length,
        seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL
    );
}

// PyBind11 module definition
PYBIND11_MODULE(_crm_torch_ext, m) {
    m.doc() = "CRM Torch C++ Extension - End-to-End Differentiable Catheter Dynamics";

    m.def("crm_step", &crm_step,
          "Differentiable catheter dynamics step",
          py::arg("currents"),
          py::arg("insertion_length"),
          py::arg("seed_v"),
          py::arg("seed_w"),
          py::arg("seed_p"),
          py::arg("seed_R"),
          py::arg("seed_xf"),
          py::arg("seed_mL"),
          py::arg("seed_nL"));

    m.def("initialize_params", &crm_torch::initialize_params,
          "Initialize CRM parameters from files",
          py::arg("param_file"),
          py::arg("config_file"));

    m.def("set_timestep", &crm_torch::set_timestep,
          "Set dynamics timestep (seconds)",
          py::arg("dt"));

    m.def("set_integrator", &crm_torch::set_integrator,
          "Set integrator type ('abm4' or 'rk4')",
          py::arg("integrator"));

    m.def("set_integration_step_size", &crm_torch::set_integration_step_size,
          "Set integration step size (mm)",
          py::arg("step_size"));

    m.def("set_damping", &crm_torch::set_damping,
          "Set damping coefficients [vx, vy, vz, wx, wy, wz]",
          py::arg("damping"));
}
