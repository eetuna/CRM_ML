/**
 * crm_torch_binding.cpp
 *
 * PyTorch C++ extension entry point for CRM physics simulation.
 * Provides batched, differentiable dynamics stepping.
 *
 * Phase 2A: Uses Python binding calls (crm_python module).
 */

#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include "torch_utils.hpp"
#include "dynamics_op.hpp"

namespace py = pybind11;

// PyBind11 module definition
PYBIND11_MODULE(_crm_torch_ext, m) {
    m.doc() = "CRM physics C++ extension for PyTorch (Phase 2A)";

    m.def("dynamics_forward",
          &crm_torch::dynamics_forward,
          py::arg("currents"),
          py::arg("insertion_length"),
          py::arg("seed_v"),
          py::arg("seed_w"),
          py::arg("seed_p"),
          py::arg("seed_R"),
          py::arg("seed_xf"),
          py::arg("seed_mL"),
          py::arg("seed_nL"),
          py::arg("param_file"),
          py::arg("config_file"),
          "Batched forward dynamics stepping");

    m.def("dynamics_backward",
          &crm_torch::dynamics_backward,
          py::arg("grad_output"),
          py::arg("grad_next_v"),
          py::arg("grad_next_w"),
          py::arg("grad_next_p"),
          py::arg("grad_next_R"),
          py::arg("grad_next_xf"),
          py::arg("grad_next_mL"),
          py::arg("grad_next_nL"),
          py::arg("currents"),
          py::arg("insertion_length"),
          py::arg("seed_v"),
          py::arg("seed_w"),
          py::arg("seed_p"),
          py::arg("seed_R"),
          py::arg("seed_xf"),
          py::arg("seed_mL"),
          py::arg("seed_nL"),
          py::arg("param_file"),
          py::arg("config_file"),
          py::arg("eps_seed") = 1e-4,
          "Compute gradients via implicit differentiation with multi-step support");
}
