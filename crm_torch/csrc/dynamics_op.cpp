/**
 * dynamics_op.cpp
 *
 * Phase 2A Implementation: Forward dynamics using Python binding calls.
 *
 * This implementation calls the existing crm_python.CRMDynamics.step_from_seed()
 * for each batch element. While this doesn't achieve full parallelization due to
 * Python's GIL, it provides a working baseline for correctness validation.
 */

#include "dynamics_op.hpp"
#include "torch_utils.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>

namespace py = pybind11;

namespace crm_torch {

torch::Tensor dynamics_forward(
    torch::Tensor currents,
    torch::Tensor insertion_length,
    torch::Tensor seed_v,
    torch::Tensor seed_w,
    torch::Tensor seed_p,
    torch::Tensor seed_R,
    torch::Tensor seed_xf,
    torch::Tensor seed_mL,
    torch::Tensor seed_nL,
    const std::string& param_file,
    const std::string& config_file
) {
    // Validate all inputs are on CPU and float64
    validate_tensor(currents, "currents");
    validate_tensor(insertion_length, "insertion_length");
    validate_tensor(seed_v, "seed_v");
    validate_tensor(seed_w, "seed_w");
    validate_tensor(seed_p, "seed_p");
    validate_tensor(seed_R, "seed_R");
    validate_tensor(seed_xf, "seed_xf");
    validate_tensor(seed_mL, "seed_mL");
    validate_tensor(seed_nL, "seed_nL");

    int64_t batch_size = get_batch_size(currents);

    // Validate shapes
    if (currents.size(1) != 3) {
        throw std::runtime_error("currents must have shape (B, 3)");
    }
    if (seed_v.dim() != 3 || seed_v.size(0) != batch_size || seed_v.size(2) != 3) {
        throw std::runtime_error("seed_v must have shape (B, num_sets, 3)");
    }

    int64_t num_sets = seed_v.size(1);
    int64_t output_dim = 3 + 3 * num_sets;  // tip_pos + velocities

    // Allocate output tensor
    auto next_state = allocate_output(batch_size, output_dim);

    // Get raw pointers for batch processing
    double* output_ptr = next_state.data_ptr<double>();

    // Import crm_python module (acquires GIL)
    py::gil_scoped_acquire acquire;
    py::module_ crm_python = py::module_::import("crm_ml_rl.wrappers.crm_python");
    py::object CRMDynamics = crm_python.attr("CRMDynamics");

    // Create single dynamics instance (reused for all batch elements)
    py::object dyn = CRMDynamics();
    bool loaded = dyn.attr("load_parameters")(param_file, config_file).cast<bool>();
    if (!loaded) {
        throw std::runtime_error("Failed to load CRM parameters from " + param_file);
    }

    // Process each batch element sequentially (GIL prevents true parallelization)
    // Note: OpenMP would not help here due to Python GIL, but we keep the structure
    // for future C++ implementation (Option 2B)
    for (int64_t i = 0; i < batch_size; ++i) {
        try {
            // Extract per-sample tensors as numpy arrays
            double curr_i[3];
            for (int j = 0; j < 3; ++j) {
                curr_i[j] = currents[i][j].item<double>();
            }

            double ins_i = get_scalar(insertion_length, i);

            // Create numpy arrays for seed state
            py::array_t<double> v_np(std::vector<py::ssize_t>{num_sets, 3});
            py::array_t<double> w_np(std::vector<py::ssize_t>{num_sets, 3});
            py::array_t<double> p_np(std::vector<py::ssize_t>{num_sets, 3});
            py::array_t<double> R_np(std::vector<py::ssize_t>{num_sets, 9});
            py::array_t<double> xf_np(std::vector<py::ssize_t>{15});
            py::array_t<double> mL_np(std::vector<py::ssize_t>{num_sets, 3});
            py::array_t<double> nL_np(std::vector<py::ssize_t>{num_sets, 3});

            auto v_buf = v_np.mutable_unchecked<2>();
            auto w_buf = w_np.mutable_unchecked<2>();
            auto p_buf = p_np.mutable_unchecked<2>();
            auto R_buf = R_np.mutable_unchecked<2>();
            auto xf_buf = xf_np.mutable_unchecked<1>();
            auto mL_buf = mL_np.mutable_unchecked<2>();
            auto nL_buf = nL_np.mutable_unchecked<2>();

            // Copy seed data from torch tensors to numpy arrays
            for (int64_t j = 0; j < num_sets; ++j) {
                for (int64_t k = 0; k < 3; ++k) {
                    v_buf(j, k) = seed_v[i][j][k].item<double>();
                    w_buf(j, k) = seed_w[i][j][k].item<double>();
                    p_buf(j, k) = seed_p[i][j][k].item<double>();
                    mL_buf(j, k) = seed_mL[i][j][k].item<double>();
                    nL_buf(j, k) = seed_nL[i][j][k].item<double>();
                }
                for (int64_t k = 0; k < 9; ++k) {
                    R_buf(j, k) = seed_R[i][j][k].item<double>();
                }
            }

            for (int64_t k = 0; k < 15; ++k) {
                xf_buf(k) = seed_xf[i][k].item<double>();
            }

            // Create currents numpy array
            py::array_t<double> curr_np(std::vector<py::ssize_t>{3});
            auto curr_buf = curr_np.mutable_unchecked<1>();
            for (int j = 0; j < 3; ++j) {
                curr_buf(j) = curr_i[j];
            }

            // Call step_from_seed
            py::dict result = dyn.attr("step_from_seed")(
                curr_np, ins_i, v_np, w_np, p_np, R_np, xf_np,
                mL_np, nL_np, py::none()  // dt=None (use default)
            ).cast<py::dict>();

            // Extract output: tip_position (3) + tip_velocity (3) for each actuator
            py::array_t<double> tip_pos = result["tip_position"].cast<py::array_t<double>>();
            py::array_t<double> tip_vel = result["tip_velocity"].cast<py::array_t<double>>();

            auto pos_buf = tip_pos.unchecked<1>();
            auto vel_buf = tip_vel.unchecked<1>();

            // Pack output: [tip_pos(3), vel_0(3), ...]
            int64_t out_idx = i * output_dim;
            for (int64_t k = 0; k < 3; ++k) {
                output_ptr[out_idx + k] = pos_buf(k);
            }

            // For now, only extract velocity for actuator 0
            // TODO: Extract velocities for all actuators when multi-actuator support is needed
            for (int64_t k = 0; k < 3; ++k) {
                output_ptr[out_idx + 3 + k] = vel_buf(k);
            }

            // Fill remaining velocity slots with zeros if num_sets > 1
            for (int64_t j = 1; j < num_sets; ++j) {
                for (int64_t k = 0; k < 3; ++k) {
                    output_ptr[out_idx + 3 + j * 3 + k] = 0.0;
                }
            }

        } catch (const std::exception& e) {
            throw std::runtime_error("Batch element " + std::to_string(i) +
                                   " failed: " + std::string(e.what()));
        }
    }

    return next_state;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dynamics_backward(
    torch::Tensor grad_output,
    torch::Tensor currents,
    torch::Tensor insertion_length,
    torch::Tensor seed_v,
    torch::Tensor seed_w,
    torch::Tensor seed_p,
    torch::Tensor seed_R,
    torch::Tensor seed_xf,
    torch::Tensor seed_mL,
    torch::Tensor seed_nL,
    const std::string& param_file,
    const std::string& config_file,
    double eps_seed
) {
    // Phase 3A: Compute gradients using Option A's implicit linearization
    // Currently implements: grad_currents (∂Loss/∂currents)
    // Seed gradients: Set to zero for MVP (can be added in Phase 3B)

    // Validate inputs
    validate_tensor(grad_output, "grad_output");
    validate_tensor(currents, "currents");

    int64_t batch_size = get_batch_size(currents);
    int64_t num_sets = seed_v.size(1);

    // Allocate gradient tensors
    auto grad_currents = torch::zeros_like(currents);
    auto grad_insertion = torch::zeros_like(insertion_length);
    auto grad_seed_v = torch::zeros_like(seed_v);
    auto grad_seed_w = torch::zeros_like(seed_w);
    auto grad_seed_p = torch::zeros_like(seed_p);
    auto grad_seed_R = torch::zeros_like(seed_R);
    auto grad_seed_xf = torch::zeros_like(seed_xf);
    auto grad_seed_mL = torch::zeros_like(seed_mL);
    auto grad_seed_nL = torch::zeros_like(seed_nL);

    // Get raw pointers for gradient accumulation
    double* grad_curr_ptr = grad_currents.data_ptr<double>();

    // Acquire GIL for Python calls
    py::gil_scoped_acquire acquire;
    py::module_ crm_python = py::module_::import("crm_ml_rl.wrappers.crm_python");
    py::object CRMDynamics = crm_python.attr("CRMDynamics");

    // Create single dynamics instance (reused for all batch elements)
    py::object dyn = CRMDynamics();
    bool loaded = dyn.attr("load_parameters")(param_file, config_file).cast<bool>();
    if (!loaded) {
        throw std::runtime_error("Failed to load CRM parameters from " + param_file);
    }

    // Process each batch element sequentially
    for (int64_t i = 0; i < batch_size; ++i) {
        try {
            // Extract currents for this sample
            double curr_i[3];
            for (int j = 0; j < 3; ++j) {
                curr_i[j] = currents[i][j].item<double>();
            }
            double ins_i = get_scalar(insertion_length, i);

            // Extract seed state for this sample
            py::array_t<double> v_np(std::vector<py::ssize_t>{num_sets, 3});
            py::array_t<double> w_np(std::vector<py::ssize_t>{num_sets, 3});
            py::array_t<double> p_np(std::vector<py::ssize_t>{num_sets, 3});
            py::array_t<double> R_np(std::vector<py::ssize_t>{num_sets, 9});
            py::array_t<double> xf_np(std::vector<py::ssize_t>{15});
            py::array_t<double> mL_np(std::vector<py::ssize_t>{num_sets, 3});
            py::array_t<double> nL_np(std::vector<py::ssize_t>{num_sets, 3});

            auto v_buf = v_np.mutable_unchecked<2>();
            auto w_buf = w_np.mutable_unchecked<2>();
            auto p_buf = p_np.mutable_unchecked<2>();
            auto R_buf = R_np.mutable_unchecked<2>();
            auto xf_buf = xf_np.mutable_unchecked<1>();
            auto mL_buf = mL_np.mutable_unchecked<2>();
            auto nL_buf = nL_np.mutable_unchecked<2>();

            // Copy seed data from torch tensors to numpy arrays
            for (int64_t j = 0; j < num_sets; ++j) {
                for (int64_t k = 0; k < 3; ++k) {
                    v_buf(j, k) = seed_v[i][j][k].item<double>();
                    w_buf(j, k) = seed_w[i][j][k].item<double>();
                    p_buf(j, k) = seed_p[i][j][k].item<double>();
                    mL_buf(j, k) = seed_mL[i][j][k].item<double>();
                    nL_buf(j, k) = seed_nL[i][j][k].item<double>();
                }
                for (int64_t k = 0; k < 9; ++k) {
                    R_buf(j, k) = seed_R[i][j][k].item<double>();
                }
            }
            for (int64_t k = 0; k < 15; ++k) {
                xf_buf(k) = seed_xf[i][k].item<double>();
            }

            // Create currents numpy array
            py::array_t<double> curr_np(std::vector<py::ssize_t>{3});
            auto curr_buf = curr_np.mutable_unchecked<1>();
            for (int j = 0; j < 3; ++j) {
                curr_buf(j) = curr_i[j];
            }

            // Call Option A's implicit linearization
            // This computes A = ∂y/∂x_seed and B = ∂y/∂currents
            py::dict result = dyn.attr("linearize_full_seed_action_from_seed_implicit")(
                curr_np, ins_i,
                v_np, w_np, p_np, R_np, xf_np,
                mL_np, nL_np,
                eps_seed,  // eps_residual_x
                eps_seed,  // eps_residual_theta
                eps_seed,  // eps_g_x
                eps_seed,  // eps_g_theta
                false      // return_debug
            ).cast<py::dict>();

            // Extract B Jacobian: ∂y/∂currents (shape: 6 x 3)
            py::array_t<double> B_np = result["B"].cast<py::array_t<double>>();
            auto B_buf = B_np.unchecked<2>();

            // Verify B shape
            if (B_buf.shape(0) != 6 || B_buf.shape(1) != 3) {
                throw std::runtime_error(
                    "Expected B to have shape (6, 3), got (" +
                    std::to_string(B_buf.shape(0)) + ", " +
                    std::to_string(B_buf.shape(1)) + ")"
                );
            }

            // Extract grad_output for this sample (shape: 6)
            double grad_y[6];
            for (int j = 0; j < 6; ++j) {
                grad_y[j] = grad_output[i][j].item<double>();
            }

            // Compute grad_currents = B^T @ grad_y (vector-Jacobian product)
            // B is (6, 3), grad_y is (6,) -> result is (3,)
            // For each current dimension j:
            //   grad_currents[j] = sum_k B[k, j] * grad_y[k]
            for (int j = 0; j < 3; ++j) {
                double grad_u_j = 0.0;
                for (int k = 0; k < 6; ++k) {
                    grad_u_j += B_buf(k, j) * grad_y[k];
                }
                grad_curr_ptr[i * 3 + j] = grad_u_j;
            }

            // Phase 3B: Compute seed gradients from A Jacobian
            // Extract A Jacobian: ∂y/∂seed_state (shape: 6 x num_seed_components)
            py::array_t<double> A_np = result["A"].cast<py::array_t<double>>();
            auto A_buf = A_np.unchecked<2>();

            // Verify A shape (should have 6 rows for output dimension)
            if (A_buf.shape(0) != 6) {
                throw std::runtime_error(
                    "Expected A to have 6 rows (output dim), got " +
                    std::to_string(A_buf.shape(0))
                );
            }

            int64_t num_seed_components = A_buf.shape(1);

            // Compute seed gradients: grad_seed = A^T @ grad_y
            // grad_y is (6,), A is (6, num_seed_components)
            // Result: (num_seed_components,)

            // Track current position in A matrix columns
            int64_t seed_offset = 0;

            // 1. grad_seed_v (num_sets * 3 components)
            for (int64_t j = 0; j < num_sets; ++j) {
                for (int64_t k = 0; k < 3; ++k) {
                    double grad_v_jk = 0.0;
                    for (int64_t l = 0; l < 6; ++l) {
                        grad_v_jk += A_buf(l, seed_offset) * grad_y[l];
                    }
                    grad_seed_v.data_ptr<double>()[i * num_sets * 3 + j * 3 + k] = grad_v_jk;
                    seed_offset++;
                }
            }

            // 2. grad_seed_w (num_sets * 3 components)
            for (int64_t j = 0; j < num_sets; ++j) {
                for (int64_t k = 0; k < 3; ++k) {
                    double grad_w_jk = 0.0;
                    for (int64_t l = 0; l < 6; ++l) {
                        grad_w_jk += A_buf(l, seed_offset) * grad_y[l];
                    }
                    grad_seed_w.data_ptr<double>()[i * num_sets * 3 + j * 3 + k] = grad_w_jk;
                    seed_offset++;
                }
            }

            // 3. grad_seed_p (num_sets * 3 components)
            for (int64_t j = 0; j < num_sets; ++j) {
                for (int64_t k = 0; k < 3; ++k) {
                    double grad_p_jk = 0.0;
                    for (int64_t l = 0; l < 6; ++l) {
                        grad_p_jk += A_buf(l, seed_offset) * grad_y[l];
                    }
                    grad_seed_p.data_ptr<double>()[i * num_sets * 3 + j * 3 + k] = grad_p_jk;
                    seed_offset++;
                }
            }

            // 4. grad_seed_R (num_sets * 9 components)
            for (int64_t j = 0; j < num_sets; ++j) {
                for (int64_t k = 0; k < 9; ++k) {
                    double grad_R_jk = 0.0;
                    for (int64_t l = 0; l < 6; ++l) {
                        grad_R_jk += A_buf(l, seed_offset) * grad_y[l];
                    }
                    grad_seed_R.data_ptr<double>()[i * num_sets * 9 + j * 9 + k] = grad_R_jk;
                    seed_offset++;
                }
            }

            // 5. grad_seed_xf (15 components)
            for (int64_t k = 0; k < 15; ++k) {
                double grad_xf_k = 0.0;
                for (int64_t l = 0; l < 6; ++l) {
                    grad_xf_k += A_buf(l, seed_offset) * grad_y[l];
                }
                grad_seed_xf.data_ptr<double>()[i * 15 + k] = grad_xf_k;
                seed_offset++;
            }

            // 6. grad_seed_mL (num_sets * 3 components)
            for (int64_t j = 0; j < num_sets; ++j) {
                for (int64_t k = 0; k < 3; ++k) {
                    double grad_mL_jk = 0.0;
                    for (int64_t l = 0; l < 6; ++l) {
                        grad_mL_jk += A_buf(l, seed_offset) * grad_y[l];
                    }
                    grad_seed_mL.data_ptr<double>()[i * num_sets * 3 + j * 3 + k] = grad_mL_jk;
                    seed_offset++;
                }
            }

            // 7. grad_seed_nL (num_sets * 3 components)
            for (int64_t j = 0; j < num_sets; ++j) {
                for (int64_t k = 0; k < 3; ++k) {
                    double grad_nL_jk = 0.0;
                    for (int64_t l = 0; l < 6; ++l) {
                        grad_nL_jk += A_buf(l, seed_offset) * grad_y[l];
                    }
                    grad_seed_nL.data_ptr<double>()[i * num_sets * 3 + j * 3 + k] = grad_nL_jk;
                    seed_offset++;
                }
            }

            // Verification: seed_offset should equal num_seed_components
            if (seed_offset != num_seed_components) {
                throw std::runtime_error(
                    "Seed component count mismatch: expected " +
                    std::to_string(num_seed_components) +
                    ", got " + std::to_string(seed_offset)
                );
            }

        } catch (const std::exception& e) {
            throw std::runtime_error("Batch element " + std::to_string(i) +
                                   " backward pass failed: " + std::string(e.what()));
        }
    }

    // Return gradients for all inputs
    // grad_insertion is zero (insertion_length typically not differentiated)
    // grad_seed_* are zero (Phase 3A MVP - currents only)
    return std::make_tuple(
        grad_currents, grad_insertion, grad_seed_v, grad_seed_w,
        grad_seed_p, grad_seed_R, grad_seed_xf, grad_seed_mL, grad_seed_nL
    );
}

} // namespace crm_torch
