// CRM Torch Extension - Custom Operator Implementation
// Option C: Native PyTorch Autograd Function

#include "crm_step_op.h"
#include <torch/extension.h>
#include <Eigen/Dense>

// Include CRM headers
#include "CRM.hpp"
#include "CRMDYN.hpp"
#include "CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp"

namespace crm_torch {

using namespace CRMCatheterModel;

torch::Tensor crm_step_forward(
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
    // Input validation
    TORCH_CHECK(currents.dim() == 1 && currents.size(0) == 3,
                "currents must be shape [3]");
    TORCH_CHECK(insertion_length.dim() == 1 && insertion_length.size(0) == 1,
                "insertion_length must be shape [1]");
    TORCH_CHECK(seed_v.dim() == 2 && seed_v.size(1) == 3,
                "seed_v must be shape [num_sets, 3]");
    TORCH_CHECK(seed_w.dim() == 2 && seed_w.size(1) == 3,
                "seed_w must be shape [num_sets, 3]");
    TORCH_CHECK(seed_p.dim() == 2 && seed_p.size(1) == 3,
                "seed_p must be shape [num_sets, 3]");
    TORCH_CHECK(seed_R.dim() == 2 && seed_R.size(1) == 9,
                "seed_R must be shape [num_sets, 9]");
    TORCH_CHECK(seed_xf.dim() == 1 && seed_xf.size(0) == 15,
                "seed_xf must be shape [15]");

    // Ensure tensors are contiguous and on CPU (required for data access)
    currents = currents.contiguous().cpu();
    insertion_length = insertion_length.contiguous().cpu();
    seed_v = seed_v.contiguous().cpu();
    seed_w = seed_w.contiguous().cpu();
    seed_p = seed_p.contiguous().cpu();
    seed_R = seed_R.contiguous().cpu();
    seed_xf = seed_xf.contiguous().cpu();
    seed_mL = seed_mL.contiguous().cpu();
    seed_nL = seed_nL.contiguous().cpu();

    // Extract dimensions
    int64_t num_sets = seed_v.size(0);
    int64_t output_dim = 3 + 3 * num_sets;  // tip_pos + velocities per set

    // Get data pointers (all tensors are float64/double)
    auto currents_acc = currents.accessor<double, 1>();
    auto ins_acc = insertion_length.accessor<double, 1>();
    auto v_acc = seed_v.accessor<double, 2>();
    auto w_acc = seed_w.accessor<double, 2>();
    auto p_acc = seed_p.accessor<double, 2>();
    auto R_acc = seed_R.accessor<double, 2>();
    auto xf_acc = seed_xf.accessor<double, 1>();

    // Copy inputs to C++ arrays (matching the Python binding format)
    double currents_arr[3];
    for (int i = 0; i < 3; i++) {
        currents_arr[i] = currents_acc[i];
    }

    double ins_len = ins_acc[0];

    double v_L[NUM_ACT_SET][3] = {};
    double w_L[NUM_ACT_SET][3] = {};
    double p_L[NUM_ACT_SET][3] = {};
    double R_L[NUM_ACT_SET][9] = {};
    double xf_local[NUM_STATES] = {};

    for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
        for (int i = 0; i < 3; i++) {
            v_L[j][i] = v_acc[j][i];
            w_L[j][i] = w_acc[j][i];
            p_L[j][i] = p_acc[j][i];
        }
        for (int i = 0; i < 9; i++) {
            R_L[j][i] = R_acc[j][i];
        }
    }

    for (int i = 0; i < NUM_STATES; i++) {
        xf_local[i] = xf_acc[i];
    }

    // TODO: Call actual CRM dynamics solver
    // For CP-C03, this is a STUB that demonstrates correct data flow
    // The actual implementation will be completed in integration phase
    //
    // Expected call:
    // DynamicsBVP(...) -> returns xf_new, x_coil
    //
    // For now, return a test pattern to verify tensor conversion works

    auto options = torch::TensorOptions().dtype(torch::kFloat64);
    torch::Tensor next_state = torch::zeros({output_dim}, options);
    auto out_acc = next_state.accessor<double, 1>();

    // Test pattern: copy tip position from seed, velocities from seed
    out_acc[0] = xf_local[0];  // tip_x
    out_acc[1] = xf_local[1];  // tip_y
    out_acc[2] = xf_local[2];  // tip_z

    for (int j = 0; j < num_sets; j++) {
        for (int i = 0; i < 3; i++) {
            out_acc[3 + j * 3 + i] = v_L[j][i];  // coil velocities
        }
    }

    return next_state;
}

std::vector<torch::Tensor> crm_step_backward(
    torch::Tensor grad_output,
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
    // Input validation
    TORCH_CHECK(grad_output.dim() == 1, "grad_output must be 1D");

    // Ensure tensors are contiguous and on CPU
    grad_output = grad_output.contiguous().cpu();

    // Extract dimensions
    int64_t num_sets = seed_v.size(0);
    int64_t output_dim = grad_output.size(0);

    // Expected: output_dim = 3 + 3 * num_sets
    TORCH_CHECK(output_dim == 3 + 3 * num_sets,
                "grad_output size must match output_dim = 3 + 3 * num_sets");

    // TODO: Call actual linearization (linearize_full_seed_action_from_seed_implicit)
    // For CP-C04 stub: use placeholder Jacobians (zeros)
    // This demonstrates correct gradient computation structure
    //
    // Actual implementation will call:
    // linearize_full_seed_action_from_seed_implicit(...) -> returns A, B matrices
    // where:
    //   B: (output_dim, control_dim) where control_dim = 4 [currents(3), insertion(1)]
    //   A: (output_dim, seed_dim) where seed_dim = sum of all seed tensor sizes

    // Compute seed dimension
    int64_t seed_dim = num_sets * 3 +  // v
                       num_sets * 3 +  // w
                       num_sets * 3 +  // p
                       num_sets * 9 +  // R
                       15 +            // xf
                       num_sets * 3 +  // mL
                       num_sets * 3;   // nL

    // Create placeholder Jacobians (stub implementation)
    // In actual implementation, these would come from linearize_* call
    auto options = torch::TensorOptions().dtype(torch::kFloat64);

    int64_t control_dim = 4;  // [currents (3), insertion_length (1)]
    torch::Tensor B = torch::zeros({output_dim, control_dim}, options);
    torch::Tensor A = torch::zeros({output_dim, seed_dim}, options);

    // Compute gradients via chain rule
    // grad_controls = B^T @ grad_output
    torch::Tensor grad_controls = torch::matmul(B.transpose(0, 1), grad_output);

    // Extract control gradients
    auto grad_currents = grad_controls.slice(0, 0, 3).clone();  // First 3 elements
    auto grad_insertion = grad_controls.slice(0, 3, 4).clone();  // 4th element

    // grad_seed = A^T @ grad_output
    torch::Tensor grad_seed_flat = torch::matmul(A.transpose(0, 1), grad_output);

    // Unflatten seed gradients (matching Python wrapper structure)
    int64_t dim_v = num_sets * 3;
    int64_t dim_w = num_sets * 3;
    int64_t dim_p = num_sets * 3;
    int64_t dim_R = num_sets * 9;
    int64_t dim_xf = 15;
    int64_t dim_mL = num_sets * 3;
    int64_t dim_nL = num_sets * 3;

    int64_t i_v0 = 0;
    int64_t i_w0 = i_v0 + dim_v;
    int64_t i_p0 = i_w0 + dim_w;
    int64_t i_R0 = i_p0 + dim_p;
    int64_t i_xf0 = i_R0 + dim_R;
    int64_t i_mL0 = i_xf0 + dim_xf;
    int64_t i_nL0 = i_mL0 + dim_mL;

    auto grad_seed_v = grad_seed_flat.slice(0, i_v0, i_w0).reshape({num_sets, 3});
    auto grad_seed_w = grad_seed_flat.slice(0, i_w0, i_p0).reshape({num_sets, 3});
    auto grad_seed_p = grad_seed_flat.slice(0, i_p0, i_R0).reshape({num_sets, 3});
    auto grad_seed_R = grad_seed_flat.slice(0, i_R0, i_xf0).reshape({num_sets, 9});
    auto grad_seed_xf = grad_seed_flat.slice(0, i_xf0, i_mL0).clone();  // [15]
    auto grad_seed_mL = grad_seed_flat.slice(0, i_mL0, i_nL0).reshape({num_sets, 3});
    auto grad_seed_nL = grad_seed_flat.slice(0, i_nL0, seed_dim).reshape({num_sets, 3});

    return {
        grad_currents,
        grad_insertion,
        grad_seed_v,
        grad_seed_w,
        grad_seed_p,
        grad_seed_R,
        grad_seed_xf,
        grad_seed_mL,
        grad_seed_nL
    };
}

} // namespace crm_torch
