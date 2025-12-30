#pragma once

#include <torch/extension.h>
#include <tuple>
#include <string>

namespace crm_torch {

/**
 * Forward dynamics operator - batched CRM physics stepping.
 *
 * This function wraps the existing CRM Python bindings and calls them
 * in parallel using OpenMP for batch processing.
 *
 * @param currents (B, 3) - Control currents per batch
 * @param insertion_length (B,) or scalar - Insertion length
 * @param seed_v (B, num_sets, 3) - Linear velocity seed
 * @param seed_w (B, num_sets, 3) - Angular velocity seed
 * @param seed_p (B, num_sets, 3) - Position seed
 * @param seed_R (B, num_sets, 9) - Rotation matrix seed (flattened)
 * @param seed_xf (B, 15) - Tip state seed
 * @param seed_mL (B, num_sets, 3) - Magnetic load seed
 * @param seed_nL (B, num_sets, 3) - Nonlinear load seed
 * @param param_file Path to CRM parameter file
 * @param config_file Path to CRM configuration file
 *
 * @return Tuple of (next_state, next_v, next_w, next_p, next_R, next_xf, next_mL, next_nL)
 *         - next_state (B, output_dim) where output_dim = 3 + 3*num_sets
 *         - next_v, next_w, next_p, next_R, next_xf, next_mL, next_nL - Updated seeds
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dynamics_forward(
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
);

/**
 * Backward dynamics operator - compute gradients via implicit differentiation.
 *
 * @param grad_output (B, output_dim) - Gradient of loss w.r.t. output
 * @param currents (B, 3) - Saved currents from forward
 * @param insertion_length (B,) or scalar - Saved insertion length
 * @param seed_v, seed_w, seed_p, seed_R, seed_xf, seed_mL, seed_nL - Saved seeds
 * @param param_file, config_file - CRM configuration files
 * @param eps_seed Epsilon for implicit differentiation
 *
 * @return Tuple of gradients: (grad_currents, grad_insertion, grad_seed_v, ...)
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dynamics_backward(
    torch::Tensor grad_output,
    torch::Tensor grad_next_v,
    torch::Tensor grad_next_w,
    torch::Tensor grad_next_p,
    torch::Tensor grad_next_R,
    torch::Tensor grad_next_xf,
    torch::Tensor grad_next_mL,
    torch::Tensor grad_next_nL,
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
);

} // namespace crm_torch
