// CRM Torch Extension - Custom Operator Header
// Option C: Native PyTorch Autograd Function

#pragma once

#include <torch/extension.h>
#include <vector>

namespace crm_torch {

/**
 * @brief Forward pass for CRM dynamics step
 *
 * Wraps the existing CRMDynamics::step_from_seed() C++ implementation.
 *
 * @param currents Control currents [3]
 * @param insertion_length Catheter insertion length [1]
 * @param seed_v Coil velocities [num_sets, 3]
 * @param seed_w Angular velocities [num_sets, 3]
 * @param seed_p Positions [num_sets, 3]
 * @param seed_R Rotation matrices (flattened) [num_sets, 9]
 * @param seed_xf Tip state [15]
 * @param seed_mL Bending moments [num_sets, 3]
 * @param seed_nL Shear forces [num_sets, 3]
 * @return std::vector<torch::Tensor> [next_state, next_v, next_w, next_p, next_R, next_xf, next_mL, next_nL, localmin]
 */
std::vector<torch::Tensor> crm_step_forward(
    torch::Tensor currents,
    torch::Tensor insertion_length,
    torch::Tensor seed_v,
    torch::Tensor seed_w,
    torch::Tensor seed_p,
    torch::Tensor seed_R,
    torch::Tensor seed_xf,
    torch::Tensor seed_mL,
    torch::Tensor seed_nL
);

/**
 * @brief Backward pass for CRM dynamics step
 *
 * Uses existing linearize_full_seed_action_from_seed_implicit() to compute
 * gradients via automatic differentiation and implicit differentiation.
 *
 * @param grad_output Gradient w.r.t. next_state [output_dim]
 * @param currents Saved from forward [3]
 * @param insertion_length Saved from forward [1]
 * @param seed_v Saved from forward [num_sets, 3]
 * @param seed_w Saved from forward [num_sets, 3]
 * @param seed_p Saved from forward [num_sets, 3]
 * @param seed_R Saved from forward [num_sets, 9]
 * @param seed_xf Saved from forward [15]
 * @param seed_mL Saved from forward [num_sets, 3]
 * @param seed_nL Saved from forward [num_sets, 3]
 * @return std::vector<torch::Tensor> Gradients for all inputs
 */
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
);

/**
 * @brief Initialize CRM parameters from files
 *
 * Must be called before using crm_step.
 *
 * @param param_file Path to parameter file
 * @param config_file Path to configuration file
 */
void initialize_params(const std::string& param_file, const std::string& config_file);

/**
 * @brief Set timestep for dynamics integration
 * @param dt Timestep in seconds
 */
void set_timestep(double dt);

/**
 * @brief Set integrator type
 * @param integrator "abm4" or "rk4"
 */
void set_integrator(const std::string& integrator);

/**
 * @brief Set integration step size
 * @param step_size Integration step size in mm
 */
void set_integration_step_size(double step_size);

/**
 * @brief Set damping coefficients
 * @param damping Array of 6 damping coefficients [vx, vy, vz, wx, wy, wz]
 */
void set_damping(const std::vector<double>& damping);

/**
 * @brief Initialize dynamics state from Forward Kinematics
 *
 * Solves the static BVP to find the physically valid starting configuration.
 *
 * @param currents Initial currents [3]
 * @param insertion_length Initial insertion length [1]
 * @return std::vector<torch::Tensor> [v, w, p, R, xf, mL, nL]
 */
std::vector<torch::Tensor> crm_initialize_from_fk(
    torch::Tensor currents,
    torch::Tensor insertion_length
);

} // namespace crm_torch
