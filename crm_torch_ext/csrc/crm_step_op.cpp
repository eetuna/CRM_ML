// CRM Torch Extension - Custom Operator Implementation
// Option C: Native PyTorch Autograd Function

#include "crm_step_op.h"
#include "crm_params.h"
#include <torch/extension.h>
#include <Eigen/Dense>
#include <cmath>

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

    // Save original seed velocities (needed for IVP integration after BVP solve)
    double v_L_seed[NUM_ACT_SET][3] = {};
    double w_L_seed[NUM_ACT_SET][3] = {};

    for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
        for (int i = 0; i < 3; i++) {
            v_L[j][i] = v_acc[j][i];
            w_L[j][i] = w_acc[j][i];
            v_L_seed[j][i] = v_acc[j][i];  // Save original
            w_L_seed[j][i] = w_acc[j][i];  // Save original
            p_L[j][i] = p_acc[j][i];
        }
        for (int i = 0; i < 9; i++) {
            R_L[j][i] = R_acc[j][i];
        }
    }

    for (int i = 0; i < NUM_STATES; i++) {
        xf_local[i] = xf_acc[i];
    }

    // Get mL and nL guess (initial guess for BVP solver)
    double mL_guess_local[NUM_ACT_SET][3] = {};
    double nL_guess_local[NUM_ACT_SET][3] = {};

    auto mL_acc = seed_mL.accessor<double, 2>();
    auto nL_acc = seed_nL.accessor<double, 2>();

    for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
        for (int i = 0; i < 3; i++) {
            mL_guess_local[j][i] = mL_acc[j][i];
            nL_guess_local[j][i] = nL_acc[j][i];
        }
    }

    // Phase 4: Call actual CRM dynamics solver
    // Get parameters from singleton
    CRMParams& params = CRMParams::getInstance();
    if (!params.isInitialized()) {
        throw std::runtime_error(
            "CRM parameters not initialized. Call initialize_params() first.");
    }

    const CRMCatheterModelParams* cparams = params.getParams();
    const CatheterConfiguration& config = params.getConfig();
    const double dt_local = params.getDt();
    const double integration_step_size = params.getIntegrationStepSize();
    const IntegratorType integrator_type = params.getIntegratorType();

    // Get damping and inertia
    double damping_local[NUM_ACT_SET][6];
    double actInertia_local[NUM_ACT_SET][9];
    params.getDamping(damping_local);
    params.getActInertia(actInertia_local);

    // Build actuation currents array
    double ActuationCurrents[NUM_ACT_SET][3] = {};
    for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
        for (int i = 0; i < 3; i++) {
            ActuationCurrents[j][i] = currents_arr[i];
        }
    }

    // Setup shooting method parameters
    ContactModeType ContactMode = ContactModeType::FREE_TIP;
    double TipForce[3] = {0.0, 0.0, 0.0};
    double TipConstraintPoint[3] = {0.0, 0.0, 0.0};

    CRMShootingMethodParams BVPParams = CRMDYNConstructShootingMethodParamSet(
        *cparams, config, ins_len, ActuationCurrents,
        ContactMode, TipConstraintPoint, TipForce, integration_step_size,
        actInertia_local, v_L, w_L, p_L, R_L, damping_local, dt_local
    );

    BVPParams.dynamics.integrator_type = integrator_type;
    BVPParams.dynamics.last_diverged = false;

    // Phase 3: Damping-compensated initial guess
    for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
        for (int i = 0; i < 3; i++) {
            mL_guess_local[j][i] += damping_local[j][i + 3] * w_L[j][i];
            nL_guess_local[j][i] += damping_local[j][i] * v_L[j][i];
        }
    }

    // DEBUG: Print key BVP parameters
    if (const char* debug = std::getenv("CRM_DEBUG_BVP")) {
        if (std::string(debug) == "1") {
            std::cout << "[CRM_DEBUG_BVP] Extension forward pass:" << std::endl;
            std::cout << "  currents: " << currents_arr[0] << ", " << currents_arr[1] << ", " << currents_arr[2] << std::endl;
            std::cout << "  insertion: " << ins_len << std::endl;
            std::cout << "  dt: " << dt_local << std::endl;
            std::cout << "  integration_step_size: " << integration_step_size << std::endl;
            std::cout << "  integrator: " << (integrator_type == IntegratorType::RK4 ? "RK4" : "ABM4") << std::endl;
            std::cout << "  xf[0:3]: " << xf_local[0] << ", " << xf_local[1] << ", " << xf_local[2] << std::endl;
            std::cout << "  xf[3:12] (R): ";
            for (int i = 3; i < 12; i++) std::cout << xf_local[i] << (i < 11 ? ", " : "");
            std::cout << std::endl;
            std::cout << "  xf[12:15] (p): " << xf_local[12] << ", " << xf_local[13] << ", " << xf_local[14] << std::endl;
            std::cout << "  v_L[0]: " << v_L[0][0] << ", " << v_L[0][1] << ", " << v_L[0][2] << std::endl;
            std::cout << "  w_L[0]: " << w_L[0][0] << ", " << w_L[0][1] << ", " << w_L[0][2] << std::endl;
            std::cout << "  p_L[0]: " << p_L[0][0] << ", " << p_L[0][1] << ", " << p_L[0][2] << std::endl;
            std::cout << "  R_L[0]: ";
            for (int i = 0; i < 9; i++) std::cout << R_L[0][i] << (i < 8 ? ", " : "");
            std::cout << std::endl;
            std::cout << "  mL_guess[0]: " << mL_guess_local[0][0] << ", " << mL_guess_local[0][1] << ", " << mL_guess_local[0][2] << std::endl;
            std::cout << "  nL_guess[0]: " << nL_guess_local[0][0] << ", " << nL_guess_local[0][1] << ", " << nL_guess_local[0][2] << std::endl;
            std::cout << "  damping[0]: " << damping_local[0][0] << ", " << damping_local[0][1] << ", " << damping_local[0][2] << std::endl;
            std::cout << "  actInertia[0]: " << actInertia_local[0][0] << ", " << actInertia_local[0][4] << ", " << actInertia_local[0][8] << std::endl;
        }
    }

    // Solve BVP
    double out_u0[3];
    double out_mL[NUM_ACT_SET][3], out_nL[NUM_ACT_SET][3];
    double out_tau[NUM_ACT_SET][3];
    double ftip_calc[3];
    double ftip_guess[3] = {0.0, 0.0, 0.0};
    int localmin;

    // Try direct solve first
    DynamicsBVP(BVPParams, xf_local, mL_guess_local, nL_guess_local, ftip_guess,
                out_u0, out_mL, out_nL, out_tau, ftip_calc, localmin);

    // If BVP failed, enter velocity continuation recovery
    if (localmin != 0) {
        // Save original velocities
        double v_L_original[NUM_ACT_SET][3];
        double w_L_original[NUM_ACT_SET][3];
        for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
            for (int i = 0; i < 3; i++) {
                v_L_original[j][i] = v_L[j][i];
                w_L_original[j][i] = w_L[j][i];
            }
        }

        // Continuation ramp: gradually increase velocity from 0% to 100% over 5 steps
        constexpr int kContinuationSteps = 5;
        bool continuation_succeeded = true;

        for (int step = 0; step <= kContinuationSteps; step++) {
            const double alpha = static_cast<double>(step) / static_cast<double>(kContinuationSteps);

            // Scale velocities
            for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
                for (int i = 0; i < 3; i++) {
                    v_L[j][i] = alpha * v_L_original[j][i];
                    w_L[j][i] = alpha * w_L_original[j][i];
                }
            }

            // Rebuild BVPParams with scaled velocities
            CRMShootingMethodParams BVPParams_ramp = CRMDYNConstructShootingMethodParamSet(
                *cparams, config, ins_len, ActuationCurrents,
                ContactMode, TipConstraintPoint, TipForce, integration_step_size,
                actInertia_local, v_L, w_L, p_L, R_L, damping_local, dt_local
            );
            BVPParams_ramp.dynamics.integrator_type = integrator_type;
            BVPParams_ramp.dynamics.last_diverged = false;

            // Recompute damping-compensated guess for this velocity level
            double mL_guess_ramp[NUM_ACT_SET][3];
            double nL_guess_ramp[NUM_ACT_SET][3];
            for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
                for (int i = 0; i < 3; i++) {
                    mL_guess_ramp[j][i] = (step > 0) ? out_mL[j][i] : mL_guess_local[j][i];
                    nL_guess_ramp[j][i] = (step > 0) ? out_nL[j][i] : nL_guess_local[j][i];
                }
            }

            // Solve at this velocity level
            int localmin_ramp;
            DynamicsBVP(BVPParams_ramp, xf_local, mL_guess_ramp, nL_guess_ramp, ftip_guess,
                        out_u0, out_mL, out_nL, out_tau, ftip_calc, localmin_ramp);

            if (localmin_ramp != 0) {
                continuation_succeeded = false;
                break;
            }
        }

        if (continuation_succeeded) {
            // Multi-pass refinement: perform 2 additional calls at 100% velocity
            for (int warmup = 0; warmup < 2; warmup++) {
                // Restore original velocities
                for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
                    for (int i = 0; i < 3; i++) {
                        v_L[j][i] = v_L_original[j][i];
                        w_L[j][i] = w_L_original[j][i];
                    }
                }

                CRMShootingMethodParams BVPParams_warmup = CRMDYNConstructShootingMethodParamSet(
                    *cparams, config, ins_len, ActuationCurrents,
                    ContactMode, TipConstraintPoint, TipForce, integration_step_size,
                    actInertia_local, v_L, w_L, p_L, R_L, damping_local, dt_local
                );
                BVPParams_warmup.dynamics.integrator_type = integrator_type;
                BVPParams_warmup.dynamics.last_diverged = false;

                // Use previous converged solution as guess
                double mL_guess_warmup[NUM_ACT_SET][3];
                double nL_guess_warmup[NUM_ACT_SET][3];
                for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
                    for (int i = 0; i < 3; i++) {
                        mL_guess_warmup[j][i] = out_mL[j][i];
                        nL_guess_warmup[j][i] = out_nL[j][i];
                    }
                }

                int localmin_warmup;
                DynamicsBVP(BVPParams_warmup, xf_local, mL_guess_warmup, nL_guess_warmup, ftip_guess,
                            out_u0, out_mL, out_nL, out_tau, ftip_calc, localmin_warmup);

                if (localmin_warmup != 0) {
                    continuation_succeeded = false;
                    break;
                }
            }

            if (continuation_succeeded) {
                localmin = 0;  // Mark as successful
            }
        }

        // Failure recovery fallback: if homotopy failed, try static reset and retry once
        if (!continuation_succeeded) {
            // Reset to zero velocity and solve
            for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
                for (int i = 0; i < 3; i++) {
                    v_L[j][i] = 0.0;
                    w_L[j][i] = 0.0;
                }
            }

            CRMShootingMethodParams BVPParams_static = CRMDYNConstructShootingMethodParamSet(
                *cparams, config, ins_len, ActuationCurrents,
                ContactMode, TipConstraintPoint, TipForce, integration_step_size,
                actInertia_local, v_L, w_L, p_L, R_L, damping_local, dt_local
            );
            BVPParams_static.dynamics.integrator_type = integrator_type;
            BVPParams_static.dynamics.last_diverged = false;

            // Use original guess for static solve
            double mL_guess_static[NUM_ACT_SET][3];
            double nL_guess_static[NUM_ACT_SET][3];
            for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
                for (int i = 0; i < 3; i++) {
                    mL_guess_static[j][i] = mL_guess_local[j][i];
                    nL_guess_static[j][i] = nL_guess_local[j][i];
                }
            }

            int localmin_static;
            DynamicsBVP(BVPParams_static, xf_local, mL_guess_static, nL_guess_static, ftip_guess,
                        out_u0, out_mL, out_nL, out_tau, ftip_calc, localmin_static);

            if (localmin_static != 0) {
                throw std::runtime_error(
                    "DynamicsBVP failed to converge (localmin=" + std::to_string(localmin_static) +
                    "). Try different initial conditions or check inputs.");
            }

            localmin = 0;  // Mark as successful
        }
    }

    // DEBUG: Confirm we reach this point
    if (const char* debug = std::getenv("CRM_DEBUG_BVP")) {
        if (std::string(debug) == "1") {
            std::cout << "[CRM_DEBUG_BVP] About to call IVP, localmin=" << localmin << std::endl;
            std::cout.flush();
        }
    }
    std::cerr << "DEBUG: Reached IVP call point!" << std::endl;

    // Solve IVP to get next state (matching Python bindings implementation)
    // Important: Restore ORIGINAL seed velocities before IVP (they may have been modified during continuation)
    // The IVP integration needs the correct initial velocities from the seed state
    for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
        for (int i = 0; i < 3; i++) {
            v_L[j][i] = v_L_seed[j][i];
            w_L[j][i] = w_L_seed[j][i];
        }
    }

    // Reconstruct BVPParams with restored velocities
    BVPParams = CRMDYNConstructShootingMethodParamSet(
        *cparams, config, ins_len, ActuationCurrents,
        ContactMode, TipConstraintPoint, TipForce, integration_step_size,
        actInertia_local, v_L, w_L, p_L, R_L, damping_local, dt_local
    );
    BVPParams.dynamics.integrator_type = integrator_type;
    BVPParams.dynamics.last_diverged = false;

    // Compute forward-integrated state including updated coil velocities
    double xf_new[NUM_STATES];
    double x_coil[NUM_ACT_SET][NUM_COIL_STATES];
    double ReportedMarkerPos[5][3];

    DYNSolverIVP(BVPParams, out_u0, out_mL, out_nL, out_tau, ftip_calc,
                 true, xf_new, x_coil, ReportedMarkerPos);

    // DEBUG: Print x_coil contents
    if (const char* debug = std::getenv("CRM_DEBUG_BVP")) {
        if (std::string(debug) == "1") {
            std::cout << "[CRM_DEBUG_BVP] After DYNSolverIVP:" << std::endl;
            std::cout << "  xf_new[0:3]: " << xf_new[0] << ", " << xf_new[1] << ", " << xf_new[2] << std::endl;
            std::cout << "  x_coil[0][0:3] (velocities): " << x_coil[0][0] << ", " << x_coil[0][1] << ", " << x_coil[0][2] << std::endl;
            std::cout << "  x_coil[0][3:6] (ang velocities): " << x_coil[0][3] << ", " << x_coil[0][4] << ", " << x_coil[0][5] << std::endl;
        }
    }

    // Extract next state from IVP solution
    // Return format: [tip_pos (3), coil_velocities (num_sets * 3)]
    auto options = torch::TensorOptions().dtype(torch::kFloat64);
    torch::Tensor next_state = torch::zeros({output_dim}, options);
    auto out_acc = next_state.accessor<double, 1>();

    // Tip position from xf_new (first 3 elements)
    out_acc[0] = xf_new[0];  // tip_x
    out_acc[1] = xf_new[1];  // tip_y
    out_acc[2] = xf_new[2];  // tip_z

    // Coil velocities from x_coil (indices 0-2 are velocities)
    // This matches Python bindings: coil_vel(j, i) = x_coil[j][i]
    for (int j = 0; j < num_sets; j++) {
        for (int i = 0; i < 3; i++) {
            out_acc[3 + j * 3 + i] = x_coil[j][i];
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
    int64_t output_dim = grad_output.size(0);

    // Expected: output_dim = 3 + 3 * num_sets
    TORCH_CHECK(output_dim == 3 + 3 * num_sets,
                "grad_output size must match output_dim = 3 + 3 * num_sets");

    // TODO CP-C04: Implement proper gradient computation
    // Current limitation: Finite differences don't work reliably for BVP solvers
    // because small perturbations can cause convergence failures.
    //
    // Proper implementation requires implicit differentiation using autodiff,
    // matching the Python bindings' linearize_full_seed_action_from_seed_implicit().
    //
    // For now, return zero gradients to allow forward-only usage.
    // Users should use zero-order optimization (e.g., CMA-ES, genetic algorithms)
    // or manual finite differences at the policy level.

    // Compute seed dimension
    int64_t seed_dim = num_sets * 3 +  // v
                       num_sets * 3 +  // w
                       num_sets * 3 +  // p
                       num_sets * 9 +  // R
                       15 +            // xf
                       num_sets * 3 +  // mL
                       num_sets * 3;   // nL

    // Return zero Jacobians (stub implementation)
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

// Parameter management functions
void initialize_params(const std::string& param_file, const std::string& config_file) {
    CRMParams& params = CRMParams::getInstance();
    params.loadFromFiles(param_file, config_file);
}

void set_timestep(double dt) {
    CRMParams& params = CRMParams::getInstance();
    params.setDt(dt);
}

void set_integrator(const std::string& integrator) {
    CRMParams& params = CRMParams::getInstance();
    if (integrator == "rk4" || integrator == "RK4") {
        params.setIntegratorType(IntegratorType::RK4);
    } else if (integrator == "abm4" || integrator == "ABM4") {
        params.setIntegratorType(IntegratorType::ABM4);
    } else {
        throw std::runtime_error("Unknown integrator type: " + integrator + ". Use 'abm4' or 'rk4'.");
    }
}

void set_integration_step_size(double step_size) {
    CRMParams& params = CRMParams::getInstance();
    params.setIntegrationStepSize(step_size);
}

void set_damping(const std::vector<double>& damping) {
    if (damping.size() != 6) {
        throw std::runtime_error("Damping must have 6 elements [vx, vy, vz, wx, wy, wz]");
    }

    CRMParams& params = CRMParams::getInstance();
    double damping_arr[NUM_ACT_SET][6];

    // Set the same damping for all actuation sets
    for (int j = 0; j < NUM_ACT_SET; j++) {
        for (int i = 0; i < 6; i++) {
            damping_arr[j][i] = damping[i];
        }
    }

    params.setDamping(damping_arr);
}

} // namespace crm_torch
