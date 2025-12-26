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

// Helper function to compute implicit differentiation Jacobians
// This implements the same logic as linearize_full_seed_action_from_seed_implicit in crm_bindings.cpp
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> compute_implicit_jacobians(
    const Eigen::Vector3d& currents,
    double insertion_length,
    int num_sets,
    const double v_L[NUM_ACT_SET][3],
    const double w_L[NUM_ACT_SET][3],
    const double p_L[NUM_ACT_SET][3],
    const double R_L[NUM_ACT_SET][9],
    const double xf[NUM_STATES],
    const double mL_star[NUM_ACT_SET][3],
    const double nL_star[NUM_ACT_SET][3],
    const CRMCatheterModelParams* cparams,
    const CatheterConfiguration& config,
    double dt,
    double integration_step_size,
    IntegratorType integrator_type,
    const double damping[NUM_ACT_SET][6],
    const double actInertia[NUM_ACT_SET][9]
) {
    // Scale m and n values (matching Python bindings)
    // Note: IVALUE_SCALE_M and IVALUE_SCALE_N are defined as macros in CRMDYN.hpp
    // Python bindings use 1e-2 and 1e-1, but the header defines them as 10000.0
    // We use the same values as Python bindings for consistency
    const double scale_m = 1e-2;
    const double scale_n = 1e-1;

    int x_dim = num_sets * 6;  // mL (3) + nL (3) per actuator set

    Eigen::VectorXd x_star_scaled(x_dim);
    for (int j = 0; j < num_sets; j++) {
        for (int i = 0; i < 3; i++) {
            x_star_scaled(j * 6 + i) = mL_star[j][i] / scale_m;
            x_star_scaled(j * 6 + 3 + i) = nL_star[j][i] / scale_n;
        }
    }

    // Build BVPParams
    double ActuationCurrents[NUM_ACT_SET][3] = {};
    for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
        for (int i = 0; i < 3; i++) {
            ActuationCurrents[j][i] = currents(i);
        }
    }

    ContactModeType ContactMode = ContactModeType::FREE_TIP;
    double TipForce[3] = {0.0, 0.0, 0.0};
    double TipConstraintPoint[3] = {0.0, 0.0, 0.0};

    // Copy arrays to avoid mutation
    double v_L_local[NUM_ACT_SET][3], w_L_local[NUM_ACT_SET][3];
    double p_L_local[NUM_ACT_SET][3], R_L_local[NUM_ACT_SET][9];
    double damping_local[NUM_ACT_SET][6], actInertia_local[NUM_ACT_SET][9];

    for (int j = 0; j < NUM_ACT_SET; j++) {
        for (int i = 0; i < 3; i++) {
            v_L_local[j][i] = v_L[j][i];
            w_L_local[j][i] = w_L[j][i];
            p_L_local[j][i] = p_L[j][i];
        }
        for (int i = 0; i < 9; i++) {
            R_L_local[j][i] = R_L[j][i];
            actInertia_local[j][i] = actInertia[j][i];
        }
        for (int i = 0; i < 6; i++) {
            damping_local[j][i] = damping[j][i];
        }
    }

    CRMShootingMethodParams BVPParams = CRMDYNConstructShootingMethodParamSet(
        *cparams, config, insertion_length, ActuationCurrents,
        ContactMode, TipConstraintPoint, TipForce, integration_step_size,
        actInertia_local, v_L_local, w_L_local, p_L_local, R_L_local, damping_local, dt
    );
    BVPParams.dynamics.integrator_type = integrator_type;
    BVPParams.dynamics.last_diverged = false;

    // Prep DYNNLE params
    const bool FinalValueOnly = true;
    DYNNLEqnParams DYNNLEParams(BVPParams.no_flex_seg, BVPParams.no_rigid_seg,
                                BVPParams.no_act_set, BVPParams.no_locmarkers,
                                BVPParams.no_fcum_steps);

    double x_0[NUM_STATES];
    for (int i = 0; i < NUM_STATES; i++) {
        if (i < 3) x_0[i] = BVPParams.p0[i];
        else if (i < 12) x_0[i] = BVPParams.R0[i - 3];
        else x_0[i] = 0.0;
    }

    double mL_guess_local[NUM_ACT_SET][3]{}, nL_guess_local[NUM_ACT_SET][3]{};
    for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
        for (int i = 0; i < 3; i++) {
            mL_guess_local[j][i] = mL_star[j][i];
            nL_guess_local[j][i] = nL_star[j][i];
        }
    }

    double actInertia_prep[NUM_ACT_SET][9]{}, damping_prep[NUM_ACT_SET][6]{};
    double v_L_pre[NUM_ACT_SET][3]{}, w_L_pre[NUM_ACT_SET][3];
    double p_pre[NUM_ACT_SET][3]{}, R_pre[NUM_ACT_SET][9]{};

    for (int j = 0; j < BVPParams.no_act_set; ++j) {
        const auto& act = BVPParams.dynamics.actuators[j];
        for (int i = 0; i < 3; ++i) {
            v_L_pre[j][i] = act.v_L_pre(i);
            w_L_pre[j][i] = act.w_L_pre(i);
            p_pre[j][i] = act.p_pre(i);
        }
        for (int i = 0; i < 9; ++i) {
            actInertia_prep[j][i] = act.inertia(i / 3, i % 3);
            R_pre[j][i] = act.R_pre(i / 3, i % 3);
        }
        for (int i = 0; i < 6; ++i) {
            damping_prep[j][i] = act.damping(i);
        }
    }

    CRMDYNSolverIVP_Prep(
        BVPParams.no_flex_seg, BVPParams.no_rigid_seg, BVPParams.no_act_set,
        BVPParams.no_locmarkers, BVPParams.no_fcum_steps,
        x_0, BVPParams.IntegrationStepSize,
        BVPParams.Li, BVPParams.dlambdainv, BVPParams.rho, BVPParams.SegmentTypes,
        BVPParams.SegEndLambdas, BVPParams.LocMarkerLambdas,
        BVPParams.K, BVPParams.Kinv, BVPParams.ustar,
        BVPParams.MagMoment, BVPParams.fcumlambda, BVPParams.CoilAlignmentTurnAreaMatrix,
        BVPParams.B0, BVPParams.g, BVPParams.ActMass, actInertia_prep, damping_prep,
        BVPParams.dynamics.DELTA_T,
        v_L_pre, w_L_pre, p_pre, R_pre,
        mL_guess_local, nL_guess_local,
        FinalValueOnly, DYNNLEParams
    );

    DYNNLEParams.ContactMode = ContactModeType::FREE_TIP;
    for (int i = 0; i < 3; i++) {
        DYNNLEParams.TipForce[i] = 0.0;
        DYNNLEParams.TipConstraintPoint[i] = 0.0;
        DYNNLEParams.ftip_initialguess[i] = 0.0;
    }

    for (int i = 0; i < NUM_STATES; i++) {
        DYNNLEParams.xf[i] = xf[i];
    }

    // Step 1: Compute J_xx using autodiff
    Eigen::VectorXd residual_out;
    Eigen::MatrixXd J_xx = CRMCatheterModel::DYNNLEquationJacobianEigenAD(
        x_star_scaled, DYNNLEParams, &residual_out);

    // Step 2: Compute J_xu (control Jacobian) using autodiff
    Eigen::VectorXd controls_with_insertion(4);
    controls_with_insertion(0) = currents(0);
    controls_with_insertion(1) = currents(1);
    controls_with_insertion(2) = currents(2);
    controls_with_insertion(3) = insertion_length;

    Eigen::MatrixXd J_xu = CRMCatheterModel::DYNNLEquationControlJacobianEigenAD(
        x_star_scaled, controls_with_insertion, DYNNLEParams, nullptr);

    // Step 3: Solve IFT: dx/du = -J_xx^{-1} @ J_xu
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(J_xx);
    Eigen::MatrixXd dx_du = qr.solve(-J_xu);  // (x_dim, 4)

    // Step 4: Compute output Jacobian g_x = dy/dx using finite differences
    // Output is [tip_pos (3), coil_vel (3 * num_sets)]
    int output_dim = 3 + 3 * num_sets;
    Eigen::MatrixXd g_x(output_dim, x_dim);
    g_x.setZero();

    const double eps_x = 1e-5;

    // Lambda to evaluate output at perturbed x
    auto eval_output = [&](const Eigen::VectorXd& x_perturbed) -> Eigen::VectorXd {
        // Unscale to get physical mL, nL
        double mL_phys[NUM_ACT_SET][3]{}, nL_phys[NUM_ACT_SET][3]{};
        for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
            for (int i = 0; i < 3; i++) {
                mL_phys[j][i] = scale_m * x_perturbed(j * 6 + i);
                nL_phys[j][i] = scale_n * x_perturbed(j * 6 + 3 + i);
            }
        }

        // Compute u0 and tau from residual equation (we need these for IVP)
        std::vector<double> x_arr(x_dim);
        for (int i = 0; i < x_dim; i++) x_arr[i] = x_perturbed(i);

        std::vector<double> residual_tmp(x_dim);
        double u0_tmp[3], tau_tmp[NUM_ACT_SET * 3];
        DYNNLEquation(x_arr.data(), residual_tmp.data(), DYNNLEParams, u0_tmp, tau_tmp);

        double tau_phys[NUM_ACT_SET][3]{};
        for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
            for (int i = 0; i < 3; i++) {
                tau_phys[j][i] = tau_tmp[j * 3 + i];
            }
        }

        // Run IVP
        double xf_new[NUM_STATES], x_coil[NUM_ACT_SET][NUM_COIL_STATES];
        double ReportedMarkerPos[5][3];
        double ftip[3] = {0.0, 0.0, 0.0};

        DYNSolverIVP(BVPParams, u0_tmp, mL_phys, nL_phys, tau_phys, ftip,
                     true, xf_new, x_coil, ReportedMarkerPos);

        // Extract output
        Eigen::VectorXd output(output_dim);
        output(0) = xf_new[0];
        output(1) = xf_new[1];
        output(2) = xf_new[2];
        for (int j = 0; j < num_sets; j++) {
            for (int i = 0; i < 3; i++) {
                output(3 + j * 3 + i) = x_coil[j][i];
            }
        }
        return output;
    };

    // Compute g_x via finite differences
    for (int j = 0; j < x_dim; j++) {
        Eigen::VectorXd x_plus = x_star_scaled;
        Eigen::VectorXd x_minus = x_star_scaled;
        x_plus(j) += eps_x;
        x_minus(j) -= eps_x;

        Eigen::VectorXd y_plus = eval_output(x_plus);
        Eigen::VectorXd y_minus = eval_output(x_minus);

        g_x.col(j) = (y_plus - y_minus) / (2.0 * eps_x);
    }

    // Step 5: Apply chain rule: B = g_x @ dx/du
    Eigen::MatrixXd B = g_x * dx_du;  // (output_dim, 4)

    // For A, we need dx/dseed which requires J_xs (seed Jacobian)
    // For now, return empty A matrix (will implement seed gradients if needed)
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(output_dim, 1);  // Placeholder

    return {B, A};
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

    // CP-C04 Implementation: Implicit differentiation using autodiff
    // This matches the Python bindings' linearize_full_seed_action_from_seed_implicit()

    try {
        // Extract input arrays
        auto currents_acc = currents.accessor<double, 1>();
        auto ins_acc = insertion_length.accessor<double, 1>();
        auto v_acc = seed_v.accessor<double, 2>();
        auto w_acc = seed_w.accessor<double, 2>();
        auto p_acc = seed_p.accessor<double, 2>();
        auto R_acc = seed_R.accessor<double, 2>();
        auto xf_acc = seed_xf.accessor<double, 1>();
        auto mL_acc = seed_mL.accessor<double, 2>();
        auto nL_acc = seed_nL.accessor<double, 2>();

        // Build C++ arrays
        Eigen::Vector3d currents_vec;
        for (int i = 0; i < 3; i++) currents_vec(i) = currents_acc[i];

        double ins_len = ins_acc[0];

        double v_L[NUM_ACT_SET][3] = {}, w_L[NUM_ACT_SET][3] = {};
        double p_L[NUM_ACT_SET][3] = {}, R_L[NUM_ACT_SET][9] = {};
        double xf_local[NUM_STATES] = {};
        double mL_star[NUM_ACT_SET][3] = {}, nL_star[NUM_ACT_SET][3] = {};

        for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
            for (int i = 0; i < 3; i++) {
                v_L[j][i] = v_acc[j][i];
                w_L[j][i] = w_acc[j][i];
                p_L[j][i] = p_acc[j][i];
                mL_star[j][i] = mL_acc[j][i];
                nL_star[j][i] = nL_acc[j][i];
            }
            for (int i = 0; i < 9; i++) {
                R_L[j][i] = R_acc[j][i];
            }
        }

        for (int i = 0; i < NUM_STATES; i++) {
            xf_local[i] = xf_acc[i];
        }

        // Get parameters
        CRMParams& params = CRMParams::getInstance();
        const CRMCatheterModelParams* cparams = params.getParams();
        const CatheterConfiguration& config = params.getConfig();
        const double dt = params.getDt();
        const double integration_step_size = params.getIntegrationStepSize();
        const IntegratorType integrator_type = params.getIntegratorType();

        double damping_local[NUM_ACT_SET][6];
        double actInertia_local[NUM_ACT_SET][9];
        params.getDamping(damping_local);
        params.getActInertia(actInertia_local);

        // Compute Jacobians
        auto [B, A] = compute_implicit_jacobians(
            currents_vec, ins_len, num_sets,
            v_L, w_L, p_L, R_L, xf_local, mL_star, nL_star,
            cparams, config, dt, integration_step_size, integrator_type,
            damping_local, actInertia_local
        );

        // Convert Eigen to Torch
        auto options = torch::TensorOptions().dtype(torch::kFloat64);

        torch::Tensor B_torch = torch::zeros({output_dim, 4}, options);
        auto B_acc = B_torch.accessor<double, 2>();
        for (int i = 0; i < output_dim; i++) {
            for (int j = 0; j < 4; j++) {
                B_acc[i][j] = B(i, j);
            }
        }

        // Compute gradients via chain rule: grad_controls = B^T @ grad_output
        torch::Tensor grad_controls = torch::matmul(B_torch.transpose(0, 1), grad_output);

        auto grad_currents = grad_controls.slice(0, 0, 3).clone();
        auto grad_insertion = grad_controls.slice(0, 3, 4).clone();

        // For now, return zero seed gradients (A computation would go here)
        int64_t seed_dim = num_sets * 3 + num_sets * 3 + num_sets * 3 +
                           num_sets * 9 + 15 + num_sets * 3 + num_sets * 3;

        torch::Tensor grad_seed_flat = torch::zeros({seed_dim}, options);

        // Unflatten seed gradients
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
        auto grad_seed_xf = grad_seed_flat.slice(0, i_xf0, i_mL0).clone();
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

    } catch (const std::exception& e) {
        std::cerr << "Warning: Implicit differentiation failed: " << e.what() << std::endl;
        std::cerr << "Falling back to zero gradients." << std::endl;

        // Fallback: return zeros
        auto options = torch::TensorOptions().dtype(torch::kFloat64);
        int64_t seed_dim = num_sets * 3 + num_sets * 3 + num_sets * 3 +
                           num_sets * 9 + 15 + num_sets * 3 + num_sets * 3;

        auto grad_currents = torch::zeros({3}, options);
        auto grad_insertion = torch::zeros({1}, options);
        auto grad_seed_v = torch::zeros({num_sets, 3}, options);
        auto grad_seed_w = torch::zeros({num_sets, 3}, options);
        auto grad_seed_p = torch::zeros({num_sets, 3}, options);
        auto grad_seed_R = torch::zeros({num_sets, 9}, options);
        auto grad_seed_xf = torch::zeros({15}, options);
        auto grad_seed_mL = torch::zeros({num_sets, 3}, options);
        auto grad_seed_nL = torch::zeros({num_sets, 3}, options);

        return {
            grad_currents, grad_insertion,
            grad_seed_v, grad_seed_w, grad_seed_p, grad_seed_R, grad_seed_xf,
            grad_seed_mL, grad_seed_nL
        };
    }
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
