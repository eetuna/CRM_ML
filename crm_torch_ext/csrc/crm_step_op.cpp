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

/**
 * @brief Orthonormalize a 3x3 rotation matrix using Gram-Schmidt process
 * @param R Pointer to 9-element array (row-major)
 */
void gram_schmidt_orthonormalize(double R[9]) {
    // Extract columns (R is row-major: [r00, r01, r02, r10, r11, r12, r20, r21, r22])
    // Column 0: R[0], R[3], R[6]
    // Column 1: R[1], R[4], R[7]
    // Column 2: R[2], R[5], R[8]

    double r0[3] = {R[0], R[3], R[6]};
    double r1[3] = {R[1], R[4], R[7]};
    double r2[3] = {R[2], R[5], R[8]};

    // Normalize r0
    double n0 = std::sqrt(r0[0]*r0[0] + r0[1]*r0[1] + r0[2]*r0[2]);
    if (n0 > 1e-12) { r0[0] /= n0; r0[1] /= n0; r0[2] /= n0; }

    // Orthogonalize r1 against r0
    double dot01 = r0[0]*r1[0] + r0[1]*r1[1] + r0[2]*r1[2];
    r1[0] -= dot01 * r0[0];
    r1[1] -= dot01 * r0[1];
    r1[2] -= dot01 * r0[2];

    // Normalize r1
    double n1 = std::sqrt(r1[0]*r1[0] + r1[1]*r1[1] + r1[2]*r1[2]);
    if (n1 > 1e-12) { r1[0] /= n1; r1[1] /= n1; r1[2] /= n1; }

    // Orthogonalize r2 against r0 and r1
    double dot02 = r0[0]*r2[0] + r0[1]*r2[1] + r0[2]*r2[2];
    double dot12 = r1[0]*r2[0] + r1[1]*r2[1] + r1[2]*r2[2];
    r2[0] -= (dot02 * r0[0] + dot12 * r1[0]);
    r2[1] -= (dot02 * r0[1] + dot12 * r1[1]);
    r2[2] -= (dot02 * r0[2] + dot12 * r1[2]);

    // Normalize r2
    double n2 = std::sqrt(r2[0]*r2[0] + r2[1]*r2[1] + r2[2]*r2[2]);
    if (n2 > 1e-12) { r2[0] /= n2; r2[1] /= n2; r2[2] /= n2; }

    // Write back to R (row-major)
    R[0] = r0[0]; R[1] = r1[0]; R[2] = r2[0];
    R[3] = r0[1]; R[4] = r1[1]; R[5] = r2[1];
    R[6] = r0[2]; R[7] = r1[2]; R[8] = r2[2];
}

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

            if (localmin_static == 0) {
                localmin = 0;  // Static solve succeeded
            } else {
                // Last ditch effort: Current/Force Continuation Ramp
                // Gradually increase currents from 0 to target
                constexpr int kCurrentRampSteps = 5;
                bool ramp_succeeded = false;

                for (int step = 1; step <= kCurrentRampSteps; step++) {
                    const double alpha = static_cast<double>(step) / static_cast<double>(kCurrentRampSteps);
                    
                    double ActuationCurrents_ramp[NUM_ACT_SET][3];
                    for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
                        for (int i = 0; i < 3; i++) {
                            ActuationCurrents_ramp[j][i] = alpha * ActuationCurrents[j][i];
                        }
                    }

                    CRMShootingMethodParams BVPParams_curr = CRMDYNConstructShootingMethodParamSet(
                        *cparams, config, ins_len, ActuationCurrents_ramp,
                        ContactMode, TipConstraintPoint, TipForce, integration_step_size,
                        actInertia_local, v_L, w_L, p_L, R_L, damping_local, dt_local
                    );
                    BVPParams_curr.dynamics.integrator_type = integrator_type;

                    double mL_guess_curr[NUM_ACT_SET][3], nL_guess_curr[NUM_ACT_SET][3];
                    for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
                        for (int i = 0; i < 3; i++) {
                            // Start from previous ramp step's solution (or original guess for step 1)
                            mL_guess_curr[j][i] = (step > 1) ? out_mL[j][i] : mL_guess_local[j][i];
                            nL_guess_curr[j][i] = (step > 1) ? out_nL[j][i] : nL_guess_local[j][i];
                        }
                    }

                    int localmin_curr;
                    DynamicsBVP(BVPParams_curr, xf_local, mL_guess_curr, nL_guess_curr, ftip_guess,
                                out_u0, out_mL, out_nL, out_tau, ftip_calc, localmin_curr);

                    if (localmin_curr != 0) {
                        ramp_succeeded = false;
                        break;
                    }
                    if (step == kCurrentRampSteps) ramp_succeeded = true;
                }

                if (ramp_succeeded) {
                    localmin = 0;
                } else {
                    // Fail soft: matching Python bindings behavior
                    if (const char* debug = std::getenv("CRM_DEBUG_BVP")) {
                        std::cerr << "Warning: DynamicsBVP failed to converge (localmin=" << localmin_static 
                                  << ") after all continuation attempts. Returning best-effort result." << std::endl;
                    }
                    localmin = localmin_static; 
                }
            }
        }
    }

    // DEBUG: Confirm we reach this point
    if (const char* debug = std::getenv("CRM_DEBUG_BVP")) {
        if (std::string(debug) == "1") {
            std::cout << "[CRM_DEBUG_BVP] About to call IVP, localmin=" << localmin << std::endl;
            std::cout.flush();
        }
    }
    
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

    // Prepare Return Tensors
    auto return_options = torch::TensorOptions().dtype(torch::kFloat64);
    
    // [0] next_state (observation)
    torch::Tensor next_state = torch::zeros({output_dim}, return_options);
    auto next_state_acc = next_state.accessor<double, 1>();

    if (localmin == 0) {
        // SUCCESS: Use IVP results
        next_state_acc[0] = xf_new[0];
        next_state_acc[1] = xf_new[1];
        next_state_acc[2] = xf_new[2];
        for (int j = 0; j < num_sets; j++) {
            for (int i = 0; i < 3; i++) next_state_acc[3 + j * 3 + i] = x_coil[j][i];
        }

        // [1] next_v
        torch::Tensor next_v = torch::zeros({num_sets, 3}, return_options);
        auto next_v_acc = next_v.accessor<double, 2>();
        for (int j = 0; j < num_sets; j++) for (int i = 0; i < 3; i++) next_v_acc[j][i] = x_coil[j][i];

        // [2] next_w
        torch::Tensor next_w = torch::zeros({num_sets, 3}, return_options);
        auto next_w_acc = next_w.accessor<double, 2>();
        for (int j = 0; j < num_sets; j++) for (int i = 0; i < 3; i++) next_w_acc[j][i] = x_coil[j][i + 3];

        // [3] next_p
        torch::Tensor next_p = torch::zeros({num_sets, 3}, return_options);
        auto next_p_acc = next_p.accessor<double, 2>();
        for (int j = 0; j < num_sets; j++) for (int i = 0; i < 3; i++) next_p_acc[j][i] = x_coil[j][i + 6];

        // [4] next_R
        torch::Tensor next_R = torch::zeros({num_sets, 9}, return_options);
        auto next_R_acc = next_R.accessor<double, 2>();
        for (int j = 0; j < num_sets; j++) {
            double R_tmp[9];
            for (int i = 0; i < 9; i++) R_tmp[i] = x_coil[j][i + 9];
            gram_schmidt_orthonormalize(R_tmp);
            for (int i = 0; i < 9; i++) next_R_acc[j][i] = R_tmp[i];
        }

        // [5] next_xf
        torch::Tensor next_xf = torch::zeros({15}, return_options);
        auto next_xf_acc = next_xf.accessor<double, 1>();
        for (int i = 0; i < 15; i++) next_xf_acc[i] = xf_new[i];
        double R_xf[9];
        for (int i = 0; i < 9; i++) R_xf[i] = xf_new[i + 3];
        gram_schmidt_orthonormalize(R_xf);
        for (int i = 0; i < 9; i++) next_xf_acc[i + 3] = R_xf[i];

        // [6] next_mL
        torch::Tensor next_mL = torch::zeros({num_sets, 3}, return_options);
        auto next_mL_acc = next_mL.accessor<double, 2>();
        for (int j = 0; j < num_sets; j++) for (int i = 0; i < 3; i++) next_mL_acc[j][i] = out_mL[j][i];

        // [7] next_nL
        torch::Tensor next_nL = torch::zeros({num_sets, 3}, return_options);
        auto next_nL_acc = next_nL.accessor<double, 2>();
        for (int j = 0; j < num_sets; j++) for (int i = 0; i < 3; i++) next_nL_acc[j][i] = out_nL[j][i];

        // [8] localmin
        torch::Tensor localmin_tensor = torch::tensor({0.0}, return_options);

        return {next_state, next_v, next_w, next_p, next_R, next_xf, next_mL, next_nL, localmin_tensor};
    } else {
        // FAILURE: Latch to original seeds, return zero velocities
        // Observation: Keep tip position from seed_xf, but zero velocities
        next_state_acc[0] = xf_local[12];
        next_state_acc[1] = xf_local[13];
        next_state_acc[2] = xf_local[14];
        // Coil velocities are already zeroed by torch::zeros

        // Return clones of inputs to maintain graph consistency
        torch::Tensor localmin_tensor = torch::tensor({static_cast<double>(localmin)}, return_options);
        return {next_state, seed_v.clone(), seed_w.clone(), seed_p.clone(), seed_R.clone(), seed_xf.clone(), seed_mL.clone(), seed_nL.clone(), localmin_tensor};
    }
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
    const double mL_seed[NUM_ACT_SET][3],
    const double nL_seed[NUM_ACT_SET][3],
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
    // Python bindings use mL_star / IVALUE_SCALE_M where IVALUE_SCALE_M = 10000.0
    // We must match this exactly.
    const double scale_m = IVALUE_SCALE_M;
    const double scale_n = IVALUE_SCALE_N;

    int x_dim = num_sets * 6;  // mL (3) + nL (3) per actuator set

    Eigen::VectorXd x_star_scaled(x_dim);
    for (int j = 0; j < num_sets; j++) {
        for (int i = 0; i < 3; i++) {
            x_star_scaled(j * 6 + i) = mL_star[j][i] / scale_m;
            x_star_scaled(j * 6 + 3 + i) = nL_star[j][i] / scale_n;
        }
    }

    if (const char* debug = std::getenv("CRM_DEBUG_BACKWARD")) {
        if (std::string(debug) == "1") {
            std::cout << "[JACOBIAN] scale_m: " << scale_m << ", IVALUE_SCALE_M: " << IVALUE_SCALE_M << std::endl;
            std::cout << "[JACOBIAN] x_star_scaled: " << x_star_scaled.transpose() << std::endl;
            std::cout << "[JACOBIAN] mL_star[0]: " << mL_star[0][0] << ", " << mL_star[0][1] << ", " << mL_star[0][2] << std::endl;
            std::cout << "[JACOBIAN] nL_star[0]: " << nL_star[0][0] << ", " << nL_star[0][1] << ", " << nL_star[0][2] << std::endl;
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

    // CRITICAL: Use the seed values (initial guess, often zeros) for IVP_Prep,
    // NOT the converged mL_star/nL_star. This matches the Python wrapper.
    double mL_guess_local[NUM_ACT_SET][3]{}, nL_guess_local[NUM_ACT_SET][3]{};
    for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
        for (int i = 0; i < 3; i++) {
            mL_guess_local[j][i] = mL_seed[j][i];
            nL_guess_local[j][i] = nL_seed[j][i];
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

    if (const char* debug = std::getenv("CRM_DEBUG_BACKWARD")) {
        if (std::string(debug) == "1") {
            std::cout << "[JACOBIAN] DYNNLEParams.xf[0:3]: " << xf[0] << ", " << xf[1] << ", " << xf[2] << std::endl;
            std::cout << "[JACOBIAN] DYNNLEParams.xf[12:15]: " << xf[12] << ", " << xf[13] << ", " << xf[14] << std::endl;
        }
    }

    // Step 1: Compute J_xx using autodiff (with fallback to finite differences)
    Eigen::MatrixXd J_xx;
    bool have_ad_jxx = false;

    try {
        Eigen::VectorXd residual_out;
        J_xx = CRMCatheterModel::DYNNLEquationJacobianEigenAD(
            x_star_scaled, DYNNLEParams, &residual_out);

        if (const char* debug = std::getenv("CRM_DEBUG_BACKWARD")) {
            if (std::string(debug) == "1") {
                std::cout << "[JACOBIAN] AD returned J_xx size: " << J_xx.rows() << "x" << J_xx.cols()
                          << ", expected: " << x_dim << "x" << x_dim << std::endl;
                std::cout << "[JACOBIAN] J_xx.allFinite(): " << J_xx.allFinite() << std::endl;
                if (!J_xx.allFinite()) {
                    std::cout << "[JACOBIAN] J_xx has inf/nan values" << std::endl;
                }
            }
        }

        if (J_xx.rows() == x_dim && J_xx.cols() == x_dim && J_xx.allFinite()) {
            have_ad_jxx = true;
        }
    } catch (const std::exception& e) {
        if (const char* debug = std::getenv("CRM_DEBUG_BACKWARD")) {
            if (std::string(debug) == "1") {
                std::cout << "[JACOBIAN] AD exception: " << e.what() << std::endl;
            }
        }
        have_ad_jxx = false;
    } catch (...) {
        if (const char* debug = std::getenv("CRM_DEBUG_BACKWARD")) {
            if (std::string(debug) == "1") {
                std::cout << "[JACOBIAN] AD unknown exception" << std::endl;
            }
        }
        have_ad_jxx = false;
    }

    if (!have_ad_jxx) {
        if (const char* debug = std::getenv("CRM_DEBUG_BACKWARD")) {
            if (std::string(debug) == "1") {
                std::cout << "[JACOBIAN] Autodiff J_xx failed, falling back to finite differences" << std::endl;
            }
        }

        // Fallback to finite differences for J_xx
        // This matches the Python wrapper approach (crm_bindings.cpp:2712-2726)
        J_xx.resize(x_dim, x_dim);
        J_xx.setZero();

        // Reduce epsilon for small scaled variables
        const double eps_residual_x = 1e-5;

        // Lambda to evaluate residual at a given x_scaled
        // CRITICAL: Must rebuild DYNNLEParams with the perturbed mL/nL values!
        auto eval_residual = [&](const Eigen::VectorXd& x_pert) -> Eigen::VectorXd {
            // Unscale x_pert to get physical mL/nL values
            double mL_pert[NUM_ACT_SET][3]{}, nL_pert[NUM_ACT_SET][3]{};
            for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
                for (int i = 0; i < 3; i++) {
                    mL_pert[j][i] = scale_m * x_pert(j * 6 + i);
                    nL_pert[j][i] = scale_n * x_pert(j * 6 + 3 + i);
                }
            }

            // Rebuild DYNNLEParams with perturbed mL/nL as guess
            const bool FinalValueOnly_local = true;
            DYNNLEqnParams DYNNLEParams_local(BVPParams.no_flex_seg, BVPParams.no_rigid_seg,
                                              BVPParams.no_act_set, BVPParams.no_locmarkers,
                                              BVPParams.no_fcum_steps);

            double x_0_local[NUM_STATES];
            for (int i = 0; i < NUM_STATES; i++) {
                if (i < 3) x_0_local[i] = BVPParams.p0[i];
                else if (i < 12) x_0_local[i] = BVPParams.R0[i - 3];
                else x_0_local[i] = 0.0;
            }

            // Use the seed values (zeros) for IVP_Prep guess, NOT the perturbed values
            double mL_guess_eval[NUM_ACT_SET][3]{}, nL_guess_eval[NUM_ACT_SET][3]{};
            for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
                for (int i = 0; i < 3; i++) {
                    mL_guess_eval[j][i] = mL_seed[j][i];
                    nL_guess_eval[j][i] = nL_seed[j][i];
                }
            }

            double actInertia_prep_eval[NUM_ACT_SET][9]{}, damping_prep_eval[NUM_ACT_SET][6]{};
            double v_L_pre_eval[NUM_ACT_SET][3]{}, w_L_pre_eval[NUM_ACT_SET][3];
            double p_pre_eval[NUM_ACT_SET][3]{}, R_pre_eval[NUM_ACT_SET][9]{};

            for (int j = 0; j < BVPParams.no_act_set; ++j) {
                const auto& act = BVPParams.dynamics.actuators[j];
                for (int i = 0; i < 3; ++i) {
                    v_L_pre_eval[j][i] = act.v_L_pre(i);
                    w_L_pre_eval[j][i] = act.w_L_pre(i);
                    p_pre_eval[j][i] = act.p_pre(i);
                }
                for (int i = 0; i < 9; ++i) {
                    actInertia_prep_eval[j][i] = act.inertia(i / 3, i % 3);
                    R_pre_eval[j][i] = act.R_pre(i / 3, i % 3);
                }
                for (int i = 0; i < 6; ++i) {
                    damping_prep_eval[j][i] = act.damping(i);
                }
            }

            CRMDYNSolverIVP_Prep(
                BVPParams.no_flex_seg, BVPParams.no_rigid_seg, BVPParams.no_act_set,
                BVPParams.no_locmarkers, BVPParams.no_fcum_steps,
                x_0_local, BVPParams.IntegrationStepSize,
                BVPParams.Li, BVPParams.dlambdainv, BVPParams.rho, BVPParams.SegmentTypes,
                BVPParams.SegEndLambdas, BVPParams.LocMarkerLambdas,
                BVPParams.K, BVPParams.Kinv, BVPParams.ustar,
                BVPParams.MagMoment, BVPParams.fcumlambda, BVPParams.CoilAlignmentTurnAreaMatrix,
                BVPParams.B0, BVPParams.g, BVPParams.ActMass, actInertia_prep_eval, damping_prep_eval,
                BVPParams.dynamics.DELTA_T,
                v_L_pre_eval, w_L_pre_eval, p_pre_eval, R_pre_eval,
                mL_guess_eval, nL_guess_eval,
                FinalValueOnly_local, DYNNLEParams_local
            );

            DYNNLEParams_local.ContactMode = ContactModeType::FREE_TIP;
            for (int i = 0; i < 3; i++) {
                DYNNLEParams_local.TipForce[i] = 0.0;
                DYNNLEParams_local.TipConstraintPoint[i] = 0.0;
                DYNNLEParams_local.ftip_initialguess[i] = 0.0;
            }

            for (int i = 0; i < NUM_STATES; i++) {
                DYNNLEParams_local.xf[i] = xf[i];
            }

            // Now evaluate residual with perturbed x_pert
            const int NLEq_Dim = x_dim;
            std::vector<double> x_arr(NLEq_Dim);
            for (int i = 0; i < NLEq_Dim; i++) {
                x_arr[i] = x_pert(i);
            }

            std::vector<double> out_y(NLEq_Dim);
            double u0_out[3];
            double tau_out[NUM_ACT_SET * 3];

            DYNNLEquation(x_arr.data(), out_y.data(), DYNNLEParams_local, u0_out, tau_out);

            Eigen::VectorXd F(NLEq_Dim);
            for (int i = 0; i < NLEq_Dim; i++) {
                F(i) = out_y[i];
            }
            return F;
        };

        // Compute J_xx via finite differences: J_xx[:, j] = (F(x + eps*ej) - F(x - eps*ej)) / (2*eps)
        for (int j = 0; j < x_dim; j++) {
            Eigen::VectorXd xp = x_star_scaled;
            Eigen::VectorXd xm = x_star_scaled;
            xp(j) += eps_residual_x;
            xm(j) -= eps_residual_x;

            Eigen::VectorXd Fp = eval_residual(xp);
            Eigen::VectorXd Fm = eval_residual(xm);

            J_xx.col(j) = (Fp - Fm) * (0.5 / eps_residual_x);

            if (const char* debug = std::getenv("CRM_DEBUG_BACKWARD")) {
                if (std::string(debug) == "1") {
                    if (j == 0 || !Fp.allFinite() || !Fm.allFinite() || Fp.norm() > 1e5) {
                        std::cout << "[JACOBIAN FD] col " << j << ": xp[j]=" << xp(j) << ", xm[j]=" << xm(j) << std::endl;
                        std::cout << "[JACOBIAN FD] Fp norm=" << Fp.norm() << ", Fm norm=" << Fm.norm() << std::endl;
                        if (Fp.norm() > 1e5) {
                            std::cout << "[JACOBIAN FD] WARNING: Large residual norm detected (Divergence?)" << std::endl;
                        }
                    }
                }
            }
        }
    }

    if (const char* debug = std::getenv("CRM_DEBUG_BACKWARD")) {
        if (std::string(debug) == "1") {
            std::cout << "[JACOBIAN] J_xx norm: " << J_xx.norm() << ", has NaN: " << (!J_xx.allFinite()) << std::endl;
        }
    }

    // Step 2: Compute J_xu (control Jacobian) using autodiff
    Eigen::VectorXd controls_with_insertion(4);
    controls_with_insertion(0) = currents(0);
    controls_with_insertion(1) = currents(1);
    controls_with_insertion(2) = currents(2);
    controls_with_insertion(3) = insertion_length;

    Eigen::MatrixXd J_xu;
    bool have_ad_jxu = false;

    try {
        J_xu = CRMCatheterModel::DYNNLEquationControlJacobianEigenAD(
            x_star_scaled, controls_with_insertion, DYNNLEParams, nullptr);

        if (const char* debug = std::getenv("CRM_DEBUG_BACKWARD")) {
            if (std::string(debug) == "1") {
                std::cout << "[JACOBIAN] AD returned J_xu size: " << J_xu.rows() << "x" << J_xu.cols()
                          << ", expected: " << x_dim << "x4" << std::endl;
                std::cout << "[JACOBIAN] J_xu.allFinite(): " << J_xu.allFinite() << std::endl;
            }
        }

        if (J_xu.rows() == x_dim && J_xu.cols() == 4 && J_xu.allFinite()) {
            have_ad_jxu = true;
        }
    } catch (const std::exception& e) {
        if (const char* debug = std::getenv("CRM_DEBUG_BACKWARD")) {
            if (std::string(debug) == "1") {
                std::cout << "[JACOBIAN] AD J_xu exception: " << e.what() << std::endl;
            }
        }
        have_ad_jxu = false;
    } catch (...) {
        if (const char* debug = std::getenv("CRM_DEBUG_BACKWARD")) {
            if (std::string(debug) == "1") {
                std::cout << "[JACOBIAN] AD J_xu unknown exception" << std::endl;
            }
        }
        have_ad_jxu = false;
    }

    if (!have_ad_jxu) {
        if (const char* debug = std::getenv("CRM_DEBUG_BACKWARD")) {
            if (std::string(debug) == "1") {
                std::cout << "[JACOBIAN] Autodiff J_xu failed, falling back to finite differences" << std::endl;
            }
        }

        // Fallback to finite differences for J_xu
        // Compute ∂F/∂u for currents and insertion_length
        J_xu.resize(x_dim, 4);
        J_xu.setZero();

        const double eps_residual_theta = 1e-5;

        // Lambda to evaluate residual at a given (x, currents, insertion)
        // We need to rebuild BVPParams and DYNNLEParams with new currents
        auto eval_residual_with_controls = [&](const Eigen::VectorXd& x_pert,
                                                 const Eigen::Vector3d& curr_pert,
                                                 double ins_pert) -> Eigen::VectorXd {
            // Rebuild BVPParams with perturbed currents/insertion
            double ActuationCurrents_local[NUM_ACT_SET][3] = {};
            for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
                for (int i = 0; i < 3; i++) {
                    ActuationCurrents_local[j][i] = curr_pert(i);
                }
            }

            ContactModeType ContactMode_local = ContactModeType::FREE_TIP;
            double TipForce_local[3] = {0.0, 0.0, 0.0};
            double TipConstraintPoint_local[3] = {0.0, 0.0, 0.0};

            // Build new BVPParams with perturbed controls
            CRMShootingMethodParams BVPParams_local = CRMDYNConstructShootingMethodParamSet(
                *cparams, config, ins_pert, ActuationCurrents_local,
                ContactMode_local, TipConstraintPoint_local, TipForce_local, integration_step_size,
                actInertia_local, v_L_local, w_L_local, p_L_local, R_L_local, damping_local, dt
            );
            BVPParams_local.dynamics.integrator_type = integrator_type;
            BVPParams_local.dynamics.last_diverged = false;

            // Prep DYNNLE params
            const bool FinalValueOnly_local = true;
            DYNNLEqnParams DYNNLEParams_local(BVPParams_local.no_flex_seg, BVPParams_local.no_rigid_seg,
                                              BVPParams_local.no_act_set, BVPParams_local.no_locmarkers,
                                              BVPParams_local.no_fcum_steps);

            double x_0_local[NUM_STATES];
            for (int i = 0; i < NUM_STATES; i++) {
                if (i < 3) x_0_local[i] = BVPParams_local.p0[i];
                else if (i < 12) x_0_local[i] = BVPParams_local.R0[i - 3];
                else x_0_local[i] = 0.0;
            }

            double mL_guess_fd[NUM_ACT_SET][3]{}, nL_guess_fd[NUM_ACT_SET][3]{};
            for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
                for (int i = 0; i < 3; i++) {
                    mL_guess_fd[j][i] = mL_seed[j][i];
                    nL_guess_fd[j][i] = nL_seed[j][i];
                }
            }

            double actInertia_prep_local[NUM_ACT_SET][9]{}, damping_prep_local[NUM_ACT_SET][6]{};
            double v_L_pre_local[NUM_ACT_SET][3]{}, w_L_pre_local[NUM_ACT_SET][3];
            double p_pre_local[NUM_ACT_SET][3]{}, R_pre_local[NUM_ACT_SET][9]{};

            for (int j = 0; j < BVPParams_local.no_act_set; ++j) {
                const auto& act = BVPParams_local.dynamics.actuators[j];
                for (int i = 0; i < 3; ++i) {
                    v_L_pre_local[j][i] = act.v_L_pre(i);
                    w_L_pre_local[j][i] = act.w_L_pre(i);
                    p_pre_local[j][i] = act.p_pre(i);
                }
                for (int i = 0; i < 9; ++i) {
                    actInertia_prep_local[j][i] = act.inertia(i / 3, i % 3);
                    R_pre_local[j][i] = act.R_pre(i / 3, i % 3);
                }
                for (int i = 0; i < 6; ++i) {
                    damping_prep_local[j][i] = act.damping(i);
                }
            }

            CRMDYNSolverIVP_Prep(
                BVPParams_local.no_flex_seg, BVPParams_local.no_rigid_seg, BVPParams_local.no_act_set,
                BVPParams_local.no_locmarkers, BVPParams_local.no_fcum_steps,
                x_0_local, BVPParams_local.IntegrationStepSize,
                BVPParams_local.Li, BVPParams_local.dlambdainv, BVPParams_local.rho, BVPParams_local.SegmentTypes,
                BVPParams_local.SegEndLambdas, BVPParams_local.LocMarkerLambdas,
                BVPParams_local.K, BVPParams_local.Kinv, BVPParams_local.ustar,
                BVPParams_local.MagMoment, BVPParams_local.fcumlambda, BVPParams_local.CoilAlignmentTurnAreaMatrix,
                BVPParams_local.B0, BVPParams_local.g, BVPParams_local.ActMass, actInertia_prep_local, damping_prep_local,
                BVPParams_local.dynamics.DELTA_T,
                v_L_pre_local, w_L_pre_local, p_pre_local, R_pre_local,
                mL_guess_fd, nL_guess_fd,
                FinalValueOnly_local, DYNNLEParams_local
            );

            DYNNLEParams_local.ContactMode = ContactModeType::FREE_TIP;
            for (int i = 0; i < 3; i++) {
                DYNNLEParams_local.TipForce[i] = 0.0;
                DYNNLEParams_local.TipConstraintPoint[i] = 0.0;
                DYNNLEParams_local.ftip_initialguess[i] = 0.0;
            }

            for (int i = 0; i < NUM_STATES; i++) {
                DYNNLEParams_local.xf[i] = xf[i];
            }

            // Evaluate residual
            const int NLEq_Dim = x_dim;
            std::vector<double> x_arr(NLEq_Dim);
            for (int i = 0; i < NLEq_Dim; i++) {
                x_arr[i] = x_pert(i);
            }

            std::vector<double> out_y(NLEq_Dim);
            double u0_out[3];
            double tau_out[NUM_ACT_SET * 3];

            DYNNLEquation(x_arr.data(), out_y.data(), DYNNLEParams_local, u0_out, tau_out);

            Eigen::VectorXd F(NLEq_Dim);
            for (int i = 0; i < NLEq_Dim; i++) {
                F(i) = out_y[i];
            }
            return F;
        };

        // Compute J_xu via finite differences
        // J_xu[:, 0:3] = ∂F/∂currents
        for (int j = 0; j < 3; j++) {
            Eigen::Vector3d curr_p = currents;
            Eigen::Vector3d curr_m = currents;
            curr_p(j) += eps_residual_theta;
            curr_m(j) -= eps_residual_theta;

            Eigen::VectorXd Fp = eval_residual_with_controls(x_star_scaled, curr_p, insertion_length);
            Eigen::VectorXd Fm = eval_residual_with_controls(x_star_scaled, curr_m, insertion_length);

            J_xu.col(j) = (Fp - Fm) * (0.5 / eps_residual_theta);
        }

        // J_xu[:, 3] = ∂F/∂insertion_length
        {
            double ins_p = insertion_length + eps_residual_theta;
            double ins_m = insertion_length - eps_residual_theta;

            Eigen::VectorXd Fp = eval_residual_with_controls(x_star_scaled, currents, ins_p);
            Eigen::VectorXd Fm = eval_residual_with_controls(x_star_scaled, currents, ins_m);

            J_xu.col(3) = (Fp - Fm) * (0.5 / eps_residual_theta);
        }
    }

    if (const char* debug = std::getenv("CRM_DEBUG_BACKWARD")) {
        if (std::string(debug) == "1") {
            std::cout << "[JACOBIAN] J_xu norm: " << J_xu.norm() << ", has NaN: " << (!J_xu.allFinite()) << std::endl;
        }
    }

    // Step 3: Solve IFT: dx/du = -J_xx^{-1} @ J_xu
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(J_xx);
    Eigen::MatrixXd dx_du = qr.solve(-J_xu);  // (x_dim, 4)

    if (const char* debug = std::getenv("CRM_DEBUG_BACKWARD")) {
        if (std::string(debug) == "1") {
            std::cout << "[JACOBIAN] dx_du norm: " << dx_du.norm() << ", has NaN: " << (!dx_du.allFinite()) << std::endl;
        }
    }

    // Step 4: Compute output Jacobians gx = dy/dx and gth = dy/dtheta using AD
    // Output is [tip_pos (3), coil_vel (3 * num_sets)]
    int output_dim = 3 + 3 * num_sets;
    int seed_dim = num_sets * 3 + num_sets * 3 + num_sets * 3 +
                   num_sets * 9 + 15 + num_sets * 3 + num_sets * 3;
    Eigen::MatrixXd gx(output_dim, x_dim);
    Eigen::MatrixXd gth(output_dim, 3 + seed_dim);

    // Assemble seed_flat for AD call
    // seed_flat layout: [v (num_sets*3), w (num_sets*3), p (num_sets*3), R (num_sets*9), xf (15), mL (num_sets*3), nL (num_sets*3)]
    Eigen::VectorXd seed_flat(seed_dim);
    int idx = 0;
    for (int j = 0; j < num_sets; j++) for (int i = 0; i < 3; i++) seed_flat(idx++) = v_L[j][i];
    for (int j = 0; j < num_sets; j++) for (int i = 0; i < 3; i++) seed_flat(idx++) = w_L[j][i];
    for (int j = 0; j < num_sets; j++) for (int i = 0; i < 3; i++) seed_flat(idx++) = p_L[j][i];
    for (int j = 0; j < num_sets; j++) for (int i = 0; i < 9; i++) seed_flat(idx++) = R_L[j][i];
    for (int i = 0; i < NUM_STATES; i++) seed_flat(idx++) = xf[i];
    for (int j = 0; j < num_sets; j++) for (int i = 0; i < 3; i++) seed_flat(idx++) = mL_seed[j][i];
    for (int j = 0; j < num_sets; j++) for (int i = 0; i < 3; i++) seed_flat(idx++) = nL_seed[j][i];

    CRMCatheterModel::dynnl_ad_eigen::DYNNLEquationOutputJacobianEigenAD(
        x_star_scaled, currents, seed_flat, DYNNLEParams, gx, gth
    );

    // Step 5: Apply chain rule: B = ∂y/∂u + (∂y/∂x) * (dx/du)
    // gth layout: [∂y/∂currents (3), ∂y/∂seed (seed_dim)]
    // We only need ∂y/∂currents for B. Insertion length grad will be added if available.
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(output_dim, 4);
    B.leftCols(3) = gth.leftCols(3) + gx * dx_du.leftCols(3);

    // Handle insertion length gradient (column 3 of B)
    // dy/d_ins = gx * (dx/d_ins) - no direct dependency of output on insertion length in eval_output_AD
    B.col(3) = gx * dx_du.col(3);

    // For A (dy/dseed), we need gth.rightCols(seed_dim) + gx * dx/dseed
    // For now, return zero A matrix (will implement if needed)
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
        auto mL_guess_acc = seed_mL.accessor<double, 2>();
        auto nL_guess_acc = seed_nL.accessor<double, 2>();

        // Build C++ arrays
        Eigen::Vector3d currents_vec;
        for (int i = 0; i < 3; i++) currents_vec(i) = currents_acc[i];

        double ins_len = ins_acc[0];

        double v_L[NUM_ACT_SET][3] = {}, w_L[NUM_ACT_SET][3] = {};
        double p_L[NUM_ACT_SET][3] = {}, R_L[NUM_ACT_SET][9] = {};
        double xf_local[NUM_STATES] = {};
        double mL_guess[NUM_ACT_SET][3] = {}, nL_guess[NUM_ACT_SET][3] = {};

        for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
            for (int i = 0; i < 3; i++) {
                v_L[j][i] = v_acc[j][i];
                w_L[j][i] = w_acc[j][i];
                p_L[j][i] = p_acc[j][i];
                mL_guess[j][i] = mL_guess_acc[j][i];
                nL_guess[j][i] = nL_guess_acc[j][i];
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

        // CRITICAL FIX: Re-solve BVP to get converged mL_star and nL_star
        // The input seed_mL and seed_nL are just guesses (often zeros)
        // We need the actual converged solution from the BVP solver
        // This matches the Python bindings approach (line 2292 in crm_bindings.cpp)

        double ActuationCurrents[NUM_ACT_SET][3] = {};
        for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
            for (int i = 0; i < 3; i++) {
                ActuationCurrents[j][i] = currents_vec(i);
            }
        }

        ContactModeType ContactMode = ContactModeType::FREE_TIP;
        double TipForce[3] = {0.0, 0.0, 0.0};
        double TipConstraintPoint[3] = {0.0, 0.0, 0.0};

        CRMShootingMethodParams BVPParams = CRMDYNConstructShootingMethodParamSet(
            *cparams, config, ins_len, ActuationCurrents,
            ContactMode, TipConstraintPoint, TipForce, integration_step_size,
            actInertia_local, v_L, w_L, p_L, R_L, damping_local, dt
        );
        BVPParams.dynamics.integrator_type = integrator_type;
        BVPParams.dynamics.last_diverged = false;

        // Apply damping compensation to guess (matching forward pass)
        // We use a separate array for this so we preserve the original seed (mL_guess)
        // for passing to compute_implicit_jacobians later.
        double mL_guess_compensated[NUM_ACT_SET][3];
        double nL_guess_compensated[NUM_ACT_SET][3];
        
        for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
            for (int i = 0; i < 3; i++) {
                mL_guess_compensated[j][i] = mL_guess[j][i] + damping_local[j][i + 3] * w_L[j][i];
                nL_guess_compensated[j][i] = nL_guess[j][i] + damping_local[j][i] * v_L[j][i];
            }
        }

        // Solve BVP to get converged mL_star and nL_star
        double out_u0[3];
        double mL_star[NUM_ACT_SET][3], nL_star[NUM_ACT_SET][3];
        double out_tau[NUM_ACT_SET][3];
        double ftip_calc[3], ftip_guess[3] = {0.0, 0.0, 0.0};
        int localmin;

        DynamicsBVP(BVPParams, xf_local, mL_guess_compensated, nL_guess_compensated, ftip_guess,
                    out_u0, mL_star, nL_star, out_tau, ftip_calc, localmin);

        if (localmin != 0) {
            // Try Current/Force Continuation Ramp in backward pass too
            constexpr int kCurrentRampSteps = 5;
            bool ramp_succeeded = false;

            for (int step = 1; step <= kCurrentRampSteps; step++) {
                const double alpha = static_cast<double>(step) / static_cast<double>(kCurrentRampSteps);
                double ActuationCurrents_ramp[NUM_ACT_SET][3];
                for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
                    for (int i = 0; i < 3; i++) ActuationCurrents_ramp[j][i] = alpha * ActuationCurrents[j][i];
                }

                CRMShootingMethodParams BVPParams_curr = CRMDYNConstructShootingMethodParamSet(
                    *cparams, config, ins_len, ActuationCurrents_ramp,
                    ContactMode, TipConstraintPoint, TipForce, integration_step_size,
                    actInertia_local, v_L, w_L, p_L, R_L, damping_local, dt
                );
                BVPParams_curr.dynamics.integrator_type = integrator_type;

                double mL_guess_curr[NUM_ACT_SET][3], nL_guess_curr[NUM_ACT_SET][3];
                for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
                    for (int i = 0; i < 3; i++) {
                        mL_guess_curr[j][i] = (step > 1) ? mL_star[j][i] : mL_guess_compensated[j][i];
                        nL_guess_curr[j][i] = (step > 1) ? nL_star[j][i] : nL_guess_compensated[j][i];
                    }
                }

                int localmin_curr;
                DynamicsBVP(BVPParams_curr, xf_local, mL_guess_curr, nL_guess_curr, ftip_guess,
                            out_u0, mL_star, nL_star, out_tau, ftip_calc, localmin_curr);

                if (localmin_curr != 0) {
                    ramp_succeeded = false;
                    break;
                }
                if (step == kCurrentRampSteps) ramp_succeeded = true;
            }

            if (ramp_succeeded) {
                localmin = 0;
            }
        }

        if (const char* debug = std::getenv("CRM_DEBUG_BACKWARD")) {
            if (std::string(debug) == "1") {
                std::cout << "[BACKWARD] BVP solve result:" << std::endl;
                std::cout << "  localmin: " << localmin << std::endl;
                if (localmin == 0) {
                    std::cout << "  mL_star[0]: " << mL_star[0][0] << ", " << mL_star[0][1] << ", " << mL_star[0][2] << std::endl;
                }
            }
        }

        if (localmin != 0) {
            std::cerr << "Warning: BVP did not converge in backward pass (localmin=" << localmin << ")" << std::endl;
            std::cerr << "Falling back to zero gradients." << std::endl;

            // Return zero gradients
            auto options = torch::TensorOptions().dtype(torch::kFloat64);
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

        // Now compute Jacobians using the converged mL_star and nL_star
        // Also pass the seed values (mL_guess, nL_guess) for IVP_Prep initialization
        auto [B, A] = compute_implicit_jacobians(
            currents_vec, ins_len, num_sets,
            v_L, w_L, p_L, R_L, xf_local, mL_star, nL_star, mL_guess, nL_guess,
            cparams, config, dt, integration_step_size, integrator_type,
            damping_local, actInertia_local
        );

        if (const char* debug = std::getenv("CRM_DEBUG_BACKWARD")) {
            if (std::string(debug) == "1") {
                std::cout << "[BACKWARD] B matrix (first row): " << B.row(0) << std::endl;
                std::cout << "[BACKWARD] B norm: " << B.norm() << std::endl;
            }
        }

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
