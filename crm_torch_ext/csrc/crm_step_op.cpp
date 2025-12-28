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
    double r0[3] = {R[0], R[3], R[6]};
    double r1[3] = {R[1], R[4], R[7]};
    double r2[3] = {R[2], R[5], R[8]};

    auto norm = [](double v[3]) { return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]); };
    
    double n0 = norm(r0);
    if (n0 > 1e-12) { r0[0] /= n0; r0[1] /= n0; r0[2] /= n0; }

    double dot01 = r0[0]*r1[0] + r0[1]*r1[1] + r0[2]*r1[2];
    r1[0] -= dot01 * r0[0]; r1[1] -= dot01 * r0[1]; r1[2] -= dot01 * r0[2];
    double n1 = norm(r1);
    if (n1 > 1e-12) { r1[0] /= n1; r1[1] /= n1; r1[2] /= n1; }

    double dot02 = r0[0]*r2[0] + r0[1]*r2[1] + r0[2]*r2[2];
    double dot12 = r1[0]*r2[0] + r1[1]*r2[1] + r1[2]*r2[2];
    r2[0] -= (dot02 * r0[0] + dot12 * r1[0]); r2[1] -= (dot02 * r0[1] + dot12 * r1[1]); r2[2] -= (dot02 * r0[2] + dot12 * r1[2]);
    double n2 = norm(r2);
    if (n2 > 1e-12) { r2[0] /= n2; r2[1] /= n2; r2[2] /= n2; }

    R[0] = r0[0]; R[1] = r1[0]; R[2] = r2[0];
    R[3] = r0[1]; R[4] = r1[1]; R[5] = r2[1];
    R[6] = r0[2]; R[7] = r1[2]; R[8] = r2[2];
}

std::vector<torch::Tensor> crm_step_forward(
    torch::Tensor currents, torch::Tensor insertion_length,
    torch::Tensor seed_v, torch::Tensor seed_w, torch::Tensor seed_p, torch::Tensor seed_R, torch::Tensor seed_xf,
    torch::Tensor seed_mL, torch::Tensor seed_nL
) {
    TORCH_CHECK(currents.dim() == 1 && currents.size(0) == 3, "currents must be [3]");
    currents = currents.contiguous().cpu();
    insertion_length = insertion_length.contiguous().cpu();
    seed_v = seed_v.contiguous().cpu(); seed_w = seed_w.contiguous().cpu(); seed_p = seed_p.contiguous().cpu();
    seed_R = seed_R.contiguous().cpu(); seed_xf = seed_xf.contiguous().cpu();
    seed_mL = seed_mL.contiguous().cpu(); seed_nL = seed_nL.contiguous().cpu();

    int64_t num_sets = seed_v.size(0), output_dim = 3 + 3 * num_sets;
    auto c_acc = currents.accessor<double, 1>();
    auto ins_acc = insertion_length.accessor<double, 1>();
    auto v_acc = seed_v.accessor<double, 2>(), w_acc = seed_w.accessor<double, 2>(), p_acc = seed_p.accessor<double, 2>(), R_acc = seed_R.accessor<double, 2>();
    auto xf_acc = seed_xf.accessor<double, 1>();
    auto mL_acc = seed_mL.accessor<double, 2>(), nL_acc = seed_nL.accessor<double, 2>();

    double Act[NUM_ACT_SET][3] = {}; for (int i = 0; i < 3; i++) Act[0][i] = c_acc[i];
    double ins_len = ins_acc[0];
    
    // Internal state accumulators
    double v_L[NUM_ACT_SET][3] = {}, w_L[NUM_ACT_SET][3] = {}, p_L[NUM_ACT_SET][3] = {}, R_L[NUM_ACT_SET][9] = {};
    double mL_guess_local[NUM_ACT_SET][3] = {}, nL_guess_local[NUM_ACT_SET][3] = {}, xf_local[NUM_STATES] = {};

    // Initialize accumulators with seed
    for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
        for (int i = 0; i < 3; i++) {
            v_L[j][i] = v_acc[j][i]; w_L[j][i] = w_acc[j][i];
            p_L[j][i] = p_acc[j][i];
            mL_guess_local[j][i] = mL_acc[j][i]; nL_guess_local[j][i] = nL_acc[j][i];
        }
        for (int i = 0; i < 9; i++) R_L[j][i] = R_acc[j][i];
    }
    for (int i = 0; i < NUM_STATES; i++) xf_local[i] = xf_acc[i];

    CRMParams& params = CRMParams::getInstance();
    const CRMCatheterModelParams* cp = params.getParams();
    const double dt_local = params.getDt();
    const double dt_safe = 0.001; // 1ms safety limit for RK4
    const int num_substeps = std::max(1, (int)std::ceil(dt_local / dt_safe));
    const double dt_sub = dt_local / num_substeps;
    const double istep = params.getIntegrationStepSize();

    double damp[NUM_ACT_SET][6], inertia[NUM_ACT_SET][9]; params.getDamping(damp); params.getActInertia(inertia);
    int localmin = 0;
    double out_u0[3], out_mL[NUM_ACT_SET][3], out_nL[NUM_ACT_SET][3], out_tau[NUM_ACT_SET][3], ftip_calc[3], ftip_guess[3] = {0,0,0};
    double xf_new[NUM_STATES], x_coil[NUM_ACT_SET][NUM_COIL_STATES], ReportedMarkerPos[5][3];

    for (int substep = 0; substep < num_substeps; ++substep) {
        double zero3[3] = {0,0,0};
        // Use dt_sub for BVP to be consistent with the physics we are about to integrate
        CRMShootingMethodParams bp = CRMDYNConstructShootingMethodParamSet(*cp, params.getConfig(), ins_len, Act, ContactModeType::FREE_TIP, zero3, zero3, istep, inertia, v_L, w_L, p_L, R_L, damp, dt_sub);
        bp.dynamics.integrator_type = IntegratorType::RK4; 

        // Damping compensation
        double mL_bvp_guess[NUM_ACT_SET][3], nL_bvp_guess[NUM_ACT_SET][3];
        for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
            for (int i = 0; i < 3; i++) {
                mL_bvp_guess[j][i] = mL_guess_local[j][i] + damp[j][i + 3] * w_L[j][i];
                nL_bvp_guess[j][i] = nL_guess_local[j][i] + damp[j][i] * v_L[j][i];
            }
        }

        DynamicsBVP(bp, xf_local, mL_bvp_guess, nL_bvp_guess, ftip_guess, out_u0, out_mL, out_nL, out_tau, ftip_calc, localmin);

        if (localmin != 0) {
            // Recovery: Velocity continuation
            constexpr int kContSteps = 5; bool ok = true;
            for (int s = 0; s <= kContSteps; s++) {
                double alpha = (double)s/kContSteps;
                double v_r[NUM_ACT_SET][3], w_r[NUM_ACT_SET][3];
                for (int j=0; j<NUM_ACT_SET; j++) for(int i=0; i<3; i++) { v_r[j][i] = alpha*v_L[j][i]; w_r[j][i] = alpha*w_L[j][i]; } 
                
                CRMShootingMethodParams bpr = CRMDYNConstructShootingMethodParamSet(*cp, params.getConfig(), ins_len, Act, ContactModeType::FREE_TIP, zero3, zero3, istep, inertia, v_r, w_r, p_L, R_L, damp, dt_sub);
                bpr.dynamics.integrator_type = IntegratorType::RK4;
                double mG[NUM_ACT_SET][3], nG[NUM_ACT_SET][3];
                for (int j=0; j<num_sets; j++) for(int i=0; i<3; i++) { mG[j][i] = (s>0)?out_mL[j][i]:mL_bvp_guess[j][i]; nG[j][i] = (s>0)?out_nL[j][i]:nL_bvp_guess[j][i]; }
                DynamicsBVP(bpr, xf_local, mG, nG, ftip_guess, out_u0, out_mL, out_nL, out_tau, ftip_calc, localmin);
                if (localmin != 0) { ok = false; break; }
            }
            if (ok) localmin = 0;
            else {
                // Recovery: Current ramp
                double mG[NUM_ACT_SET][3], nG[NUM_ACT_SET][3];
                for (int j=0; j<num_sets; j++) for(int i=0; i<3; i++) { mG[j][i] = mL_bvp_guess[j][i]; nG[j][i] = nL_bvp_guess[j][i]; }
                for (int s = 1; s <= kContSteps; s++) {
                    double alpha = (double)s/kContSteps; double cur[NUM_ACT_SET][3] = {}; for (int i=0; i<3; i++) cur[0][i] = alpha*Act[0][i];
                    double v_stat[NUM_ACT_SET][3] = {}, w_stat[NUM_ACT_SET][3] = {};
                    
                    // CRITICAL FIX: Use dt_local (20ms) for this Static Ramp to emulate Wrapper behavior
                    CRMShootingMethodParams bpc = CRMDYNConstructShootingMethodParamSet(*cp, params.getConfig(), ins_len, cur, ContactModeType::FREE_TIP, zero3, zero3, istep, inertia, v_stat, w_stat, p_L, R_L, damp, dt_local);
                    bpc.dynamics.integrator_type = IntegratorType::RK4;
                    
                    DynamicsBVP(bpc, xf_local, mG, nG, ftip_guess, out_u0, out_mL, out_nL, out_tau, ftip_calc, localmin);
                    if (localmin != 0) break;
                    for (int j=0; j<num_sets; j++) for(int i=0; i<3; i++) { mG[j][i] = out_mL[j][i]; nG[j][i] = out_nL[j][i]; }
                    if (s == kContSteps) ok = true;
                }
                if (ok) localmin = 0;
            }
        }

        if (localmin != 0) break; // Latch

        // IVP Integration (RK4 at dt_sub)
        CRMShootingMethodParams p_ivp = CRMDYNConstructShootingMethodParamSet(*cp, params.getConfig(), ins_len, Act, ContactModeType::FREE_TIP, zero3, zero3, istep, inertia, v_L, w_L, p_L, R_L, damp, dt_sub);
        p_ivp.dynamics.integrator_type = IntegratorType::RK4;
        DYNSolverIVP(p_ivp, out_u0, out_mL, out_nL, out_tau, ftip_calc, true, xf_new, x_coil, ReportedMarkerPos);

        // Update state
        for (int j = 0; j < num_sets; j++) {
            for (int i = 0; i < 3; i++) {
                v_L[j][i] = x_coil[j][i]; w_L[j][i] = x_coil[j][i+3]; p_L[j][i] = x_coil[j][i+6];
                mL_guess_local[j][i] = out_mL[j][i]; nL_guess_local[j][i] = out_nL[j][i];
            }
            for (int i = 0; i < 9; i++) R_L[j][i] = x_coil[j][i+9];
        }
        for (int i = 0; i < NUM_STATES; i++) xf_local[i] = xf_new[i];
    }

    auto opt = torch::TensorOptions().dtype(torch::kFloat64);
    if (localmin == 0) {
        torch::Tensor ns = torch::zeros({output_dim}, opt); auto ns_a = ns.accessor<double, 1>();
        ns_a[0] = xf_local[0]; ns_a[1] = xf_local[1]; ns_a[2] = xf_local[2];
        for (int j=0; j<num_sets; j++) for(int i=0; i<3; i++) ns_a[3+j*3+i] = v_L[j][i]; 

        torch::Tensor nv = torch::zeros({num_sets, 3}, opt); auto nv_a = nv.accessor<double, 2>();
        for (int j=0; j<num_sets; j++) for(int i=0; i<3; i++) nv_a[j][i] = v_L[j][i];
        torch::Tensor nw = torch::zeros({num_sets, 3}, opt); auto nw_a = nw.accessor<double, 2>();
        for (int j=0; j<num_sets; j++) for(int i=0; i<3; i++) nw_a[j][i] = w_L[j][i];
        torch::Tensor np = torch::zeros({num_sets, 3}, opt); auto np_a = np.accessor<double, 2>();
        for (int j=0; j<num_sets; j++) for(int i=0; i<3; i++) np_a[j][i] = p_L[j][i];
        torch::Tensor nR = torch::zeros({num_sets, 9}, opt); auto nR_a = nR.accessor<double, 2>();
        for (int j=0; j<num_sets; j++) { double rt[9]; for(int i=0; i<9; i++) rt[i] = R_L[j][i]; gram_schmidt_orthonormalize(rt); for(int i=0; i<9; i++) nR_a[j][i] = rt[i]; }
        
        torch::Tensor nxf = torch::zeros({15}, opt); auto nxf_a = nxf.accessor<double, 1>();
        for (int i=0; i<15; i++) nxf_a[i] = xf_local[i];
        double rtt[9]; for(int i=0; i<9; i++) rtt[i] = xf_local[i+3]; gram_schmidt_orthonormalize(rtt); for(int i=0; i<9; i++) nxf_a[i+3] = rtt[i];
        
        torch::Tensor nmL = torch::zeros({num_sets, 3}, opt); auto nmL_a = nmL.accessor<double, 2>();
        for (int j=0; j<num_sets; j++) for(int i=0; i<3; i++) nmL_a[j][i] = mL_guess_local[j][i];
        torch::Tensor nnL = torch::zeros({num_sets, 3}, opt); auto nnL_a = nnL.accessor<double, 2>();
        for (int j=0; j<num_sets; j++) for(int i=0; i<3; i++) nnL_a[j][i] = nL_guess_local[j][i];

        return {ns, nv, nw, np, nR, nxf, nmL, nnL, torch::tensor({0.0}, opt)};
    } else {
        torch::Tensor ns = torch::zeros({output_dim}, opt); auto ns_a = ns.accessor<double, 1>();
        ns_a[0] = xf_local[0]; ns_a[1] = xf_local[1]; ns_a[2] = xf_local[2];
        return {ns, seed_v.clone(), seed_w.clone(), seed_p.clone(), seed_R.clone(), seed_xf.clone(), seed_mL.clone(), seed_nL.clone(), torch::tensor({(double)localmin}, opt)};
    }
}

std::vector<torch::Tensor> crm_initialize_from_fk(torch::Tensor currents, torch::Tensor insertion_length) {
    TORCH_CHECK(currents.dim() == 1 && currents.size(0) == 3, "currents must be [3]");
    currents = currents.contiguous().cpu(); insertion_length = insertion_length.contiguous().cpu();
    auto c_acc = currents.accessor<double, 1>(), i_acc = insertion_length.accessor<double, 1>();
    double c_arr[3] = {c_acc[0], c_acc[1], c_acc[2]}, ins = i_acc[0];
    CRMParams& params = CRMParams::getInstance(); const CRMCatheterModelParams* cp = params.getParams();
    CRMForwardKinematicsData fkp; fkp.CathParams = const_cast<CRMCatheterModelParams*>(cp); fkp.CathConfig = const_cast<CatheterConfiguration*>(&params.getConfig());
    fkp.ContactMode = ContactModeType::FREE_TIP; fkp.IntegrationStepSize = params.getIntegrationStepSize(); fkp.FinalValueOnly = true;
    for(int i=0; i<3; i++) { fkp.deltau0_initialguess[i] = 0.0; fkp.ftip_initialguess[i] = 0.0; fkp.TipForce[i] = 0.0; fkp.TipConstraintPoint[i] = 0.0; }
    std::vector<double> mp(cp->no_locmarkers * 3), co(cp->no_act_set * 9), cp_data(cp->no_act_set * 3);
    fkp.ReportedMarkerPos = reinterpret_cast<double(*)[3]>(mp.data()); fkp.ReportedCoilOrient = reinterpret_cast<double(*)[9]>(co.data()); fkp.ReportedCoilPos = reinterpret_cast<double(*)[3]>(cp_data.data());
    double in_x[4] = {c_arr[0], c_arr[1], c_arr[2], ins}, out_y[15], pe;
    int localmin = CRM_ForwardKinematics(in_x, out_y, pe, fkp);
    
    if (localmin != 0) {
        std::cerr << "Warning: FK for requested currents failed (localmin=" << localmin << "). Retrying with ZERO currents for robustness..." << std::endl;
        // Zero-Current Retry: Solve for a straight rod at the target length
        double zero_x[4] = {0.0, 0.0, 0.0, ins};
        localmin = CRM_ForwardKinematics(zero_x, out_y, pe, fkp);
        
        if (localmin != 0) {
             std::cerr << "Critical Error: Even ZERO current FK failed (localmin=" << localmin << "). The model might be physically invalid at this length." << std::endl;
        } else {
             std::cerr << "Success: Initialized with ZERO current (straight rod). Transition will occur during the first step." << std::endl;
        }
    }
    auto opt = torch::TensorOptions().dtype(torch::kFloat64); int ns = cp->no_act_set;
    torch::Tensor p = torch::zeros({ns, 3}, opt); auto p_acc = p.accessor<double, 2>();
    for (int j = 0; j < ns; j++) for (int i = 0; i < 3; i++) p_acc[j][i] = fkp.ReportedCoilPos[j][i];
    torch::Tensor R = torch::zeros({ns, 9}, opt); auto R_acc = R.accessor<double, 2>();
    for (int j = 0; j < ns; j++) { double rt[9]; for(int i=0; i<9; i++) rt[i] = fkp.ReportedCoilOrient[j][i]; gram_schmidt_orthonormalize(rt); for(int i=0; i<9; i++) R_acc[j][i] = rt[i]; }
    torch::Tensor xf = torch::zeros({15}, opt); auto xf_acc = xf.accessor<double, 1>();
    for (int i = 0; i < 3; i++) xf_acc[i] = out_y[i];
    double rtt[9]; for(int i=0; i<9; i++) rtt[i] = out_y[3+i]; gram_schmidt_orthonormalize(rtt); for(int i=0; i<9; i++) xf_acc[3+i] = rtt[i];
    for (int i = 0; i < 3; i++) xf_acc[12+i] = out_y[12+i];
    return {torch::zeros({ns, 3}, opt), torch::zeros({ns, 3}, opt), p, R, xf, torch::zeros({ns, 3}, opt), torch::zeros({ns, 3}, opt)};
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> compute_implicit_jacobians(
    const Eigen::Vector3d& currents, double insertion_length, int num_sets,
    const double v_L[NUM_ACT_SET][3], const double w_L[NUM_ACT_SET][3], const double p_L[NUM_ACT_SET][3], const double R_L[NUM_ACT_SET][9],
    const double xf[NUM_STATES], const double mL_star[NUM_ACT_SET][3], const double nL_star[NUM_ACT_SET][3],
    const double mL_seed[NUM_ACT_SET][3], const double nL_seed[NUM_ACT_SET][3],
    const CRMCatheterModelParams* cparams, const CatheterConfiguration& config,
    double dt, double integration_step_size, IntegratorType integrator_type,
    const double damping[NUM_ACT_SET][6], const double actInertia[NUM_ACT_SET][9]
) {
    const double scale_m = IVALUE_SCALE_M, scale_n = IVALUE_SCALE_N; int x_dim = num_sets * 6;
    Eigen::VectorXd x_star_scaled(x_dim);
    for (int j = 0; j < num_sets; j++) for (int i = 0; i < 3; i++) { x_star_scaled(j*6+i) = mL_star[j][i]/scale_m; x_star_scaled(j*6+3+i) = nL_star[j][i]/scale_n; }
    double Act[NUM_ACT_SET][3] = {}; for (int i = 0; i < 3; i++) Act[0][i] = currents(i);
    double v_L_l[NUM_ACT_SET][3], w_L_l[NUM_ACT_SET][3], p_L_l[NUM_ACT_SET][3], R_L_l[NUM_ACT_SET][9], damp_l[NUM_ACT_SET][6], inertia_l[NUM_ACT_SET][9], zero3[3] = {0,0,0};
    for (int j = 0; j < NUM_ACT_SET; j++) { for (int i = 0; i < 3; i++) { v_L_l[j][i] = v_L[j][i]; w_L_l[j][i] = w_L[j][i]; p_L_l[j][i] = p_L[j][i]; } for (int i = 0; i < 9; i++) { R_L_l[j][i] = R_L[j][i]; inertia_l[j][i] = actInertia[j][i]; } for (int i = 0; i < 6; i++) damp_l[j][i] = damping[j][i]; }
    CRMShootingMethodParams bp = CRMDYNConstructShootingMethodParamSet(*cparams, config, insertion_length, Act, ContactModeType::FREE_TIP, zero3, zero3, integration_step_size, inertia_l, v_L_l, w_L_l, p_L_l, R_L_l, damp_l, dt);
    bp.dynamics.integrator_type = integrator_type;
    DYNNLEqnParams p(bp.no_flex_seg, bp.no_rigid_seg, bp.no_act_set, bp.no_locmarkers, bp.no_fcum_steps);
    double x0[NUM_STATES]; for(int i=0; i<3; i++) x0[i]=bp.p0[i]; for(int i=0; i<9; i++) x0[3+i]=bp.R0[i]; for(int i=12; i<15; i++) x0[i]=0;
    double mG[NUM_ACT_SET][3] = {}, nG[NUM_ACT_SET][3] = {}; for (int j=0; j<num_sets; j++) for(int i=0; i<3; i++){ mG[j][i]=mL_seed[j][i]; nG[j][i]=nL_seed[j][i]; }
    CRMDYNSolverIVP_Prep(bp.no_flex_seg, bp.no_rigid_seg, bp.no_act_set, bp.no_locmarkers, bp.no_fcum_steps, x0, bp.IntegrationStepSize, bp.Li, bp.dlambdainv, bp.rho, bp.SegmentTypes, bp.SegEndLambdas, bp.LocMarkerLambdas, bp.K, bp.Kinv, bp.ustar, bp.MagMoment, bp.fcumlambda, bp.CoilAlignmentTurnAreaMatrix, bp.B0, bp.g, bp.ActMass, inertia_l, damp_l, bp.dynamics.DELTA_T, v_L_l, w_L_l, p_L_l, R_L_l, mG, nG, true, p);
    for (int i = 0; i < NUM_STATES; i++) p.xf[i] = xf[i];
    Eigen::MatrixXd J_xx; try { Eigen::VectorXd r; J_xx = CRMCatheterModel::DYNNLEquationJacobianEigenAD(x_star_scaled, p, &r); }
    catch (...) {
        J_xx.resize(x_dim, x_dim); J_xx.setZero(); const double eps = 1e-5;
        auto eval = [&](const Eigen::VectorXd& xs) {
            double mLp[NUM_ACT_SET][3] = {}, nLp[NUM_ACT_SET][3] = {}; for (int j=0; j<num_sets; j++) for (int i=0; i<3; i++) { mLp[j][i] = xs(j*6+i)*scale_m; nLp[j][i] = xs(j*6+3+i)*scale_n; }
            DYNNLEqnParams pp = p; CRMDYNSolverIVP_Prep(bp.no_flex_seg, bp.no_rigid_seg, bp.no_act_set, bp.no_locmarkers, bp.no_fcum_steps, x0, bp.IntegrationStepSize, bp.Li, bp.dlambdainv, bp.rho, bp.SegmentTypes, bp.SegEndLambdas, bp.LocMarkerLambdas, bp.K, bp.Kinv, bp.ustar, bp.MagMoment, bp.fcumlambda, bp.CoilAlignmentTurnAreaMatrix, bp.B0, bp.g, bp.ActMass, inertia_l, damp_l, bp.dynamics.DELTA_T, v_L_l, w_L_l, p_L_l, R_L_l, mG, nG, true, pp);
            for(int i=0; i<NUM_STATES; i++) pp.xf[i]=xf[i]; std::vector<double> xa(x_dim), out(x_dim); double u0[3], t[NUM_ACT_SET*3]; for(int i=0; i<x_dim; i++) xa[i]=xs(i); DYNNLEquation(xa.data(), out.data(), pp, u0, t);
            Eigen::VectorXd res(x_dim); for(int i=0; i<x_dim; i++) res(i)=out[i]; return res;
        };
        for (int j=0; j<x_dim; j++) { Eigen::VectorXd xp=x_star_scaled, xm=x_star_scaled; xp(j)+=eps; xm(j)-=eps; J_xx.col(j) = (eval(xp)-eval(xm))*(0.5/eps); }
    }
    Eigen::VectorXd ctrl(4); ctrl << currents(0), currents(1), currents(2), insertion_length;
    Eigen::MatrixXd J_xu; try { J_xu = CRMCatheterModel::DYNNLEquationControlJacobianEigenAD(x_star_scaled, ctrl, p, nullptr); }
    catch (...) {
        J_xu.resize(x_dim, 4); J_xu.setZero(); const double eps = 1e-5;
        auto eval = [&](const Eigen::VectorXd& xs, const Eigen::Vector3d& c, double in) {
            double A[NUM_ACT_SET][3] = {}; for(int i=0; i<3; i++) A[0][i]=c(i);
            CRMShootingMethodParams bp_ = CRMDYNConstructShootingMethodParamSet(*cparams, config, in, A, ContactModeType::FREE_TIP, zero3, zero3, integration_step_size, inertia_l, v_L_l, w_L_l, p_L_l, R_L_l, damp_l, dt);
            bp_.dynamics.integrator_type = integrator_type; DYNNLEqnParams pp(bp_.no_flex_seg, bp_.no_rigid_seg, bp_.no_act_set, bp_.no_locmarkers, bp_.no_fcum_steps);
            double x0_[NUM_STATES]; for(int i=0; i<3; i++) x0_[i]=bp_.p0[i]; for(int i=0; i<9; i++) x0_[3+i]=bp_.R0[i]; for(int i=12; i<15; i++) x0_[i]=0;
            CRMDYNSolverIVP_Prep(bp_.no_flex_seg, bp_.no_rigid_seg, bp_.no_act_set, bp_.no_locmarkers, bp_.no_fcum_steps, x0_, bp_.IntegrationStepSize, bp_.Li, bp_.dlambdainv, bp_.rho, bp_.SegmentTypes, bp_.SegEndLambdas, bp_.LocMarkerLambdas, bp_.K, bp_.Kinv, bp_.ustar, bp_.MagMoment, bp_.fcumlambda, bp_.CoilAlignmentTurnAreaMatrix, bp_.B0, bp_.g, bp_.ActMass, inertia_l, damp_l, bp_.dynamics.DELTA_T, v_L_l, w_L_l, p_L_l, R_L_l, mG, nG, true, pp);
            for(int i=0; i<NUM_STATES; i++) pp.xf[i]=xf[i]; std::vector<double> xa(x_dim), out(x_dim); double u0[3], t[NUM_ACT_SET*3]; for(int i=0; i<x_dim; i++) xa[i]=xs(i); DYNNLEquation(xa.data(), out.data(), pp, u0, t);
            Eigen::VectorXd res(x_dim); for(int i=0; i<x_dim; i++) res(i)=out[i]; return res;
        };
        for(int j=0; j<3; j++) { Eigen::Vector3d cp=currents, cm=currents; cp(j)+=eps; cm(j)-=eps; J_xu.col(j) = (eval(x_star_scaled, cp, insertion_length)-eval(x_star_scaled, cm, insertion_length))*(0.5/eps); }
        double ip=insertion_length+eps, im=insertion_length-eps; J_xu.col(3) = (eval(x_star_scaled, currents, ip)-eval(x_star_scaled, currents, im))*(0.5/eps);
    }
    Eigen::MatrixXd dx_du = J_xx.colPivHouseholderQr().solve(-J_xu);
    int out_dim = 3+3*num_sets, s_dim = num_sets*21 + 15; Eigen::MatrixXd gx(out_dim, x_dim), gth(out_dim, 3+s_dim);
    Eigen::VectorXd sf(num_sets*21+15); int idx=0;
    for(int j=0; j<num_sets; j++) for(int i=0; i<3; i++) sf(idx++) = v_L[j][i];
    for(int j=0; j<num_sets; j++) for(int i=0; i<3; i++) sf(idx++) = w_L[j][i];
    for(int j=0; j<num_sets; j++) for(int i=0; i<3; i++) sf(idx++) = p_L[j][i];
    for(int j=0; j<num_sets; j++) for(int i=0; i<9; i++) sf(idx++) = R_L[j][i];
    for(int i=0; i<15; i++) sf(idx++) = xf[i];
    for(int j=0; j<num_sets; j++) for(int i=0; i<3; i++) sf(idx++) = mL_seed[j][i];
    for(int j=0; j<num_sets; j++) for(int i=0; i<3; i++) sf(idx++) = nL_seed[j][i];
    CRMCatheterModel::dynnl_ad_eigen::DYNNLEquationOutputJacobianEigenAD(x_star_scaled, currents, sf, p, gx, gth);
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(out_dim, 4);
    B.leftCols(3) = gth.leftCols(3) + gx * dx_du.leftCols(3); B.col(3) = gx * dx_du.col(3);
    return {B, Eigen::MatrixXd::Zero(out_dim, 1)};
}

void initialize_params(const std::string& p, const std::string& c) { CRMParams::getInstance().loadFromFiles(p, c); }
void set_timestep(double dt) { CRMParams::getInstance().setDt(dt); }
void set_integrator(const std::string& i) { CRMParams::getInstance().setIntegratorType(i == "rk4" || i == "RK4" ? IntegratorType::RK4 : IntegratorType::ABM4); }
void set_integration_step_size(double s) { CRMParams::getInstance().setIntegrationStepSize(s); }
void set_damping(const std::vector<double>& d) { double da[NUM_ACT_SET][6]; for (int j=0; j<NUM_ACT_SET; j++) for (int i=0; i<6; i++) da[j][i] = d[i]; CRMParams::getInstance().setDamping(da); }

} // namespace crm_torch
