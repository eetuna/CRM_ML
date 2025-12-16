#pragma once

#include <Eigen/Dense>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>

#include <array>
#include <cmath>

#include "CRM.hpp"
#include "CRMDYN.hpp"
#include "CRM_BVPIVP_APIDeclarations.hpp"

namespace CRMCatheterModel {

inline constexpr double kCoilTStep = 0.001;

inline double scalar_value(double x) { return x; }
template <size_t N, typename T>
inline double scalar_value(const autodiff::detail::Real<N, T>& x) { return static_cast<double>(autodiff::val(x)); }

template <typename Scalar>
inline void wHatT(const Scalar in_w[3], Scalar out_what[9])
{
    out_what[0] = Scalar(0);
    out_what[1] = -in_w[2];
    out_what[2] = in_w[1];
    out_what[3] = in_w[2];
    out_what[4] = Scalar(0);
    out_what[5] = -in_w[0];
    out_what[6] = -in_w[1];
    out_what[7] = in_w[0];
    out_what[8] = Scalar(0);
}

template <typename Scalar>
inline void RodriguesExpandedT(const Scalar in_w[3], const Scalar in_theta, Scalar out_R[9])
{
    // This matches the double implementation in src/CRM_IVPSolver.cpp.
    Scalar w_hat[9];
    wHatT(in_w, w_hat);

    Scalar w_hat_sq[9];
    mMult_AB<3, 3, 3>(w_hat, w_hat, w_hat_sq);

    const Scalar s = sin(in_theta);
    const Scalar c = cos(in_theta);

    // R = I + sin(theta) * w_hat + (1-cos(theta)) * w_hat^2
    out_R[0] = Scalar(1) + s * w_hat[0] + (Scalar(1) - c) * w_hat_sq[0];
    out_R[1] = Scalar(0) + s * w_hat[1] + (Scalar(1) - c) * w_hat_sq[1];
    out_R[2] = Scalar(0) + s * w_hat[2] + (Scalar(1) - c) * w_hat_sq[2];
    out_R[3] = Scalar(0) + s * w_hat[3] + (Scalar(1) - c) * w_hat_sq[3];
    out_R[4] = Scalar(1) + s * w_hat[4] + (Scalar(1) - c) * w_hat_sq[4];
    out_R[5] = Scalar(0) + s * w_hat[5] + (Scalar(1) - c) * w_hat_sq[5];
    out_R[6] = Scalar(0) + s * w_hat[6] + (Scalar(1) - c) * w_hat_sq[6];
    out_R[7] = Scalar(0) + s * w_hat[7] + (Scalar(1) - c) * w_hat_sq[7];
    out_R[8] = Scalar(1) + s * w_hat[8] + (Scalar(1) - c) * w_hat_sq[8];
}

template <typename Scalar>
inline void SE3_Analytical_StepT(
    const Scalar in_R_n[9], const Scalar in_p_n[3], const Scalar in_u_n[3], const Scalar h, Scalar out_R_np1[9], Scalar out_p_np1[3])
{
    // Matches SE3_Analytical_Step(double...) structure, but templated.
    Scalar u[3] = {in_u_n[0], in_u_n[1], in_u_n[2]};
    const Scalar umagsq = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];

    // Always use the Rodrigues form; avoid branching on AD types.
    const Scalar umag = sqrt(umagsq + Scalar(1e-24));
    const Scalar umag_inv = Scalar(1) / umag;
    Scalar unorm[3] = {u[0] * umag_inv, u[1] * umag_inv, u[2] * umag_inv};

    const Scalar theta = h * umag;
    Scalar Rdelta[9];
    RodriguesExpandedT(unorm, theta, Rdelta);
    mMult_AB<3, 3, 3>(in_R_n, Rdelta, out_R_np1);

    // p_{n+1} = p_n + R_n * V(theta) * e3 * h (analytic screw step)
    // This matches the original implementation in src/CRM_IVPSolver.cpp.
    const Scalar s = sin(theta);
    const Scalar c = cos(theta);

    // Compute pdelta = (I - Rdelta) * (w x v) / ||w||^2 + w w^T v * h / ||w||^2
    // Here v = e3 (0,0,1), w = u.
    Scalar e3[3] = {Scalar(0), Scalar(0), Scalar(1)};
    Scalar w_hat[9];
    wHatT(u, w_hat);
    Scalar wxv[3];
    mMult_AB<3, 3, 1>(w_hat, e3, wxv);

    Scalar ImRdelta[9];
    ImRdelta[0] = Scalar(1) - Rdelta[0];
    ImRdelta[1] = -Rdelta[1];
    ImRdelta[2] = -Rdelta[2];
    ImRdelta[3] = -Rdelta[3];
    ImRdelta[4] = Scalar(1) - Rdelta[4];
    ImRdelta[5] = -Rdelta[5];
    ImRdelta[6] = -Rdelta[6];
    ImRdelta[7] = -Rdelta[7];
    ImRdelta[8] = Scalar(1) - Rdelta[8];

    Scalar ImRdeltawxv[3];
    mMult_AB<3, 3, 1>(ImRdelta, wxv, ImRdeltawxv);

    Scalar wwTv[3];
    wwTv[0] = u[0] * (u[0] * e3[0] + u[1] * e3[1] + u[2] * e3[2]);
    wwTv[1] = u[1] * (u[0] * e3[0] + u[1] * e3[1] + u[2] * e3[2]);
    wwTv[2] = u[2] * (u[0] * e3[0] + u[1] * e3[1] + u[2] * e3[2]);

    Scalar ImRuxvpuuTvds[3];
    for (int i = 0; i < 3; ++i) {
        ImRuxvpuuTvds[i] = ImRdeltawxv[i] + wwTv[i] * h;
    }

    const Scalar umagsq_inv = Scalar(1) / (umagsq + Scalar(1e-24));
    Scalar pdelta[3];
    for (int i = 0; i < 3; ++i) pdelta[i] = ImRuxvpuuTvds[i] * umagsq_inv;

    Scalar Rnpd[3];
    mMult_AB<3, 3, 1>(in_R_n, pdelta, Rnpd);
    for (int i = 0; i < 3; ++i) out_p_np1[i] = in_p_n[i] + Rnpd[i];
}

template <typename Scalar>
inline void CoilIntegradT(
    const Scalar in_twist[6],
    const Scalar in_n[3],
    const double g[3],
    const Scalar R[9],
    double actMass,
    const double actInertia[9],
    const double damping[6],
    const double in_B0[3],
    const double in_muhat[9],
    const Scalar in_mL[3],
    Scalar twistdot[6])
{
    Scalar RTg[3];
    mMult_ATB<3, 3, 1>(R, g, RTg);

    Scalar v[3] = {in_twist[0], in_twist[1], in_twist[2]};
    Scalar w[3] = {in_twist[3], in_twist[4], in_twist[5]};

    Scalar w_hat[9];
    wHatT(w, w_hat);
    Scalar w_v[3];
    mMult_AB<3, 3, 1>(w_hat, v, w_v);

    Scalar vdot[3];
    for (int i = 0; i < 3; ++i) {
        const Scalar damping_vec = Scalar(damping[i]) * v[i];
        vdot[i] = RTg[i] - in_n[i] / Scalar(actMass) - w_v[i] - damping_vec;
    }

    Scalar inertiaw[3];
    mMult_AB<3, 3, 1>(actInertia, w, inertiaw);
    Scalar w_inertia_w[3];
    mMult_AB<3, 3, 1>(w_hat, inertiaw, w_inertia_w);

    Scalar RscTB0[3];
    mMult_ATB<3, 3, 1>(R, in_B0, RscTB0);

    Scalar Tb[3];
    mMult_AB<3, 3, 1>(in_muhat, RscTB0, Tb);

    Scalar tau[3];
    mSub_AB<3, 1>(Tb, in_mL, tau);

    Scalar residual_w[3];
    for (int i = 0; i < 3; ++i) {
        const Scalar damping_w = Scalar(damping[i + 3]) * w[i];
        residual_w[i] = tau[i] - w_inertia_w[i] - damping_w;
    }

    Scalar wdot[3];
    wdot[0] = residual_w[0] / Scalar(actInertia[0]);
    wdot[1] = residual_w[1] / Scalar(actInertia[4]);
    wdot[2] = residual_w[2] / Scalar(actInertia[8]);

    for (int i = 0; i < 3; ++i) {
        twistdot[i] = vdot[i];
        twistdot[i + 3] = wdot[i];
    }
}

template <typename Scalar>
inline void DYNSE3_TimeSpaceT(
    const Scalar in_R_n[9],
    const Scalar in_p_n[3],
    const Scalar h,
    const Scalar in_twist_n[6],
    Scalar out_R_np1[9],
    Scalar out_p_np1[3])
{
    // Use the same closed-form as SE3_Analytical_StepT with u := w and v := v.
    const Scalar v_n[3] = {in_twist_n[0], in_twist_n[1], in_twist_n[2]};
    const Scalar w_n[3] = {in_twist_n[3], in_twist_n[4], in_twist_n[5]};

    const Scalar umagsq = w_n[0] * w_n[0] + w_n[1] * w_n[1] + w_n[2] * w_n[2];
    const Scalar umag = sqrt(umagsq + Scalar(1e-24));
    const Scalar umag_inv = Scalar(1) / umag;
    Scalar unorm[3] = {w_n[0] * umag_inv, w_n[1] * umag_inv, w_n[2] * umag_inv};
    const Scalar theta = h * umag;

    Scalar Rdelta[9];
    RodriguesExpandedT(unorm, theta, Rdelta);
    mMult_AB<3, 3, 3>(in_R_n, Rdelta, out_R_np1);

    Scalar what[9];
    wHatT(w_n, what);
    Scalar wxv[3];
    mMult_AB<3, 3, 1>(what, v_n, wxv);

    Scalar ImRdelta[9];
    ImRdelta[0] = Scalar(1) - Rdelta[0];
    ImRdelta[1] = -Rdelta[1];
    ImRdelta[2] = -Rdelta[2];
    ImRdelta[3] = -Rdelta[3];
    ImRdelta[4] = Scalar(1) - Rdelta[4];
    ImRdelta[5] = -Rdelta[5];
    ImRdelta[6] = -Rdelta[6];
    ImRdelta[7] = -Rdelta[7];
    ImRdelta[8] = Scalar(1) - Rdelta[8];

    Scalar ImRdeltawxv[3];
    mMult_AB<3, 3, 1>(ImRdelta, wxv, ImRdeltawxv);

    Scalar wwTv[3];
    wwTv[0] = w_n[0] * (w_n[0] * v_n[0] + w_n[1] * v_n[1] + w_n[2] * v_n[2]);
    wwTv[1] = w_n[1] * (w_n[0] * v_n[0] + w_n[1] * v_n[1] + w_n[2] * v_n[2]);
    wwTv[2] = w_n[2] * (w_n[0] * v_n[0] + w_n[1] * v_n[1] + w_n[2] * v_n[2]);

    Scalar ImRuxvpuuTvds[3];
    for (int i = 0; i < 3; ++i) {
        ImRuxvpuuTvds[i] = ImRdeltawxv[i] + wwTv[i] * h;
    }

    const Scalar umagsq_inv = Scalar(1) / (umagsq + Scalar(1e-24));
    Scalar pdelta[3];
    for (int i = 0; i < 3; ++i) pdelta[i] = ImRuxvpuuTvds[i] * umagsq_inv;

    Scalar Rnpd[3];
    mMult_AB<3, 3, 1>(in_R_n, pdelta, Rnpd);
    for (int i = 0; i < 3; ++i) out_p_np1[i] = in_p_n[i] + Rnpd[i];
}

template <typename Scalar>
inline void RK2_coildynT(
    const Scalar in_x_n[NUM_COIL_STATES],
    const Scalar in_n[3],
    const double g[3],
    double actMass,
    const double actInertia[9],
    const double damping[6],
    const double in_B0[3],
    const double in_muhat[9],
    const Scalar in_mL[3],
    Scalar out_x_np1[NUM_COIL_STATES],
    Scalar out_xdot_n[6])
{
    Scalar R_n[9], p_n[3], twist_n[6];
    for (int i = 0; i < 3; ++i) p_n[i] = in_x_n[i + 6];
    for (int i = 0; i < 9; ++i) R_n[i] = in_x_n[i + 9];
    for (int i = 0; i < 6; ++i) twist_n[i] = in_x_n[i];

    Scalar k1[6];
    Scalar x_n_p_k1o2[6];
	    CoilIntegradT(twist_n, in_n, g, R_n, actMass, actInertia, damping, in_B0, in_muhat, in_mL, out_xdot_n);
	    for (int i = 0; i < 6; ++i) {
	        k1[i] = Scalar(kCoilTStep) * out_xdot_n[i];
	        x_n_p_k1o2[i] = twist_n[i] + k1[i] * Scalar(0.5);
	    }

	    Scalar R_half[9], p_half[3];
	    DYNSE3_TimeSpaceT(R_n, p_n, Scalar(kCoilTStep) * Scalar(0.5), twist_n, R_half, p_half);

    Scalar xdot_half[6];
    CoilIntegradT(x_n_p_k1o2, in_n, g, R_half, actMass, actInertia, damping, in_B0, in_muhat, in_mL, xdot_half);

	    Scalar R_np1[9], p_np1[3];
	    DYNSE3_TimeSpaceT(R_n, p_n, Scalar(kCoilTStep), x_n_p_k1o2, R_np1, p_np1);

	    for (int i = 0; i < 6; ++i) out_x_np1[i] = twist_n[i] + Scalar(kCoilTStep) * xdot_half[i];
	    for (int i = 0; i < 3; ++i) out_x_np1[i + 6] = p_np1[i];
	    for (int i = 0; i < 9; ++i) out_x_np1[i + 9] = R_np1[i];
	}

template <typename Scalar>
inline void ABM4_coildynT(
    const Scalar in_x_n[NUM_COIL_STATES],
    const Scalar in_xdot_nm1[6],
    const Scalar in_xdot_nm2[6],
    const Scalar in_xdot_nm3[6],
    const Scalar in_x_nm1[NUM_COIL_STATES],
    const Scalar in_x_nm2[NUM_COIL_STATES],
    const Scalar in_x_nm3[NUM_COIL_STATES],
    const Scalar in_n[3],
    const double g[3],
    double actMass,
    const double actInertia[9],
    const double damping[6],
    const double in_B0[3],
    const double in_muhat[9],
    const Scalar in_mL[3],
    Scalar out_x_np1[NUM_COIL_STATES],
    Scalar out_xdot_n[6])
{
    constexpr double P_COEFF_N = 55.0 / 24.0, P_COEFF_Nm1 = -59.0 / 24.0, P_COEFF_Nm2 = 37.0 / 24.0, P_COEFF_Nm3 = -9.0 / 24.0;
    constexpr double C_COEFF_Np1 = 9.0 / 24.0, C_COEFF_N = 19.0 / 24.0, C_COEFF_Nm1 = -5.0 / 24.0, C_COEFF_Nm2 = 1.0 / 24.0;

    Scalar twist_n[6];
    for (int i = 0; i < 6; ++i) twist_n[i] = in_x_n[i];

    CoilIntegradT(twist_n, in_n, g, in_x_n + 9, actMass, actInertia, damping, in_B0, in_muhat, in_mL, out_xdot_n);

	    Scalar twist_hat[6];
	    for (int i = 0; i < 6; ++i) {
	        twist_hat[i] = twist_n[i]
	            + Scalar(kCoilTStep) * (Scalar(P_COEFF_N) * out_xdot_n[i]
	                + Scalar(P_COEFF_Nm1) * in_xdot_nm1[i]
	                + Scalar(P_COEFF_Nm2) * in_xdot_nm2[i]
	                + Scalar(P_COEFF_Nm3) * in_xdot_nm3[i]);
	    }

    Scalar R_hat[9], p_hat[3];
    {
        Scalar u_pred[6];
	        for (int i = 0; i < 6; ++i) {
	            u_pred[i] = Scalar(P_COEFF_N) * in_x_n[i]
	                + Scalar(P_COEFF_Nm1) * in_x_nm1[i]
	                + Scalar(P_COEFF_Nm2) * in_x_nm2[i]
	                + Scalar(P_COEFF_Nm3) * in_x_nm3[i];
	        }
	        DYNSE3_TimeSpaceT(in_x_n + 9, in_x_n + 6, Scalar(kCoilTStep), u_pred, R_hat, p_hat);
	    }

    Scalar xdot_hat[6];
    CoilIntegradT(twist_hat, in_n, g, R_hat, actMass, actInertia, damping, in_B0, in_muhat, in_mL, xdot_hat);

	    Scalar twist_corr[6];
	    for (int i = 0; i < 6; ++i) {
	        twist_corr[i] = twist_n[i]
	            + Scalar(kCoilTStep) * (Scalar(C_COEFF_Np1) * xdot_hat[i]
	                + Scalar(C_COEFF_N) * out_xdot_n[i]
	                + Scalar(C_COEFF_Nm1) * in_xdot_nm1[i]
	                + Scalar(C_COEFF_Nm2) * in_xdot_nm2[i]);
	    }

    Scalar R_np1[9], p_np1[3];
    {
        Scalar u_corr[6];
	        for (int i = 0; i < 6; ++i) {
	            u_corr[i] = Scalar(C_COEFF_Np1) * twist_hat[i]
	                + Scalar(C_COEFF_N) * twist_n[i]
	                + Scalar(C_COEFF_Nm1) * in_x_nm1[i]
	                + Scalar(C_COEFF_Nm2) * in_x_nm2[i];
	        }
	        DYNSE3_TimeSpaceT(in_x_n + 9, in_x_n + 6, Scalar(kCoilTStep), u_corr, R_np1, p_np1);
	    }

    for (int i = 0; i < 6; ++i) out_x_np1[i] = twist_corr[i];
    for (int i = 0; i < 3; ++i) out_x_np1[i + 6] = p_np1[i];
    for (int i = 0; i < 9; ++i) out_x_np1[i + 9] = R_np1[i];
}

template <typename Scalar>
inline void CoilDynamicsT(
    const Scalar in_coil_state[NUM_COIL_STATES],
    const Scalar in_n[3],
    const double g[3],
    double actMass,
    const double actInertia[9],
    const double damping[6],
    double DELTA_T,
    const double in_B0[3],
    const double in_muhat[9],
    const Scalar in_mL[3],
    Scalar out_coil_state[NUM_COIL_STATES],
    Scalar out_xdot_n[6])
{
    Scalar x_nm3[NUM_COIL_STATES]{};
    Scalar x_nm2[NUM_COIL_STATES]{};
    Scalar x_nm1[NUM_COIL_STATES]{};
    Scalar x_n[NUM_COIL_STATES]{};
    Scalar x_np1[NUM_COIL_STATES]{};
    Scalar xdot_nm3[6]{};
    Scalar xdot_nm2[6]{};
    Scalar xdot_nm1[6]{};
    Scalar xdot_n[6]{};

    for (int i = 0; i < NUM_COIL_STATES; i++) x_n[i] = in_coil_state[i];

	    const int N = static_cast<int>(std::ceil(DELTA_T / kCoilTStep));
    for (int idx = 0; idx < N; idx++) {
        if (idx < 3) {
            RK2_coildynT(x_n, in_n, g, actMass, actInertia, damping, in_B0, in_muhat, in_mL, x_np1, xdot_n);
        } else {
            ABM4_coildynT(x_n, xdot_nm1, xdot_nm2, xdot_nm3, x_nm1, x_nm2, x_nm3, in_n, g, actMass, actInertia, damping, in_B0, in_muhat, in_mL, x_np1, xdot_n);
            if (std::isnan(scalar_value(x_n[0]))) {
                // Keep behavior similar to original (warning only).
            }
        }

        for (int i = 0; i < NUM_COIL_STATES; i++) {
            x_nm3[i] = x_nm2[i];
            x_nm2[i] = x_nm1[i];
            x_nm1[i] = x_n[i];
            x_n[i] = x_np1[i];
        }
        for (int i = 0; i < 6; i++) {
            xdot_nm3[i] = xdot_nm2[i];
            xdot_nm2[i] = xdot_nm1[i];
            xdot_nm1[i] = xdot_n[i];
        }
    }

    for (int i = 0; i < NUM_COIL_STATES; i++) out_coil_state[i] = x_n[i];
    for (int i = 0; i < 6; ++i) out_xdot_n[i] = xdot_n[i];
}

template <typename Scalar>
inline void CRMIntegrand_dynT(
    double s,
    const Scalar in_p[3],
    const Scalar in_R[9],
    const Scalar in_u[3],
    const CRMIntegrandParams& in_Params,
    const Scalar in_nL[3],
    Scalar out_udot[3])
{
    const double Length = in_Params.Li;
    const double deltalambdainv = in_Params.dlambdainv;
    const auto& K = in_Params.K;
    const auto& Kinv = in_Params.Kinv;
    const auto& ustar = in_Params.ustar;
    const auto& l = in_Params.l;

    // interpolate fcum (double), then promote to Scalar
    const double lambda = Length - s;
    double ix = lambda * deltalambdainv;
    double ird_f = floor(ix);
    if (ird_f < 0) ird_f = 0;
    int ird = static_cast<int>(ird_f);
    double iru_f = ceil(ix);
    if (iru_f > in_Params.no_fcum_steps) iru_f = in_Params.no_fcum_steps;
    int iru = static_cast<int>(iru_f);
    double ixmird = ix - ird;
    double irumix = iru - ix;

    Scalar fcum[3];
    for (int i = 0; i < 3; i++) {
        fcum[i] = Scalar(in_Params.fcumlambda[iru][i] * ixmird + in_Params.fcumlambda[ird][i] * irumix);
    }
    for (int i = 0; i < 3; i++) fcum[i] = fcum[i] + Scalar(in_Params.ftip[i]);

    Scalar nL_spatial[3];
    mMult_AB<3, 3, 1>(in_R, in_nL, nL_spatial);
    for (int i = 0; i < 3; i++) fcum[i] = fcum[i] + nL_spatial[i];

    Scalar u_hat[9];
    wHatT(in_u, u_hat);

    Scalar e3hatRT[9];
    e3hatRT[0] = -in_R[1]; e3hatRT[1] = -in_R[4]; e3hatRT[2] = -in_R[7];
    e3hatRT[3] = in_R[0];  e3hatRT[4] = in_R[3];  e3hatRT[5] = in_R[6];
    e3hatRT[6] = Scalar(0); e3hatRT[7] = Scalar(0); e3hatRT[8] = Scalar(0);

    Scalar e3hatRTfcum[3];
    mMult_AB<3, 3, 1>(e3hatRT, fcum, e3hatRTfcum);

    Scalar RTl[3];
    mMult_ATB<3, 3, 1>(in_R, l, RTl);

    Scalar umustar[3];
    mSub_AB<3, 1>(in_u, ustar, umustar);

    Scalar Kumustar[3], uhatKumustar[3];
    mMult_AB<3, 3, 1>(K, umustar, Kumustar);
    mMult_AB<3, 3, 1>(u_hat, Kumustar, uhatKumustar);

    Scalar sumterm[3];
    mAdd_ABC<3, 1>(uhatKumustar, e3hatRTfcum, RTl, sumterm);

    Scalar KinvSum[3];
    mMult_AB<3, 3, 1>(Kinv, sumterm, KinvSum);

    Scalar ustardot[3] = {Scalar(0), Scalar(0), Scalar(0)};
    mSub_AB<3, 1>(ustardot, KinvSum, out_udot);
}

	template <typename Scalar>
	inline void CRMFlexible_IVP_Back_T(
	    int SegmentIndex,
	    const Scalar in_p[3],
	    const Scalar in_R[9],
	    const DYNNLEqnParams& in_params,
	    const Scalar in_u[3],
	    const Scalar in_n_L[3],
	    Scalar out_u[3],
	    Scalar out_p[3],
    Scalar out_R[9])
{
    // Minimal ABM4 integration of (p,R,u) using analytic SE3 step, matching CRMFlexible_IVP_Back.
    const int fsegno = SegmentIndex >> 1;
    const double* SegBounds = in_params.SegBounds;
    const int* SegSteps = in_params.SegSteps;
    const double h = -1.0 * (SegBounds[SegmentIndex + 1] - SegBounds[SegmentIndex]) / (SegSteps[fsegno] * 1.0);
    const int N = SegSteps[fsegno];

    // Build integrand params (same as original).
    CRMIntegrandParams IntegrandParams;
    IntegrandParams.dlambdainv = in_params.dlambdainv;
    IntegrandParams.Li = in_params.InsertedLength;
    static double l_zero[3] = {0.0, 0.0, 0.0};
    IntegrandParams.l = l_zero;
	    IntegrandParams.no_fcum_steps = in_params.no_fcum_steps;
	    IntegrandParams.fcumlambda = in_params.fcumlambda;
	    static double ftip_zero[3] = {0.0, 0.0, 0.0};
	    IntegrandParams.ftip = ftip_zero;
	    IntegrandParams.g = const_cast<double*>(in_params.g);
	    IntegrandParams.K = in_params.K[fsegno];
	    IntegrandParams.Kinv = in_params.Kinv[fsegno];
	    IntegrandParams.ustar = in_params.ustar[fsegno];
	    IntegrandParams.rho = in_params.rho[SegmentIndex];

    // State history for ABM4.
    Scalar p_nm3[3], p_nm2[3], p_nm1[3], p_n[3], p_np1[3];
    Scalar R_nm3[9], R_nm2[9], R_nm1[9], R_n[9], R_np1[9];
    Scalar u_nm3[3], u_nm2[3], u_nm1[3], u_n[3], u_np1[3];

    for (int i = 0; i < 3; ++i) {
        p_n[i] = in_p[i];
        u_n[i] = in_u[i];
    }
    for (int i = 0; i < 9; ++i) R_n[i] = in_R[i];

    Scalar udot_nm3[3]{}, udot_nm2[3]{}, udot_nm1[3]{}, udot_n[3]{}, udot_np1_hat[3]{};

    double t_n = SegBounds[SegmentIndex + 1];
    for (int idx = 0; idx < N; idx++) {
        // Compute udot at current.
        CRMIntegrand_dynT(t_n, p_n, R_n, u_n, IntegrandParams, in_n_L, udot_n);

        if (idx < 3) {
            // RK2
            Scalar u_mid[3];
            for (int i = 0; i < 3; ++i) u_mid[i] = u_n[i] + Scalar(0.5) * Scalar(h) * udot_n[i];
            Scalar R_mid[9], p_mid[3];
            SE3_Analytical_StepT(R_n, p_n, u_n, Scalar(h) * Scalar(0.5), R_mid, p_mid);
            Scalar udot_mid[3];
            CRMIntegrand_dynT(t_n + 0.5 * h, p_mid, R_mid, u_mid, IntegrandParams, in_n_L, udot_mid);
            for (int i = 0; i < 3; ++i) u_np1[i] = u_n[i] + Scalar(h) * udot_mid[i];
            SE3_Analytical_StepT(R_n, p_n, u_mid, Scalar(h), R_np1, p_np1);
        } else {
            // ABM4 predictor/corrector on u, analytic SE3 update driven by predicted/corrected u.
            const Scalar P_COEFF_N = Scalar(55.0 / 24.0), P_COEFF_Nm1 = Scalar(-59.0 / 24.0), P_COEFF_Nm2 = Scalar(37.0 / 24.0), P_COEFF_Nm3 = Scalar(-9.0 / 24.0);
            const Scalar C_COEFF_Np1 = Scalar(9.0 / 24.0), C_COEFF_N = Scalar(19.0 / 24.0), C_COEFF_Nm1 = Scalar(-5.0 / 24.0), C_COEFF_Nm2 = Scalar(1.0 / 24.0);

            Scalar u_hat[3];
            for (int i = 0; i < 3; ++i) {
                u_hat[i] = u_n[i] + Scalar(h) * (P_COEFF_N * udot_n[i] + P_COEFF_Nm1 * udot_nm1[i] + P_COEFF_Nm2 * udot_nm2[i] + P_COEFF_Nm3 * udot_nm3[i]);
            }
            Scalar u_pred[3];
            for (int i = 0; i < 3; ++i) {
                u_pred[i] = (P_COEFF_N * u_n[i] + P_COEFF_Nm1 * u_nm1[i] + P_COEFF_Nm2 * u_nm2[i] + P_COEFF_Nm3 * u_nm3[i]);
            }
            Scalar R_hat[9], p_hat[3];
            SE3_Analytical_StepT(R_n, p_n, u_pred, Scalar(h), R_hat, p_hat);

            CRMIntegrand_dynT(t_n + h, p_hat, R_hat, u_hat, IntegrandParams, in_n_L, udot_np1_hat);
            for (int i = 0; i < 3; ++i) {
                u_np1[i] = u_n[i] + Scalar(h) * (C_COEFF_Np1 * udot_np1_hat[i] + C_COEFF_N * udot_n[i] + C_COEFF_Nm1 * udot_nm1[i] + C_COEFF_Nm2 * udot_nm2[i]);
            }
            Scalar u_corr[3];
            for (int i = 0; i < 3; ++i) {
                u_corr[i] = (C_COEFF_Np1 * u_hat[i] + C_COEFF_N * u_n[i] + C_COEFF_Nm1 * u_nm1[i] + C_COEFF_Nm2 * u_nm2[i]);
            }
            SE3_Analytical_StepT(R_n, p_n, u_corr, Scalar(h), R_np1, p_np1);
        }

        t_n = t_n + h;

        // Shift history
        for (int i = 0; i < 3; ++i) {
            p_nm3[i] = p_nm2[i]; p_nm2[i] = p_nm1[i]; p_nm1[i] = p_n[i]; p_n[i] = p_np1[i];
            u_nm3[i] = u_nm2[i]; u_nm2[i] = u_nm1[i]; u_nm1[i] = u_n[i]; u_n[i] = u_np1[i];
            udot_nm3[i] = udot_nm2[i]; udot_nm2[i] = udot_nm1[i]; udot_nm1[i] = udot_n[i];
        }
        for (int i = 0; i < 9; ++i) {
            R_nm3[i] = R_nm2[i]; R_nm2[i] = R_nm1[i]; R_nm1[i] = R_n[i]; R_n[i] = R_np1[i];
        }
    }

    for (int i = 0; i < 3; ++i) {
        out_p[i] = p_n[i];
        out_u[i] = u_n[i];
    }
    for (int i = 0; i < 9; ++i) out_R[i] = R_n[i];
}

template <typename Scalar>
inline autodiff::VectorXreal DYNNLEquationResidualAD(
    const autodiff::VectorXreal& in_x,
    const DYNNLEqnParams* ParamsPtr)
{
    // Current AD residual implementation supports NUM_ACT_SET == 1 (current repo configuration).
    constexpr int NLEq_Dim = NUM_DYN_RESIDUAL;
    autodiff::VectorXreal out_y(NLEq_Dim);

    const DYNNLEqnParams& Params = *ParamsPtr;

    Scalar m_L[NUM_ACT_SET][3], n_L[NUM_ACT_SET][3], n_0[3];
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; i++) {
            m_L[j][i] = Scalar(IVALUE_SCALE_M) * in_x(i + j * 6);
            n_L[j][i] = Scalar(IVALUE_SCALE_N) * in_x(i + j * 6 + 3);
        }
    }
    for (int i = 0; i < 3; i++) n_0[i] = Scalar(Params.TipForce[i]);

    Scalar x_coil[NUM_ACT_SET][NUM_COIL_STATES]{};
    Scalar out_x_coil[NUM_ACT_SET][NUM_COIL_STATES]{};

    double muhat[NUM_ACT_SET][9];
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        double mu[3] = {Params.MagMoment[j][0], Params.MagMoment[j][1], Params.MagMoment[j][2]};
        double muhattemp[9];
        wHat(mu, muhattemp);
        for (int i = 0; i < 9; ++i) muhat[j][i] = muhattemp[i];
    }

    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; ++i) {
            x_coil[j][i] = Scalar(Params.v_L_pre[j][i]);
            x_coil[j][i + 3] = Scalar(Params.w_L_pre[j][i]);
            x_coil[j][i + 6] = Scalar(Params.p_pre[j][i]);
        }
        for (int i = 0; i < 9; ++i) x_coil[j][i + 9] = Scalar(Params.R_pre[j][i]);
    }

    auto& K = Params.K;
    auto& Kinv = Params.Kinv;
    auto& ustar = Params.ustar;
    auto& SegBounds = Params.SegBounds;
    auto& NUM_SEGMENTS = Params.no_segments;

    Scalar net_mL[3]{}, tau[3]{}, K2invResidual[3]{}, u_t[3]{}, du[3]{};
    Scalar tau_0[3] = {Scalar(0), Scalar(0), Scalar(0)};
    Scalar p_t[3]{}, R_t[9]{}, u_tau[3]{}, p_[3]{}, R_[9]{};
    int fsegi = 0, actno = 0, actseg = 0, actno_mn = 0;
    Scalar u_L[3]{}, u_f[3]{}, p_f[3]{}, R_f[9]{};
    Scalar out_xdot[6]{}, residual[NUM_ACT_SET][6]{};

    Scalar p_L[3]{}, R_L[9]{};
    Scalar v1[3]{}, v2[3]{}, v3[3]{}, v_val[3]{};

    Scalar p_d[3];
    Scalar R_d[9];
    for (int i = 0; i < 3; ++i) p_d[i] = Scalar(Params.xi[i]);
    for (int i = 0; i < 9; ++i) R_d[i] = Scalar(Params.xi[3 + i]);

    Scalar net_nL[3]{};

    for (int segi = NUM_SEGMENTS - 1; segi >= 0; --segi) {
        if (segi % 2 == 0) {
            fsegi = segi >> 1;
            if (segi == NUM_SEGMENTS - 1) {
                mMult_AB<3, 3, 1>(Kinv[fsegi], tau_0, K2invResidual);
                mAdd_AB<3, 1>(ustar[fsegi], K2invResidual, u_t);

                for (int i = 0; i < 3; ++i) p_t[i] = Scalar(Params.xf[i]);
                for (int i = 0; i < 9; ++i) R_t[i] = Scalar(Params.xf[3 + i]);
                CRMFlexible_IVP_Back_T(segi, p_t, R_t, Params, u_t, n_0, u_tau, p_, R_);

                mSub_AB<3, 1>(u_tau, ustar[fsegi], du);
                mMult_AB<3, 3, 1>(K[fsegi], du, tau);
                actno = fsegi - 1;
                mSub_AB<3, 1>(m_L[actno], tau, net_mL);
            } else {
                actno = fsegi;
                mMult_AB<3, 3, 1>(Kinv[fsegi], m_L[actno], K2invResidual);
                mAdd_AB<3, 1>(ustar[fsegi], K2invResidual, u_L);
                actseg = segi + 1;
                const double RigidSegmentLength = SegBounds[actseg + 1] - SegBounds[actseg];
                for (int i = 0; i < 9; ++i) R_L[i] = out_x_coil[actno][i + 9];
                for (int i = 0; i < 3; ++i) p_L[i] = out_x_coil[actno][i + 6] - R_L[i * 3 + 2] * Scalar(RigidSegmentLength) * Scalar(0.5);
                CRMFlexible_IVP_Back_T(segi, p_L, R_L, Params, u_L, n_L[actno], u_f, p_f, R_f);

                actno = fsegi - 1;
                if (actno > -1) {
                    mSub_AB<3, 1>(u_f, ustar[fsegi], du);
                    mMult_AB<3, 3, 1>(K[fsegi], du, tau);
                    mSub_AB<3, 1>(m_L[actno], tau, net_mL);
                }
            }
        } else {
            actno = (segi - 1) >> 1;
            if (actno < NUM_ACT_SET - 1) {
                mSub_AB<3, 1>(n_L[actno], n_L[actno + 1], net_nL);
            } else {
                mSub_AB<3, 1>(n_L[actno], n_0, net_nL);
            }
            CoilDynamicsT(x_coil[actno], net_nL, Params.g, Params.ActMass[actno], Params.actInertia[actno], Params.damping[actno], Params.DELTA_T, Params.B0, muhat[actno], net_mL, out_x_coil[actno], out_xdot);

            if (actno < NUM_ACT_SET - 1) {
                const double RigidSegmentLength = SegBounds[segi + 1] - SegBounds[segi];
                actno_mn = actno + 1;
                for (int i = 0; i < 9; ++i) R_L[i] = out_x_coil[actno][i + 9];
                for (int i = 0; i < 3; ++i) p_L[i] = out_x_coil[actno][i + 6] + R_L[i * 3 + 2] * Scalar(RigidSegmentLength) * Scalar(0.5);
                for (int i = 0; i < 3; ++i) residual[actno_mn][i] = (p_f[i] - p_L[i]);
                for (int i = 0; i < 3; ++i) {
                    v1[i] = R_f[i * 3] - R_L[i * 3];
                    v2[i] = R_f[1 + i * 3] - R_L[1 + i * 3];
                    v3[i] = R_f[2 + i * 3] - R_L[2 + i * 3];
                }
                v_val[0] = vNormSq<3>(v1);
                v_val[1] = vNormSq<3>(v2);
                v_val[2] = vNormSq<3>(v3);
                for (int i = 0; i < 3; ++i) residual[actno_mn][i + 3] = sqrt(v_val[i] + Scalar(1e-24));
            }
        }
    }

    // last segment residual
    actno = 0;
    for (int i = 0; i < 3; ++i) residual[actno][i] = (p_f[i] - p_d[i]);
    for (int i = 0; i < 3; ++i) {
        v1[i] = R_f[i * 3] - R_d[i * 3];
        v2[i] = R_f[1 + i * 3] - R_d[1 + i * 3];
        v3[i] = R_f[2 + i * 3] - R_d[2 + i * 3];
    }
    v_val[0] = vNormSq<3>(v1);
    v_val[1] = vNormSq<3>(v2);
    v_val[2] = vNormSq<3>(v3);
    for (int i = 0; i < 3; ++i) residual[actno][i + 3] = sqrt(v_val[i] + Scalar(1e-24));

    for (int i = 0; i < NUM_ACT_SET; ++i) {
        for (int j = 0; j < 3; ++j) out_y(j + i * 6) = Scalar(RESIDUAL_SCALE_P) * residual[i][j];
        for (int j = 3; j < 6; ++j) out_y(j + i * 6) = Scalar(RESIDUAL_SCALE_R) * residual[i][j];
    }
    return out_y;
}

	inline Eigen::MatrixXd DYNNLEquationJacobianAD(
	    const Eigen::VectorXd& x_scaled,
	    DYNNLEqnParams& Params,
	    Eigen::VectorXd* out_residual = nullptr)
	{
    using autodiff::VectorXreal;
    using autodiff::real;
    using autodiff::jacobian;
    using autodiff::wrt;
    using autodiff::at;

    VectorXreal x(x_scaled.size());
	    for (int i = 0; i < x_scaled.size(); ++i) x(i) = x_scaled(i);

	    VectorXreal y;
	    Eigen::MatrixXd J;
	    jacobian(DYNNLEquationResidualAD<real>, wrt(x), at(x, &Params), y, J);
	    if (out_residual) {
	        out_residual->resize(y.size());
	        for (int i = 0; i < y.size(); ++i) (*out_residual)(i) = autodiff::val(y(i));
	    }
	    return J;
	}

} // namespace CRMCatheterModel
