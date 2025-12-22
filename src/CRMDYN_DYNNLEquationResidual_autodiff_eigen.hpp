#pragma once

#include <Eigen/Dense>

#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>

#include <cmath>
#include <string>
#include <vector>

#include "CRMDYN.hpp"

namespace CRMCatheterModel {

namespace dynnl_ad_eigen {

inline constexpr double kCoilTStep = 0.001;
inline constexpr double kEps = 1.0e-12;

template <typename Scalar>
using Vec3 = Eigen::Matrix<Scalar, 3, 1>;

template <typename Scalar>
using Vec6 = Eigen::Matrix<Scalar, 6, 1>;

template <typename Scalar>
using Mat3 = Eigen::Matrix<Scalar, 3, 3, Eigen::RowMajor>;

// ============================================================================
// Parameter Gradient Support (Task A1)
// ============================================================================
// Learnable parameters (theta):
//   - damping[6], K_diag[3], ustar[3], actMass[1], MagMoment[3]
// ============================================================================

inline constexpr int NUM_LEARNABLE_PARAMS = 16;

inline constexpr int THETA_OFFSET_DAMPING = 0;
inline constexpr int THETA_OFFSET_K_DIAG = 6;
inline constexpr int THETA_OFFSET_USTAR = 9;
inline constexpr int THETA_OFFSET_ACTMASS = 12;
inline constexpr int THETA_OFFSET_MAGMOMENT = 13;

template <typename Scalar>
struct DYNNLEqnParamsAD {
    Eigen::Matrix<Scalar, 6, 1> damping;
    Eigen::Matrix<Scalar, 3, 1> K_diag;
    Eigen::Matrix<Scalar, 3, 1> ustar;
    Scalar actMass;
    Eigen::Matrix<Scalar, 3, 1> MagMoment;

    Eigen::Vector3d B0;
    Eigen::Vector3d g;
    Eigen::Matrix3d actInertia;
    double DELTA_T;

    const DYNNLEqnParams* base_params = nullptr;

    Mat3<Scalar> getK() const {
        Mat3<Scalar> K = Mat3<Scalar>::Zero();
        K(0, 0) = K_diag(0);
        K(1, 1) = K_diag(1);
        K(2, 2) = K_diag(2);
        return K;
    }

    Mat3<Scalar> getKinv() const {
        Mat3<Scalar> Kinv = Mat3<Scalar>::Zero();
        Kinv(0, 0) = Scalar(1.0) / (K_diag(0) + Scalar(1e-12));
        Kinv(1, 1) = Scalar(1.0) / (K_diag(1) + Scalar(1e-12));
        Kinv(2, 2) = Scalar(1.0) / (K_diag(2) + Scalar(1e-12));
        return Kinv;
    }

    Mat3<Scalar> getMuHat() const {
        Mat3<Scalar> muhat;
        muhat << Scalar(0), -MagMoment(2), MagMoment(1),
                 MagMoment(2), Scalar(0), -MagMoment(0),
                 -MagMoment(1), MagMoment(0), Scalar(0);
        return muhat;
    }
};

inline Eigen::VectorXd packLearnableParams(const DYNNLEqnParams& params, int flex_seg_index = 0) {
    Eigen::VectorXd theta(NUM_LEARNABLE_PARAMS);
    theta.setZero();

    for (int i = 0; i < 6; ++i) {
        theta(THETA_OFFSET_DAMPING + i) = params.damping[0][i];
    }

    if (params.no_flex_seg > flex_seg_index) {
        theta(THETA_OFFSET_K_DIAG + 0) = params.K[flex_seg_index](0, 0);
        theta(THETA_OFFSET_K_DIAG + 1) = params.K[flex_seg_index](1, 1);
        theta(THETA_OFFSET_K_DIAG + 2) = params.K[flex_seg_index](2, 2);
    }

    if (params.no_flex_seg > flex_seg_index) {
        for (int i = 0; i < 3; ++i) {
            theta(THETA_OFFSET_USTAR + i) = params.ustar[flex_seg_index](i);
        }
    }

    theta(THETA_OFFSET_ACTMASS) = params.ActMass[0];

    for (int i = 0; i < 3; ++i) {
        theta(THETA_OFFSET_MAGMOMENT + i) = params.MagMoment[0](i);
    }

    return theta;
}

inline std::vector<std::string> getLearnableParamNames() {
    return {
        "damping_v0", "damping_v1", "damping_v2",
        "damping_w0", "damping_w1", "damping_w2",
        "K_diag_0", "K_diag_1", "K_diag_2",
        "ustar_0", "ustar_1", "ustar_2",
        "actMass",
        "MagMoment_0", "MagMoment_1", "MagMoment_2"
    };
}

template <typename Scalar>
DYNNLEqnParamsAD<Scalar> unpackToADParams(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& theta,
    const DYNNLEqnParams& base_params)
{
    DYNNLEqnParamsAD<Scalar> ad_params;
    ad_params.base_params = &base_params;

    for (int i = 0; i < 6; ++i) {
        ad_params.damping(i) = theta(THETA_OFFSET_DAMPING + i);
    }
    for (int i = 0; i < 3; ++i) {
        ad_params.K_diag(i) = theta(THETA_OFFSET_K_DIAG + i);
    }
    for (int i = 0; i < 3; ++i) {
        ad_params.ustar(i) = theta(THETA_OFFSET_USTAR + i);
    }
    ad_params.actMass = theta(THETA_OFFSET_ACTMASS);
    for (int i = 0; i < 3; ++i) {
        ad_params.MagMoment(i) = theta(THETA_OFFSET_MAGMOMENT + i);
    }

    for (int i = 0; i < 3; ++i) {
        ad_params.B0(i) = base_params.B0[i];
        ad_params.g(i) = base_params.g[i];
    }

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            ad_params.actInertia(r, c) = base_params.actInertia[0][r * 3 + c];
        }
    }
    ad_params.DELTA_T = base_params.DELTA_T;

    return ad_params;
}

template <typename T>
inline T sinT(const T& x)
{
    using std::sin;
    return sin(x);
}

template <typename T>
inline T cosT(const T& x)
{
    using std::cos;
    return cos(x);
}

template <typename T>
inline T sqrtT(const T& x)
{
    using std::sqrt;
    return sqrt(x);
}

template <typename Scalar>
inline Mat3<Scalar> wHat(const Vec3<Scalar>& w)
{
    Mat3<Scalar> W;
    W << Scalar(0), -w(2), w(1),
        w(2), Scalar(0), -w(0),
        -w(1), w(0), Scalar(0);
    return W;
}

template <typename Scalar>
inline Mat3<Scalar> RodriguesExpanded(const Vec3<Scalar>& w_unit, const Scalar theta)
{
    const Mat3<Scalar> W = wHat(w_unit);
    const Mat3<Scalar> I = Mat3<Scalar>::Identity();
    return I + sinT(theta) * W + (Scalar(1) - cosT(theta)) * (W * W);
}

template <typename Scalar>
inline void SE3_Analytical_Step(const Mat3<Scalar>& R_n,
                                const Vec3<Scalar>& p_n,
                                const Vec3<Scalar>& u_n,
                                const Scalar h,
                                Mat3<Scalar>& R_np1,
                                Vec3<Scalar>& p_np1)
{
    const Scalar umagsq = u_n.squaredNorm();
    if (autodiff::val(umagsq) < kEps) {
        R_np1 = R_n;
        p_np1 = p_n + R_n.col(2) * h;
        return;
    }

    const Scalar umag = sqrtT(umagsq);
    const Vec3<Scalar> unorm = u_n / umag;
    const Scalar delsumag = h * umag;
    const Mat3<Scalar> Rdelta = RodriguesExpanded(unorm, delsumag);

    // This matches CRM_IVPSolver.cpp::SE3_Analytical_Step.
    const Vec3<Scalar> uu3dels = u_n * (u_n(2) * h);
    Vec3<Scalar> ImRuxv;
    ImRuxv(0) = Rdelta(0, 1) * u_n(0) - Rdelta(0, 0) * u_n(1) + u_n(1);
    ImRuxv(1) = Rdelta(1, 1) * u_n(0) - Rdelta(1, 0) * u_n(1) - u_n(0);
    ImRuxv(2) = Rdelta(2, 1) * u_n(0) - Rdelta(2, 0) * u_n(1);

    const Vec3<Scalar> pdelta = (ImRuxv + uu3dels) / umagsq;
    R_np1 = R_n * Rdelta;
    p_np1 = p_n + R_n * pdelta;
}

template <typename Scalar>
inline void DYNSE3_TimeSpace(const Mat3<Scalar>& R_n,
                             const Vec3<Scalar>& p_n,
                             const Scalar h,
                             const Vec6<Scalar>& twist_n,
                             Mat3<Scalar>& R_np1,
                             Vec3<Scalar>& p_np1)
{
    const Vec3<Scalar> v_n = twist_n.template head<3>();
    const Vec3<Scalar> w_n = twist_n.template tail<3>();

    const Scalar wmagsq = w_n.squaredNorm();
    const Vec3<Scalar> p_dot = R_n * v_n;
    if (autodiff::val(wmagsq) < kEps) {
        R_np1 = R_n;
        p_np1 = p_n + p_dot * h;
        return;
    }

    const Scalar wmag = sqrtT(wmagsq);
    const Vec3<Scalar> wunit = w_n / wmag;
    const Scalar delsumag = h * wmag;
    const Mat3<Scalar> Rdelta = RodriguesExpanded(wunit, delsumag);
    R_np1 = R_n * Rdelta;

    const Mat3<Scalar> W = wHat(w_n);
    const Vec3<Scalar> wxv = W * v_n;

    Mat3<Scalar> ImRdelta = Mat3<Scalar>::Identity() - Rdelta;
    const Vec3<Scalar> ImRdeltawxv = ImRdelta * wxv;
    const Scalar wTv = w_n.dot(v_n);
    const Vec3<Scalar> wwTv = w_n * wTv;
    const Vec3<Scalar> pdelta = (ImRdeltawxv + wwTv * h) / wmagsq;

    p_np1 = p_n + R_n * pdelta;
}

template <typename Scalar>
inline void CoilIntegrand(const Vec6<Scalar>& twist,
                          const Vec3<Scalar>& n_L,
                          const Eigen::Vector3d& g,
                          const Mat3<Scalar>& R,
                          const Scalar actMass,
                          const Eigen::Matrix3d& actInertia,
                          const Eigen::Matrix<Scalar, 6, 1>& damping,
                          const Eigen::Vector3d& B0,
                          const Mat3<Scalar>& muhat,
                          const Vec3<Scalar>& m_L,
                          Vec6<Scalar>& twistdot)
{
    const Vec3<Scalar> v = twist.template head<3>();
    const Vec3<Scalar> w = twist.template tail<3>();

    const Vec3<Scalar> RTg = R.transpose() * g.template cast<Scalar>();
    const Mat3<Scalar> w_hat = wHat(w);
    const Vec3<Scalar> w_v = w_hat * v;

    Vec3<Scalar> vdot;
    for (int i = 0; i < 3; ++i) {
        const Scalar damp_v = Scalar(damping(i)) * v(i);
        vdot(i) = RTg(i) - n_L(i) / Scalar(actMass) - w_v(i) - damp_v;
    }

    const Vec3<Scalar> inertiaw = actInertia.template cast<Scalar>() * w;
    const Vec3<Scalar> w_inertia_w = w_hat * inertiaw;

    const Vec3<Scalar> RscTB0 = R.transpose() * B0.template cast<Scalar>();
    const Vec3<Scalar> Tb = muhat * RscTB0;
    const Vec3<Scalar> tau = Tb - m_L;

    Vec3<Scalar> residual_w;
    for (int i = 0; i < 3; ++i) {
        const Scalar damp_w = Scalar(damping(3 + i)) * w(i);
        residual_w(i) = (tau(i) - w_inertia_w(i) - damp_w);
    }

    // actInertia is diagonal in this model.
    Vec3<Scalar> wdot;
    wdot(0) = residual_w(0) / Scalar(actInertia(0, 0));
    wdot(1) = residual_w(1) / Scalar(actInertia(1, 1));
    wdot(2) = residual_w(2) / Scalar(actInertia(2, 2));

    twistdot.template head<3>() = vdot;
    twistdot.template tail<3>() = wdot;
}

template <typename Scalar>
inline void CoilDynamics(const Vec6<Scalar>& v0w0,
                         const Vec3<Scalar>& p0,
                         const Mat3<Scalar>& R0,
                         const Vec3<Scalar>& n_L,
                         const Eigen::Vector3d& g,
                         const Scalar actMass,
                         const Eigen::Matrix3d& actInertia,
                         const Eigen::Matrix<Scalar, 6, 1>& damping,
                         const double DELTA_T,
                         const Eigen::Vector3d& B0,
                         const Mat3<Scalar>& muhat,
                         const Vec3<Scalar>& m_L,
                         Vec6<Scalar>& v1w1,
                         Vec3<Scalar>& p1,
                         Mat3<Scalar>& R1,
                         Vec6<Scalar>& out_xdot_n)
{
    // ABM4 with RK2 warmup, matching CoilDynamics_Defs.cpp (t_step fixed at 0.001).
    const int N = static_cast<int>(std::ceil(DELTA_T / kCoilTStep));

    Vec6<Scalar> twist_nm3 = Vec6<Scalar>::Zero();
    Vec6<Scalar> twist_nm2 = Vec6<Scalar>::Zero();
    Vec6<Scalar> twist_nm1 = Vec6<Scalar>::Zero();
    Vec6<Scalar> twist_n = v0w0;
    Vec6<Scalar> twist_np1 = twist_n;

    Vec6<Scalar> xdot_nm3 = Vec6<Scalar>::Zero();
    Vec6<Scalar> xdot_nm2 = Vec6<Scalar>::Zero();
    Vec6<Scalar> xdot_nm1 = Vec6<Scalar>::Zero();
    Vec6<Scalar> xdot_n = Vec6<Scalar>::Zero();

    Vec3<Scalar> p_nm3 = Vec3<Scalar>::Zero();
    Vec3<Scalar> p_nm2 = Vec3<Scalar>::Zero();
    Vec3<Scalar> p_nm1 = Vec3<Scalar>::Zero();
    Vec3<Scalar> p_n = p0;
    Vec3<Scalar> p_np1 = p_n;

    Mat3<Scalar> R_nm3 = Mat3<Scalar>::Identity();
    Mat3<Scalar> R_nm2 = Mat3<Scalar>::Identity();
    Mat3<Scalar> R_nm1 = Mat3<Scalar>::Identity();
    Mat3<Scalar> R_n = R0;
    Mat3<Scalar> R_np1 = R_n;

    for (int idx = 0; idx < N; ++idx) {
        CoilIntegrand(twist_n, n_L, g, R_n, actMass, actInertia, damping, B0, muhat, m_L, xdot_n);
        out_xdot_n = xdot_n;

        const Scalar h = Scalar(kCoilTStep);
        if (idx < 3) {
            // RK2
            const Vec6<Scalar> k1 = h * xdot_n;
            const Vec6<Scalar> twist_half = twist_n + k1 * Scalar(0.5);
            Mat3<Scalar> R_half;
            Vec3<Scalar> p_half;
            DYNSE3_TimeSpace(R_n, p_n, h * Scalar(0.5), twist_n, R_half, p_half);
            Vec6<Scalar> xdot_half;
            CoilIntegrand(twist_half, n_L, g, R_half, actMass, actInertia, damping, B0, muhat, m_L, xdot_half);
            twist_np1 = twist_n + h * xdot_half;
            DYNSE3_TimeSpace(R_n, p_n, h, twist_half, R_np1, p_np1);
        } else {
            // ABM4 predictor/corrector on twist, analytic SE3 update driven by predicted/corrected twists.
            const Scalar Pn = Scalar(55.0 / 24.0), Pnm1 = Scalar(-59.0 / 24.0), Pnm2 = Scalar(37.0 / 24.0), Pnm3 = Scalar(-9.0 / 24.0);
            const Scalar Cnp1 = Scalar(9.0 / 24.0), Cn = Scalar(19.0 / 24.0), Cnm1 = Scalar(-5.0 / 24.0), Cnm2 = Scalar(1.0 / 24.0);

            const Vec6<Scalar> twist_hat = twist_n + h * (Pn * xdot_n + Pnm1 * xdot_nm1 + Pnm2 * xdot_nm2 + Pnm3 * xdot_nm3);
            const Vec6<Scalar> u_pred = Pn * twist_n + Pnm1 * twist_nm1 + Pnm2 * twist_nm2 + Pnm3 * twist_nm3;
            Mat3<Scalar> R_hat;
            Vec3<Scalar> p_hat;
            DYNSE3_TimeSpace(R_n, p_n, h, u_pred, R_hat, p_hat);

            Vec6<Scalar> xdot_hat;
            CoilIntegrand(twist_hat, n_L, g, R_hat, actMass, actInertia, damping, B0, muhat, m_L, xdot_hat);

            const Vec6<Scalar> twist_corr = twist_n + h * (Cnp1 * xdot_hat + Cn * xdot_n + Cnm1 * xdot_nm1 + Cnm2 * xdot_nm2);
            const Vec6<Scalar> u_corr = Cnp1 * twist_hat + Cn * twist_n + Cnm1 * twist_nm1 + Cnm2 * twist_nm2;
            DYNSE3_TimeSpace(R_n, p_n, h, u_corr, R_np1, p_np1);

            twist_np1 = twist_corr;
        }

        twist_nm3 = twist_nm2;
        twist_nm2 = twist_nm1;
        twist_nm1 = twist_n;
        twist_n = twist_np1;

        R_nm3 = R_nm2;
        R_nm2 = R_nm1;
        R_nm1 = R_n;
        R_n = R_np1;

        p_nm3 = p_nm2;
        p_nm2 = p_nm1;
        p_nm1 = p_n;
        p_n = p_np1;

        xdot_nm3 = xdot_nm2;
        xdot_nm2 = xdot_nm1;
        xdot_nm1 = xdot_n;
    }

    v1w1 = twist_n;
    p1 = p_n;
    R1 = R_n;
}

template <typename Scalar>
inline Vec3<Scalar> interpolate_fcum(const DYNNLEqnParams& Params, const double s)
{
    const double Length = Params.InsertedLength;
    const double deltalambdainv = Params.dlambdainv;
    const double lambda = Length - s;
    double ix = lambda * deltalambdainv;
    double ird_f = std::floor(ix);
    if (ird_f < 0.0) ird_f = 0.0;
    int ird = static_cast<int>(ird_f);
    double iru_f = std::ceil(ix);
    if (iru_f > Params.no_fcum_steps) iru_f = Params.no_fcum_steps;
    int iru = static_cast<int>(iru_f);
    const double ixmird = ix - ird;
    const double irumix = iru - ix;

    Vec3<Scalar> fcum;
    for (int i = 0; i < 3; ++i) {
        const double v = Params.fcumlambda[iru](i) * ixmird + Params.fcumlambda[ird](i) * irumix;
        fcum(i) = Scalar(v);
    }
    return fcum;
}

template <typename Scalar>
inline Vec3<Scalar> CRMIntegrand_dyn(const DYNNLEqnParams& Params,
                                     const int fsegno,
                                     const int SegmentIndex,
                                     const double s,
                                     const Mat3<Scalar>& R,
                                     const Vec3<Scalar>& u,
                                     const Vec3<Scalar>& nL_local)
{
    Vec3<Scalar> fcum = interpolate_fcum<Scalar>(Params, s);
    // DYNNLEquation path uses ftip=0 inside CRMFlexible_IVP_Back (tip force enters via boundary condition n_0).

    const Vec3<Scalar> nL_spatial = R * nL_local;
    fcum += nL_spatial;

    // Build e3hat * R^T explicitly (same as existing).
    Mat3<Scalar> e3hatRT;
    e3hatRT << -R(0, 1), -R(1, 1), -R(2, 1),
        R(0, 0), R(1, 0), R(2, 0),
        Scalar(0), Scalar(0), Scalar(0);

    const Vec3<Scalar> e3hatRTfcum = e3hatRT * fcum;

    const Mat3<Scalar> K = Params.K[fsegno].template cast<Scalar>();
    const Mat3<Scalar> Kinv = Params.Kinv[fsegno].template cast<Scalar>();
    const Vec3<Scalar> ustar = Params.ustar[fsegno].template cast<Scalar>();

    const Vec3<Scalar> umustar = u - ustar;
    const Vec3<Scalar> Kumustar = K * umustar;
    const Vec3<Scalar> uhatKumustar = wHat(u) * Kumustar;

    const Vec3<Scalar> RTl = Vec3<Scalar>::Zero();  // l is zero in DYNNLEquation path

    const Vec3<Scalar> sumterm = uhatKumustar + e3hatRTfcum + RTl;
    const Vec3<Scalar> udot = -Kinv * sumterm;
    (void)SegmentIndex;
    return udot;
}

template <typename Scalar>
inline void CRMFlexible_IVP_Back(const int SegmentIndex,
                                 const Vec3<Scalar>& in_p,
                                 const Mat3<Scalar>& in_R,
                                 const DYNNLEqnParams& Params,
                                 const Vec3<Scalar>& in_u,
                                 const Vec3<Scalar>& in_nL,
                                 Vec3<Scalar>& out_u,
                                 Vec3<Scalar>& out_p,
                                 Mat3<Scalar>& out_R)
{
    const int fsegno = SegmentIndex >> 1;
    const double h = -1.0 * (Params.SegBounds[SegmentIndex + 1] - Params.SegBounds[SegmentIndex]) / (Params.SegSteps[fsegno] * 1.0);
    const int N = Params.SegSteps[fsegno];

    Vec3<Scalar> u_nm3 = Vec3<Scalar>::Zero();
    Vec3<Scalar> u_nm2 = Vec3<Scalar>::Zero();
    Vec3<Scalar> u_nm1 = Vec3<Scalar>::Zero();
    Vec3<Scalar> u_n = in_u;
    Vec3<Scalar> u_np1 = u_n;

    Vec3<Scalar> udot_nm3 = Vec3<Scalar>::Zero();
    Vec3<Scalar> udot_nm2 = Vec3<Scalar>::Zero();
    Vec3<Scalar> udot_nm1 = Vec3<Scalar>::Zero();
    Vec3<Scalar> udot_n = Vec3<Scalar>::Zero();

    Vec3<Scalar> p_nm3 = Vec3<Scalar>::Zero();
    Vec3<Scalar> p_nm2 = Vec3<Scalar>::Zero();
    Vec3<Scalar> p_nm1 = Vec3<Scalar>::Zero();
    Vec3<Scalar> p_n = in_p;
    Vec3<Scalar> p_np1 = p_n;

    Mat3<Scalar> R_nm3 = Mat3<Scalar>::Identity();
    Mat3<Scalar> R_nm2 = Mat3<Scalar>::Identity();
    Mat3<Scalar> R_nm1 = Mat3<Scalar>::Identity();
    Mat3<Scalar> R_n = in_R;
    Mat3<Scalar> R_np1 = R_n;

    double t_n = Params.SegBounds[SegmentIndex + 1];
    for (int idx = 0; idx < N; ++idx) {
        udot_n = CRMIntegrand_dyn<Scalar>(Params, fsegno, SegmentIndex, t_n, R_n, u_n, in_nL);

        if (idx < 3) {
            // RK2
            const Vec3<Scalar> u_mid = u_n + Scalar(0.5) * Scalar(h) * udot_n;
            Mat3<Scalar> R_mid;
            Vec3<Scalar> p_mid;
            SE3_Analytical_Step(R_n, p_n, u_n, Scalar(h) * Scalar(0.5), R_mid, p_mid);
            const Vec3<Scalar> udot_mid = CRMIntegrand_dyn<Scalar>(Params, fsegno, SegmentIndex, t_n + 0.5 * h, R_mid, u_mid, in_nL);
            u_np1 = u_n + Scalar(h) * udot_mid;
            SE3_Analytical_Step(R_n, p_n, u_mid, Scalar(h), R_np1, p_np1);
        } else {
            const Scalar Pn = Scalar(55.0 / 24.0), Pnm1 = Scalar(-59.0 / 24.0), Pnm2 = Scalar(37.0 / 24.0), Pnm3 = Scalar(-9.0 / 24.0);
            const Scalar Cnp1 = Scalar(9.0 / 24.0), Cn = Scalar(19.0 / 24.0), Cnm1 = Scalar(-5.0 / 24.0), Cnm2 = Scalar(1.0 / 24.0);

            const Vec3<Scalar> u_hat = u_n + Scalar(h) * (Pn * udot_n + Pnm1 * udot_nm1 + Pnm2 * udot_nm2 + Pnm3 * udot_nm3);
            const Vec3<Scalar> u_n_pred = Pn * u_n + Pnm1 * u_nm1 + Pnm2 * u_nm2 + Pnm3 * u_nm3;

            Mat3<Scalar> R_hat;
            Vec3<Scalar> p_hat;
            SE3_Analytical_Step(R_n, p_n, u_n_pred, Scalar(h), R_hat, p_hat);

            const Vec3<Scalar> udot_hat = CRMIntegrand_dyn<Scalar>(Params, fsegno, SegmentIndex, t_n + h, R_hat, u_hat, in_nL);
            const Vec3<Scalar> u_corr = u_n + Scalar(h) * (Cnp1 * udot_hat + Cn * udot_n + Cnm1 * udot_nm1 + Cnm2 * udot_nm2);
            const Vec3<Scalar> u_n_corr = Cnp1 * u_hat + Cn * u_n + Cnm1 * u_nm1 + Cnm2 * u_nm2;
            SE3_Analytical_Step(R_n, p_n, u_n_corr, Scalar(h), R_np1, p_np1);

            u_np1 = u_corr;
        }

        // shift
        u_nm3 = u_nm2;
        u_nm2 = u_nm1;
        u_nm1 = u_n;
        u_n = u_np1;

        udot_nm3 = udot_nm2;
        udot_nm2 = udot_nm1;
        udot_nm1 = udot_n;

        p_nm3 = p_nm2;
        p_nm2 = p_nm1;
        p_nm1 = p_n;
        p_n = p_np1;

        R_nm3 = R_nm2;
        R_nm2 = R_nm1;
        R_nm1 = R_n;
        R_n = R_np1;

        t_n += h;
    }

    out_u = u_n;
    out_p = p_n;
    out_R = R_n;
}

template <typename Scalar>
inline Vec3<Scalar> CRMIntegrand_dynAD(const DYNNLEqnParams& Params,
                                       const int fsegno,
                                       const int SegmentIndex,
                                       const double s,
                                       const Mat3<Scalar>& R,
                                       const Vec3<Scalar>& u,
                                       const Vec3<Scalar>& nL_local,
                                       const Mat3<Scalar>& K,
                                       const Mat3<Scalar>& Kinv,
                                       const Vec3<Scalar>& ustar)
{
    Vec3<Scalar> fcum = interpolate_fcum<Scalar>(Params, s);

    const Vec3<Scalar> nL_spatial = R * nL_local;
    fcum += nL_spatial;

    Mat3<Scalar> e3hatRT;
    e3hatRT << -R(0, 1), -R(1, 1), -R(2, 1),
        R(0, 0), R(1, 0), R(2, 0),
        Scalar(0), Scalar(0), Scalar(0);

    const Vec3<Scalar> e3hatRTfcum = e3hatRT * fcum;

    const Vec3<Scalar> umustar = u - ustar;
    const Vec3<Scalar> Kumustar = K * umustar;
    const Vec3<Scalar> uhatKumustar = wHat(u) * Kumustar;

    const Vec3<Scalar> RTl = Vec3<Scalar>::Zero();

    const Vec3<Scalar> sumterm = uhatKumustar + e3hatRTfcum + RTl;
    const Vec3<Scalar> udot = -Kinv * sumterm;
    (void)SegmentIndex;
    (void)fsegno;
    return udot;
}

template <typename Scalar>
inline void CRMFlexible_IVP_BackAD(const int SegmentIndex,
                                   const Vec3<Scalar>& in_p,
                                   const Mat3<Scalar>& in_R,
                                   const DYNNLEqnParams& Params,
                                   const Vec3<Scalar>& in_u,
                                   const Vec3<Scalar>& in_nL,
                                   const Mat3<Scalar>& K,
                                   const Mat3<Scalar>& Kinv,
                                   const Vec3<Scalar>& ustar,
                                   Vec3<Scalar>& out_u,
                                   Vec3<Scalar>& out_p,
                                   Mat3<Scalar>& out_R)
{
    const int fsegno = SegmentIndex >> 1;
    const double h = -1.0 * (Params.SegBounds[SegmentIndex + 1] - Params.SegBounds[SegmentIndex]) / (Params.SegSteps[fsegno] * 1.0);
    const int N = Params.SegSteps[fsegno];

    Vec3<Scalar> u_nm3 = Vec3<Scalar>::Zero();
    Vec3<Scalar> u_nm2 = Vec3<Scalar>::Zero();
    Vec3<Scalar> u_nm1 = Vec3<Scalar>::Zero();
    Vec3<Scalar> u_n = in_u;
    Vec3<Scalar> u_np1 = u_n;

    Vec3<Scalar> udot_nm3 = Vec3<Scalar>::Zero();
    Vec3<Scalar> udot_nm2 = Vec3<Scalar>::Zero();
    Vec3<Scalar> udot_nm1 = Vec3<Scalar>::Zero();
    Vec3<Scalar> udot_n = Vec3<Scalar>::Zero();

    Vec3<Scalar> p_nm3 = Vec3<Scalar>::Zero();
    Vec3<Scalar> p_nm2 = Vec3<Scalar>::Zero();
    Vec3<Scalar> p_nm1 = Vec3<Scalar>::Zero();
    Vec3<Scalar> p_n = in_p;
    Vec3<Scalar> p_np1 = p_n;

    Mat3<Scalar> R_nm3 = Mat3<Scalar>::Identity();
    Mat3<Scalar> R_nm2 = Mat3<Scalar>::Identity();
    Mat3<Scalar> R_nm1 = Mat3<Scalar>::Identity();
    Mat3<Scalar> R_n = in_R;
    Mat3<Scalar> R_np1 = R_n;

    double t_n = Params.SegBounds[SegmentIndex + 1];
    for (int idx = 0; idx < N; ++idx) {
        udot_n = CRMIntegrand_dynAD<Scalar>(Params, fsegno, SegmentIndex, t_n, R_n, u_n, in_nL, K, Kinv, ustar);

        if (idx < 3) {
            const Vec3<Scalar> u_mid = u_n + Scalar(0.5) * Scalar(h) * udot_n;
            Mat3<Scalar> R_mid;
            Vec3<Scalar> p_mid;
            SE3_Analytical_Step(R_n, p_n, u_n, Scalar(h) * Scalar(0.5), R_mid, p_mid);
            const Vec3<Scalar> udot_mid = CRMIntegrand_dynAD<Scalar>(Params, fsegno, SegmentIndex, t_n + 0.5 * h, R_mid, u_mid, in_nL, K, Kinv, ustar);
            u_np1 = u_n + Scalar(h) * udot_mid;
            SE3_Analytical_Step(R_n, p_n, u_mid, Scalar(h), R_np1, p_np1);
        } else {
            const Scalar Pn = Scalar(55.0 / 24.0), Pnm1 = Scalar(-59.0 / 24.0), Pnm2 = Scalar(37.0 / 24.0), Pnm3 = Scalar(-9.0 / 24.0);
            const Scalar Cnp1 = Scalar(9.0 / 24.0), Cn = Scalar(19.0 / 24.0), Cnm1 = Scalar(-5.0 / 24.0), Cnm2 = Scalar(1.0 / 24.0);

            const Vec3<Scalar> u_hat = u_n + Scalar(h) * (Pn * udot_n + Pnm1 * udot_nm1 + Pnm2 * udot_nm2 + Pnm3 * udot_nm3);
            const Vec3<Scalar> u_n_pred = Pn * u_n + Pnm1 * u_nm1 + Pnm2 * u_nm2 + Pnm3 * u_nm3;

            Mat3<Scalar> R_hat;
            Vec3<Scalar> p_hat;
            SE3_Analytical_Step(R_n, p_n, u_n_pred, Scalar(h), R_hat, p_hat);

            const Vec3<Scalar> udot_hat = CRMIntegrand_dynAD<Scalar>(Params, fsegno, SegmentIndex, t_n + h, R_hat, u_hat, in_nL, K, Kinv, ustar);
            const Vec3<Scalar> u_corr = u_n + Scalar(h) * (Cnp1 * udot_hat + Cn * udot_n + Cnm1 * udot_nm1 + Cnm2 * udot_nm2);
            const Vec3<Scalar> u_n_corr = Cnp1 * u_hat + Cn * u_n + Cnm1 * u_nm1 + Cnm2 * u_nm2;
            SE3_Analytical_Step(R_n, p_n, u_n_corr, Scalar(h), R_np1, p_np1);

            u_np1 = u_corr;
        }

        u_nm3 = u_nm2;
        u_nm2 = u_nm1;
        u_nm1 = u_n;
        u_n = u_np1;

        udot_nm3 = udot_nm2;
        udot_nm2 = udot_nm1;
        udot_nm1 = udot_n;

        p_nm3 = p_nm2;
        p_nm2 = p_nm1;
        p_nm1 = p_n;
        p_n = p_np1;

        R_nm3 = R_nm2;
        R_nm2 = R_nm1;
        R_nm1 = R_n;
        R_n = R_np1;

        t_n += h;
    }

    out_u = u_n;
    out_p = p_n;
    out_R = R_n;
}

template <typename Scalar>
inline Eigen::Matrix<Scalar, NUM_DYN_RESIDUAL, 1> DYNNLEquationResidualEigenAD(const autodiff::VectorXreal& x_scaled,
                                                                              const DYNNLEqnParams* ParamsPtr)
{
    static_assert(NUM_ACT_SET == 1, "This Eigen+autodiff residual currently supports NUM_ACT_SET==1.");
    using Resid = Eigen::Matrix<Scalar, NUM_DYN_RESIDUAL, 1>;

    const DYNNLEqnParams& Params = *ParamsPtr;
    // Unpack NLE variables (scaled) -> physical m_L, n_L.
    Vec3<Scalar> m_L;
    Vec3<Scalar> n_L;
    for (int i = 0; i < 3; ++i) {
        m_L(i) = Scalar(IVALUE_SCALE_M) * x_scaled(i);
        n_L(i) = Scalar(IVALUE_SCALE_N) * x_scaled(3 + i);
    }

    // tip force (free tip) is fixed; note: in existing DYNNLEquation they use Params.TipForce.
    Vec3<Scalar> n_0;
    for (int i = 0; i < 3; ++i) n_0(i) = Scalar(Params.TipForce[i]);

    // root (desired) configuration.
    Vec3<Scalar> p_d;
    Mat3<Scalar> R_d;
    for (int i = 0; i < 3; ++i) p_d(i) = Scalar(Params.xi[i]);
    for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) R_d(r, c) = Scalar(Params.xi[3 + r * 3 + c]);

    // start from tip state Params.xf (stored in DYNNLEqnParams).
    Vec3<Scalar> p_t;
    Mat3<Scalar> R_t;
    for (int i = 0; i < 3; ++i) p_t(i) = Scalar(Params.xf[i]);
    for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) R_t(r, c) = Scalar(Params.xf[3 + r * 3 + c]);

    // initial tau_0 = 0 and compute u_t at tip for last flexible segment.
    Vec3<Scalar> tau_0 = Vec3<Scalar>::Zero();

    // Aliases to segment bounds.
    const int NUM_SEGMENTS = Params.no_segments;

    Vec3<Scalar> u_f = Vec3<Scalar>::Zero();
    Vec3<Scalar> p_f = p_t;
    Mat3<Scalar> R_f = R_t;

    // Working vars.
    Vec3<Scalar> tau = Vec3<Scalar>::Zero();
    Vec3<Scalar> u_t = Vec3<Scalar>::Zero();
    Vec3<Scalar> u_tau = Vec3<Scalar>::Zero();
    Vec3<Scalar> p_ = Vec3<Scalar>::Zero();
    Mat3<Scalar> R_ = Mat3<Scalar>::Identity();

    Vec6<Scalar> xdot_dummy = Vec6<Scalar>::Zero();
    Vec6<Scalar> vw1 = Vec6<Scalar>::Zero();
    Vec3<Scalar> p1 = Vec3<Scalar>::Zero();
    Mat3<Scalar> R1 = Mat3<Scalar>::Identity();

    // Downstream coil state seed.
    Vec6<Scalar> vw_coil0;
    Vec3<Scalar> p_coil0;
    Mat3<Scalar> R_coil0;
    for (int i = 0; i < 3; ++i) {
        vw_coil0(i) = Scalar(Params.v_L_pre[0][i]);
        vw_coil0(3 + i) = Scalar(Params.w_L_pre[0][i]);
        p_coil0(i) = Scalar(Params.p_pre[0][i]);
    }
    for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) R_coil0(r, c) = Scalar(Params.R_pre[0][r * 3 + c]);

    const Scalar actMass = Scalar(Params.ActMass[0]);
    Eigen::Matrix3d actInertia;
    for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) actInertia(r, c) = Params.actInertia[0][r * 3 + c];
    Eigen::Matrix<Scalar, 6, 1> damping;
    for (int i = 0; i < 6; ++i) damping(i) = Scalar(Params.damping[0][i]);

    // Magnetic moment hat.
    Vec3<Scalar> mu;
    for (int i = 0; i < 3; ++i) mu(i) = Scalar(Params.MagMoment[0](i));
    Mat3<Scalar> muhat;
    muhat << Scalar(0), -mu(2), mu(1),
        mu(2), Scalar(0), -mu(0),
        -mu(1), mu(0), Scalar(0);

    // Iterate segments distal->proximal (same direction as original: segi = NUM_SEGMENTS-1..0).
    Vec3<Scalar> net_mL = Vec3<Scalar>::Zero();
    Vec3<Scalar> net_nL = Vec3<Scalar>::Zero();

    // Coil output cache for segments above.
    bool have_out_coil = false;
    Vec6<Scalar> vw_coil_out = vw_coil0;
    Vec3<Scalar> p_coil_out = p_coil0;
    Mat3<Scalar> R_coil_out = R_coil0;

    for (int segi = NUM_SEGMENTS - 1; segi >= 0; --segi) {
        if (segi % 2 == 0) {
            // flexible segment
            const int fsegi = segi >> 1;
            const Vec3<Scalar> ustar = Params.ustar[fsegi].template cast<Scalar>();
            const Mat3<Scalar> Kinv = Params.Kinv[fsegi].template cast<Scalar>();

            if (segi == NUM_SEGMENTS - 1) {
                // last segment: tau_0 = 0, n_0 is tip force.
                u_t = ustar + Kinv * tau_0;
                CRMFlexible_IVP_Back(segi, p_t, R_t, Params, u_t, n_0, u_tau, p_, R_);
            } else {
                // non-free tip flexible segments with coil on top (see CoilDynamics_Defs.cpp).
                // u_L = ustar + Kinv * m_L (moment at lower side of upper coil)
                const Vec3<Scalar> u_L = ustar + Kinv * m_L;
                const int actseg = segi + 1;
                const double RigidSegmentLength = Params.SegBounds[actseg + 1] - Params.SegBounds[actseg];

                if (!have_out_coil) {
                    // If coil output is missing, keep behavior deterministic (this indicates a segment ordering mismatch).
                    // We propagate using the current carried pose.
                    CRMFlexible_IVP_Back(segi, p_f, R_f, Params, u_L, n_L, u_tau, p_, R_);
                } else {
                    // Starting pose for this flex segment is half a rigid segment "above" coil center.
                    const Vec3<Scalar> p_L = p_coil_out - R_coil_out.col(2) * Scalar(RigidSegmentLength) * Scalar(0.5);
                    const Mat3<Scalar> R_L = R_coil_out;
                    CRMFlexible_IVP_Back(segi, p_L, R_L, Params, u_L, n_L, u_f, p_f, R_f);
                    u_tau = u_f;
                    p_ = p_f;
                    R_ = R_f;
                }
            }

            // tau = K * (u_tau - ustar)
            const Mat3<Scalar> K = Params.K[fsegi].template cast<Scalar>();
            tau = K * (u_tau - ustar);

            // next coil is below this flex segment (actno = fsegi-1).
            // With NUM_ACT_SET==1, only one coil exists; original uses actno = fsegi-1.
            // We assume fsegi==1 for last flex (so actno==0). For fsegi==0, there is no coil below.
            // If actno < 0, no coil: just propagate.
            const int actno = fsegi - 1;
            if (actno >= 0) {
                // net torque applied to the downward coil: m_L - tau
                net_mL = m_L - tau;
            } else {
                // no coil below: just update carried state
                p_f = p_;
                R_f = R_;
                u_f = u_tau;
            }
        } else {
            // rigid (actuator) segment: integrate coil dynamics using net_mL from previous flexible step.
            const int actno = (segi - 1) >> 1;
            (void)actno;

            // net_nL = n_L - n_0 for NUM_ACT_SET==1 (see DYNNLEquation).
            net_nL = n_L - n_0;

            CoilDynamics(vw_coil0, p_coil0, R_coil0, net_nL, Eigen::Vector3d(Params.g[0], Params.g[1], Params.g[2]),
                         actMass, actInertia, damping, Params.DELTA_T,
                         Eigen::Vector3d(Params.B0[0], Params.B0[1], Params.B0[2]),
                         muhat, net_mL,
                         vw_coil_out, p_coil_out, R_coil_out, xdot_dummy);
            have_out_coil = true;
        }
    }

    // If we didn't return inside loop, compute last residual against root config.
    Vec3<Scalar> res_p = p_f - p_d;
    Vec3<Scalar> res_r;
    for (int k = 0; k < 3; ++k) {
        const Vec3<Scalar> diff = R_f.col(k) - R_d.col(k);
        res_r(k) = sqrtT(diff.squaredNorm() + Scalar(1e-24));
    }
    Resid out;
    out.template head<3>() = Scalar(RESIDUAL_SCALE_P) * res_p;
    out.template tail<3>() = Scalar(RESIDUAL_SCALE_R) * res_r;
    return out;
}

template <typename Scalar>
inline Eigen::Matrix<Scalar, NUM_DYN_RESIDUAL, 1> DYNNLEquationResidualWithParamsAD(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x_scaled,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& theta,
    const DYNNLEqnParams* ParamsPtr)
{
    static_assert(NUM_ACT_SET == 1, "This Eigen+autodiff residual currently supports NUM_ACT_SET==1.");
    using Resid = Eigen::Matrix<Scalar, NUM_DYN_RESIDUAL, 1>;

    const DYNNLEqnParams& Params = *ParamsPtr;
    DYNNLEqnParamsAD<Scalar> ad_params = unpackToADParams<Scalar>(theta, Params);

    Vec3<Scalar> m_L;
    Vec3<Scalar> n_L;
    for (int i = 0; i < 3; ++i) {
        m_L(i) = Scalar(IVALUE_SCALE_M) * x_scaled(i);
        n_L(i) = Scalar(IVALUE_SCALE_N) * x_scaled(3 + i);
    }

    Vec3<Scalar> n_0;
    for (int i = 0; i < 3; ++i) n_0(i) = Scalar(Params.TipForce[i]);

    Vec3<Scalar> p_d;
    Mat3<Scalar> R_d;
    for (int i = 0; i < 3; ++i) p_d(i) = Scalar(Params.xi[i]);
    for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) R_d(r, c) = Scalar(Params.xi[3 + r * 3 + c]);

    Vec3<Scalar> p_t;
    Mat3<Scalar> R_t;
    for (int i = 0; i < 3; ++i) p_t(i) = Scalar(Params.xf[i]);
    for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) R_t(r, c) = Scalar(Params.xf[3 + r * 3 + c]);

    Vec3<Scalar> tau_0 = Vec3<Scalar>::Zero();

    const int NUM_SEGMENTS = Params.no_segments;

    Vec3<Scalar> u_f = Vec3<Scalar>::Zero();
    Vec3<Scalar> p_f = p_t;
    Mat3<Scalar> R_f = R_t;

    Vec3<Scalar> tau = Vec3<Scalar>::Zero();
    Vec3<Scalar> u_t = Vec3<Scalar>::Zero();
    Vec3<Scalar> u_tau = Vec3<Scalar>::Zero();
    Vec3<Scalar> p_ = Vec3<Scalar>::Zero();
    Mat3<Scalar> R_ = Mat3<Scalar>::Identity();

    Vec6<Scalar> xdot_dummy = Vec6<Scalar>::Zero();
    Vec6<Scalar> vw1 = Vec6<Scalar>::Zero();
    Vec3<Scalar> p1 = Vec3<Scalar>::Zero();
    Mat3<Scalar> R1 = Mat3<Scalar>::Identity();

    Vec6<Scalar> vw_coil0;
    Vec3<Scalar> p_coil0;
    Mat3<Scalar> R_coil0;
    for (int i = 0; i < 3; ++i) {
        vw_coil0(i) = Scalar(Params.v_L_pre[0][i]);
        vw_coil0(3 + i) = Scalar(Params.w_L_pre[0][i]);
        p_coil0(i) = Scalar(Params.p_pre[0][i]);
    }
    for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) R_coil0(r, c) = Scalar(Params.R_pre[0][r * 3 + c]);

    const Scalar& actMass = ad_params.actMass;
    const Eigen::Matrix3d& actInertia = ad_params.actInertia;
    const Eigen::Matrix<Scalar, 6, 1>& damping = ad_params.damping;
    const Mat3<Scalar> muhat = ad_params.getMuHat();

    const Mat3<Scalar> K = ad_params.getK();
    const Mat3<Scalar> Kinv = ad_params.getKinv();
    const Vec3<Scalar>& ustar = ad_params.ustar;

    Vec3<Scalar> net_mL = Vec3<Scalar>::Zero();
    Vec3<Scalar> net_nL = Vec3<Scalar>::Zero();

    bool have_out_coil = false;
    Vec6<Scalar> vw_coil_out = vw_coil0;
    Vec3<Scalar> p_coil_out = p_coil0;
    Mat3<Scalar> R_coil_out = R_coil0;

    for (int segi = NUM_SEGMENTS - 1; segi >= 0; --segi) {
        if (segi % 2 == 0) {
            const int fsegi = segi >> 1;

            if (segi == NUM_SEGMENTS - 1) {
                u_t = ustar + Kinv * tau_0;
                CRMFlexible_IVP_BackAD(segi, p_t, R_t, Params, u_t, n_0, K, Kinv, ustar, u_tau, p_, R_);
            } else {
                const Vec3<Scalar> u_L = ustar + Kinv * m_L;
                const int actseg = segi + 1;
                const double RigidSegmentLength = Params.SegBounds[actseg + 1] - Params.SegBounds[actseg];

                if (!have_out_coil) {
                    CRMFlexible_IVP_BackAD(segi, p_f, R_f, Params, u_L, n_L, K, Kinv, ustar, u_tau, p_, R_);
                } else {
                    const Vec3<Scalar> p_L = p_coil_out - R_coil_out.col(2) * Scalar(RigidSegmentLength) * Scalar(0.5);
                    const Mat3<Scalar> R_L = R_coil_out;
                    CRMFlexible_IVP_BackAD(segi, p_L, R_L, Params, u_L, n_L, K, Kinv, ustar, u_f, p_f, R_f);
                    u_tau = u_f;
                    p_ = p_f;
                    R_ = R_f;
                }
            }

            tau = K * (u_tau - ustar);

            const int actno = fsegi - 1;
            if (actno >= 0) {
                net_mL = m_L - tau;
            } else {
                p_f = p_;
                R_f = R_;
                u_f = u_tau;
            }
        } else {
            const int actno = (segi - 1) >> 1;
            (void)actno;

            net_nL = n_L - n_0;

            CoilDynamics(vw_coil0, p_coil0, R_coil0, net_nL, ad_params.g,
                         actMass, actInertia, damping, ad_params.DELTA_T,
                         ad_params.B0, muhat, net_mL,
                         vw_coil_out, p_coil_out, R_coil_out, xdot_dummy);
            have_out_coil = true;
        }
    }

    Vec3<Scalar> res_p = p_f - p_d;
    Vec3<Scalar> res_r;
    for (int k = 0; k < 3; ++k) {
        const Vec3<Scalar> diff = R_f.col(k) - R_d.col(k);
        res_r(k) = sqrtT(diff.squaredNorm() + Scalar(1e-24));
    }
    Resid out;
    out.template head<3>() = Scalar(RESIDUAL_SCALE_P) * res_p;
    out.template tail<3>() = Scalar(RESIDUAL_SCALE_R) * res_r;
    return out;
}

} // namespace dynnl_ad_eigen

inline Eigen::VectorXd DYNNLEquationResidualEigenDouble(const Eigen::VectorXd& x_scaled, DYNNLEqnParams& Params)
{
    // Convenience wrapper for FD comparisons in bindings/tests.
    double x_raw[NUM_DYN_RESIDUAL];
    for (int i = 0; i < NUM_DYN_RESIDUAL; ++i) x_raw[i] = x_scaled(i);
    double y_raw[NUM_DYN_RESIDUAL];
    double u0[3];
    double tau[NUM_ACT_SET * 3];
    DYNNLEquation(x_raw, y_raw, Params, u0, tau);
    Eigen::VectorXd y(NUM_DYN_RESIDUAL);
    for (int i = 0; i < NUM_DYN_RESIDUAL; ++i) y(i) = y_raw[i];
    return y;
}

inline Eigen::MatrixXd DYNNLEquationJacobianEigenAD(const Eigen::VectorXd& x_scaled, DYNNLEqnParams& Params, Eigen::VectorXd* out_residual = nullptr)
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
    jacobian(dynnl_ad_eigen::DYNNLEquationResidualEigenAD<real>, wrt(x), at(x, &Params), y, J);
    if (out_residual) {
        out_residual->resize(y.size());
        for (int i = 0; i < y.size(); ++i) (*out_residual)(i) = autodiff::val(y(i));
    }
    return J;
}

inline Eigen::MatrixXd DYNNLEquationParameterJacobianEigenAD(
    const Eigen::VectorXd& x_scaled,
    DYNNLEqnParams& Params,
    Eigen::VectorXd* out_residual = nullptr,
    Eigen::VectorXd* out_theta = nullptr)
{
    using autodiff::VectorXreal;
    using autodiff::real;
    using autodiff::jacobian;
    using autodiff::wrt;
    using autodiff::at;

    Eigen::VectorXd theta_d = dynnl_ad_eigen::packLearnableParams(Params, 0);

    VectorXreal x_ad(x_scaled.size());
    for (int i = 0; i < x_scaled.size(); ++i) x_ad(i) = x_scaled(i);

    VectorXreal theta_ad(theta_d.size());
    for (int i = 0; i < theta_d.size(); ++i) theta_ad(i) = theta_d(i);

    VectorXreal y_ad;
    Eigen::MatrixXd J_theta;

    auto residual_fn = [&](const VectorXreal& theta_) -> VectorXreal {
        return dynnl_ad_eigen::DYNNLEquationResidualWithParamsAD<real>(x_ad, theta_, &Params);
    };

    jacobian(residual_fn, wrt(theta_ad), at(theta_ad), y_ad, J_theta);

    if (out_residual) {
        out_residual->resize(y_ad.size());
        for (int i = 0; i < y_ad.size(); ++i) (*out_residual)(i) = autodiff::val(y_ad(i));
    }

    if (out_theta) {
        *out_theta = theta_d;
    }

    return J_theta;
}

inline void DYNNLEquationFullJacobiansEigenAD(
    const Eigen::VectorXd& x_scaled,
    DYNNLEqnParams& Params,
    Eigen::MatrixXd& J_x,
    Eigen::MatrixXd& J_theta,
    Eigen::VectorXd* out_residual = nullptr,
    Eigen::VectorXd* out_theta = nullptr)
{
    using autodiff::VectorXreal;
    using autodiff::real;
    using autodiff::jacobian;
    using autodiff::wrt;
    using autodiff::at;

    Eigen::VectorXd theta_d = dynnl_ad_eigen::packLearnableParams(Params, 0);

    VectorXreal x(x_scaled.size());
    for (int i = 0; i < x_scaled.size(); ++i) x(i) = x_scaled(i);

    VectorXreal theta(theta_d.size());
    for (int i = 0; i < theta_d.size(); ++i) theta(i) = theta_d(i);

    Eigen::VectorXd residual_x;
    J_x = DYNNLEquationJacobianEigenAD(x_scaled, Params, &residual_x);

    VectorXreal y;
    auto residual_fn = [&x, &Params](const VectorXreal& theta_) -> VectorXreal {
        return dynnl_ad_eigen::DYNNLEquationResidualWithParamsAD<real>(x, theta_, &Params);
    };

    jacobian(residual_fn, wrt(theta), at(theta), y, J_theta);

    if (out_residual) {
        *out_residual = residual_x;
    }

    if (out_theta) {
        *out_theta = theta_d;
    }
}

} // namespace CRMCatheterModel
