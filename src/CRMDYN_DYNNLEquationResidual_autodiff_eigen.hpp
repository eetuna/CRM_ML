#pragma once

#include <Eigen/Dense>

#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>

#include <cmath>
#include <string>
#include <stdexcept>
#include <vector>

#include "CRMDYN.hpp"
#include "CRM_DynamicsContext.hpp"
#include "CRM_DynamicsContext_AD_impl.hpp"

namespace CRMCatheterModel {

namespace dynnl_ad_eigen {

inline constexpr double kCoilTStep = 0.001;
inline constexpr double kEps = 1.0e-12;

// Phase 3.4: Adaptive stepping thresholds for RK4 integrator
// Threshold: 1000 rad/s² (2.5x above critical acceleration for typical actuator inertia ~2.4e-4)
inline constexpr double kAngularAccelThreshold = 1000.0;  // rad/s²
inline constexpr int kMaxSubdivisionLevels = 4;           // 2^4 = 16x refinement max

template <typename Scalar>
using Vec3 = Eigen::Matrix<Scalar, 3, 1>;

template <typename Scalar>
using Vec6 = Eigen::Matrix<Scalar, 6, 1>;

template <typename Scalar>
using Mat3 = Eigen::Matrix<Scalar, 3, 3, Eigen::RowMajor>;

// Phase 3.4: Check if angular acceleration is within safe bounds for adaptive stepping
template <typename Scalar>
inline bool is_acceleration_safe(const Vec6<Scalar>& xdot) {
    const Vec3<Scalar> wdot = xdot.template tail<3>();  // Angular acceleration (last 3 components)
    const double wdot_mag = std::sqrt(static_cast<double>(wdot.squaredNorm()));
    return std::isfinite(wdot_mag) && wdot_mag < kAngularAccelThreshold;
}

// ============================================================================
// Parameter Gradient Support (Task A1)
// ============================================================================
// Learnable parameters (theta):
//   - damping[6], K_diag[3], ustar[3], actMass[1], MagMoment[3]
// ============================================================================


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
                         Vec6<Scalar>& out_xdot_n,
                         bool* out_diverged = nullptr)
{
    // ABM4 with RK2 warmup, matching CoilDynamics_Defs.cpp (t_step fixed at 0.001).
    const int N = static_cast<int>(std::ceil(DELTA_T / kCoilTStep));
    constexpr double kDivergenceThreshold = 1e6;  // Phase 3: Soft failure mode

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

        // Phase 3: Check for divergence (soft failure mode)
        if (out_diverged != nullptr) {
            const double twist_mag = std::sqrt(static_cast<double>(twist_n.squaredNorm()));
            const double p_mag = std::sqrt(static_cast<double>(p_n.squaredNorm()));
            if (!std::isfinite(twist_mag) || !std::isfinite(p_mag) ||
                twist_mag > kDivergenceThreshold || p_mag > kDivergenceThreshold) {
                *out_diverged = true;
                return;  // Early exit on divergence
            }
        }
    }

    v1w1 = twist_n;
    p1 = p_n;
    R1 = R_n;

    // Phase 3: Final divergence check
    if (out_diverged != nullptr) {
        *out_diverged = false;  // Success
    }
}

/**
 * @brief Recursive RK4 step with adaptive subdivision (Phase 3.4)
 *
 * Performs a single RK4 integration step. If angular acceleration exceeds
 * the threshold, recursively subdivides the step into two half-steps.
 *
 * @return true if step succeeded, false if max subdivisions exceeded with unsafe acceleration
 */
template <typename Scalar>
inline bool rk4_step_adaptive(
    const Vec6<Scalar>& twist_in,
    const Mat3<Scalar>& R_in,
    const Vec3<Scalar>& p_in,
    const Vec3<Scalar>& n_L,
    const Eigen::Vector3d& g,
    const Scalar actMass,
    const Eigen::Matrix3d& actInertia,
    const Eigen::Matrix<Scalar, 6, 1>& damping,
    const Eigen::Vector3d& B0,
    const Mat3<Scalar>& muhat,
    const Vec3<Scalar>& m_L,
    const Scalar h,              // Step size
    int subdivision_level,       // Current recursion depth
    Vec6<Scalar>& twist_out,     // Output state
    Mat3<Scalar>& R_out,
    Vec3<Scalar>& p_out,
    Vec6<Scalar>& xdot_final)    // Output derivative
{
    // Compute RK4 stage 1
    Vec6<Scalar> k1;
    CoilIntegrand(twist_in, n_L, g, R_in, actMass, actInertia, damping, B0, muhat, m_L, k1);

    // Check acceleration magnitude - subdivide if needed
    if (!is_acceleration_safe(k1) && subdivision_level < kMaxSubdivisionLevels) {
        // Subdivide: two half-steps
        const Scalar h_half = h * Scalar(0.5);

        // First half-step
        Vec6<Scalar> twist_mid, xdot_mid;
        Mat3<Scalar> R_mid;
        Vec3<Scalar> p_mid;
        if (!rk4_step_adaptive(twist_in, R_in, p_in, n_L, g, actMass, actInertia,
                               damping, B0, muhat, m_L, h_half, subdivision_level + 1,
                               twist_mid, R_mid, p_mid, xdot_mid)) {
            return false;  // Subdivision failed
        }

        // Second half-step
        if (!rk4_step_adaptive(twist_mid, R_mid, p_mid, n_L, g, actMass, actInertia,
                               damping, B0, muhat, m_L, h_half, subdivision_level + 1,
                               twist_out, R_out, p_out, xdot_final)) {
            return false;  // Subdivision failed
        }

        return true;  // Successful subdivision
    }

    // If acceleration is safe OR max subdivisions reached, proceed with normal RK4

    // RK4 Stage 2
    const Vec6<Scalar> twist_2 = twist_in + h * k1 * Scalar(0.5);
    Mat3<Scalar> R_2;
    Vec3<Scalar> p_2;
    DYNSE3_TimeSpace(R_in, p_in, h * Scalar(0.5), twist_in, R_2, p_2);

    Vec6<Scalar> k2;
    CoilIntegrand(twist_2, n_L, g, R_2, actMass, actInertia, damping, B0, muhat, m_L, k2);

    // RK4 Stage 3 - use twist_for_R3 for SE3 update to match original implementation
    const Vec6<Scalar> twist_3 = twist_in + h * k2 * Scalar(0.5);
    Mat3<Scalar> R_3;
    Vec3<Scalar> p_3;
    const Vec6<Scalar> twist_for_R3 = twist_in + h * k1 * Scalar(0.5);
    DYNSE3_TimeSpace(R_in, p_in, h * Scalar(0.5), twist_for_R3, R_3, p_3);

    Vec6<Scalar> k3;
    CoilIntegrand(twist_3, n_L, g, R_3, actMass, actInertia, damping, B0, muhat, m_L, k3);

    // RK4 Stage 4 - use twist_for_R4 for SE3 update to match original implementation
    const Vec6<Scalar> twist_4 = twist_in + h * k3;
    Mat3<Scalar> R_4;
    Vec3<Scalar> p_4;
    const Vec6<Scalar> twist_for_R4 = twist_in + h * k2;
    DYNSE3_TimeSpace(R_in, p_in, h, twist_for_R4, R_4, p_4);

    Vec6<Scalar> k4;
    CoilIntegrand(twist_4, n_L, g, R_4, actMass, actInertia, damping, B0, muhat, m_L, k4);

    // RK4 update
    twist_out = twist_in + h * (k1 + Scalar(2.0)*k2 + Scalar(2.0)*k3 + k4) / Scalar(6.0);

    // SE3 update using weighted average twist (midpoint rule for stability)
    const Vec6<Scalar> twist_avg = (twist_in + twist_out) * Scalar(0.5);
    DYNSE3_TimeSpace(R_in, p_in, h, twist_avg, R_out, p_out);

    xdot_final = k1;  // Return initial derivative for diagnostics

    // Check if max subdivisions reached but still unsafe
    if (!is_acceleration_safe(k1) && subdivision_level >= kMaxSubdivisionLevels) {
        return false;  // Max subdivisions exceeded with unsafe acceleration
    }

    return true;  // Success
}

/**
 * @brief RK4 integrator for coil dynamics (Task A1.7 Phase 3)
 *
 * This is a more stable alternative to the ABM4 integrator above. RK4 has no
 * history dependence and is more robust to stiff systems and large perturbations
 * (e.g., during AutoDiff parameter sweeps).
 *
 * Phase 3.4: Now includes adaptive step-size subdivision when angular acceleration
 * exceeds threshold (1000 rad/s²). Subdivision recurses up to 4 levels (16x refinement).
 *
 * Interface matches CoilDynamics exactly for drop-in replacement.
 *
 * @tparam Scalar Numeric type (double or autodiff::real)
 */
template <typename Scalar>
inline void CoilDynamicsRK4(const Vec6<Scalar>& v0w0,
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
                            Vec6<Scalar>& out_xdot_n,
                            bool* out_diverged = nullptr)
{
    // Use same substep size as ABM4 for consistency
    const int N = static_cast<int>(std::ceil(DELTA_T / kCoilTStep));
    const Scalar h = Scalar(kCoilTStep);
    constexpr double kDivergenceThreshold = 1e6;  // Phase 3: Soft failure mode

    Vec6<Scalar> twist_n = v0w0;
    Vec3<Scalar> p_n = p0;
    Mat3<Scalar> R_n = R0;

    Vec6<Scalar> twist_np1 = twist_n;
    Vec3<Scalar> p_np1 = p_n;
    Mat3<Scalar> R_np1 = R_n;

    for (int idx = 0; idx < N; ++idx) {
        // Phase 3.4: Adaptive RK4 step with subdivision if angular acceleration exceeds threshold
        Vec6<Scalar> xdot_final;
        bool step_success = rk4_step_adaptive(
            twist_n, R_n, p_n,
            n_L, g, actMass, actInertia, damping, B0, muhat, m_L,
            h,          // Step size
            0,          // Initial subdivision level
            twist_np1, R_np1, p_np1,
            xdot_final
        );

        out_xdot_n = xdot_final;  // Save for output

        if (!step_success) {
            // Adaptive stepping failed - max subdivisions exceeded with unsafe acceleration
            if (out_diverged != nullptr) {
                *out_diverged = true;
            }
            return;  // Early exit
        }

        // Phase 3: Check for divergence (soft failure mode - magnitude check)
        if (out_diverged != nullptr) {
            const double twist_mag = std::sqrt(static_cast<double>(twist_np1.squaredNorm()));
            const double p_mag = std::sqrt(static_cast<double>(p_np1.squaredNorm()));
            if (!std::isfinite(twist_mag) || !std::isfinite(p_mag) ||
                twist_mag > kDivergenceThreshold || p_mag > kDivergenceThreshold) {
                *out_diverged = true;
                return;  // Early exit on divergence
            }
        }

        // Update state for next iteration
        twist_n = twist_np1;
        R_n = R_np1;
        p_n = p_np1;
    }

    v1w1 = twist_n;
    p1 = p_n;
    R1 = R_n;

    // Phase 3: Final divergence check
    if (out_diverged != nullptr) {
        *out_diverged = false;  // Success
    }
}

/**
 * @brief Dispatcher for coil dynamics integrator (Task A1.7 Phase 3.4)
 *
 * Calls either CoilDynamics (ABM4) or CoilDynamicsRK4 based on the selected
 * integrator type. This allows runtime selection of integration method.
 *
 * @tparam Scalar Numeric type (double or autodiff::real)
 */
template <typename Scalar>
inline void CoilDynamicsDispatch(IntegratorType integrator,
                                 const Vec6<Scalar>& v0w0,
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
                                 Vec6<Scalar>& out_xdot_n,
                                 bool* out_diverged = nullptr)
{
    switch (integrator) {
        case IntegratorType::RK4:
            CoilDynamicsRK4(v0w0, p0, R0, n_L, g, actMass, actInertia, damping, DELTA_T, B0, muhat, m_L, v1w1, p1, R1, out_xdot_n, out_diverged);
            break;
        case IntegratorType::ABM4:
        default:
            CoilDynamics(v0w0, p0, R0, n_L, g, actMass, actInertia, damping, DELTA_T, B0, muhat, m_L, v1w1, p1, R1, out_xdot_n, out_diverged);
            break;
    }
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
                                   const DynamicsContextAD<Scalar>& ctx,
                                   const Vec3<Scalar>& in_u,
                                   const Vec3<Scalar>& in_nL,
                                   Vec3<Scalar>& out_u,
                                   Vec3<Scalar>& out_p,
                                   Mat3<Scalar>& out_R)
{
    // Guard against null geometry pointer (indicates incomplete context initialization)
    if (!ctx.geometry) {
        throw std::runtime_error(
            "CRMFlexible_IVP_BackAD: geometry pointer is null; "
            "context must be initialized via DynamicsContextAD::from_params()");
    }

    // Extract geometry data and learnable parameters
    const DYNNLEqnParams& Params = *ctx.geometry;
    const Mat3<Scalar> K = ctx.learnable.getK();
    const Mat3<Scalar> Kinv = ctx.learnable.getKinv();
    const Vec3<Scalar>& ustar = ctx.learnable.ustar;

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

// Phase 2 Task 2.1: Templated forward IVP for AD gradient computation
template <typename Scalar>
inline void CRMFlexible_IVP_ForwardAD(const int SegmentIndex,
                                      const Vec3<Scalar>& in_p,
                                      const Mat3<Scalar>& in_R,
                                      const DynamicsContextAD<Scalar>& ctx,
                                      const Vec3<Scalar>& in_u,
                                      const Vec3<Scalar>& in_nL,
                                      Vec3<Scalar>& out_u,
                                      Vec3<Scalar>& out_p,
                                      Mat3<Scalar>& out_R)
{
    // Guard against null geometry pointer (indicates incomplete context initialization)
    if (!ctx.geometry) {
        throw std::runtime_error(
            "CRMFlexible_IVP_ForwardAD: geometry pointer is null; "
            "context must be initialized via DynamicsContextAD::from_params()");
    }

    // Extract geometry data and learnable parameters
    const DYNNLEqnParams& Params = *ctx.geometry;
    const Mat3<Scalar> K = ctx.learnable.getK();
    const Mat3<Scalar> Kinv = ctx.learnable.getKinv();
    const Vec3<Scalar>& ustar = ctx.learnable.ustar;

    const int fsegno = SegmentIndex >> 1;
    const double h = (Params.SegBounds[SegmentIndex + 1] - Params.SegBounds[SegmentIndex]) / (Params.SegSteps[fsegno] * 1.0);
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

    double t_n = Params.SegBounds[SegmentIndex];
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

// Phase 2 Task 2.2: Output mapping evaluation using AD
// This function computes the output y = g(x*, theta) where:
//   x* are the solved m_L, n_L values
//   theta are the parameters (currents, seed state)
//   y = [u_tip (3), v_coil (3)]
//
// Note: This is a simplified version that evaluates output at the equilibrium point
// The full forward dynamics (CRMIVP_DYN_AD) would be implemented here in a complete solution
// Task 4.9: Returns dynamic-sized output for multi-actuator support
template <typename Scalar>
inline Eigen::Matrix<Scalar, Eigen::Dynamic, 1> eval_output_AD(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x_scaled,
    const DynamicsContextAD<Scalar>& ctx)
{
    if (!ctx.geometry) {
        throw std::runtime_error(
            "eval_output_AD: geometry pointer is null");
    }

    const DYNNLEqnParams& Params = *ctx.geometry;
    const int num_sets = Params.dynamics.size();
    const int output_dim = 3 + 3 * num_sets;  // tip_position + velocity_per_actuator

    // Unpack x_scaled to get m_L, n_L
    std::array<Vec3<Scalar>, NUM_ACT_SET> m_L_all, n_L_all;
    for (int j = 0; j < num_sets && j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; ++i) {
            m_L_all[j](i) = Scalar(IVALUE_SCALE_M) * x_scaled(i + j * 6);
            n_L_all[j](i) = Scalar(IVALUE_SCALE_N) * x_scaled(i + j * 6 + 3);
        }
    }

    // Get tip force
    Vec3<Scalar> n_0;
    for (int i = 0; i < 3; ++i) n_0(i) = Scalar(Params.TipForce[i]);

    // Get initial root configuration
    Vec3<Scalar> p_0;
    Mat3<Scalar> R_0;
    for (int i = 0; i < 3; ++i) p_0(i) = Scalar(Params.xi[i]);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            R_0(r, c) = Scalar(Params.xi[3 + r * 3 + c]);
        }
    }

    // Get tip state
    Vec3<Scalar> p_t;
    Mat3<Scalar> R_t;
    for (int i = 0; i < 3; ++i) p_t(i) = Scalar(Params.xf[i]);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            R_t(r, c) = Scalar(Params.xf[3 + r * 3 + c]);
        }
    }

    // Extract learnable parameters
    const Mat3<Scalar> K = ctx.learnable.getK();
    const Mat3<Scalar> Kinv = ctx.learnable.getKinv();
    const Vec3<Scalar>& ustar = ctx.learnable.ustar;
    const Scalar& actMass = ctx.learnable.actMass;
    const Eigen::Matrix<Scalar, 6, 1>& damping = ctx.learnable.damping;
    const Mat3<Scalar> muhat = ctx.learnable.getMuHat();

    const int NUM_SEGMENTS = Params.no_segments;

    // Integrate backward from tip to get u at tip
    Vec3<Scalar> tau_0 = Vec3<Scalar>::Zero();
    Vec3<Scalar> u_t = ustar + Kinv * tau_0;

    // Backward pass through last flexible segment to get state at first actuator
    Vec3<Scalar> u_tau, p_flex;
    Mat3<Scalar> R_flex_mat;
    const int last_flex_seg = NUM_SEGMENTS - 1;
    CRMFlexible_IVP_BackAD(last_flex_seg, p_t, R_t, ctx, u_t, n_0, u_tau, p_flex, R_flex_mat);

    // Compute moment at boundary
    Vec3<Scalar> tau = K * (u_tau - ustar);

    // Get first actuator state
    const auto& act0 = Params.dynamics.actuators[0];
    Vec6<Scalar> vw_coil;
    Vec3<Scalar> p_coil;
    Mat3<Scalar> R_coil;
    for (int i = 0; i < 3; ++i) {
        vw_coil(i) = Scalar(act0.v_L_pre(i));
        vw_coil(3 + i) = Scalar(act0.w_L_pre(i));
        p_coil(i) = Scalar(act0.p_pre(i));
    }
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            R_coil(r, c) = Scalar(act0.R_pre(r, c));
        }
    }

    // Compute net forces/moments on first actuator
    Vec3<Scalar> net_nL;
    if (num_sets > 1) {
        net_nL = n_L_all[0] - n_L_all[1];
    } else {
        net_nL = n_L_all[0] - n_0;
    }

    Vec3<Scalar> net_mL = m_L_all[0] - tau;

    // Integrate actuator dynamics
    Vec6<Scalar> vw_out, xdot_dummy = Vec6<Scalar>::Zero();
    Vec3<Scalar> p_out;
    Mat3<Scalar> R_out;

    CoilDynamicsDispatch(ctx.integrator_type,
                        vw_coil, p_coil, R_coil, net_nL,
                        ctx.g, actMass, ctx.actInertia, damping,
                        ctx.DELTA_T, ctx.B0, muhat, net_mL,
                        vw_out, p_out, R_out, xdot_dummy);

    // Forward integrate through flexible segment to get new tip position
    // Start from actuator boundary state (p_out, R_out) and integrate to tip
    Vec3<Scalar> u_boundary = ustar + Kinv * tau;  // Curvature at actuator/flex boundary
    Vec3<Scalar> p_tip_new, u_tip_new;
    Mat3<Scalar> R_tip_new;

    // Integrate through the last flexible segment (actuator -> tip)
    CRMFlexible_IVP_ForwardAD(last_flex_seg, p_out, R_out, ctx, u_boundary, n_0,
                              u_tip_new, p_tip_new, R_tip_new);

    // Task 4.9: Build output with dynamic sizing for multi-actuator support
    // Output: [p_tip (3), v_coil_0 (3), v_coil_1 (3), ...]
    // Size: 3 + 3*num_sets
    //
    // IMPORTANT: This output must match the FD version in crm_bindings.cpp eval_output lambda:
    // y(0-2) = tip position (xf_new[0-2])
    // y(3-5) = actuator 0 linear velocity (x_coil[0][0-2])
    // y(6-8) = actuator 1 linear velocity (x_coil[1][0-2]) [if num_sets > 1]
    // ... etc
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> y(output_dim);

    // Tip position
    y(0) = p_tip_new(0);  // New tip position X (after forward integration)
    y(1) = p_tip_new(1);  // New tip position Y
    y(2) = p_tip_new(2);  // New tip position Z

    // Actuator 0 velocity (currently only this is computed)
    y(3) = vw_out(0);     // Actuator 0 linear velocity X
    y(4) = vw_out(1);     // Actuator 0 linear velocity Y
    y(5) = vw_out(2);     // Actuator 0 linear velocity Z

    // TODO: For NUM_ACT_SET > 1, loop through remaining actuators
    // and compute their dynamics to fill y(6+), y(9+), etc.
    // This requires implementing actuator chaining logic.

    return y;
}

// Task 4.7: Overload of eval_output_AD that accepts parameters as AD variables
// This enables differentiation w.r.t. currents and seed state
// Task 4.9: Returns dynamic-sized output for multi-actuator support
template <typename Scalar>
inline Eigen::Matrix<Scalar, Eigen::Dynamic, 1> eval_output_AD_with_params(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x_scaled,
    const Eigen::Matrix<Scalar, 3, 1>& currents_ad,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& seed_flat_ad,
    const DynamicsContextAD<Scalar>& ctx)
{
    if (!ctx.geometry) {
        throw std::runtime_error(
            "eval_output_AD_with_params: geometry pointer is null");
    }

    const DYNNLEqnParams& Params = *ctx.geometry;
    const int num_sets = Params.dynamics.size();
    const int output_dim = 3 + 3 * num_sets;  // tip_position + velocity_per_actuator

    // Unpack x_scaled to get m_L, n_L (same as before)
    std::array<Vec3<Scalar>, NUM_ACT_SET> m_L_all, n_L_all;
    for (int j = 0; j < num_sets && j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; ++i) {
            m_L_all[j](i) = Scalar(IVALUE_SCALE_M) * x_scaled(i + j * 6);
            n_L_all[j](i) = Scalar(IVALUE_SCALE_N) * x_scaled(i + j * 6 + 3);
        }
    }

    // Get tip force (constant, not differentiable)
    Vec3<Scalar> n_0;
    for (int i = 0; i < 3; ++i) n_0(i) = Scalar(Params.TipForce[i]);

    // Get initial root configuration (constant)
    Vec3<Scalar> p_0;
    Mat3<Scalar> R_0;
    for (int i = 0; i < 3; ++i) p_0(i) = Scalar(Params.xi[i]);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            R_0(r, c) = Scalar(Params.xi[3 + r * 3 + c]);
        }
    }

    // Extract tip state from seed_flat_ad instead of Params
    // seed_flat layout: [v (num_sets*3), w (num_sets*3), p (num_sets*3),
    //                    R (num_sets*9), xf (15), mL (num_sets*3), nL (num_sets*3)]
    const int dim_v = num_sets * 3;
    const int dim_w = num_sets * 3;
    const int dim_p = num_sets * 3;
    const int dim_R = num_sets * 9;
    const int dim_xf = 15; // NUM_STATES

    int offset = dim_v + dim_w + dim_p + dim_R;

    Vec3<Scalar> p_t;
    Mat3<Scalar> R_t;
    for (int i = 0; i < 3; ++i) p_t(i) = seed_flat_ad(offset + i);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            R_t(r, c) = seed_flat_ad(offset + 3 + r * 3 + c);
        }
    }

    // Extract learnable parameters (same as before - from context)
    const Mat3<Scalar> K = ctx.learnable.getK();
    const Mat3<Scalar> Kinv = ctx.learnable.getKinv();
    const Vec3<Scalar>& ustar = ctx.learnable.ustar;
    const Scalar& actMass = ctx.learnable.actMass;
    const Eigen::Matrix<Scalar, 6, 1>& damping = ctx.learnable.damping;
    const Mat3<Scalar> muhat = ctx.learnable.getMuHat();

    const int NUM_SEGMENTS = Params.no_segments;

    // Integrate backward from tip to get u at tip
    Vec3<Scalar> tau_0 = Vec3<Scalar>::Zero();
    Vec3<Scalar> u_t = ustar + Kinv * tau_0;

    // Backward pass through last flexible segment to get state at first actuator
    Vec3<Scalar> u_tau, p_flex;
    Mat3<Scalar> R_flex_mat;
    const int last_flex_seg = NUM_SEGMENTS - 1;
    CRMFlexible_IVP_BackAD(last_flex_seg, p_t, R_t, ctx, u_t, n_0, u_tau, p_flex, R_flex_mat);

    // Compute moment at boundary
    Vec3<Scalar> tau = K * (u_tau - ustar);

    // Extract first actuator state from seed_flat_ad instead of Params
    // seed layout: v, w, p, R, xf, mL, nL
    offset = 0;
    Vec6<Scalar> vw_coil;
    Vec3<Scalar> p_coil;
    Mat3<Scalar> R_coil;

    // v for actuator 0
    for (int i = 0; i < 3; ++i) {
        vw_coil(i) = seed_flat_ad(offset + 0 * 3 + i);  // v[0][i]
    }
    // w for actuator 0
    offset = dim_v;
    for (int i = 0; i < 3; ++i) {
        vw_coil(3 + i) = seed_flat_ad(offset + 0 * 3 + i);  // w[0][i]
    }
    // p for actuator 0
    offset = dim_v + dim_w;
    for (int i = 0; i < 3; ++i) {
        p_coil(i) = seed_flat_ad(offset + 0 * 3 + i);  // p[0][i]
    }
    // R for actuator 0
    offset = dim_v + dim_w + dim_p;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            R_coil(r, c) = seed_flat_ad(offset + 0 * 9 + r * 3 + c);  // R[0][r*3+c]
        }
    }

    // Compute net forces/moments on first actuator
    Vec3<Scalar> net_nL;
    if (num_sets > 1) {
        net_nL = n_L_all[0] - n_L_all[1];
    } else {
        net_nL = n_L_all[0] - n_0;
    }

    Vec3<Scalar> net_mL = m_L_all[0] - tau;

    // Integrate actuator dynamics
    // NOTE: currents_ad is not directly used here because magnetic effects
    // are pre-computed in B0 and muhat. For full current differentiation,
    // we would need to compute magnetic field from currents_ad.
    // For now, this provides differentiation w.r.t. seed state.
    Vec6<Scalar> vw_out, xdot_dummy = Vec6<Scalar>::Zero();
    Vec3<Scalar> p_out;
    Mat3<Scalar> R_out;

    CoilDynamicsDispatch(ctx.integrator_type,
                        vw_coil, p_coil, R_coil, net_nL,
                        ctx.g, actMass, ctx.actInertia, damping,
                        ctx.DELTA_T, ctx.B0, muhat, net_mL,
                        vw_out, p_out, R_out, xdot_dummy);

    // Forward integrate through flexible segment to get new tip position
    Vec3<Scalar> u_boundary = ustar + Kinv * tau;
    Vec3<Scalar> p_tip_new, u_tip_new;
    Mat3<Scalar> R_tip_new;

    CRMFlexible_IVP_ForwardAD(last_flex_seg, p_out, R_out, ctx, u_boundary, n_0,
                              u_tip_new, p_tip_new, R_tip_new);

    // Task 4.9: Build output with dynamic sizing
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> y(output_dim);
    y(0) = p_tip_new(0);
    y(1) = p_tip_new(1);
    y(2) = p_tip_new(2);
    y(3) = vw_out(0);
    y(4) = vw_out(1);
    y(5) = vw_out(2);

    // TODO: For NUM_ACT_SET > 1, compute remaining actuator velocities

    return y;
}

// Phase 2 Task 2.2: Wrapper for computing output Jacobians using autodiff
// Computes gx = ∂y/∂x and gθ = ∂y/∂θ where y = g(x*, θ)
// Task 4.9: Updated to handle dynamic output size (3 + 3*num_sets)
inline void DYNNLEquationOutputJacobianEigenAD(
    const Eigen::VectorXd& x_scaled,
    const Eigen::Vector3d& currents,
    const Eigen::VectorXd& seed_flat,
    DYNNLEqnParams& Params,
    Eigen::MatrixXd& out_gx,    // output_dim × x_dim
    Eigen::MatrixXd& out_gth)   // output_dim × theta_dim
{
    using autodiff::VectorXreal;
    using autodiff::real;
    using autodiff::jacobian;
    using autodiff::wrt;
    using autodiff::at;

    // Create base context from params (double precision)
    dynnl_ad_eigen::DynamicsContextAD<double> ctx_base = dynnl_ad_eigen::DynamicsContextAD<double>::from_params(Params, 0);

    const int num_sets = Params.dynamics.size();
    const int output_dim = 3 + 3 * num_sets;  // tip_position + velocity_per_actuator
    const int x_dim = static_cast<int>(x_scaled.size());
    const int curr_dim = 3;
    const int seed_dim = static_cast<int>(seed_flat.size());
    const int theta_dim = curr_dim + seed_dim;

    out_gx.resize(output_dim, x_dim);
    out_gth.resize(output_dim, theta_dim);

    // Convert inputs to AD types
    VectorXreal x_ad(x_dim);
    for (int i = 0; i < x_dim; ++i) x_ad(i) = x_scaled(i);

    // Compute Jacobian w.r.t. x (state variables m_L, n_L)
    VectorXreal y_x;
    auto output_fn_x = [&ctx_base](const VectorXreal& x_) -> VectorXreal {
        dynnl_ad_eigen::DynamicsContextAD<real> ctx_ad(ctx_base);
        return dynnl_ad_eigen::eval_output_AD(x_, ctx_ad);
    };

    jacobian(output_fn_x, wrt(x_ad), at(x_ad), y_x, out_gx);

    // Task 4.7: Compute Jacobian w.r.t. θ (parameters: currents + seed)
    // θ = [currents (3), seed_flat (seed_dim)]
    VectorXreal theta_ad(theta_dim);
    for (int i = 0; i < curr_dim; ++i) theta_ad(i) = currents(i);
    for (int i = 0; i < seed_dim; ++i) theta_ad(curr_dim + i) = seed_flat(i);

    VectorXreal y_theta;
    auto output_fn_theta = [&ctx_base, &x_ad](const VectorXreal& th_) -> VectorXreal {
        dynnl_ad_eigen::DynamicsContextAD<real> ctx_ad(ctx_base);

        // Split theta into currents and seed
        Eigen::Matrix<real, 3, 1> curr_ad;
        for (int i = 0; i < 3; ++i) curr_ad(i) = th_(i);

        const int seed_size = static_cast<int>(th_.size()) - 3;
        VectorXreal seed_ad(seed_size);
        for (int i = 0; i < seed_size; ++i) seed_ad(i) = th_(3 + i);

        return dynnl_ad_eigen::eval_output_AD_with_params(x_ad, curr_ad, seed_ad, ctx_ad);
    };

    jacobian(output_fn_theta, wrt(theta_ad), at(theta_ad), y_theta, out_gth);
}

// New overload: Accept DynamicsContextAD for consistent parameter handling (Task A1.7 Phase 2)
template <typename Scalar>
inline Eigen::Matrix<Scalar, NUM_DYN_RESIDUAL, 1> DYNNLEquationResidualEigenAD(const autodiff::VectorXreal& x_scaled,
                                                                              const DynamicsContextAD<Scalar>& ctx)
{
    using Resid = Eigen::Matrix<Scalar, NUM_DYN_RESIDUAL, 1>;

    // Guard against null geometry pointer (indicates incomplete context initialization)
    if (!ctx.geometry) {
        throw std::runtime_error(
            "DYNNLEquationResidualEigenAD: geometry pointer is null; "
            "context must be initialized via DynamicsContextAD::from_params()");
    }

    const DYNNLEqnParams& Params = *ctx.geometry;
    // Unpack NLE variables (scaled) -> physical m_L, n_L for each actuator.
    // Layout: [m_L_0[3], n_L_0[3], m_L_1[3], n_L_1[3], ...]
    std::array<Vec3<Scalar>, NUM_ACT_SET> m_L_all, n_L_all;
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; ++i) {
            m_L_all[j](i) = Scalar(IVALUE_SCALE_M) * x_scaled(i + j*6);
            n_L_all[j](i) = Scalar(IVALUE_SCALE_N) * x_scaled(i + j*6 + 3);
        }
    }
    // For single-actuator compatibility, use first actuator
    Vec3<Scalar>& m_L = m_L_all[0];
    Vec3<Scalar>& n_L = n_L_all[0];

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

    // Extract learnable parameters from context
    const Scalar& actMass = ctx.learnable.actMass;
    const Eigen::Matrix3d& actInertia = ctx.actInertia;
    const Eigen::Matrix<Scalar, 6, 1>& damping = ctx.learnable.damping;
    const Mat3<Scalar> muhat = ctx.learnable.getMuHat();

    // Iterate segments distal->proximal (same direction as original: segi = NUM_SEGMENTS-1..0).
    Vec3<Scalar> net_mL = Vec3<Scalar>::Zero();
    Vec3<Scalar> net_nL = Vec3<Scalar>::Zero();

    // Coil output cache for segments above.
    bool have_out_coil = false;
    Vec6<Scalar> vw_coil_out = Vec6<Scalar>::Zero();
    Vec3<Scalar> p_coil_out = Vec3<Scalar>::Zero();
    Mat3<Scalar> R_coil_out = Mat3<Scalar>::Identity();

    for (int segi = NUM_SEGMENTS - 1; segi >= 0; --segi) {
        if (segi % 2 == 0) {
            // flexible segment
            const int fsegi = segi >> 1;
            const Vec3<Scalar>& ustar = ctx.learnable.ustar;
            const Mat3<Scalar> Kinv = ctx.learnable.getKinv();

            if (segi == NUM_SEGMENTS - 1) {
                // last segment: tau_0 = 0, n_0 is tip force.
                u_t = ustar + Kinv * tau_0;
                CRMFlexible_IVP_BackAD(segi, p_t, R_t, ctx, u_t, n_0, u_tau, p_, R_);

                // Calculate tau and net_mL for actuator below
                const Mat3<Scalar> K = ctx.learnable.getK();
                tau = K * (u_tau - ustar);
                const int actno = fsegi - 1;
                net_mL = m_L_all[actno] - tau;
            } else {
                // non-free tip flexible segments with coil on top
                const int actno = fsegi;
                // u_L = ustar + Kinv * m_L[actno]
                const Vec3<Scalar> u_L = ustar + Kinv * m_L_all[actno];
                const int actseg = segi + 1;
                const double RigidSegmentLength = Params.SegBounds[actseg + 1] - Params.SegBounds[actseg];

                // Starting pose for this flex segment is half a rigid segment "above" coil center
                const Vec3<Scalar> p_L = p_coil_out - R_coil_out.col(2) * Scalar(RigidSegmentLength) * Scalar(0.5);
                const Mat3<Scalar> R_L = R_coil_out;
                CRMFlexible_IVP_BackAD(segi, p_L, R_L, ctx, u_L, n_L_all[actno], u_f, p_f, R_f);

                const int actno_below = fsegi - 1;
                if (actno_below >= 0) {
                    // Calculate moment at upper side of coil and net torque for actuator below
                    const Mat3<Scalar> K = ctx.learnable.getK();
                    tau = K * (u_f - ustar);
                    net_mL = m_L_all[actno_below] - tau;
                }
            }
        } else {
            // rigid (actuator) segment: integrate coil dynamics
            const int actno = (segi - 1) >> 1;

            // Calculate net force: n_L[actno] - n_L[actno+1] or n_L[actno] - n_0 for last actuator
            if (actno < NUM_ACT_SET - 1) {
                net_nL = n_L_all[actno] - n_L_all[actno + 1];
            } else {
                net_nL = n_L_all[actno] - n_0;
            }

            // Get actuator-specific initial state and parameters
            const auto& act = Params.dynamics.actuators[actno];
            Vec6<Scalar> vw_coil;
            Vec3<Scalar> p_coil;
            Mat3<Scalar> R_coil;
            for (int i = 0; i < 3; ++i) {
                vw_coil(i) = Scalar(act.v_L_pre(i));
                vw_coil(3 + i) = Scalar(act.w_L_pre(i));
                p_coil(i) = Scalar(act.p_pre(i));
            }
            for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) {
                R_coil(r, c) = Scalar(act.R_pre(r, c));
            }

            CoilDynamicsDispatch(ctx.integrator_type,
                                 vw_coil, p_coil, R_coil, net_nL,
                                 ctx.g, actMass, actInertia, damping,
                                 ctx.DELTA_T, ctx.B0, muhat, net_mL,
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
    const DynamicsContextAD<Scalar>& ctx)
{
    using Resid = Eigen::Matrix<Scalar, NUM_DYN_RESIDUAL, 1>;

    // Guard against null geometry pointer (indicates incomplete context initialization)
    if (!ctx.geometry) {
        throw std::runtime_error(
            "DYNNLEquationResidualWithParamsAD: geometry pointer is null; "
            "context must be initialized via DynamicsContextAD::from_params()");
    }

    // Geometry data accessed via ctx.geometry pointer
    const DYNNLEqnParams& Params = *ctx.geometry;

    // Unpack NLE variables (scaled) -> physical m_L, n_L for each actuator.
    // Layout: [m_L_0[3], n_L_0[3], m_L_1[3], n_L_1[3], ...]
    std::array<Vec3<Scalar>, NUM_ACT_SET> m_L_all, n_L_all;
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; ++i) {
            m_L_all[j](i) = Scalar(IVALUE_SCALE_M) * x_scaled(i + j*6);
            n_L_all[j](i) = Scalar(IVALUE_SCALE_N) * x_scaled(i + j*6 + 3);
        }
    }
    // For single-actuator compatibility, use first actuator
    Vec3<Scalar>& m_L = m_L_all[0];
    Vec3<Scalar>& n_L = n_L_all[0];

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

    // Extract learnable parameters from context
    const Scalar& actMass = ctx.learnable.actMass;
    const Eigen::Matrix3d& actInertia = ctx.actInertia;
    const Eigen::Matrix<Scalar, 6, 1>& damping = ctx.learnable.damping;
    const Mat3<Scalar> muhat = ctx.learnable.getMuHat();

    const Mat3<Scalar> K = ctx.learnable.getK();
    const Mat3<Scalar> Kinv = ctx.learnable.getKinv();
    const Vec3<Scalar>& ustar = ctx.learnable.ustar;

    Vec3<Scalar> net_mL = Vec3<Scalar>::Zero();
    Vec3<Scalar> net_nL = Vec3<Scalar>::Zero();

    bool have_out_coil = false;
    Vec6<Scalar> vw_coil_out = Vec6<Scalar>::Zero();
    Vec3<Scalar> p_coil_out = Vec3<Scalar>::Zero();
    Mat3<Scalar> R_coil_out = Mat3<Scalar>::Identity();

    for (int segi = NUM_SEGMENTS - 1; segi >= 0; --segi) {
        if (segi % 2 == 0) {
            const int fsegi = segi >> 1;

            if (segi == NUM_SEGMENTS - 1) {
                u_t = ustar + Kinv * tau_0;
                CRMFlexible_IVP_BackAD(segi, p_t, R_t, ctx, u_t, n_0, u_tau, p_, R_);

                tau = K * (u_tau - ustar);
                const int actno = fsegi - 1;
                net_mL = m_L_all[actno] - tau;
            } else {
                const int actno = fsegi;
                const Vec3<Scalar> u_L = ustar + Kinv * m_L_all[actno];
                const int actseg = segi + 1;
                const double RigidSegmentLength = Params.SegBounds[actseg + 1] - Params.SegBounds[actseg];

                const Vec3<Scalar> p_L = p_coil_out - R_coil_out.col(2) * Scalar(RigidSegmentLength) * Scalar(0.5);
                const Mat3<Scalar> R_L = R_coil_out;
                CRMFlexible_IVP_BackAD(segi, p_L, R_L, ctx, u_L, n_L_all[actno], u_f, p_f, R_f);

                const int actno_below = fsegi - 1;
                if (actno_below >= 0) {
                    tau = K * (u_f - ustar);
                    net_mL = m_L_all[actno_below] - tau;
                }
            }
        } else {
            const int actno = (segi - 1) >> 1;

            if (actno < NUM_ACT_SET - 1) {
                net_nL = n_L_all[actno] - n_L_all[actno + 1];
            } else {
                net_nL = n_L_all[actno] - n_0;
            }

            const auto& act = Params.dynamics.actuators[actno];
            Vec6<Scalar> vw_coil;
            Vec3<Scalar> p_coil;
            Mat3<Scalar> R_coil;
            for (int i = 0; i < 3; ++i) {
                vw_coil(i) = Scalar(act.v_L_pre(i));
                vw_coil(3 + i) = Scalar(act.w_L_pre(i));
                p_coil(i) = Scalar(act.p_pre(i));
            }
            for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) {
                R_coil(r, c) = Scalar(act.R_pre(r, c));
            }

            CoilDynamicsDispatch(ctx.integrator_type,
                                 vw_coil, p_coil, R_coil, net_nL,
                                 ctx.g, actMass, actInertia, damping,
                                 ctx.DELTA_T, ctx.B0, muhat, net_mL,
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

// ============================================================================
// Control Input Gradient Support (Task 1.5)
// ============================================================================
// Compute ∂F/∂u where u = actuation currents (3D)
// MagMoment = CoilAlignmentTurnAreaMatrix * currents
// ============================================================================

// New overload: Accept DynamicsContextAD for consistent parameter handling (Task A1.7 Phase 2)
template <typename Scalar>
inline Eigen::Matrix<Scalar, NUM_DYN_RESIDUAL, 1> DYNNLEquationResidualWithControlsAD(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x_scaled,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& currents,
    const DynamicsContextAD<Scalar>& ctx)
{
    using Resid = Eigen::Matrix<Scalar, NUM_DYN_RESIDUAL, 1>;

    // Guard against null geometry pointer (indicates incomplete context initialization)
    if (!ctx.geometry) {
        throw std::runtime_error(
            "DYNNLEquationResidualWithControlsAD: geometry pointer is null; "
            "context must be initialized via DynamicsContextAD::from_params()");
    }

    const DYNNLEqnParams& Params = *ctx.geometry;

    // Compute MagMoment from currents using CoilAlignmentTurnAreaMatrix for each actuator
    // currents layout: [i0[3], i1[3], ...] for NUM_ACT_SET actuators
    std::array<Vec3<Scalar>, NUM_ACT_SET> MagMoment_all;
    std::array<Mat3<Scalar>, NUM_ACT_SET> muhat_all;
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        const Eigen::Matrix3d& CATAM = Params.CoilAlignmentTurnAreaMatrix[j];
        Eigen::Matrix<Scalar, 3, 1> currents_j;
        for (int i = 0; i < 3; ++i) {
            currents_j(i) = currents(i + j*3);
        }
        MagMoment_all[j] = CATAM.template cast<Scalar>() * currents_j;
        const Vec3<Scalar>& MM = MagMoment_all[j];
        muhat_all[j] << Scalar(0), -MM(2), MM(1),
                        MM(2), Scalar(0), -MM(0),
                        -MM(1), MM(0), Scalar(0);
    }
    // For single-actuator compatibility, use first actuator
    Mat3<Scalar>& muhat = muhat_all[0];

    // Unpack NLE variables (scaled) -> physical m_L, n_L for each actuator.
    // Layout: [m_L_0[3], n_L_0[3], m_L_1[3], n_L_1[3], ...]
    std::array<Vec3<Scalar>, NUM_ACT_SET> m_L_all, n_L_all;
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; ++i) {
            m_L_all[j](i) = Scalar(IVALUE_SCALE_M) * x_scaled(i + j*6);
            n_L_all[j](i) = Scalar(IVALUE_SCALE_N) * x_scaled(i + j*6 + 3);
        }
    }
    // For single-actuator compatibility, use first actuator
    Vec3<Scalar>& m_L = m_L_all[0];
    Vec3<Scalar>& n_L = n_L_all[0];

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

    // Extract learnable parameters from context
    const Scalar& actMass = ctx.learnable.actMass;
    const Eigen::Matrix3d& actInertia = ctx.actInertia;
    const Eigen::Matrix<Scalar, 6, 1>& damping = ctx.learnable.damping;

    Vec3<Scalar> net_mL = Vec3<Scalar>::Zero();
    Vec3<Scalar> net_nL = Vec3<Scalar>::Zero();

    bool have_out_coil = false;
    Vec6<Scalar> vw_coil_out = Vec6<Scalar>::Zero();
    Vec3<Scalar> p_coil_out = Vec3<Scalar>::Zero();
    Mat3<Scalar> R_coil_out = Mat3<Scalar>::Identity();

    for (int segi = NUM_SEGMENTS - 1; segi >= 0; --segi) {
        if (segi % 2 == 0) {
            const int fsegi = segi >> 1;
            const Vec3<Scalar>& ustar = ctx.learnable.ustar;
            const Mat3<Scalar> Kinv = ctx.learnable.getKinv();
            const Mat3<Scalar> K = ctx.learnable.getK();

            if (segi == NUM_SEGMENTS - 1) {
                u_t = ustar + Kinv * tau_0;
                CRMFlexible_IVP_BackAD(segi, p_t, R_t, ctx, u_t, n_0, u_tau, p_, R_);

                tau = K * (u_tau - ustar);
                const int actno = fsegi - 1;
                net_mL = m_L_all[actno] - tau;
            } else {
                const int actno = fsegi;
                const Vec3<Scalar> u_L = ustar + Kinv * m_L_all[actno];
                const int actseg = segi + 1;
                const double RigidSegmentLength = Params.SegBounds[actseg + 1] - Params.SegBounds[actseg];

                const Vec3<Scalar> p_L = p_coil_out - R_coil_out.col(2) * Scalar(RigidSegmentLength) * Scalar(0.5);
                const Mat3<Scalar> R_L = R_coil_out;
                CRMFlexible_IVP_BackAD(segi, p_L, R_L, ctx, u_L, n_L_all[actno], u_f, p_f, R_f);

                const int actno_below = fsegi - 1;
                if (actno_below >= 0) {
                    tau = K * (u_f - ustar);
                    net_mL = m_L_all[actno_below] - tau;
                }
            }
        } else {
            const int actno = (segi - 1) >> 1;

            if (actno < NUM_ACT_SET - 1) {
                net_nL = n_L_all[actno] - n_L_all[actno + 1];
            } else {
                net_nL = n_L_all[actno] - n_0;
            }

            const auto& act = Params.dynamics.actuators[actno];
            Vec6<Scalar> vw_coil;
            Vec3<Scalar> p_coil;
            Mat3<Scalar> R_coil;
            for (int i = 0; i < 3; ++i) {
                vw_coil(i) = Scalar(act.v_L_pre(i));
                vw_coil(3 + i) = Scalar(act.w_L_pre(i));
                p_coil(i) = Scalar(act.p_pre(i));
            }
            for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) {
                R_coil(r, c) = Scalar(act.R_pre(r, c));
            }

            CoilDynamicsDispatch(ctx.integrator_type,
                                 vw_coil, p_coil, R_coil, net_nL,
                                 ctx.g, actMass, actInertia, damping,
                                 ctx.DELTA_T, ctx.B0, muhat_all[actno], net_mL,
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

// =============================================================================
// Legacy API Wrappers for Python Bindings
// =============================================================================
// These maintain backward compatibility with the Python bindings while using
// the new LearnableParamsAD implementation internally.

inline constexpr int NUM_LEARNABLE_PARAMS = 16;

inline constexpr int THETA_OFFSET_DAMPING = 0;
inline constexpr int THETA_OFFSET_K_DIAG = 6;
inline constexpr int THETA_OFFSET_USTAR = 9;
inline constexpr int THETA_OFFSET_ACTMASS = 12;
inline constexpr int THETA_OFFSET_MAGMOMENT = 13;

inline Eigen::VectorXd packLearnableParams(const DYNNLEqnParams& params, int flex_seg_index = 0) {
    LearnableParamsAD<double> learnable(params, flex_seg_index);
    return learnable.to_vector();
}

inline std::vector<std::string> getLearnableParamNames() {
    return LearnableParamsAD<double>::get_param_names();
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

    // Create base context from params (double precision)
    dynnl_ad_eigen::DynamicsContextAD<double> ctx_base = dynnl_ad_eigen::DynamicsContextAD<double>::from_params(Params, 0);

    VectorXreal x(x_scaled.size());
    for (int i = 0; i < x_scaled.size(); ++i) x(i) = x_scaled(i);

    VectorXreal y;
    Eigen::MatrixXd J;

    auto residual_fn = [&ctx_base](const VectorXreal& x_) -> VectorXreal {
        // Create AD context from base
        dynnl_ad_eigen::DynamicsContextAD<real> ctx_ad(ctx_base);
        return dynnl_ad_eigen::DYNNLEquationResidualEigenAD<real>(x_, ctx_ad);
    };

    jacobian(residual_fn, wrt(x), at(x), y, J);
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

    // Create base context from params (double precision)
    dynnl_ad_eigen::DynamicsContextAD<double> ctx_base = dynnl_ad_eigen::DynamicsContextAD<double>::from_params(Params, 0);
    Eigen::VectorXd theta_d = ctx_base.learnable.to_vector();

    VectorXreal x_ad(x_scaled.size());
    for (int i = 0; i < x_scaled.size(); ++i) x_ad(i) = x_scaled(i);

    VectorXreal theta_ad(theta_d.size());
    for (int i = 0; i < theta_d.size(); ++i) theta_ad(i) = theta_d(i);

    VectorXreal y_ad;
    Eigen::MatrixXd J_theta;

    auto residual_fn = [&](const VectorXreal& theta_) -> VectorXreal {
        // Create AD context from base, then update learnable params
        dynnl_ad_eigen::DynamicsContextAD<real> ctx_ad(ctx_base);
        ctx_ad.learnable.from_vector(theta_);
        return dynnl_ad_eigen::DYNNLEquationResidualWithParamsAD<real>(x_ad, ctx_ad);
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

    // Create base context from params (double precision)
    dynnl_ad_eigen::DynamicsContextAD<double> ctx_base = dynnl_ad_eigen::DynamicsContextAD<double>::from_params(Params, 0);
    Eigen::VectorXd theta_d = ctx_base.learnable.to_vector();

    VectorXreal x(x_scaled.size());
    for (int i = 0; i < x_scaled.size(); ++i) x(i) = x_scaled(i);

    VectorXreal theta(theta_d.size());
    for (int i = 0; i < theta_d.size(); ++i) theta(i) = theta_d(i);

    Eigen::VectorXd residual_x;
    J_x = DYNNLEquationJacobianEigenAD(x_scaled, Params, &residual_x);

    VectorXreal y;
    auto residual_fn = [&x, &ctx_base](const VectorXreal& theta_) -> VectorXreal {
        // Create AD context from base, then update learnable params
        dynnl_ad_eigen::DynamicsContextAD<real> ctx_ad(ctx_base);
        ctx_ad.learnable.from_vector(theta_);
        return dynnl_ad_eigen::DYNNLEquationResidualWithParamsAD<real>(x, ctx_ad);
    };

    jacobian(residual_fn, wrt(theta), at(theta), y, J_theta);

    if (out_residual) {
        *out_residual = residual_x;
    }

    if (out_theta) {
        *out_theta = theta_d;
    }
}

inline Eigen::MatrixXd DYNNLEquationControlJacobianEigenAD(
    const Eigen::VectorXd& x_scaled,
    const Eigen::VectorXd& currents,
    DYNNLEqnParams& Params,
    Eigen::VectorXd* out_residual = nullptr)
{
    using autodiff::VectorXreal;
    using autodiff::real;
    using autodiff::jacobian;
    using autodiff::wrt;
    using autodiff::at;

    // Create base context from params (double precision)
    dynnl_ad_eigen::DynamicsContextAD<double> ctx_base = dynnl_ad_eigen::DynamicsContextAD<double>::from_params(Params, 0);

    // Convert inputs to autodiff types
    VectorXreal x_ad(x_scaled.size());
    for (int i = 0; i < x_scaled.size(); ++i) x_ad(i) = x_scaled(i);

    VectorXreal u_ad(currents.size());
    for (int i = 0; i < currents.size(); ++i) u_ad(i) = currents(i);

    VectorXreal y_ad;
    Eigen::MatrixXd J_u;

    auto residual_fn = [&x_ad, &ctx_base](const VectorXreal& u_) -> VectorXreal {
        // Create AD context from base
        dynnl_ad_eigen::DynamicsContextAD<real> ctx_ad(ctx_base);
        return dynnl_ad_eigen::DYNNLEquationResidualWithControlsAD<real>(x_ad, u_, ctx_ad);
    };

    jacobian(residual_fn, wrt(u_ad), at(u_ad), y_ad, J_u);

    if (out_residual) {
        out_residual->resize(y_ad.size());
        for (int i = 0; i < y_ad.size(); ++i) (*out_residual)(i) = autodiff::val(y_ad(i));
    }

    return J_u;
}
} // namespace CRMCatheterModel
