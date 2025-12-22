#pragma once

#include <vector>
#include <Eigen/Dense>
#include <autodiff/forward/real.hpp>
#include "CRM_DynamicsContext.hpp"

namespace CRMCatheterModel {
namespace dynnl_ad_eigen {

/**
 * @brief Templated container for dynamics parameters of a single actuator.
 *
 * This is the AD-compatible version of ActuatorDynamicsParams. It uses a template
 * parameter Scalar that can be either double or autodiff::real, enabling automatic
 * differentiation through the dynamics equations.
 *
 * @tparam Scalar Numeric type (double or autodiff::real)
 */
template <typename Scalar>
struct ActuatorDynamicsParamsAD {
    Eigen::Matrix<Scalar, 6, 1> damping;   ///< Linear (3) + angular (3) damping coefficients
    Eigen::Matrix<Scalar, 3, 3> inertia;   ///< 3x3 actuator inertia matrix
    Scalar mass;                            ///< Actuator mass
    Eigen::Matrix<Scalar, 3, 1> v_L_pre;   ///< Linear velocity (previous timestep)
    Eigen::Matrix<Scalar, 3, 1> w_L_pre;   ///< Angular velocity (previous timestep)
    Eigen::Matrix<Scalar, 3, 1> p_pre;     ///< Position (previous timestep)
    Eigen::Matrix<Scalar, 3, 3> R_pre;     ///< Rotation matrix (previous timestep)
    Eigen::Matrix<Scalar, 3, 1> m_L;       ///< Moment at coil
    Eigen::Matrix<Scalar, 3, 1> n_L;       ///< Force at coil

    /**
     * @brief Default constructor - initializes all values to zero.
     */
    ActuatorDynamicsParamsAD()
        : damping(Eigen::Matrix<Scalar, 6, 1>::Zero()),
          inertia(Eigen::Matrix<Scalar, 3, 3>::Zero()),
          mass(Scalar(0.0)),
          v_L_pre(Eigen::Matrix<Scalar, 3, 1>::Zero()),
          w_L_pre(Eigen::Matrix<Scalar, 3, 1>::Zero()),
          p_pre(Eigen::Matrix<Scalar, 3, 1>::Zero()),
          R_pre(Eigen::Matrix<Scalar, 3, 3>::Identity()),
          m_L(Eigen::Matrix<Scalar, 3, 1>::Zero()),
          n_L(Eigen::Matrix<Scalar, 3, 1>::Zero())
    {}

    /**
     * @brief Construct from non-templated ActuatorDynamicsParams.
     *
     * This constructor enables conversion from the base double-precision version
     * to the templated version, which is useful when Scalar = autodiff::real.
     */
    explicit ActuatorDynamicsParamsAD(const ActuatorDynamicsParams& params)
        : mass(Scalar(params.mass))
    {
        // Copy damping
        for (int i = 0; i < 6; ++i) {
            damping(i) = Scalar(params.damping(i));
        }

        // Copy inertia matrix
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                inertia(row, col) = Scalar(params.inertia(row, col));
            }
        }

        // Copy velocities
        for (int i = 0; i < 3; ++i) {
            v_L_pre(i) = Scalar(params.v_L_pre(i));
            w_L_pre(i) = Scalar(params.w_L_pre(i));
        }

        // Copy position
        for (int i = 0; i < 3; ++i) {
            p_pre(i) = Scalar(params.p_pre(i));
        }

        // Copy rotation matrix
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                R_pre(row, col) = Scalar(params.R_pre(row, col));
            }
        }

        // Copy moments and forces
        for (int i = 0; i < 3; ++i) {
            m_L(i) = Scalar(params.m_L(i));
            n_L(i) = Scalar(params.n_L(i));
        }
    }
};

/**
 * @brief Templated container for all dynamics-related parameters.
 *
 * This is the AD-compatible version of DynamicsContext. It enables automatic
 * differentiation through the entire dynamics solver by using templated Eigen
 * matrices and vectors.
 *
 * Design rationale:
 * - Supports both double and autodiff::real types
 * - Zero-copy conversion from DynamicsContext when Scalar = double
 * - Compatible with existing AD infrastructure in CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp
 *
 * @tparam Scalar Numeric type (double or autodiff::real)
 */
template <typename Scalar>
struct DynamicsContextAD {
    std::vector<ActuatorDynamicsParamsAD<Scalar>> actuators;  ///< Dynamics params for each actuator
    double DELTA_T;  ///< Time step (not differentiated, remains double)

    /**
     * @brief Default constructor - initializes empty context.
     */
    DynamicsContextAD() : DELTA_T(0.0) {}

    /**
     * @brief Construct from non-templated DynamicsContext.
     *
     * This constructor enables conversion from the base double-precision version
     * to the templated version. When Scalar = double, this is essentially a copy.
     * When Scalar = autodiff::real, this converts all values to AD types.
     */
    explicit DynamicsContextAD(const DynamicsContext& ctx)
        : DELTA_T(ctx.DELTA_T)
    {
        actuators.reserve(ctx.size());
        for (int i = 0; i < ctx.size(); ++i) {
            actuators.emplace_back(ctx.actuators[i]);
        }
    }

    /**
     * @brief Resize the actuator vector to accommodate no_act_set actuators.
     * @param no_act_set Number of actuators in the system
     */
    void resize(int no_act_set) {
        actuators.resize(no_act_set);
    }

    /**
     * @brief Get the number of actuators in this context.
     * @return Number of actuators
     */
    int size() const {
        return static_cast<int>(actuators.size());
    }

    /**
     * @brief Check if the context is properly initialized.
     * @return true if at least one actuator exists and DELTA_T > 0
     */
    bool is_valid() const {
        return !actuators.empty() && DELTA_T > 0.0;
    }
};

/**
 * @brief Convert a DynamicsContext to DynamicsContextAD<Scalar>.
 *
 * This is a convenience function that creates a templated AD-compatible context
 * from the base double-precision version. It's equivalent to using the constructor
 * but provides a clearer API for conversion.
 *
 * @tparam Scalar Target numeric type (double or autodiff::real)
 * @param ctx Source context (double-precision)
 * @return AD-compatible context
 */
template <typename Scalar>
inline DynamicsContextAD<Scalar> convertToAD(const DynamicsContext& ctx) {
    return DynamicsContextAD<Scalar>(ctx);
}

/**
 * @brief Specialized conversion for double (no-op conversion).
 *
 * When Scalar = double, this provides an optimized path that just copies
 * the data without type conversion overhead.
 */
template <>
inline DynamicsContextAD<double> convertToAD<double>(const DynamicsContext& ctx) {
    DynamicsContextAD<double> result;
    result.DELTA_T = ctx.DELTA_T;
    result.actuators.reserve(ctx.size());

    for (int i = 0; i < ctx.size(); ++i) {
        ActuatorDynamicsParamsAD<double> params;
        params.damping = ctx.actuators[i].damping;
        params.inertia = ctx.actuators[i].inertia;
        params.mass = ctx.actuators[i].mass;
        params.v_L_pre = ctx.actuators[i].v_L_pre;
        params.w_L_pre = ctx.actuators[i].w_L_pre;
        params.p_pre = ctx.actuators[i].p_pre;
        params.R_pre = ctx.actuators[i].R_pre;
        params.m_L = ctx.actuators[i].m_L;
        params.n_L = ctx.actuators[i].n_L;
        result.actuators.push_back(params);
    }

    return result;
}

} // namespace dynnl_ad_eigen
} // namespace CRMCatheterModel
