#pragma once

#include <vector>
#include <string>
#include <stdexcept>
#include <Eigen/Dense>
#include <autodiff/forward/real.hpp>
#include "CRM_DynamicsContext.hpp"

namespace CRMCatheterModel {

namespace dynnl_ad_eigen {

// Type aliases for cleaner code
template <typename Scalar>
using Vec3 = Eigen::Matrix<Scalar, 3, 1>;

template <typename Scalar>
using Mat3 = Eigen::Matrix<Scalar, 3, 3, Eigen::RowMajor>;

/**
 * @brief Container for all learnable parameters in the CRM dynamics model.
 *
 * This struct holds the 16 learnable parameters that can be optimized via gradient descent.
 * It supports both double and autodiff::real types for automatic differentiation.
 *
 * Parameter breakdown (16 total):
 * - damping[6]: Linear (3) + angular (3) damping coefficients (per-actuator)
 * - K_diag[3]: Diagonal of stiffness matrix K (per-flexible-segment)
 * - ustar[3]: Rest curvature (per-flexible-segment)
 * - actMass: Actuator mass (per-actuator)
 * - MagMoment[3]: Magnetic moment vector (per-actuator)
 *
 * Note: For NUM_ACT_SET==1 systems, all parameters from actuator[0] and flex_seg[0] are used.
 *
 * @tparam Scalar Numeric type (double or autodiff::real)
 */
template <typename Scalar>
struct LearnableParamsAD {
    // Actuator parameters (from actuator[0])
    Eigen::Matrix<Scalar, 6, 1> damping;   ///< Linear (3) + angular (3) damping coefficients
    Scalar actMass;                         ///< Actuator mass
    Eigen::Matrix<Scalar, 3, 1> MagMoment; ///< Magnetic moment vector

    // Flexible segment parameters (from flex_seg[0] for single-actuator systems)
    Eigen::Matrix<Scalar, 3, 1> K_diag;    ///< Diagonal of stiffness matrix K
    Eigen::Matrix<Scalar, 3, 1> ustar;     ///< Rest curvature

    /**
     * @brief Default constructor - zero-initialize all parameters.
     */
    LearnableParamsAD()
        : damping(Eigen::Matrix<Scalar, 6, 1>::Zero()),
          actMass(Scalar(0.0)),
          MagMoment(Eigen::Matrix<Scalar, 3, 1>::Zero()),
          K_diag(Eigen::Matrix<Scalar, 3, 1>::Zero()),
          ustar(Eigen::Matrix<Scalar, 3, 1>::Zero())
    {}

    /**
     * @brief Construct from DYNNLEqnParams by extracting learnable parameters.
     *
     * Extracts parameters from:
     * - actuator[0] for damping, actMass, MagMoment
     * - flex_seg[flex_seg_idx] for K_diag, ustar
     *
     * @param params Source parameter struct
     * @param flex_seg_idx Which flexible segment to extract K and ustar from (default 0)
     */
    explicit LearnableParamsAD(const DYNNLEqnParams& params, int flex_seg_idx = 0);

    /**
     * @brief Reconstruct full 3x3 stiffness matrix K from diagonal.
     * @return Diagonal matrix with K_diag on the diagonal
     */
    Mat3<Scalar> getK() const {
        Mat3<Scalar> K = Mat3<Scalar>::Zero();
        K(0, 0) = K_diag(0);
        K(1, 1) = K_diag(1);
        K(2, 2) = K_diag(2);
        return K;
    }

    /**
     * @brief Compute inverse of stiffness matrix Kinv from diagonal.
     * @return Diagonal matrix with 1/K_diag(i) on the diagonal (with regularization)
     */
    Mat3<Scalar> getKinv() const {
        Mat3<Scalar> Kinv = Mat3<Scalar>::Zero();
        Kinv(0, 0) = Scalar(1.0) / (K_diag(0) + Scalar(1e-12));
        Kinv(1, 1) = Scalar(1.0) / (K_diag(1) + Scalar(1e-12));
        Kinv(2, 2) = Scalar(1.0) / (K_diag(2) + Scalar(1e-12));
        return Kinv;
    }

    /**
     * @brief Compute skew-symmetric matrix from magnetic moment vector.
     * @return 3x3 skew-symmetric matrix muhat such that muhat * v = MagMoment x v
     */
    Mat3<Scalar> getMuHat() const {
        Mat3<Scalar> muhat;
        muhat << Scalar(0), -MagMoment(2), MagMoment(1),
                 MagMoment(2), Scalar(0), -MagMoment(0),
                 -MagMoment(1), MagMoment(0), Scalar(0);
        return muhat;
    }

    /**
     * @brief Serialize all learnable parameters to a flat vector.
     *
     * Parameter order (16 total):
     * [0-5]:   damping[6]
     * [6-8]:   K_diag[3]
     * [9-11]:  ustar[3]
     * [12]:    actMass
     * [13-15]: MagMoment[3]
     *
     * This order matches THETA_OFFSET_* constants in the legacy code.
     *
     * @return 16-element vector of learnable parameters
     */
    Eigen::Matrix<Scalar, 16, 1> to_vector() const {
        Eigen::Matrix<Scalar, 16, 1> theta;
        theta.template segment<6>(0) = damping;
        theta.template segment<3>(6) = K_diag;
        theta.template segment<3>(9) = ustar;
        theta(12) = actMass;
        theta.template segment<3>(13) = MagMoment;
        return theta;
    }

    /**
     * @brief Deserialize learnable parameters from a flat vector.
     *
     * This is the inverse of to_vector(). It updates all fields from the input vector.
     *
     * @param theta 16-element vector of learnable parameters
     * @throws std::invalid_argument if theta.size() != 16
     */
    void from_vector(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& theta) {
        if (theta.size() != 16) {
            throw std::invalid_argument(
                "LearnableParamsAD::from_vector: expected 16 parameters, got " +
                std::to_string(theta.size()));
        }
        damping = theta.template segment<6>(0);
        K_diag = theta.template segment<3>(6);
        ustar = theta.template segment<3>(9);
        actMass = theta(12);
        MagMoment = theta.template segment<3>(13);
    }

    /**
     * @brief Get names of all learnable parameters for debugging/logging.
     * @return Vector of 16 parameter names
     */
    static std::vector<std::string> get_param_names() {
        return {
            "damping_v0", "damping_v1", "damping_v2",
            "damping_w0", "damping_w1", "damping_w2",
            "K_diag_0", "K_diag_1", "K_diag_2",
            "ustar_0", "ustar_1", "ustar_2",
            "actMass",
            "MagMoment_0", "MagMoment_1", "MagMoment_2"
        };
    }
};

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
    IntegratorType integrator_type;  ///< Which integrator to use (not differentiated)

    // NEW: Learnable parameters (replaces shadow struct DYNNLEqnParamsAD)
    LearnableParamsAD<Scalar> learnable;  ///< All 16 learnable parameters

    // NEW: Non-learnable physical constants (remain double for efficiency)
    Eigen::Vector3d B0;           ///< Magnetic field vector
    Eigen::Vector3d g;            ///< Gravity vector
    Eigen::Matrix3d actInertia;   ///< Actuator inertia matrix (3x3)

    // NEW: Pointer to geometry data (not owned, read-only access)
    const DYNNLEqnParams* geometry;  ///< Pointer for accessing xi, xf, TipForce, SegBounds, etc. Must outlive this context.

    /**
     * @brief Default constructor - initializes empty context with ABM4 (legacy default).
     */
    DynamicsContextAD()
        : DELTA_T(0.0),
          integrator_type(IntegratorType::ABM4),
          learnable(),
          B0(Eigen::Vector3d::Zero()),
          g(Eigen::Vector3d::Zero()),
          actInertia(Eigen::Matrix3d::Zero()),
          geometry(nullptr)
    {}

    /**
     * @brief Construct from non-templated DynamicsContext.
     *
     * This constructor enables conversion from the base double-precision version
     * to the templated version. When Scalar = double, this is essentially a copy.
     * When Scalar = autodiff::real, this converts all values to AD types.
     *
     * Note: This constructor does NOT populate learnable, B0, g, actInertia, or geometry.
     * Use the DYNNLEqnParams constructor for full initialization.
     */
    explicit DynamicsContextAD(const DynamicsContext& ctx)
        : DELTA_T(ctx.DELTA_T),
          integrator_type(ctx.integrator_type),
          learnable(),
          B0(Eigen::Vector3d::Zero()),
          g(Eigen::Vector3d::Zero()),
          actInertia(Eigen::Matrix3d::Zero()),
          geometry(nullptr)
    {
        actuators.reserve(ctx.size());
        for (int i = 0; i < ctx.size(); ++i) {
            actuators.emplace_back(ctx.actuators[i]);
        }
    }

    /**
     * @brief Construct from DYNNLEqnParams with full initialization.
     *
     * This is the preferred constructor for AD residual functions. It extracts:
     * - Learnable parameters from actuator[0] and flex_seg[flex_seg_idx]
     * - Physical constants (B0, g, actInertia)
     * - Dynamics state from DynamicsContext
     * - Geometry pointer for read-only access
     *
     * @param params Source parameter struct
     * @param flex_seg_idx Which flexible segment to extract K and ustar from (default 0)
     */
    explicit DynamicsContextAD(const DYNNLEqnParams& params, int flex_seg_idx = 0);

    /**
     * @brief Static factory method for clarity.
     *
     * This is equivalent to the constructor but provides clearer syntax:
     * `auto ctx = DynamicsContextAD<real>::from_params(params);`
     *
     * @param params Source parameter struct
     * @param flex_seg_idx Which flexible segment to extract K and ustar from (default 0)
     * @return Fully initialized DynamicsContextAD
     */
    static DynamicsContextAD from_params(const DYNNLEqnParams& params, int flex_seg_idx = 0) {
        return DynamicsContextAD(params, flex_seg_idx);
    }

    /**
     * @brief Template copy constructor for converting double→AD types.
     *
     * This enables creating a DynamicsContextAD<autodiff::real> from a
     * DynamicsContextAD<double>, which is useful in the gradient computation lambda.
     *
     * @tparam OtherScalar Source scalar type (typically double)
     * @param other Source context
     */
    template <typename OtherScalar>
    explicit DynamicsContextAD(const DynamicsContextAD<OtherScalar>& other)
        : DELTA_T(other.DELTA_T),
          integrator_type(other.integrator_type),
          B0(other.B0),
          g(other.g),
          actInertia(other.actInertia),
          geometry(other.geometry)
    {
        // Convert actuators
        actuators.reserve(other.actuators.size());
        for (const auto& act : other.actuators) {
            actuators.emplace_back();
            auto& dst = actuators.back();
            // Convert each field from OtherScalar to Scalar
            for (int i = 0; i < 6; ++i) dst.damping(i) = Scalar(act.damping(i));
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    dst.inertia(r, c) = Scalar(act.inertia(r, c));
                }
            }
            dst.mass = Scalar(act.mass);
            for (int i = 0; i < 3; ++i) {
                dst.v_L_pre(i) = Scalar(act.v_L_pre(i));
                dst.w_L_pre(i) = Scalar(act.w_L_pre(i));
                dst.p_pre(i) = Scalar(act.p_pre(i));
            }
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    dst.R_pre(r, c) = Scalar(act.R_pre(r, c));
                }
            }
            for (int i = 0; i < 3; ++i) {
                dst.m_L(i) = Scalar(act.m_L(i));
                dst.n_L(i) = Scalar(act.n_L(i));
            }
        }

        // Convert learnable parameters
        for (int i = 0; i < 6; ++i) learnable.damping(i) = Scalar(other.learnable.damping(i));
        learnable.actMass = Scalar(other.learnable.actMass);
        for (int i = 0; i < 3; ++i) {
            learnable.MagMoment(i) = Scalar(other.learnable.MagMoment(i));
            learnable.K_diag(i) = Scalar(other.learnable.K_diag(i));
            learnable.ustar(i) = Scalar(other.learnable.ustar(i));
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

    /**
     * @brief Check if the context is valid for AD residual computation.
     * @return true if basic validity checks pass AND geometry pointer is non-null
     *
     * This is a stricter check than is_valid() since AD residual functions
     * require the geometry pointer to be properly initialized.
     */
    bool is_valid_for_ad() const {
        return is_valid() && geometry != nullptr;
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
    result.integrator_type = ctx.integrator_type;
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

// =============================================================================
// Template Implementation Note
// =============================================================================
// The constructors that take DYNNLEqnParams are implemented in
// CRM_DynamicsContext_AD_impl.hpp, which must be included AFTER CRMDYN.hpp

} // namespace dynnl_ad_eigen
} // namespace CRMCatheterModel
