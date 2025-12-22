#pragma once

#include <vector>
#include <Eigen/Dense>

namespace CRMCatheterModel {

/**
 * @brief Container for dynamics parameters of a single actuator.
 *
 * This struct replaces the legacy fixed-size arrays (e.g., double damping[NUM_ACT_SET][6])
 * with type-safe Eigen containers. This eliminates memory aliasing issues that caused the
 * "Ghost Value" bug where magnetic moment values leaked into damping array storage.
 */
struct ActuatorDynamicsParams {
    Eigen::Matrix<double, 6, 1> damping;   ///< Linear (3) + angular (3) damping coefficients
    Eigen::Matrix3d inertia;                ///< 3x3 actuator inertia matrix
    double mass;                            ///< Actuator mass
    Eigen::Vector3d v_L_pre;                ///< Linear velocity (previous timestep)
    Eigen::Vector3d w_L_pre;                ///< Angular velocity (previous timestep)
    Eigen::Vector3d p_pre;                  ///< Position (previous timestep)
    Eigen::Matrix3d R_pre;                  ///< Rotation matrix (previous timestep)
    Eigen::Vector3d m_L;                    ///< Moment at coil
    Eigen::Vector3d n_L;                    ///< Force at coil

    /**
     * @brief Default constructor - initializes all values to zero.
     */
    ActuatorDynamicsParams()
        : damping(Eigen::Matrix<double, 6, 1>::Zero()),
          inertia(Eigen::Matrix3d::Zero()),
          mass(0.0),
          v_L_pre(Eigen::Vector3d::Zero()),
          w_L_pre(Eigen::Vector3d::Zero()),
          p_pre(Eigen::Vector3d::Zero()),
          R_pre(Eigen::Matrix3d::Identity()),
          m_L(Eigen::Vector3d::Zero()),
          n_L(Eigen::Vector3d::Zero())
    {}
};

/**
 * @brief Modern container for all dynamics-related parameters.
 *
 * This struct centralizes dynamics state and eliminates the memory layout issues
 * present in the legacy CRMIVPCoreParams fixed-size arrays. All storage is heap-allocated
 * via std::vector, preventing compiler padding mismatches and NUM_ACT_SET synchronization errors.
 *
 * Design rationale:
 * - Prevents "Ghost Value" bug (magnetic moment overwriting damping)
 * - Supports dynamic resizing for multi-actuator systems
 * - Compatible with Eigen::Map for zero-copy data transfer
 * - Clear ownership semantics (no raw pointer arithmetic)
 */
struct DynamicsContext {
    std::vector<ActuatorDynamicsParams> actuators;  ///< Dynamics params for each actuator [no_act_set]
    double DELTA_T;                                  ///< Time step for integration

    /**
     * @brief Default constructor - initializes empty context.
     */
    DynamicsContext() : DELTA_T(0.0) {}

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

} // namespace CRMCatheterModel
