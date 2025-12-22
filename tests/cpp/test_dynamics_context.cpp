#include "CRM_DynamicsContext.hpp"
#include <iostream>
#include <cmath>

using namespace CRMCatheterModel;

// Simple test helper
#define TEST_ASSERT(condition, message) \
    if (!(condition)) { \
        std::cerr << "FAILED: " << message << std::endl; \
        return 1; \
    } else { \
        std::cout << "PASSED: " << message << std::endl; \
    }

int main() {
    std::cout << "### Testing DynamicsContext..." << std::endl << std::endl;

    int fail_count = 0;

    // Test 1: Default construction
    {
        std::cout << "Test 1: Default construction" << std::endl;
        DynamicsContext ctx;
        TEST_ASSERT(ctx.size() == 0, "Default context should have zero actuators");
        TEST_ASSERT(ctx.DELTA_T == 0.0, "Default DELTA_T should be zero");
        TEST_ASSERT(!ctx.is_valid(), "Default context should be invalid");
    }

    // Test 2: Resize and basic initialization
    {
        std::cout << "\nTest 2: Resize and initialization" << std::endl;
        DynamicsContext ctx;
        ctx.resize(3);
        TEST_ASSERT(ctx.size() == 3, "Context should have 3 actuators after resize");

        ctx.DELTA_T = 0.05;
        TEST_ASSERT(ctx.is_valid(), "Context should be valid with actuators and DELTA_T > 0");
    }

    // Test 3: ActuatorDynamicsParams default initialization
    {
        std::cout << "\nTest 3: ActuatorDynamicsParams default values" << std::endl;
        ActuatorDynamicsParams params;

        TEST_ASSERT(params.mass == 0.0, "Default mass should be zero");
        TEST_ASSERT(params.damping.norm() == 0.0, "Default damping should be zero vector");
        TEST_ASSERT(params.inertia.norm() == 0.0, "Default inertia should be zero matrix");
        TEST_ASSERT(params.v_L_pre.norm() == 0.0, "Default v_L_pre should be zero vector");
        TEST_ASSERT(params.w_L_pre.norm() == 0.0, "Default w_L_pre should be zero vector");
        TEST_ASSERT(params.p_pre.norm() == 0.0, "Default p_pre should be zero vector");
        TEST_ASSERT(params.R_pre.isIdentity(), "Default R_pre should be identity matrix");
        TEST_ASSERT(params.m_L.norm() == 0.0, "Default m_L should be zero vector");
        TEST_ASSERT(params.n_L.norm() == 0.0, "Default n_L should be zero vector");
    }

    // Test 4: Data assignment and retrieval
    {
        std::cout << "\nTest 4: Data assignment and retrieval" << std::endl;
        DynamicsContext ctx;
        ctx.resize(2);
        ctx.DELTA_T = 0.01;

        // Set first actuator parameters
        ctx.actuators[0].mass = 0.5;
        ctx.actuators[0].damping << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
        ctx.actuators[0].inertia <<
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0;
        ctx.actuators[0].v_L_pre << 0.1, 0.2, 0.3;
        ctx.actuators[0].w_L_pre << 0.4, 0.5, 0.6;
        ctx.actuators[0].p_pre << 10.0, 20.0, 30.0;

        // Verify first actuator
        TEST_ASSERT(ctx.actuators[0].mass == 0.5, "Actuator 0 mass should be 0.5");
        TEST_ASSERT(ctx.actuators[0].damping(0) == 1.0, "Actuator 0 damping[0] should be 1.0");
        TEST_ASSERT(ctx.actuators[0].damping(5) == 6.0, "Actuator 0 damping[5] should be 6.0");
        TEST_ASSERT(ctx.actuators[0].inertia.isIdentity(), "Actuator 0 inertia should be identity");
        TEST_ASSERT(ctx.actuators[0].v_L_pre(0) == 0.1, "Actuator 0 v_L_pre[0] should be 0.1");
        TEST_ASSERT(ctx.actuators[0].p_pre(2) == 30.0, "Actuator 0 p_pre[2] should be 30.0");

        // Verify second actuator is still default-initialized
        TEST_ASSERT(ctx.actuators[1].mass == 0.0, "Actuator 1 mass should be default (0.0)");
        TEST_ASSERT(ctx.actuators[1].damping.norm() == 0.0, "Actuator 1 damping should be zero vector");
    }

    // Test 5: Multiple resize operations
    {
        std::cout << "\nTest 5: Multiple resize operations" << std::endl;
        DynamicsContext ctx;
        ctx.resize(1);
        ctx.actuators[0].mass = 1.0;

        ctx.resize(3);  // Grow
        TEST_ASSERT(ctx.size() == 3, "Context should have 3 actuators after second resize");
        TEST_ASSERT(ctx.actuators[0].mass == 1.0, "First actuator data should be preserved after resize");
        TEST_ASSERT(ctx.actuators[2].mass == 0.0, "New actuators should be default-initialized");

        ctx.resize(1);  // Shrink
        TEST_ASSERT(ctx.size() == 1, "Context should have 1 actuator after shrink");
        TEST_ASSERT(ctx.actuators[0].mass == 1.0, "First actuator data should survive shrink");
    }

    // Test 6: Rotation matrix handling
    {
        std::cout << "\nTest 6: Rotation matrix operations" << std::endl;
        DynamicsContext ctx;
        ctx.resize(1);

        // Set a valid rotation matrix (90-degree rotation around z-axis)
        ctx.actuators[0].R_pre <<
            0.0, -1.0, 0.0,
            1.0,  0.0, 0.0,
            0.0,  0.0, 1.0;

        // Check determinant (should be 1 for valid rotation)
        double det = ctx.actuators[0].R_pre.determinant();
        TEST_ASSERT(std::abs(det - 1.0) < 1e-10, "Rotation matrix determinant should be 1.0");

        // Check orthogonality (R * R^T = I)
        Eigen::Matrix3d identity = ctx.actuators[0].R_pre * ctx.actuators[0].R_pre.transpose();
        TEST_ASSERT(identity.isIdentity(1e-10), "Rotation matrix should be orthogonal");
    }

    // Test 7: Memory independence (no aliasing)
    {
        std::cout << "\nTest 7: Memory independence between actuators" << std::endl;
        DynamicsContext ctx;
        ctx.resize(2);

        // Set different values for each actuator
        ctx.actuators[0].damping << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;
        ctx.actuators[1].damping << 2.0, 2.0, 2.0, 2.0, 2.0, 2.0;

        // Verify no cross-contamination
        TEST_ASSERT(ctx.actuators[0].damping(0) == 1.0, "Actuator 0 damping should remain 1.0");
        TEST_ASSERT(ctx.actuators[1].damping(0) == 2.0, "Actuator 1 damping should remain 2.0");

        // Modify one and verify the other is unaffected
        ctx.actuators[0].mass = 99.0;
        TEST_ASSERT(ctx.actuators[1].mass == 0.0, "Actuator 1 mass should be unaffected");
    }

    std::cout << "\n### All DynamicsContext tests passed!" << std::endl;
    return 0;
}
