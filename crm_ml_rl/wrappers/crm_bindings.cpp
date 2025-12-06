/**
 * pybind11 bindings for CRM C++ physics engine.
 *
 * This file creates Python bindings for the Cosserat Rod Model
 * dynamics and kinematics solvers.
 *
 * Build with CMake (see CMakeLists.txt) or manually:
 *   c++ -O3 -Wall -shared -std=c++17 -fPIC \
 *       $(python3 -m pybind11 --includes) \
 *       -I../../src -I../../numerical -I/usr/include/eigen3 \
 *       crm_bindings.cpp -L../../build -lCRMCPPLib \
 *       -o crm_python$(python3-config --extension-suffix)
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <stdexcept>
#include <memory>

// Include CRM headers
#include "CRM.hpp"
#include "CRMDYN.hpp"

namespace py = pybind11;
using namespace CRMCatheterModel;


/**
 * Python wrapper for CRMCatheterModelParams
 *
 * Uses unique_ptr to manage dynamically allocated params to avoid double-free issues.
 */
class PyCatheterParams {
public:
    std::unique_ptr<CRMCatheterModelParams> params;
    CatheterConfiguration config;
    bool initialized = false;

    PyCatheterParams() : params(std::make_unique<CRMCatheterModelParams>(2, 1, 1, 5)) {}

    bool loadFromFiles(const std::string& param_file, const std::string& config_file) {
        try {
            // Load into a new params object
            CRMCatheterModelParams loaded = Load_CRMCatheterModelParams(param_file.c_str());
            // Use placement new or reset with a copy
            params = std::make_unique<CRMCatheterModelParams>(loaded);
            config = Load_CatheterConfiguration(config_file.c_str());
            initialized = true;
            return true;
        } catch (const std::exception& e) {
            py::print("Error loading parameters:", e.what());
            return false;
        }
    }

    // Getters for key parameters
    int getNumFlexSeg() const { return params->no_flex_seg; }
    int getNumRigidSeg() const { return params->no_rigid_seg; }
    int getNumActSet() const { return params->no_act_set; }
    int getNumSegments() const { return params->no_segments; }
    int getNumLocMarkers() const { return params->no_locmarkers; }

    py::array_t<double> getB0() const {
        py::array_t<double> result(3);
        auto buf = result.request();
        double* ptr = static_cast<double*>(buf.ptr);
        for (int i = 0; i < 3; i++) ptr[i] = config.B0[i];
        return result;
    }

    py::array_t<double> getP0() const {
        py::array_t<double> result(3);
        auto buf = result.request();
        double* ptr = static_cast<double*>(buf.ptr);
        for (int i = 0; i < 3; i++) ptr[i] = config.p0[i];
        return result;
    }

    py::array_t<double> getSegLengths() const {
        int n = params->no_segments;
        py::array_t<double> result(n);
        auto buf = result.request();
        double* ptr = static_cast<double*>(buf.ptr);
        for (int i = 0; i < n; i++) {
            ptr[i] = params->SegLengths[i];
        }
        return result;
    }

    // Return pointer to underlying params for use by other wrappers
    CRMCatheterModelParams* getParams() { return params.get(); }
    const CRMCatheterModelParams* getParams() const { return params.get(); }
};


/**
 * Wrapper class for CRM Forward Kinematics
 */
class CRMKinematicsWrapper {
public:
    PyCatheterParams catheter;
    double integrationStepSize = 0.2;  // mm
    bool initialized = false;

    CRMKinematicsWrapper() {}

    bool loadParameters(const std::string& param_file, const std::string& config_file) {
        initialized = catheter.loadFromFiles(param_file, config_file);
        return initialized;
    }

    /**
     * Forward kinematics: compute tip position from currents and insertion length.
     *
     * Args:
     *     currents: Applied currents (NUM_ACT_SET * 3,) - flattened [I0x, I0y, I0z, I1x, ...]
     *     insertion_length: Inserted length of catheter (mm)
     *
     * Returns:
     *     Dictionary with:
     *         - tip_position: (3,) tip position in spatial coordinates
     *         - tip_rotation: (9,) tip rotation matrix (row-major)
     *         - delta_u0: (3,) delta curvature at base
     *         - potential_energy: scalar
     *         - converged: bool
     */
    py::dict forwardKinematics(py::array_t<double> currents, double insertion_length) {
        if (!initialized) {
            throw std::runtime_error("Parameters not loaded. Call load_parameters first.");
        }

        auto curr_buf = currents.request();
        int num_currents = catheter.getParams()->no_act_set * 3;

        if (curr_buf.size != num_currents) {
            throw std::runtime_error("currents must have " + std::to_string(num_currents) + " elements");
        }

        double* curr_ptr = static_cast<double*>(curr_buf.ptr);

        // Build input vector: currents + insertion_length
        int x_dim = num_currents + 1;
        std::vector<double> in_x(x_dim);
        for (int i = 0; i < num_currents; i++) {
            in_x[i] = curr_ptr[i];
        }
        in_x[num_currents] = insertion_length;

        // Setup FK parameters
        CRMForwardKinematicsData FKParams;
        CRMCatheterModelParams* cparams = catheter.getParams();
        FKParams.CathParams = cparams;
        FKParams.CathConfig = &catheter.config;
        FKParams.ContactMode = ContactModeType::FREE_TIP;
        FKParams.TipForce[0] = FKParams.TipForce[1] = FKParams.TipForce[2] = 0.0;
        FKParams.TipConstraintPoint[0] = FKParams.TipConstraintPoint[1] = FKParams.TipConstraintPoint[2] = 0.0;
        FKParams.deltau0_initialguess[0] = FKParams.deltau0_initialguess[1] = FKParams.deltau0_initialguess[2] = 0.0;
        FKParams.ftip_initialguess[0] = FKParams.ftip_initialguess[1] = FKParams.ftip_initialguess[2] = 0.0;
        FKParams.IntegrationStepSize = integrationStepSize;
        FKParams.FinalValueOnly = true;

        // Allocate marker storage
        std::vector<double> markerPosData(cparams->no_locmarkers * 3);
        FKParams.ReportedMarkerPos = reinterpret_cast<double(*)[3]>(markerPosData.data());

        std::vector<double> coilOrientData(cparams->no_act_set * 9);
        FKParams.ReportedCoilOrient = reinterpret_cast<double(*)[9]>(coilOrientData.data());

        std::vector<double> coilPosData(cparams->no_act_set * 3);
        FKParams.ReportedCoilPos = reinterpret_cast<double(*)[3]>(coilPosData.data());

        // Output: p[3], R[9], deltau0[3] for FREE_TIP
        int y_dim = 3 + 9 + 3;
        std::vector<double> out_y(y_dim);

        double potentialEnergy;

        // Call FK
        int localmin = CRM_ForwardKinematics(in_x.data(), out_y.data(), potentialEnergy, FKParams);

        // Pack results - create array with explicit copy
        std::vector<ssize_t> shape3 = {3};
        std::vector<ssize_t> shape9 = {9};

        auto tip_pos = py::array_t<double>(shape3);
        auto tip_rot = py::array_t<double>(shape9);
        auto delta_u0 = py::array_t<double>(shape3);

        // Use unchecked for faster access
        auto pos_acc = tip_pos.mutable_unchecked<1>();
        auto rot_acc = tip_rot.mutable_unchecked<1>();
        auto u0_acc = delta_u0.mutable_unchecked<1>();

        pos_acc(0) = out_y[0];
        pos_acc(1) = out_y[1];
        pos_acc(2) = out_y[2];

        for (int i = 0; i < 9; i++) {
            rot_acc(i) = out_y[3 + i];
        }

        u0_acc(0) = out_y[12];
        u0_acc(1) = out_y[13];
        u0_acc(2) = out_y[14];

        py::dict result;
        result["tip_position"] = tip_pos;
        result["tip_rotation"] = tip_rot;
        result["delta_u0"] = delta_u0;
        result["potential_energy"] = potentialEnergy;
        result["converged"] = (localmin == 0);

        return result;
    }

    /**
     * Compute analytical Jacobian at current configuration.
     */
    py::array_t<double> computeJacobian(py::array_t<double> currents, double insertion_length) {
        if (!initialized) {
            throw std::runtime_error("Parameters not loaded.");
        }

        // First compute FK to get output
        auto fk_result = forwardKinematics(currents, insertion_length);

        CRMCatheterModelParams* cparams = catheter.getParams();
        auto curr_buf = currents.request();
        int num_currents = cparams->no_act_set * 3;
        double* curr_ptr = static_cast<double*>(curr_buf.ptr);

        // Build input vector
        int x_dim = num_currents + 1;
        VectorXd in_x(x_dim);
        for (int i = 0; i < num_currents; i++) {
            in_x(i) = curr_ptr[i];
        }
        in_x(num_currents) = insertion_length;

        // Build FK output vector
        py::array_t<double> tip_pos = fk_result["tip_position"].cast<py::array_t<double>>();
        py::array_t<double> tip_rot = fk_result["tip_rotation"].cast<py::array_t<double>>();
        py::array_t<double> delta_u0 = fk_result["delta_u0"].cast<py::array_t<double>>();

        int y_dim = 3 + 9 + 3;
        VectorXd in_FKouty(y_dim);
        auto pos_buf = tip_pos.request();
        auto rot_buf = tip_rot.request();
        auto u0_buf = delta_u0.request();
        double* pos_ptr = static_cast<double*>(pos_buf.ptr);
        double* rot_ptr = static_cast<double*>(rot_buf.ptr);
        double* u0_ptr = static_cast<double*>(u0_buf.ptr);

        for (int i = 0; i < 3; i++) in_FKouty(i) = pos_ptr[i];
        for (int i = 0; i < 9; i++) in_FKouty(3 + i) = rot_ptr[i];
        for (int i = 0; i < 3; i++) in_FKouty(12 + i) = u0_ptr[i];

        // Setup FK parameters
        CRMForwardKinematicsData FKParams;
        FKParams.CathParams = cparams;
        FKParams.CathConfig = &catheter.config;
        FKParams.ContactMode = ContactModeType::FREE_TIP;
        FKParams.TipForce[0] = FKParams.TipForce[1] = FKParams.TipForce[2] = 0.0;
        FKParams.IntegrationStepSize = integrationStepSize;
        FKParams.FinalValueOnly = true;

        std::vector<double> markerPosData(cparams->no_locmarkers * 3);
        FKParams.ReportedMarkerPos = reinterpret_cast<double(*)[3]>(markerPosData.data());
        std::vector<double> coilOrientData(cparams->no_act_set * 9);
        FKParams.ReportedCoilOrient = reinterpret_cast<double(*)[9]>(coilOrientData.data());
        std::vector<double> coilPosData(cparams->no_act_set * 3);
        FKParams.ReportedCoilPos = reinterpret_cast<double(*)[3]>(coilPosData.data());

        // Compute Jacobian
        MatrixXd J = CRM_FKJacobian_Analytical(in_x, in_FKouty, FKParams);

        // Convert to numpy array
        int rows = J.rows();
        int cols = J.cols();
        py::array_t<double> result({rows, cols});
        auto res_buf = result.request();
        double* res_ptr = static_cast<double*>(res_buf.ptr);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                res_ptr[i * cols + j] = J(i, j);
            }
        }

        return result;
    }

    bool isInitialized() const { return initialized; }
};


/**
 * Wrapper class for CRM Dynamics simulation.
 */
class CRMDynamicsWrapper {
public:
    PyCatheterParams catheter;
    double dt = 0.02;  // Default timestep (50Hz)
    double integrationStepSize = 0.2;  // mm
    bool initialized = false;

    // State storage for dynamics
    double v_L[NUM_ACT_SET][3];
    double w_L[NUM_ACT_SET][3];
    double p_L[NUM_ACT_SET][3];
    double R_L[NUM_ACT_SET][9];
    double mL_guess[NUM_ACT_SET][3];
    double nL_guess[NUM_ACT_SET][3];
    double xf[NUM_STATES];  // tip state
    double damping[NUM_ACT_SET][6];
    double actInertia[NUM_ACT_SET][9];

    CRMDynamicsWrapper() {
        // Initialize state to zeros
        for (int j = 0; j < NUM_ACT_SET; j++) {
            for (int i = 0; i < 3; i++) {
                v_L[j][i] = 0.0;
                w_L[j][i] = 0.0;
                p_L[j][i] = 0.0;
                mL_guess[j][i] = 0.0;
                nL_guess[j][i] = 0.0;
            }
            for (int i = 0; i < 9; i++) {
                R_L[j][i] = (i == 0 || i == 4 || i == 8) ? 1.0 : 0.0;  // Identity
                actInertia[j][i] = 0.0;
            }
            for (int i = 0; i < 6; i++) {
                damping[j][i] = 10.0;  // Default damping
            }
        }
        for (int i = 0; i < NUM_STATES; i++) {
            xf[i] = 0.0;
        }
    }

    bool loadParameters(const std::string& param_file, const std::string& config_file) {
        initialized = catheter.loadFromFiles(param_file, config_file);
        if (initialized) {
            // Initialize inertia based on loaded parameters
            CRMCatheterModelParams* cparams = catheter.getParams();
            for (int i = 0; i < cparams->no_act_set && i < NUM_ACT_SET; i++) {
                double mass = cparams->ActMass[i];
                double r_out = cparams->OuterRadius[0];
                double r_in = cparams->InnerRadius[0];
                double seg_len = cparams->SegLengths[2*i+1];

                double I_zz = 0.5 * mass * (r_out*r_out + r_in*r_in);
                double I_xx = 0.25 * mass * (r_out*r_out + r_in*r_in) + (1.0/12.0) * mass * seg_len * seg_len;

                actInertia[i][0] = I_xx;
                actInertia[i][4] = I_xx;
                actInertia[i][8] = I_zz;
            }
        }
        return initialized;
    }

    void setDamping(py::array_t<double> damping_values) {
        auto buf = damping_values.request();
        double* ptr = static_cast<double*>(buf.ptr);

        for (int j = 0; j < NUM_ACT_SET; j++) {
            for (int i = 0; i < 6 && (j*6 + i) < buf.size; i++) {
                damping[j][i] = ptr[j*6 + i];
            }
        }
    }

    void setTimestep(double timestep) {
        dt = timestep;
    }

    /**
     * Reset dynamics state to zeros (use initializeFromKinematics for proper initialization).
     */
    void reset() {
        for (int j = 0; j < NUM_ACT_SET; j++) {
            for (int i = 0; i < 3; i++) {
                v_L[j][i] = 0.0;
                w_L[j][i] = 0.0;
                p_L[j][i] = 0.0;
                mL_guess[j][i] = 0.0;
                nL_guess[j][i] = 0.0;
            }
            for (int i = 0; i < 9; i++) {
                R_L[j][i] = (i == 0 || i == 4 || i == 8) ? 1.0 : 0.0;
            }
        }
        for (int i = 0; i < NUM_STATES; i++) {
            xf[i] = 0.0;
        }
    }

    /**
     * Initialize dynamics state from a valid forward kinematics solution.
     *
     * This MUST be called before stepDynamics() to ensure the solver starts
     * from a physically valid configuration and avoids convergence issues.
     *
     * Args:
     *     currents: Applied currents (NUM_ACT_SET * 3,)
     *     insertion_length: Inserted length (mm)
     *
     * Returns:
     *     True if initialization succeeded, false otherwise.
     */
    bool initializeFromKinematics(py::array_t<double> currents, double insertion_length) {
        if (!initialized) {
            throw std::runtime_error("Parameters not loaded.");
        }

        auto curr_buf = currents.request();
        double* curr_ptr = static_cast<double*>(curr_buf.ptr);

        // Build input vector: currents + insertion_length
        CRMCatheterModelParams* cparams = catheter.getParams();
        int num_currents = cparams->no_act_set * 3;
        int x_dim = num_currents + 1;
        std::vector<double> in_x(x_dim);
        for (int i = 0; i < num_currents; i++) {
            in_x[i] = (i < curr_buf.size) ? curr_ptr[i] : 0.0;
        }
        in_x[num_currents] = insertion_length;

        // Setup FK parameters
        CRMForwardKinematicsData FKParams;
        FKParams.CathParams = cparams;
        FKParams.CathConfig = &catheter.config;
        FKParams.ContactMode = ContactModeType::FREE_TIP;
        FKParams.TipForce[0] = FKParams.TipForce[1] = FKParams.TipForce[2] = 0.0;
        FKParams.TipConstraintPoint[0] = FKParams.TipConstraintPoint[1] = FKParams.TipConstraintPoint[2] = 0.0;
        FKParams.deltau0_initialguess[0] = FKParams.deltau0_initialguess[1] = FKParams.deltau0_initialguess[2] = 0.0;
        FKParams.ftip_initialguess[0] = FKParams.ftip_initialguess[1] = FKParams.ftip_initialguess[2] = 0.0;
        FKParams.IntegrationStepSize = integrationStepSize;
        FKParams.FinalValueOnly = false;  // Need intermediate values for coil states

        // Allocate marker and coil storage
        std::vector<double> markerPosData(cparams->no_locmarkers * 3);
        FKParams.ReportedMarkerPos = reinterpret_cast<double(*)[3]>(markerPosData.data());

        std::vector<double> coilOrientData(cparams->no_act_set * 9);
        FKParams.ReportedCoilOrient = reinterpret_cast<double(*)[9]>(coilOrientData.data());

        std::vector<double> coilPosData(cparams->no_act_set * 3);
        FKParams.ReportedCoilPos = reinterpret_cast<double(*)[3]>(coilPosData.data());

        // Output: p[3], R[9], deltau0[3] for FREE_TIP
        int y_dim = 3 + 9 + 3;
        std::vector<double> out_y(y_dim);

        double potentialEnergy;

        // Call FK to get valid configuration
        int localmin = CRM_ForwardKinematics(in_x.data(), out_y.data(), potentialEnergy, FKParams);

        if (localmin != 0) {
            // FK didn't converge, return false but don't throw
            py::print("Warning: FK did not converge during dynamics initialization");
            return false;
        }

        // Initialize tip state (xf) from FK output
        // FK output layout: p[0..2], R[0..8], deltau0[0..2]
        // xf layout (same as FK output): p[0..2], R[0..8], u[0..2]
        // Position from FK
        xf[0] = out_y[0];
        xf[1] = out_y[1];
        xf[2] = out_y[2];

        // Rotation matrix from FK
        for (int i = 0; i < 9; i++) {
            xf[3 + i] = out_y[3 + i];
        }

        // Curvature (u) from delta_u0
        xf[12] = out_y[12];
        xf[13] = out_y[13];
        xf[14] = out_y[14];

        // Initialize coil states from FK-reported coil positions and orientations
        for (int j = 0; j < cparams->no_act_set && j < NUM_ACT_SET; j++) {
            // Zero velocities (static equilibrium)
            v_L[j][0] = 0.0;
            v_L[j][1] = 0.0;
            v_L[j][2] = 0.0;
            w_L[j][0] = 0.0;
            w_L[j][1] = 0.0;
            w_L[j][2] = 0.0;

            // Coil position from FK
            p_L[j][0] = FKParams.ReportedCoilPos[j][0];
            p_L[j][1] = FKParams.ReportedCoilPos[j][1];
            p_L[j][2] = FKParams.ReportedCoilPos[j][2];

            // Coil rotation from FK
            for (int i = 0; i < 9; i++) {
                R_L[j][i] = FKParams.ReportedCoilOrient[j][i];
            }

            // Reset internal force/moment guesses
            mL_guess[j][0] = 0.0;
            mL_guess[j][1] = 0.0;
            mL_guess[j][2] = 0.0;
            nL_guess[j][0] = 0.0;
            nL_guess[j][1] = 0.0;
            nL_guess[j][2] = 0.0;
        }

        // Validate rotation matrix orthogonality
        double det = R_L[0][0] * (R_L[0][4] * R_L[0][8] - R_L[0][5] * R_L[0][7])
                   - R_L[0][1] * (R_L[0][3] * R_L[0][8] - R_L[0][5] * R_L[0][6])
                   + R_L[0][2] * (R_L[0][3] * R_L[0][7] - R_L[0][4] * R_L[0][6]);

        if (std::abs(det - 1.0) > 0.01) {
            py::print("Warning: Coil rotation matrix det =", det, "(should be 1.0)");
        }

        return true;
    }

    /**
     * Step dynamics forward in time.
     *
     * Args:
     *     currents: Applied currents (NUM_ACT_SET * 3,)
     *     insertion_length: Inserted length (mm)
     *
     * Returns:
     *     Dictionary with tip_position, tip_velocity, coil_states, converged
     */
    py::dict stepDynamics(py::array_t<double> currents, double insertion_length) {
        if (!initialized) {
            throw std::runtime_error("Parameters not loaded.");
        }

        auto curr_buf = currents.request();
        double* curr_ptr = static_cast<double*>(curr_buf.ptr);

        // Build actuation currents array
        double ActuationCurrents[NUM_ACT_SET][3];
        for (int i = 0; i < NUM_ACT_SET; i++) {
            for (int j = 0; j < 3; j++) {
                int idx = i * 3 + j;
                ActuationCurrents[i][j] = (idx < curr_buf.size) ? curr_ptr[idx] : 0.0;
            }
        }

        // Setup shooting method parameters
        ContactModeType ContactMode = ContactModeType::FREE_TIP;
        double TipForce[3] = {0.0, 0.0, 0.0};
        double TipConstraintPoint[3] = {0.0, 0.0, 0.0};

        CRMShootingMethodParams BVPParams = CRMDYNConstructShootingMethodParamSet(
            *catheter.getParams(), catheter.config, insertion_length, ActuationCurrents,
            ContactMode, TipConstraintPoint, TipForce, integrationStepSize,
            actInertia, v_L, w_L, p_L, R_L, damping, dt
        );

        // Solve BVP
        double out_u0[3];
        double out_mL[NUM_ACT_SET][3], out_nL[NUM_ACT_SET][3];
        double out_tau[NUM_ACT_SET][3];
        double ftip_calc[3];
        double ftip_guess[3] = {0.0, 0.0, 0.0};
        int localmin;

        DynamicsBVP(BVPParams, xf, mL_guess, nL_guess, ftip_guess,
                    out_u0, out_mL, out_nL, out_tau, ftip_calc, localmin);

        // Solve IVP
        double xf_new[NUM_STATES];
        double x_coil[NUM_ACT_SET][NUM_COIL_STATES];
        double ReportedMarkerPos[5][3];  // Assuming max 5 markers

        DYNSolverIVP(BVPParams, out_u0, out_mL, out_nL, out_tau, ftip_calc,
                     true, xf_new, x_coil, ReportedMarkerPos);

        // Update state
        for (int j = 0; j < NUM_ACT_SET; j++) {
            for (int i = 0; i < 3; i++) {
                v_L[j][i] = x_coil[j][i];
                w_L[j][i] = x_coil[j][i + 3];
                p_L[j][i] = x_coil[j][i + 6];
                mL_guess[j][i] = out_mL[j][i];
                nL_guess[j][i] = out_nL[j][i];
            }
            for (int i = 0; i < 9; i++) {
                R_L[j][i] = x_coil[j][i + 9];
            }
        }
        for (int i = 0; i < NUM_STATES; i++) {
            xf[i] = xf_new[i];
        }

        // Extract tip position from xf (indices 0-2 are position)
        std::vector<ssize_t> shape3 = {3};
        auto tip_pos = py::array_t<double>(shape3);
        auto pos_buf = tip_pos.mutable_unchecked<1>();
        pos_buf(0) = xf[0];
        pos_buf(1) = xf[1];
        pos_buf(2) = xf[2];

        // Tip velocity from coil state
        auto tip_vel = py::array_t<double>(shape3);
        auto vel_buf = tip_vel.mutable_unchecked<1>();
        vel_buf(0) = v_L[0][0];
        vel_buf(1) = v_L[0][1];
        vel_buf(2) = v_L[0][2];

        py::dict result;
        result["tip_position"] = tip_pos;
        result["tip_velocity"] = tip_vel;
        result["converged"] = (localmin == 0);

        return result;
    }

    /**
     * Get current tip position.
     */
    py::array_t<double> getTipPosition() const {
        // Use explicit shape specification to ensure correct array creation
        std::vector<ssize_t> shape = {3};
        auto result = py::array_t<double>(shape);
        auto buf = result.mutable_unchecked<1>();
        // xf layout: p[0..2], R[3..11], u[12..14]
        buf(0) = xf[0];
        buf(1) = xf[1];
        buf(2) = xf[2];
        return result;
    }

    bool isInitialized() const { return initialized; }
};


// Python module definition
PYBIND11_MODULE(crm_python, m) {
    m.doc() = "Python bindings for CRM C++ physics engine";

    // PyCatheterParams
    py::class_<PyCatheterParams>(m, "CatheterParams")
        .def(py::init<>())
        .def("load_from_files", &PyCatheterParams::loadFromFiles,
             py::arg("param_file"), py::arg("config_file"),
             "Load catheter parameters from files")
        .def_property_readonly("num_flex_seg", &PyCatheterParams::getNumFlexSeg)
        .def_property_readonly("num_rigid_seg", &PyCatheterParams::getNumRigidSeg)
        .def_property_readonly("num_act_set", &PyCatheterParams::getNumActSet)
        .def_property_readonly("num_segments", &PyCatheterParams::getNumSegments)
        .def_property_readonly("num_loc_markers", &PyCatheterParams::getNumLocMarkers)
        .def_property_readonly("B0", &PyCatheterParams::getB0)
        .def_property_readonly("p0", &PyCatheterParams::getP0)
        .def_property_readonly("seg_lengths", &PyCatheterParams::getSegLengths)
        .def_readonly("initialized", &PyCatheterParams::initialized);

    // CRMKinematicsWrapper
    py::class_<CRMKinematicsWrapper>(m, "CRMKinematics")
        .def(py::init<>())
        .def("load_parameters", &CRMKinematicsWrapper::loadParameters,
             py::arg("param_file"), py::arg("config_file"),
             "Load catheter parameters from files")
        .def("forward_kinematics", &CRMKinematicsWrapper::forwardKinematics,
             py::arg("currents"), py::arg("insertion_length"),
             "Compute forward kinematics")
        .def("compute_jacobian", &CRMKinematicsWrapper::computeJacobian,
             py::arg("currents"), py::arg("insertion_length"),
             "Compute analytical Jacobian")
        .def_readwrite("integration_step_size", &CRMKinematicsWrapper::integrationStepSize)
        .def("is_initialized", &CRMKinematicsWrapper::isInitialized);

    // CRMDynamicsWrapper
    py::class_<CRMDynamicsWrapper>(m, "CRMDynamics")
        .def(py::init<>())
        .def("load_parameters", &CRMDynamicsWrapper::loadParameters,
             py::arg("param_file"), py::arg("config_file"),
             "Load catheter parameters from files")
        .def("set_damping", &CRMDynamicsWrapper::setDamping,
             py::arg("damping_values"),
             "Set damping coefficients")
        .def("set_timestep", &CRMDynamicsWrapper::setTimestep,
             py::arg("dt"),
             "Set simulation timestep")
        .def("reset", &CRMDynamicsWrapper::reset,
             "Reset dynamics state to zeros")
        .def("initialize_from_kinematics", &CRMDynamicsWrapper::initializeFromKinematics,
             py::arg("currents"), py::arg("insertion_length"),
             "Initialize dynamics from FK solution (MUST call before step)")
        .def("step", &CRMDynamicsWrapper::stepDynamics,
             py::arg("currents"), py::arg("insertion_length"),
             "Step dynamics forward")
        .def("get_tip_position", &CRMDynamicsWrapper::getTipPosition,
             "Get current tip position")
        .def_readwrite("dt", &CRMDynamicsWrapper::dt)
        .def_readwrite("integration_step_size", &CRMDynamicsWrapper::integrationStepSize)
        .def("is_initialized", &CRMDynamicsWrapper::isInitialized);
}
