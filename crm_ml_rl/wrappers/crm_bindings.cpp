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
#include <array>
#include <optional>
#include <vector>
#include <string>
#include <stdexcept>
#include <memory>
#include <limits>
#include <cstdlib>
#include <cmath>

// Include CRM headers
#include "CRM.hpp"
#include "CRMDYN.hpp"
#include "CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp"

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
            if (const char* debug = std::getenv("CRM_DEBUG_PARAM_LOAD")) {
                if (std::string(debug) == "1") {
                    std::cout << "[CRM_DEBUG_PARAM_LOAD] PyCatheterParams seg_lengths:";
                    for (int i = 0; i < params->no_segments; ++i) {
                        std::cout << " " << params->SegLengths[i];
                    }
                    std::cout << "\n[CRM_DEBUG_PARAM_LOAD] PyCatheterParams g: "
                              << config.g[0] << " " << config.g[1] << " " << config.g[2] << "\n";
                }
            }
            return true;
        } catch (const std::exception& e) {
            py::print("Error loading parameters:", e.what());
            return false;
        }
    }

    // Getters for key data/simulation_parameters
    int getNumFlexSeg() const { return params->no_flex_seg; }
    int getNumRigidSeg() const { return params->no_rigid_seg; }
    int getNumActSet() const { return params->no_act_set; }
    int getNumSegments() const { return params->no_segments; }
    int getNumLocMarkers() const { return params->no_locmarkers; }

    py::array_t<double> getB0() const {
        py::array_t<double> result({3}, {static_cast<ssize_t>(sizeof(double))});
        auto buf = result.mutable_unchecked<1>();
        for (int i = 0; i < 3; i++) buf(i) = config.B0[i];
        return result;
    }

    py::array_t<double> getP0() const {
        py::array_t<double> result({3}, {static_cast<ssize_t>(sizeof(double))});
        auto buf = result.mutable_unchecked<1>();
        for (int i = 0; i < 3; i++) buf(i) = config.p0[i];
        return result;
    }

    py::array_t<double> getSegLengths() const {
        int n = params->no_segments;
        py::array_t<double> result({n}, {static_cast<ssize_t>(sizeof(double))});
        auto buf = result.mutable_unchecked<1>();
        for (int i = 0; i < n; i++) {
            buf(i) = params->SegLengths[i];
        }
        if (const char* debug = std::getenv("CRM_DEBUG_PARAM_LOAD")) {
            if (std::string(debug) == "1") {
                std::cout << "[CRM_DEBUG_PARAM_LOAD] getSegLengths:";
                for (int i = 0; i < n; ++i) std::cout << " " << params->SegLengths[i];
                std::cout << "\n";
            }
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

    bool loadParams(const std::string& param_file, const std::string& config_file) {
        initialized = catheter.loadFromFiles(param_file, config_file);
        return initialized;
    }

    py::dict get_config_snapshot() const {
        py::dict out;
        const CRMCatheterModelParams* cparams = catheter.getParams();
        py::array_t<double> B0({3}, {static_cast<ssize_t>(sizeof(double))});
        py::array_t<double> g({3}, {static_cast<ssize_t>(sizeof(double))});
        py::array_t<double> p0({3}, {static_cast<ssize_t>(sizeof(double))});
        py::array_t<double> R0({9}, {static_cast<ssize_t>(sizeof(double))});
        py::array_t<double> seg_lengths({cparams ? cparams->no_segments : 0}, {static_cast<ssize_t>(sizeof(double))});

        if (cparams) {
            auto b = B0.mutable_unchecked<1>();
            auto gg = g.mutable_unchecked<1>();
            auto pp = p0.mutable_unchecked<1>();
            auto rr = R0.mutable_unchecked<1>();
            for (int i = 0; i < 3; i++) {
                b(i) = catheter.config.B0[i];
                gg(i) = catheter.config.g[i];
                pp(i) = catheter.config.p0[i];
            }
            for (int i = 0; i < 9; i++) rr(i) = catheter.config.R0[i];
            auto sl = seg_lengths.mutable_unchecked<1>();
            for (int i = 0; i < cparams->no_segments; i++) sl(i) = cparams->SegLengths[i];
        }

        out["B0"] = B0;
        out["g"] = g;
        out["p0"] = p0;
        out["R0"] = R0;
        out["seg_lengths"] = seg_lengths;

        if (const char* debug = std::getenv("CRM_DEBUG_PARAM_LOAD")) {
            if (std::string(debug) == "1") {
                std::cout << "[CRM_DEBUG_PARAM_LOAD] get_config_snapshot g: "
                          << catheter.config.g[0] << " " << catheter.config.g[1] << " " << catheter.config.g[2] << "\n";
                std::cout << "[CRM_DEBUG_PARAM_LOAD] get_config_snapshot R0:";
                for (int i = 0; i < 9; ++i) std::cout << " " << catheter.config.R0[i];
                std::cout << "\n[CRM_DEBUG_PARAM_LOAD] get_config_snapshot seg_lengths:";
                for (int i = 0; i < cparams->no_segments; ++i) std::cout << " " << cparams->SegLengths[i];
                std::cout << "\n";
            }
        }
        return out;
    }

private:
    void fillFKParams(CRMForwardKinematicsData& FKParams, CRMCatheterModelParams* cparams, bool finalValueOnly) {
        FKParams.CathParams = cparams;
        FKParams.CathConfig = &catheter.config;
        FKParams.ContactMode = ContactModeType::FREE_TIP;
        FKParams.TipForce[0] = FKParams.TipForce[1] = FKParams.TipForce[2] = 0.0;
        FKParams.TipConstraintPoint[0] = FKParams.TipConstraintPoint[1] = FKParams.TipConstraintPoint[2] = 0.0;
        FKParams.deltau0_initialguess[0] = FKParams.deltau0_initialguess[1] = FKParams.deltau0_initialguess[2] = 0.0;
        FKParams.ftip_initialguess[0] = FKParams.ftip_initialguess[1] = FKParams.ftip_initialguess[2] = 0.0;
        FKParams.IntegrationStepSize = integrationStepSize;
        FKParams.FinalValueOnly = finalValueOnly;
    }

    void applyDeltaU0Guess(CRMForwardKinematicsData& FKParams, const py::array_t<double>& deltau0_guess) {
        if (deltau0_guess.size() <= 0) return;
        auto buf = deltau0_guess.request();
        if (buf.size < 3) return;
        const double* ptr = static_cast<double*>(buf.ptr);
        for (int i = 0; i < 3; ++i) FKParams.deltau0_initialguess[i] = ptr[i];
    }

public:
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
            throw std::runtime_error("Params not loaded. Call load_parameters first.");
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

        // Setup FK data/simulation_parameters
        CRMForwardKinematicsData FKParams;
        CRMCatheterModelParams* cparams = catheter.getParams();
        fillFKParams(FKParams, cparams, /*finalValueOnly=*/true);

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
     * Forward kinematics with an explicit delta_u0 initial guess (warm-start).
     *
     * Matches the MATLAB usage pattern:
     *   FKParams.deltau0_initialguess = previous_FKsolution(13:15);
     */
    py::dict forwardKinematicsWithGuess(py::array_t<double> currents, double insertion_length, py::array_t<double> deltau0_initialguess) {
        if (!initialized) {
            throw std::runtime_error("Params not loaded. Call load_parameters first.");
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
        for (int i = 0; i < num_currents; i++) in_x[i] = curr_ptr[i];
        in_x[num_currents] = insertion_length;

        CRMForwardKinematicsData FKParams;
        CRMCatheterModelParams* cparams = catheter.getParams();
        fillFKParams(FKParams, cparams, /*finalValueOnly=*/true);
        applyDeltaU0Guess(FKParams, deltau0_initialguess);

        std::vector<double> markerPosData(cparams->no_locmarkers * 3);
        FKParams.ReportedMarkerPos = reinterpret_cast<double(*)[3]>(markerPosData.data());
        std::vector<double> coilOrientData(cparams->no_act_set * 9);
        FKParams.ReportedCoilOrient = reinterpret_cast<double(*)[9]>(coilOrientData.data());
        std::vector<double> coilPosData(cparams->no_act_set * 3);
        FKParams.ReportedCoilPos = reinterpret_cast<double(*)[3]>(coilPosData.data());

        int y_dim = 3 + 9 + 3;
        std::vector<double> out_y(y_dim);
        double potentialEnergy;
        int localmin = CRM_ForwardKinematics(in_x.data(), out_y.data(), potentialEnergy, FKParams);

        std::vector<ssize_t> shape3 = {3};
        std::vector<ssize_t> shape9 = {9};
        auto tip_pos = py::array_t<double>(shape3);
        auto tip_rot = py::array_t<double>(shape9);
        auto delta_u0 = py::array_t<double>(shape3);
        auto pos_acc = tip_pos.mutable_unchecked<1>();
        auto rot_acc = tip_rot.mutable_unchecked<1>();
        auto u0_acc = delta_u0.mutable_unchecked<1>();
        pos_acc(0) = out_y[0];
        pos_acc(1) = out_y[1];
        pos_acc(2) = out_y[2];
        for (int i = 0; i < 9; i++) rot_acc(i) = out_y[3 + i];
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
     * Fast path: compute FK and analytical Jacobian in one call (no second FK solve).
     *
     * Returns dict:
     *   tip_position, tip_rotation, delta_u0, potential_energy, converged, jacobian
     */
    py::dict fkAndJacobian(py::array_t<double> currents, double insertion_length, py::array_t<double> deltau0_initialguess) {
        if (!initialized) {
            throw std::runtime_error("Params not loaded.");
        }

        auto curr_buf = currents.request();
        CRMCatheterModelParams* cparams = catheter.getParams();
        int num_currents = cparams->no_act_set * 3;
        if (curr_buf.size != num_currents) {
            throw std::runtime_error("currents must have " + std::to_string(num_currents) + " elements");
        }
        double* curr_ptr = static_cast<double*>(curr_buf.ptr);

        int x_dim = num_currents + 1;
        VectorXd in_x(x_dim);
        for (int i = 0; i < num_currents; ++i) in_x(i) = curr_ptr[i];
        in_x(num_currents) = insertion_length;

        CRMForwardKinematicsData FKParams;
        fillFKParams(FKParams, cparams, /*finalValueOnly=*/true);
        applyDeltaU0Guess(FKParams, deltau0_initialguess);

        std::vector<double> markerPosData(cparams->no_locmarkers * 3);
        FKParams.ReportedMarkerPos = reinterpret_cast<double(*)[3]>(markerPosData.data());
        std::vector<double> coilOrientData(cparams->no_act_set * 9);
        FKParams.ReportedCoilOrient = reinterpret_cast<double(*)[9]>(coilOrientData.data());
        std::vector<double> coilPosData(cparams->no_act_set * 3);
        FKParams.ReportedCoilPos = reinterpret_cast<double(*)[3]>(coilPosData.data());

        const int y_dim = 3 + 9 + 3;
        std::vector<double> out_y(y_dim);
        double potentialEnergy;
        int localmin = CRM_ForwardKinematics(in_x.data(), out_y.data(), potentialEnergy, FKParams);

        VectorXd in_FKouty(y_dim);
        for (int i = 0; i < y_dim; ++i) in_FKouty(i) = out_y[i];

        MatrixXd J = CRM_FKJacobian_Analytical(in_x, in_FKouty, FKParams);

        std::vector<ssize_t> shape3 = {3};
        std::vector<ssize_t> shape9 = {9};
        auto tip_pos = py::array_t<double>(shape3);
        auto tip_rot = py::array_t<double>(shape9);
        auto delta_u0 = py::array_t<double>(shape3);
        auto pos_acc = tip_pos.mutable_unchecked<1>();
        auto rot_acc = tip_rot.mutable_unchecked<1>();
        auto u0_acc = delta_u0.mutable_unchecked<1>();
        pos_acc(0) = out_y[0];
        pos_acc(1) = out_y[1];
        pos_acc(2) = out_y[2];
        for (int i = 0; i < 9; i++) rot_acc(i) = out_y[3 + i];
        u0_acc(0) = out_y[12];
        u0_acc(1) = out_y[13];
        u0_acc(2) = out_y[14];

        py::array_t<double> jacobian({J.rows(), J.cols()});
        auto jac_buf = jacobian.request();
        double* jac_ptr = static_cast<double*>(jac_buf.ptr);
        for (int r = 0; r < J.rows(); ++r) {
            for (int c = 0; c < J.cols(); ++c) {
                jac_ptr[r * J.cols() + c] = J(r, c);
            }
        }

        py::dict result;
        result["tip_position"] = tip_pos;
        result["tip_rotation"] = tip_rot;
        result["delta_u0"] = delta_u0;
        result["potential_energy"] = potentialEnergy;
        result["converged"] = (localmin == 0);
        result["jacobian"] = jacobian;
        return result;
    }

    /**
     * Compute analytical Jacobian at current configuration.
     */
    py::array_t<double> computeJacobian(py::array_t<double> currents, double insertion_length) {
        if (!initialized) {
            throw std::runtime_error("Params not loaded.");
        }

        // First compute FK to get output
        // If you need FK+J efficiently, prefer `fk_and_jacobian` to avoid a second FK solve.
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

        // Setup FK data/simulation_parameters
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

    // Integrator selection (Task A1.7 Phase 3)
    IntegratorType integrator_type = IntegratorType::ABM4;  // Default: ABM4 (legacy)

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
    double last_fk_out[NUM_STATES];
    bool have_last_fk = false;

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
            // Default damping values tuned for stable coil dynamics (Task A1.7)
            // [linear_x, linear_y, linear_z, angular_x, angular_y, angular_z]
            damping[j][0] = 12.1761626666366;
            damping[j][1] = 12.1761626666366;
            damping[j][2] = 284.429938756989;
            damping[j][3] = 0.0304776127617393;
            damping[j][4] = 0.0304776127617393;
            damping[j][5] = 0.00502712804532508;
        }
        for (int i = 0; i < NUM_STATES; i++) {
            xf[i] = 0.0;
        }
        for (int i = 0; i < NUM_STATES; i++) {
            last_fk_out[i] = 0.0;
        }
    }

    py::dict get_seed_state() const {
        const int num_sets = catheter.getParams() ? catheter.getParams()->no_act_set : NUM_ACT_SET;

        py::array_t<double> v_out({num_sets, 3}, {static_cast<ssize_t>(3 * sizeof(double)), static_cast<ssize_t>(sizeof(double))});
        py::array_t<double> w_out({num_sets, 3}, {static_cast<ssize_t>(3 * sizeof(double)), static_cast<ssize_t>(sizeof(double))});
        py::array_t<double> p_out({num_sets, 3}, {static_cast<ssize_t>(3 * sizeof(double)), static_cast<ssize_t>(sizeof(double))});
        py::array_t<double> R_out({num_sets, 9}, {static_cast<ssize_t>(9 * sizeof(double)), static_cast<ssize_t>(sizeof(double))});
        py::array_t<double> xf_out({NUM_STATES}, {static_cast<ssize_t>(sizeof(double))});
        py::array_t<double> mL_out({num_sets, 3}, {static_cast<ssize_t>(3 * sizeof(double)), static_cast<ssize_t>(sizeof(double))});
        py::array_t<double> nL_out({num_sets, 3}, {static_cast<ssize_t>(3 * sizeof(double)), static_cast<ssize_t>(sizeof(double))});

        auto v = v_out.mutable_unchecked<2>();
        auto w = w_out.mutable_unchecked<2>();
        auto p = p_out.mutable_unchecked<2>();
        auto R = R_out.mutable_unchecked<2>();
        auto xfacc = xf_out.mutable_unchecked<1>();
        auto mL = mL_out.mutable_unchecked<2>();
        auto nL = nL_out.mutable_unchecked<2>();

        for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
            for (int i = 0; i < 3; i++) {
                v(j, i) = v_L[j][i];
                w(j, i) = w_L[j][i];
                p(j, i) = p_L[j][i];
                mL(j, i) = mL_guess[j][i];
                nL(j, i) = nL_guess[j][i];
            }
            for (int i = 0; i < 9; i++) {
                R(j, i) = R_L[j][i];
            }
        }
        for (int i = 0; i < NUM_STATES; i++) xfacc(i) = xf[i];

        py::dict out;
        out["v"] = v_out;
        out["w"] = w_out;
        out["p"] = p_out;
        out["R"] = R_out;
        out["xf"] = xf_out;
        out["mL"] = mL_out;
        out["nL"] = nL_out;
        return out;
    }

    void set_seed_state(
        py::array_t<double> v_in,
        py::array_t<double> w_in,
        py::array_t<double> p_in,
        py::array_t<double> R_in,
        py::array_t<double> xf_in,
        py::array_t<double> mL_in = py::array_t<double>(),
        py::array_t<double> nL_in = py::array_t<double>()
    ) {
        if (!initialized) {
            throw std::runtime_error("Params not loaded.");
        }

        const int num_sets = catheter.getParams() ? catheter.getParams()->no_act_set : NUM_ACT_SET;

        auto vbuf = v_in.request(); const double* vptr = static_cast<double*>(vbuf.ptr);
        auto wbuf = w_in.request(); const double* wptr = static_cast<double*>(wbuf.ptr);
        auto pbuf = p_in.request(); const double* pptr = static_cast<double*>(pbuf.ptr);
        auto Rbuf = R_in.request(); const double* Rptr = static_cast<double*>(Rbuf.ptr);
        auto xfbuf = xf_in.request(); const double* xfptr = static_cast<double*>(xfbuf.ptr);

        for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
            for (int i = 0; i < 3; i++) {
                const ssize_t idx3 = j * 3 + i;
                if (idx3 < vbuf.size) v_L[j][i] = vptr[idx3];
                if (idx3 < wbuf.size) w_L[j][i] = wptr[idx3];
                if (idx3 < pbuf.size) p_L[j][i] = pptr[idx3];
            }
            for (int i = 0; i < 9; i++) {
                const ssize_t idx9 = j * 9 + i;
                if (idx9 < Rbuf.size) R_L[j][i] = Rptr[idx9];
            }
        }

        for (int i = 0; i < NUM_STATES && i < xfbuf.size; i++) xf[i] = xfptr[i];

        if (mL_in.size() > 0) {
            auto mbuf = mL_in.request(); const double* mptr = static_cast<double*>(mbuf.ptr);
            for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
                for (int i = 0; i < 3; i++) {
                    const ssize_t idx = j * 3 + i;
                    if (idx < mbuf.size) mL_guess[j][i] = mptr[idx];
                }
            }
        }

        if (nL_in.size() > 0) {
            auto nbuf = nL_in.request(); const double* nptr = static_cast<double*>(nbuf.ptr);
            for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
                for (int i = 0; i < 3; i++) {
                    const ssize_t idx = j * 3 + i;
                    if (idx < nbuf.size) nL_guess[j][i] = nptr[idx];
                }
            }
        }
    }

    bool loadParams(const std::string& param_file, const std::string& config_file) {
        initialized = catheter.loadFromFiles(param_file, config_file);
        if (initialized) {
            // Initialize inertia based on loaded data/simulation_parameters
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
            if (const char* debug = std::getenv("CRM_DEBUG_PARAM_LOAD")) {
                if (std::string(debug) == "1") {
                    std::cout << "[CRM_DEBUG_PARAM_LOAD] CRMDynamicsWrapper seg_lengths:";
                    for (int i = 0; i < cparams->no_segments; ++i) {
                        std::cout << " " << cparams->SegLengths[i];
                    }
                    std::cout << "\n[CRM_DEBUG_PARAM_LOAD] CRMDynamicsWrapper g: "
                              << catheter.config.g[0] << " " << catheter.config.g[1] << " " << catheter.config.g[2] << "\n";
                    for (int i = 0; i < cparams->no_act_set && i < NUM_ACT_SET; i++) {
                        std::cout << "[CRM_DEBUG_PARAM_LOAD] act " << i
                                  << " mass=" << cparams->ActMass[i]
                                  << " r_out=" << cparams->OuterRadius[0]
                                  << " r_in=" << cparams->InnerRadius[0]
                                  << " seg_len=" << cparams->SegLengths[2 * i + 1]
                                  << " I_xx=" << actInertia[i][0]
                                  << " I_zz=" << actInertia[i][8]
                                  << "\n";
                    }
                }
            }
        }
        return initialized;
    }

    py::dict get_config_snapshot() const {
        py::dict out;
        const CRMCatheterModelParams* cparams = catheter.getParams();
        py::array_t<double> B0({3}, {static_cast<ssize_t>(sizeof(double))});
        py::array_t<double> g({3}, {static_cast<ssize_t>(sizeof(double))});
        py::array_t<double> p0({3}, {static_cast<ssize_t>(sizeof(double))});
        py::array_t<double> R0({9}, {static_cast<ssize_t>(sizeof(double))});
        py::array_t<double> seg_lengths({cparams ? cparams->no_segments : 0}, {static_cast<ssize_t>(sizeof(double))});

        if (cparams) {
            auto b = B0.mutable_unchecked<1>();
            auto gg = g.mutable_unchecked<1>();
            auto pp = p0.mutable_unchecked<1>();
            auto rr = R0.mutable_unchecked<1>();
            for (int i = 0; i < 3; i++) {
                b(i) = catheter.config.B0[i];
                gg(i) = catheter.config.g[i];
                pp(i) = catheter.config.p0[i];
            }
            for (int i = 0; i < 9; i++) rr(i) = catheter.config.R0[i];
            auto sl = seg_lengths.mutable_unchecked<1>();
            for (int i = 0; i < cparams->no_segments; i++) sl(i) = cparams->SegLengths[i];
        }

        out["B0"] = B0;
        out["g"] = g;
        out["p0"] = p0;
        out["R0"] = R0;
        out["seg_lengths"] = seg_lengths;
        return out;
    }

    /**
     * Debug-only: seed internal dynamics state manually.
     *
     * This is not used in normal initialization; useful for diagnostics.
     */
    void debugSeedState(
        py::array_t<double> v_in,
        py::array_t<double> w_in,
        py::array_t<double> p_in,
        py::array_t<double> R_in,
        py::array_t<double> xf_in,
        py::array_t<double> mL_in = py::array_t<double>(),
        py::array_t<double> nL_in = py::array_t<double>()
    ) {
        int num_curr_sets = catheter.getParams() ? catheter.getParams()->no_act_set : NUM_ACT_SET;

        auto vbuf = v_in.request(); const double* vptr = static_cast<double*>(vbuf.ptr);
        auto wbuf = w_in.request(); const double* wptr = static_cast<double*>(wbuf.ptr);
        auto pbuf = p_in.request(); const double* pptr = static_cast<double*>(pbuf.ptr);
        auto Rbuf = R_in.request(); const double* Rptr = static_cast<double*>(Rbuf.ptr);
        auto xfbuf = xf_in.request(); const double* xfptr = static_cast<double*>(xfbuf.ptr);

        for (int j = 0; j < num_curr_sets && j < NUM_ACT_SET; j++) {
            for (int i = 0; i < 3; i++) {
                ssize_t idx = j * 3 + i;
                if (idx < vbuf.size) v_L[j][i] = vptr[idx];
                if (idx < wbuf.size) w_L[j][i] = wptr[idx];
                if (idx < pbuf.size) p_L[j][i] = pptr[idx];
            }
            for (int i = 0; i < 9; i++) {
                ssize_t idx = j * 9 + i;
                if (idx < Rbuf.size) R_L[j][i] = Rptr[idx];
            }
        }

        for (int i = 0; i < NUM_STATES && i < xfbuf.size; i++) {
            xf[i] = xfptr[i];
        }

        if (mL_in.size() > 0) {
            auto mbuf = mL_in.request();
            const double* mptr = static_cast<double*>(mbuf.ptr);
            for (int j = 0; j < num_curr_sets && j < NUM_ACT_SET; j++) {
                for (int i = 0; i < 3; i++) {
                    ssize_t idx = j * 3 + i;
                    if (idx < mbuf.size) mL_guess[j][i] = mptr[idx];
                }
            }
        }
        if (nL_in.size() > 0) {
            auto nbuf = nL_in.request();
            const double* nptr = static_cast<double*>(nbuf.ptr);
            for (int j = 0; j < num_curr_sets && j < NUM_ACT_SET; j++) {
                for (int i = 0; i < 3; i++) {
                    ssize_t idx = j * 3 + i;
                    if (idx < nbuf.size) nL_guess[j][i] = nptr[idx];
                }
            }
        }
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
     * Set the integrator type for coil dynamics integration (Task A1.7 Phase 3).
     * @param integrator String: "abm4" (default, legacy) or "rk4" (more stable)
     */
    void set_integrator(const std::string& integrator) {
        if (integrator == "rk4" || integrator == "RK4") {
            integrator_type = IntegratorType::RK4;
        } else if (integrator == "abm4" || integrator == "ABM4") {
            integrator_type = IntegratorType::ABM4;
        } else {
            throw std::runtime_error("Unknown integrator type: " + integrator + ". Use 'abm4' or 'rk4'.");
        }
    }

    /**
     * Get the current integrator type.
     * @return String: "abm4" or "rk4"
     */
    std::string get_integrator() const {
        switch (integrator_type) {
            case IntegratorType::RK4:
                return "rk4";
            case IntegratorType::ABM4:
            default:
                return "abm4";
        }
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
            throw std::runtime_error("Params not loaded.");
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

        // Setup FK data/simulation_parameters
        CRMForwardKinematicsData FKParams;
        FKParams.CathParams = cparams;
        FKParams.CathConfig = &catheter.config;
        FKParams.ContactMode = ContactModeType::FREE_TIP;
        FKParams.TipForce[0] = FKParams.TipForce[1] = FKParams.TipForce[2] = 0.0;
        FKParams.TipConstraintPoint[0] = FKParams.TipConstraintPoint[1] = FKParams.TipConstraintPoint[2] = 0.0;
        FKParams.deltau0_initialguess[0] = FKParams.deltau0_initialguess[1] = FKParams.deltau0_initialguess[2] = 0.0;
        FKParams.ftip_initialguess[0] = FKParams.ftip_initialguess[1] = FKParams.ftip_initialguess[2] = 0.0;
        FKParams.IntegrationStepSize = integrationStepSize;
        // Match the kinematics wrapper path to avoid corrupt outputs.
        // Coil orientation/position outputs are still populated during IVP.
        FKParams.FinalValueOnly = true;

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

        if (const char* debug = std::getenv("CRM_DEBUG_FK_INIT")) {
            if (std::string(debug) == "1") {
                std::cout << "[CRM_DEBUG_FK_INIT] out_y:";
                for (int i = 0; i < y_dim; ++i) std::cout << " " << out_y[i];
                std::cout << "\n";
            }
        }

        for (int i = 0; i < NUM_STATES; i++) {
            last_fk_out[i] = out_y[i];
        }
        have_last_fk = true;

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

    py::array_t<double> get_last_fk_output() const {
        if (!have_last_fk) {
            return py::array_t<double>();
        }
        py::array_t<double> out({NUM_STATES}, {static_cast<ssize_t>(sizeof(double))});
        auto acc = out.mutable_unchecked<1>();
        for (int i = 0; i < NUM_STATES; i++) acc(i) = last_fk_out[i];
        return out;
    }

    /**
     * Initialize dynamics directly from explicit seeds (mirrors CRMDYN_test.cpp).
     */
    bool initializeFromSeed(
        py::array_t<double> currents,
        double insertion_length,
        py::array_t<double> p_in,
        py::array_t<double> R_in,
        py::array_t<double> xf_in,
        py::array_t<double> mL_in = py::array_t<double>(),
        py::array_t<double> nL_in = py::array_t<double>()
    ) {
        if (!initialized) {
            throw std::runtime_error("Params not loaded.");
        }
        auto curr_buf = currents.request(); const double* cptr = static_cast<double*>(curr_buf.ptr);
        auto pbuf = p_in.request(); const double* pptr = static_cast<double*>(pbuf.ptr);
        auto Rbuf = R_in.request(); const double* Rptr = static_cast<double*>(Rbuf.ptr);
        auto xfbuf = xf_in.request(); const double* xfptr = static_cast<double*>(xfbuf.ptr);

        int num_curr_sets = catheter.getParams() ? catheter.getParams()->no_act_set : NUM_ACT_SET;
        for (int j = 0; j < num_curr_sets && j < NUM_ACT_SET; j++) {
            for (int i = 0; i < 3; i++) {
                ssize_t idx = j * 3 + i;
                if (idx < pbuf.size) p_L[j][i] = pptr[idx];
            }
            for (int i = 0; i < 9; i++) {
                ssize_t idx = j * 9 + i;
                if (idx < Rbuf.size) R_L[j][i] = Rptr[idx];
            }
        }
        for (int i = 0; i < NUM_STATES && i < xfbuf.size; i++) {
            xf[i] = xfptr[i];
        }
        if (mL_in.size() > 0) {
            auto mbuf = mL_in.request(); const double* mptr = static_cast<double*>(mbuf.ptr);
            for (int j = 0; j < num_curr_sets && j < NUM_ACT_SET; j++) {
                for (int i = 0; i < 3; i++) {
                    ssize_t idx = j * 3 + i;
                    if (idx < mbuf.size) mL_guess[j][i] = mptr[idx];
                }
            }
        }
        if (nL_in.size() > 0) {
            auto nbuf = nL_in.request(); const double* nptr = static_cast<double*>(nbuf.ptr);
            for (int j = 0; j < num_curr_sets && j < NUM_ACT_SET; j++) {
                for (int i = 0; i < 3; i++) {
                    ssize_t idx = j * 3 + i;
                    if (idx < nbuf.size) nL_guess[j][i] = nptr[idx];
                }
            }
        }

        // No direct C API for seed-only init; we seed internal buffers here and report success.
        // A following stepDynamics() call will pick up these seeds.
        (void)cptr;  // currents are consumed in stepDynamics
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
            throw std::runtime_error("Params not loaded.");
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

        // Setup shooting method data/simulation_parameters
        ContactModeType ContactMode = ContactModeType::FREE_TIP;
        double TipForce[3] = {0.0, 0.0, 0.0};
        double TipConstraintPoint[3] = {0.0, 0.0, 0.0};

        CRMShootingMethodParams BVPParams = CRMDYNConstructShootingMethodParamSet(
            *catheter.getParams(), catheter.config, insertion_length, ActuationCurrents,
            ContactMode, TipConstraintPoint, TipForce, integrationStepSize,
            actInertia, v_L, w_L, p_L, R_L, damping, dt
        );

        // Task A1.7 Phase 3: Set integrator type
        BVPParams.dynamics.integrator_type = integrator_type;
        BVPParams.dynamics.last_diverged = false;

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

        // Update state only if converged to avoid corrupting internal state with garbage
        const bool diverged = BVPParams.dynamics.last_diverged;
        if (localmin == 0 && !diverged) {
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
        result["converged"] = (localmin == 0) && !diverged;
        result["localmin"] = localmin;
        result["diverged"] = diverged;

        return result;
    }

    py::dict step_from_seed(
        py::array_t<double> currents,
        double insertion_length,
        py::array_t<double> v_in,
        py::array_t<double> w_in,
        py::array_t<double> p_in,
        py::array_t<double> R_in,
        py::array_t<double> xf_in,
        py::array_t<double> mL_in = py::array_t<double>(),
        py::array_t<double> nL_in = py::array_t<double>(),
        std::optional<double> dt_override = std::nullopt
    ) {
        if (!initialized) {
            throw std::runtime_error("Params not loaded.");
        }

        const double dt_local = dt_override.has_value() ? *dt_override : dt;

        const int num_sets = catheter.getParams() ? catheter.getParams()->no_act_set : NUM_ACT_SET;
        auto curr_buf = currents.request();
        const double* curr_ptr = static_cast<double*>(curr_buf.ptr);

        auto vbuf = v_in.request(); const double* vptr = static_cast<double*>(vbuf.ptr);
        auto wbuf = w_in.request(); const double* wptr = static_cast<double*>(wbuf.ptr);
        auto pbuf = p_in.request(); const double* pptr = static_cast<double*>(pbuf.ptr);
        auto Rbuf = R_in.request(); const double* Rptr = static_cast<double*>(Rbuf.ptr);
        auto xfbuf = xf_in.request(); const double* xfptr = static_cast<double*>(xfbuf.ptr);

        double v_L_local[NUM_ACT_SET][3]{};
        double w_L_local[NUM_ACT_SET][3]{};
        double p_L_local[NUM_ACT_SET][3]{};
        double R_L_local[NUM_ACT_SET][9]{};
        double xf_local[NUM_STATES]{};
        double mL_guess_local[NUM_ACT_SET][3]{};
        double nL_guess_local[NUM_ACT_SET][3]{};
        const double kMnZeroEps = 1e-12;
        auto sum_abs_mn = [](const double mn[NUM_ACT_SET][3], int sets) {
            double total = 0.0;
            for (int j = 0; j < sets && j < NUM_ACT_SET; j++) {
                for (int i = 0; i < 3; i++) {
                    total += std::abs(mn[j][i]);
                }
            }
            return total;
        };

        for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
            for (int i = 0; i < 3; i++) {
                const ssize_t idx3 = j * 3 + i;
                v_L_local[j][i] = (idx3 < vbuf.size) ? vptr[idx3] : 0.0;
                w_L_local[j][i] = (idx3 < wbuf.size) ? wptr[idx3] : 0.0;
                p_L_local[j][i] = (idx3 < pbuf.size) ? pptr[idx3] : 0.0;
            }
            for (int i = 0; i < 9; i++) {
                const ssize_t idx9 = j * 9 + i;
                R_L_local[j][i] = (idx9 < Rbuf.size) ? Rptr[idx9] : ((i == 0 || i == 4 || i == 8) ? 1.0 : 0.0);
            }
        }

        for (int i = 0; i < NUM_STATES; i++) {
            xf_local[i] = (i < xfbuf.size) ? xfptr[i] : 0.0;
        }

        double mL_input_abs = 0.0;
        if (mL_in.size() > 0) {
            auto mbuf = mL_in.request(); const double* mptr = static_cast<double*>(mbuf.ptr);
            for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
                for (int i = 0; i < 3; i++) {
                    const ssize_t idx = j * 3 + i;
                    if (idx < mbuf.size) {
                        mL_guess_local[j][i] = mptr[idx];
                        mL_input_abs += std::abs(mptr[idx]);
                    }
                }
            }
        }

        double nL_input_abs = 0.0;
        if (nL_in.size() > 0) {
            auto nbuf = nL_in.request(); const double* nptr = static_cast<double*>(nbuf.ptr);
            for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
                for (int i = 0; i < 3; i++) {
                    const ssize_t idx = j * 3 + i;
                    if (idx < nbuf.size) {
                        nL_guess_local[j][i] = nptr[idx];
                        nL_input_abs += std::abs(nptr[idx]);
                    }
                }
            }
        }

        const double mL_internal_abs = sum_abs_mn(mL_guess, num_sets);
        const double nL_internal_abs = sum_abs_mn(nL_guess, num_sets);
        const bool use_internal_mL = (mL_in.size() == 0 || mL_input_abs <= kMnZeroEps) && (mL_internal_abs > kMnZeroEps);
        const bool use_internal_nL = (nL_in.size() == 0 || nL_input_abs <= kMnZeroEps) && (nL_internal_abs > kMnZeroEps);
        if (use_internal_mL || use_internal_nL) {
            for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
                for (int i = 0; i < 3; i++) {
                    if (use_internal_mL) mL_guess_local[j][i] = mL_guess[j][i];
                    if (use_internal_nL) nL_guess_local[j][i] = nL_guess[j][i];
                }
            }
        }

        // Build actuation currents array
        double ActuationCurrents[NUM_ACT_SET][3];
        for (int i = 0; i < NUM_ACT_SET; i++) {
            for (int j = 0; j < 3; j++) {
                const int idx = i * 3 + j;
                ActuationCurrents[i][j] = (idx < curr_buf.size) ? curr_ptr[idx] : 0.0;
            }
        }

        // Copy constant parameters to locals to avoid any accidental mutation.
        double actInertia_local[NUM_ACT_SET][9];
        double damping_local[NUM_ACT_SET][6];
        for (int j = 0; j < NUM_ACT_SET; j++) {
            for (int i = 0; i < 9; i++) actInertia_local[j][i] = actInertia[j][i];
            for (int i = 0; i < 6; i++) damping_local[j][i] = damping[j][i];
        }

        // Setup shooting method data/simulation_parameters
        ContactModeType ContactMode = ContactModeType::FREE_TIP;
        double TipForce[3] = {0.0, 0.0, 0.0};
        double TipConstraintPoint[3] = {0.0, 0.0, 0.0};

        CRMShootingMethodParams BVPParams = CRMDYNConstructShootingMethodParamSet(
            *catheter.getParams(), catheter.config, insertion_length, ActuationCurrents,
            ContactMode, TipConstraintPoint, TipForce, integrationStepSize,
            actInertia_local, v_L_local, w_L_local, p_L_local, R_L_local, damping_local, dt_local
        );

        // Task A1.7 Phase 3: Set integrator type
        BVPParams.dynamics.integrator_type = integrator_type;
        BVPParams.dynamics.last_diverged = false;

        // Solve BVP
        double out_u0[3];
        double out_mL[NUM_ACT_SET][3], out_nL[NUM_ACT_SET][3];
        double out_tau[NUM_ACT_SET][3];
        double ftip_calc[3];
        double ftip_guess[3] = {0.0, 0.0, 0.0};
        int localmin;

        DynamicsBVP(BVPParams, xf_local, mL_guess_local, nL_guess_local, ftip_guess,
                    out_u0, out_mL, out_nL, out_tau, ftip_calc, localmin);

        // Solve IVP
        double xf_new[NUM_STATES];
        double x_coil[NUM_ACT_SET][NUM_COIL_STATES];
        double ReportedMarkerPos[5][3];

        DYNSolverIVP(BVPParams, out_u0, out_mL, out_nL, out_tau, ftip_calc,
                     true, xf_new, x_coil, ReportedMarkerPos);

        // Tip position from xf (indices 0-2 are position)
        py::array_t<double> tip_pos({3});
        auto pos = tip_pos.mutable_unchecked<1>();
        // Tip velocity from v (coil velocity proxy)
        py::array_t<double> tip_vel({3});
        auto vel = tip_vel.mutable_unchecked<1>();

        const bool diverged = BVPParams.dynamics.last_diverged;
        const bool converged = (localmin == 0) && !diverged;
        if (converged) {
            pos(0) = xf_new[0];
            pos(1) = xf_new[1];
            pos(2) = xf_new[2];
            vel(0) = x_coil[0][0];
            vel(1) = x_coil[0][1];
            vel(2) = x_coil[0][2];
        } else {
            pos(0) = xf_local[0];
            pos(1) = xf_local[1];
            pos(2) = xf_local[2];
            vel(0) = v_L_local[0][0];
            vel(1) = v_L_local[0][1];
            vel(2) = v_L_local[0][2];
        }

        // Return next seed (or the original seed on failure)
        py::array_t<double> v_next({num_sets, 3});
        py::array_t<double> w_next({num_sets, 3});
        py::array_t<double> p_next({num_sets, 3});
        py::array_t<double> R_next({num_sets, 9});
        py::array_t<double> xf_next({NUM_STATES});
        py::array_t<double> mL_next({num_sets, 3});
        py::array_t<double> nL_next({num_sets, 3});

        auto vout = v_next.mutable_unchecked<2>();
        auto wout = w_next.mutable_unchecked<2>();
        auto pout = p_next.mutable_unchecked<2>();
        auto Rout = R_next.mutable_unchecked<2>();
        auto xfout = xf_next.mutable_unchecked<1>();
        auto mLout = mL_next.mutable_unchecked<2>();
        auto nLout = nL_next.mutable_unchecked<2>();

        for (int i = 0; i < NUM_STATES; i++) xfout(i) = converged ? xf_new[i] : xf_local[i];

        for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
            for (int i = 0; i < 3; i++) {
                if (converged) {
                    vout(j, i) = x_coil[j][i];
                    wout(j, i) = x_coil[j][i + 3];
                    pout(j, i) = x_coil[j][i + 6];
                    mLout(j, i) = out_mL[j][i];
                    nLout(j, i) = out_nL[j][i];
                } else {
                    vout(j, i) = v_L_local[j][i];
                    wout(j, i) = w_L_local[j][i];
                    pout(j, i) = p_L_local[j][i];
                    mLout(j, i) = mL_guess_local[j][i];
                    nLout(j, i) = nL_guess_local[j][i];
                }
            }
            for (int i = 0; i < 9; i++) {
                if (converged) {
                    Rout(j, i) = x_coil[j][i + 9];
                } else {
                    Rout(j, i) = R_L_local[j][i];
                }
            }
        }

        py::dict result;
        result["tip_position"] = tip_pos;
        result["tip_velocity"] = tip_vel;
        result["converged"] = converged;
        result["localmin"] = localmin;
        result["diverged"] = diverged;
        result["next_v"] = v_next;
        result["next_w"] = w_next;
        result["next_p"] = p_next;
        result["next_R"] = R_next;
        result["next_xf"] = xf_next;
        result["next_mL"] = mL_next;
        result["next_nL"] = nL_next;
        return result;
    }

    py::dict linearize_action_from_seed(
        py::array_t<double> currents,
        double insertion_length,
        py::array_t<double> v_in,
        py::array_t<double> w_in,
        py::array_t<double> p_in,
        py::array_t<double> R_in,
        py::array_t<double> xf_in,
        py::array_t<double> mL_in = py::array_t<double>(),
        py::array_t<double> nL_in = py::array_t<double>(),
        double eps = 1e-4
    ) {
        py::dict base = step_from_seed(currents, insertion_length, v_in, w_in, p_in, R_in, xf_in, mL_in, nL_in, std::nullopt);
        if (!base["converged"].cast<bool>()) {
            py::array_t<double> next_state({6});
            auto ns = next_state.mutable_unchecked<1>();
            for (int i = 0; i < 6; i++) ns(i) = 0.0;

            py::array_t<double> B_out({6, 3});
            auto Bout = B_out.mutable_unchecked<2>();
            for (int i = 0; i < 6; i++) {
                for (int j = 0; j < 3; j++) {
                    Bout(i, j) = 0.0;
                }
            }

            py::dict result;
            result["next_state"] = next_state;
            result["B"] = B_out;
            result["base"] = base;
            result["converged"] = false;
            return result;
        }
        auto tip_pos_base = base["tip_position"].cast<py::array_t<double>>().request();
        auto tip_vel_base = base["tip_velocity"].cast<py::array_t<double>>().request();
        const double* pos_ptr = static_cast<double*>(tip_pos_base.ptr);
        const double* vel_ptr = static_cast<double*>(tip_vel_base.ptr);

        Eigen::Matrix<double, 6, 1> y0;
        for (int i = 0; i < 3; i++) y0(i) = pos_ptr[i];
        for (int i = 0; i < 3; i++) y0(3 + i) = vel_ptr[i];

        Eigen::Matrix<double, 6, 3> B;
        B.setZero();

        auto curr_buf = currents.request();
        const double* curr_ptr = static_cast<double*>(curr_buf.ptr);
        std::array<double, 3> u0{};
        for (int i = 0; i < 3; i++) u0[i] = (i < curr_buf.size) ? curr_ptr[i] : 0.0;

        bool perturb_failed = false;
        for (int k = 0; k < 3; k++) {
            std::array<double, 3> u_plus = u0;
            std::array<double, 3> u_minus = u0;
            u_plus[k] += eps;
            u_minus[k] -= eps;

            py::array_t<double> u_plus_arr(3);
            py::array_t<double> u_minus_arr(3);
            auto up = u_plus_arr.mutable_unchecked<1>();
            auto um = u_minus_arr.mutable_unchecked<1>();
            for (int i = 0; i < 3; i++) {
                up(i) = u_plus[i];
                um(i) = u_minus[i];
            }

            py::dict out_plus = step_from_seed(u_plus_arr, insertion_length, v_in, w_in, p_in, R_in, xf_in, mL_in, nL_in, std::nullopt);
            py::dict out_minus = step_from_seed(u_minus_arr, insertion_length, v_in, w_in, p_in, R_in, xf_in, mL_in, nL_in, std::nullopt);
            if (!out_plus["converged"].cast<bool>() || !out_minus["converged"].cast<bool>()) {
                perturb_failed = true;
                break;
            }

            auto pos_p = out_plus["tip_position"].cast<py::array_t<double>>().request();
            auto vel_p = out_plus["tip_velocity"].cast<py::array_t<double>>().request();
            auto pos_m = out_minus["tip_position"].cast<py::array_t<double>>().request();
            auto vel_m = out_minus["tip_velocity"].cast<py::array_t<double>>().request();
            const double* pos_p_ptr = static_cast<double*>(pos_p.ptr);
            const double* vel_p_ptr = static_cast<double*>(vel_p.ptr);
            const double* pos_m_ptr = static_cast<double*>(pos_m.ptr);
            const double* vel_m_ptr = static_cast<double*>(vel_m.ptr);

            Eigen::Matrix<double, 6, 1> yp, ym;
            for (int i = 0; i < 3; i++) yp(i) = pos_p_ptr[i];
            for (int i = 0; i < 3; i++) yp(3 + i) = vel_p_ptr[i];
            for (int i = 0; i < 3; i++) ym(i) = pos_m_ptr[i];
            for (int i = 0; i < 3; i++) ym(3 + i) = vel_m_ptr[i];

            B.col(k) = (yp - ym) * (0.5 / eps);
        }
        if (perturb_failed) {
            B.setZero();
        }

        py::array_t<double> next_state({6});
        auto ns = next_state.mutable_unchecked<1>();
        for (int i = 0; i < 6; i++) ns(i) = y0(i);

        py::array_t<double> B_out({6, 3});
        auto Bout = B_out.mutable_unchecked<2>();
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 3; j++) {
                Bout(i, j) = B(i, j);
            }
        }

        py::dict result;
        result["next_state"] = next_state;
        result["B"] = B_out;
        result["base"] = base;
        result["converged"] = !perturb_failed;
        return result;
    }

    py::dict linearize_full_seed_action_from_seed(
        py::array_t<double> currents,
        double insertion_length,
        py::array_t<double> v_in,
        py::array_t<double> w_in,
        py::array_t<double> p_in,
        py::array_t<double> R_in,
        py::array_t<double> xf_in,
        py::array_t<double> mL_in = py::array_t<double>(),
        py::array_t<double> nL_in = py::array_t<double>(),
        double eps_u = 1e-4,
        double eps_seed = 1e-4
    ) {
        auto get_state6 = [](const py::dict& out) {
            auto tip_pos = out["tip_position"].cast<py::array_t<double>>().request();
            auto tip_vel = out["tip_velocity"].cast<py::array_t<double>>().request();
            const double* pos_ptr = static_cast<double*>(tip_pos.ptr);
            const double* vel_ptr = static_cast<double*>(tip_vel.ptr);
            Eigen::Matrix<double, 6, 1> y;
            for (int i = 0; i < 3; i++) y(i) = pos_ptr[i];
            for (int i = 0; i < 3; i++) y(3 + i) = vel_ptr[i];
            return y;
        };

        // Determine num_act_set from v_in, and validate shapes.
        auto vbuf = v_in.request();
        if (vbuf.ndim != 2 || vbuf.shape[1] != 3) {
            throw std::runtime_error("v_in must have shape (num_act_set, 3)");
        }
        const int num_sets = static_cast<int>(vbuf.shape[0]);

        auto wbuf = w_in.request();
        if (wbuf.ndim != 2 || wbuf.shape[0] != num_sets || wbuf.shape[1] != 3) {
            throw std::runtime_error("w_in must have shape (num_act_set, 3)");
        }
        auto pbuf = p_in.request();
        if (pbuf.ndim != 2 || pbuf.shape[0] != num_sets || pbuf.shape[1] != 3) {
            throw std::runtime_error("p_in must have shape (num_act_set, 3)");
        }
        auto Rbuf = R_in.request();
        if (Rbuf.ndim != 2 || Rbuf.shape[0] != num_sets || Rbuf.shape[1] != 9) {
            throw std::runtime_error("R_in must have shape (num_act_set, 9)");
        }
        auto xfbuf = xf_in.request();
        if (xfbuf.ndim != 1 || xfbuf.shape[0] != NUM_STATES) {
            throw std::runtime_error("xf_in must have shape (NUM_STATES,)");
        }

        const double kMnZeroEps = 1e-12;
        auto sum_abs_mn = [&](const double mn[NUM_ACT_SET][3]) {
            double total = 0.0;
            for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
                for (int i = 0; i < 3; i++) total += std::abs(mn[j][i]);
            }
            return total;
        };
        auto ensure_mn = [&](const py::array_t<double>& arr, const double fallback[NUM_ACT_SET][3]) {
            if (arr.size() != 0) {
                auto abuf = arr.request();
                if (abuf.ndim != 2 || abuf.shape[0] != num_sets || abuf.shape[1] != 3) {
                    throw std::runtime_error("mL/nL must have shape (num_act_set, 3) when provided");
                }
                const double* aptr = static_cast<double*>(abuf.ptr);
                double input_abs = 0.0;
                for (int j = 0; j < num_sets; j++) {
                    for (int i = 0; i < 3; i++) {
                        const ssize_t idx = j * 3 + i;
                        if (idx < abuf.size) input_abs += std::abs(aptr[idx]);
                    }
                }
                if (input_abs > kMnZeroEps) {
                    return arr;
                }
            }
            const bool use_fallback = sum_abs_mn(fallback) > kMnZeroEps;
            py::array_t<double> out({num_sets, 3});
            auto o = out.mutable_unchecked<2>();
            for (int j = 0; j < num_sets; j++) {
                for (int i = 0; i < 3; i++) {
                    o(j, i) = use_fallback ? fallback[j][i] : 0.0;
                }
            }
            return out;
        };

        py::array_t<double> mL_use = ensure_mn(mL_in, mL_guess);
        py::array_t<double> nL_use = ensure_mn(nL_in, nL_guess);

        py::dict base = step_from_seed(currents, insertion_length, v_in, w_in, p_in, R_in, xf_in, mL_use, nL_use, std::nullopt);
        const bool ok = base["converged"].cast<bool>();
        if (!ok) {
            py::array_t<double> next_state({6});
            auto ns = next_state.mutable_unchecked<1>();
            for (int i = 0; i < 6; i++) ns(i) = 0.0;

            py::array_t<double> B_out({6, 3});
            auto Bout = B_out.mutable_unchecked<2>();
            for (int i = 0; i < 6; i++) {
                for (int j = 0; j < 3; j++) {
                    Bout(i, j) = 0.0;
                }
            }

            const int seed_dim = num_sets * 3 + num_sets * 3 + num_sets * 3 + num_sets * 9 + NUM_STATES + num_sets * 3 + num_sets * 3;
            py::array_t<double> A_out({6, seed_dim});
            auto Aout = A_out.mutable_unchecked<2>();
            for (int i = 0; i < 6; i++) {
                for (int j = 0; j < seed_dim; j++) {
                    Aout(i, j) = 0.0;
                }
            }

            py::dict result;
            result["next_state"] = next_state;
            result["B"] = B_out;
            result["A"] = A_out;
            result["base"] = base;
            result["converged"] = false;
            return result;
        }
        const Eigen::Matrix<double, 6, 1> y0 = get_state6(base);

        Eigen::Matrix<double, 6, 3> B;
        B.setZero();

        const int dim_v = num_sets * 3;
        const int dim_w = num_sets * 3;
        const int dim_p = num_sets * 3;
        const int dim_R = num_sets * 9;
        const int dim_xf = NUM_STATES;
        const int dim_mL = num_sets * 3;
        const int dim_nL = num_sets * 3;
        const int seed_dim = dim_v + dim_w + dim_p + dim_R + dim_xf + dim_mL + dim_nL;

        Eigen::MatrixXd A(6, seed_dim);
        A.setZero();

        bool perturb_failed = false;
        if (ok) {
            // B: central differences w.r.t currents.
            auto curr_buf = currents.request();
            const double* curr_ptr = static_cast<double*>(curr_buf.ptr);
            std::array<double, 3> u0{};
            for (int i = 0; i < 3; i++) u0[i] = (i < curr_buf.size) ? curr_ptr[i] : 0.0;

            for (int k = 0; k < 3; k++) {
                std::array<double, 3> u_plus = u0;
                std::array<double, 3> u_minus = u0;
                u_plus[k] += eps_u;
                u_minus[k] -= eps_u;

                py::array_t<double> u_plus_arr(3);
                py::array_t<double> u_minus_arr(3);
                auto up = u_plus_arr.mutable_unchecked<1>();
                auto um = u_minus_arr.mutable_unchecked<1>();
                for (int i = 0; i < 3; i++) {
                    up(i) = u_plus[i];
                    um(i) = u_minus[i];
                }

                py::dict out_plus = step_from_seed(u_plus_arr, insertion_length, v_in, w_in, p_in, R_in, xf_in, mL_use, nL_use, std::nullopt);
                py::dict out_minus = step_from_seed(u_minus_arr, insertion_length, v_in, w_in, p_in, R_in, xf_in, mL_use, nL_use, std::nullopt);
                if (!out_plus["converged"].cast<bool>() || !out_minus["converged"].cast<bool>()) {
                    perturb_failed = true;
                    break;
                }

                const Eigen::Matrix<double, 6, 1> yp = get_state6(out_plus);
                const Eigen::Matrix<double, 6, 1> ym = get_state6(out_minus);
                B.col(k) = (yp - ym) * (0.5 / eps_u);
            }

            if (!perturb_failed) {
                auto copy2 = [](py::array_t<double> dst, const py::array_t<double>& src) {
                    auto d = dst.mutable_unchecked<2>();
                    auto s = src.unchecked<2>();
                    for (ssize_t i = 0; i < s.shape(0); i++) for (ssize_t j = 0; j < s.shape(1); j++) d(i, j) = s(i, j);
                };
                auto copy1 = [](py::array_t<double> dst, const py::array_t<double>& src) {
                    auto d = dst.mutable_unchecked<1>();
                    auto s = src.unchecked<1>();
                    for (ssize_t i = 0; i < s.shape(0); i++) d(i) = s(i);
                };

                const int i_v0 = 0;
                const int i_w0 = i_v0 + dim_v;
                const int i_p0 = i_w0 + dim_w;
                const int i_R0 = i_p0 + dim_p;
                const int i_xf0 = i_R0 + dim_R;
                const int i_mL0 = i_xf0 + dim_xf;
                const int i_nL0 = i_mL0 + dim_mL;

                auto bump2 = [&](const py::array_t<double>& src, py::array_t<double>& plus, py::array_t<double>& minus, int flat_idx, int stride) {
                    plus = py::array_t<double>({num_sets, stride});
                    minus = py::array_t<double>({num_sets, stride});
                    copy2(plus, src);
                    copy2(minus, src);
                    const int row = flat_idx / stride;
                    const int col2 = flat_idx % stride;
                    auto p = plus.mutable_unchecked<2>();
                    auto m = minus.mutable_unchecked<2>();
                    p(row, col2) += eps_seed;
                    m(row, col2) -= eps_seed;
                };

                // A: central differences w.r.t seed components.
                for (int col = 0; col < seed_dim; col++) {
                    py::array_t<double> v_plus, v_minus, w_plus, w_minus, p_plus, p_minus, R_plus, R_minus, xf_plus, xf_minus, mL_plus, mL_minus, nL_plus, nL_minus;

                    const py::array_t<double>* vP = &v_in;
                    const py::array_t<double>* vM = &v_in;
                    const py::array_t<double>* wP = &w_in;
                    const py::array_t<double>* wM = &w_in;
                    const py::array_t<double>* pP = &p_in;
                    const py::array_t<double>* pM = &p_in;
                    const py::array_t<double>* RP = &R_in;
                    const py::array_t<double>* RM = &R_in;
                    const py::array_t<double>* xfP = &xf_in;
                    const py::array_t<double>* xfM = &xf_in;
                    const py::array_t<double>* mLP = &mL_use;
                    const py::array_t<double>* mLM = &mL_use;
                    const py::array_t<double>* nLP = &nL_use;
                    const py::array_t<double>* nLM = &nL_use;

                    if (col >= i_v0 && col < i_w0) {
                        bump2(v_in, v_plus, v_minus, col - i_v0, 3);
                        vP = &v_plus;
                        vM = &v_minus;
                    } else if (col >= i_w0 && col < i_p0) {
                        bump2(w_in, w_plus, w_minus, col - i_w0, 3);
                        wP = &w_plus;
                        wM = &w_minus;
                    } else if (col >= i_p0 && col < i_R0) {
                        bump2(p_in, p_plus, p_minus, col - i_p0, 3);
                        pP = &p_plus;
                        pM = &p_minus;
                    } else if (col >= i_R0 && col < i_xf0) {
                        bump2(R_in, R_plus, R_minus, col - i_R0, 9);
                        RP = &R_plus;
                        RM = &R_minus;
                    } else if (col >= i_xf0 && col < i_mL0) {
                        const int idx = col - i_xf0;
                        xf_plus = py::array_t<double>({NUM_STATES});
                        xf_minus = py::array_t<double>({NUM_STATES});
                        copy1(xf_plus, xf_in);
                        copy1(xf_minus, xf_in);
                        auto xp = xf_plus.mutable_unchecked<1>();
                        auto xm = xf_minus.mutable_unchecked<1>();
                        xp(idx) += eps_seed;
                        xm(idx) -= eps_seed;
                        xfP = &xf_plus;
                        xfM = &xf_minus;
                    } else if (col >= i_mL0 && col < i_nL0) {
                        bump2(mL_use, mL_plus, mL_minus, col - i_mL0, 3);
                        mLP = &mL_plus;
                        mLM = &mL_minus;
                    } else if (col >= i_nL0 && col < seed_dim) {
                        bump2(nL_use, nL_plus, nL_minus, col - i_nL0, 3);
                        nLP = &nL_plus;
                        nLM = &nL_minus;
                    } else {
                        throw std::runtime_error("seed index out of range");
                    }

                    py::dict out_plus = step_from_seed(currents, insertion_length, *vP, *wP, *pP, *RP, *xfP, *mLP, *nLP, std::nullopt);
                    py::dict out_minus = step_from_seed(currents, insertion_length, *vM, *wM, *pM, *RM, *xfM, *mLM, *nLM, std::nullopt);
                    if (!out_plus["converged"].cast<bool>() || !out_minus["converged"].cast<bool>()) {
                        perturb_failed = true;
                        break;
                    }

                    const Eigen::Matrix<double, 6, 1> yp = get_state6(out_plus);
                    const Eigen::Matrix<double, 6, 1> ym = get_state6(out_minus);
                    A.col(col) = (yp - ym) * (0.5 / eps_seed);
                }
            }
        }

        if (perturb_failed) {
            B.setZero();
            A.setZero();
        }

        py::array_t<double> next_state({6});
        auto ns = next_state.mutable_unchecked<1>();
        for (int i = 0; i < 6; i++) ns(i) = y0(i);

        py::array_t<double> B_out({6, 3});
        auto Bout = B_out.mutable_unchecked<2>();
        for (int i = 0; i < 6; i++) for (int j = 0; j < 3; j++) Bout(i, j) = B(i, j);

        py::array_t<double> A_out({6, seed_dim});
        auto Aout = A_out.mutable_unchecked<2>();
        for (int i = 0; i < 6; i++) for (int j = 0; j < seed_dim; j++) Aout(i, j) = A(i, j);

        py::dict result;
        result["next_state"] = next_state;
        result["B"] = B_out;
        result["A"] = A_out;
        result["seed_dim"] = seed_dim;
        result["base"] = base;
        result["converged"] = !perturb_failed;
        return result;
    }

    py::dict linearize_full_seed_action_from_seed_implicit(
        py::array_t<double> currents,
        double insertion_length,
        py::array_t<double> v_in,
        py::array_t<double> w_in,
        py::array_t<double> p_in,
        py::array_t<double> R_in,
        py::array_t<double> xf_in,
        py::array_t<double> mL_in = py::array_t<double>(),
        py::array_t<double> nL_in = py::array_t<double>(),
        double eps_residual_x = 1e-5,
        double eps_residual_theta = 1e-5,
        double eps_g_x = 1e-5,
        double eps_g_theta = 1e-5,
        bool return_debug = false
    ) {
        // Implicit differentiation with residual Jacobians.
        // Jxx = dF/dx uses autodiff (Eigen+dual numbers) when available; otherwise falls back to finite differences.

        auto get_state6 = [](const py::dict& out) {
            auto tip_pos = out["tip_position"].cast<py::array_t<double>>().request();
            auto tip_vel = out["tip_velocity"].cast<py::array_t<double>>().request();
            const double* pos_ptr = static_cast<double*>(tip_pos.ptr);
            const double* vel_ptr = static_cast<double*>(tip_vel.ptr);
            Eigen::Matrix<double, 6, 1> y;
            for (int i = 0; i < 3; i++) y(i) = pos_ptr[i];
            for (int i = 0; i < 3; i++) y(3 + i) = vel_ptr[i];
            return y;
        };

        // Determine num_act_set and validate seed shapes.
        auto vbuf = v_in.request();
        if (vbuf.ndim != 2 || vbuf.shape[1] != 3) {
            throw std::runtime_error("v_in must have shape (num_act_set, 3)");
        }
        const int num_sets = static_cast<int>(vbuf.shape[0]);

        auto wbuf = w_in.request();
        auto pbuf = p_in.request();
        auto Rbuf = R_in.request();
        auto xfbuf = xf_in.request();
        if (wbuf.ndim != 2 || wbuf.shape[0] != num_sets || wbuf.shape[1] != 3) {
            throw std::runtime_error("w_in must have shape (num_act_set, 3)");
        }
        if (pbuf.ndim != 2 || pbuf.shape[0] != num_sets || pbuf.shape[1] != 3) {
            throw std::runtime_error("p_in must have shape (num_act_set, 3)");
        }
        if (Rbuf.ndim != 2 || Rbuf.shape[0] != num_sets || Rbuf.shape[1] != 9) {
            throw std::runtime_error("R_in must have shape (num_act_set, 9)");
        }
        if (xfbuf.ndim != 1 || xfbuf.shape[0] != NUM_STATES) {
            throw std::runtime_error("xf_in must have shape (NUM_STATES,)");
        }

        const double kMnZeroEps = 1e-12;
        auto sum_abs_mn = [&](const double mn[NUM_ACT_SET][3]) {
            double total = 0.0;
            for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
                for (int i = 0; i < 3; i++) total += std::abs(mn[j][i]);
            }
            return total;
        };
        auto ensure_mn = [&](const py::array_t<double>& arr, const double fallback[NUM_ACT_SET][3]) {
            if (arr.size() != 0) {
                auto abuf = arr.request();
                if (abuf.ndim != 2 || abuf.shape[0] != num_sets || abuf.shape[1] != 3) {
                    throw std::runtime_error("mL/nL must have shape (num_act_set, 3) when provided");
                }
                const double* aptr = static_cast<double*>(abuf.ptr);
                double input_abs = 0.0;
                for (int j = 0; j < num_sets; j++) {
                    for (int i = 0; i < 3; i++) {
                        const ssize_t idx = j * 3 + i;
                        if (idx < abuf.size) input_abs += std::abs(aptr[idx]);
                    }
                }
                if (input_abs > kMnZeroEps) {
                    return arr;
                }
            }
            const bool use_fallback = sum_abs_mn(fallback) > kMnZeroEps;
            py::array_t<double> out({num_sets, 3});
            auto o = out.mutable_unchecked<2>();
            for (int j = 0; j < num_sets; j++) {
                for (int i = 0; i < 3; i++) {
                    o(j, i) = use_fallback ? fallback[j][i] : 0.0;
                }
            }
            return out;
        };

        py::array_t<double> mL_use = ensure_mn(mL_in, mL_guess);
        py::array_t<double> nL_use = ensure_mn(nL_in, nL_guess);

        // Base solve: get y0 and x* (as next_mL/next_nL) by running the full step once.
        py::dict base = step_from_seed(currents, insertion_length, v_in, w_in, p_in, R_in, xf_in, mL_use, nL_use, std::nullopt);
        const bool ok = base["converged"].cast<bool>();

        const Eigen::Matrix<double, 6, 1> y0 = get_state6(base);

        // x* is the solved internal variables (mL,nL) from DynamicsBVP output (returned as next_mL/next_nL).
        auto mL_star_arr = base["next_mL"].cast<py::array_t<double>>();
        auto nL_star_arr = base["next_nL"].cast<py::array_t<double>>();
        auto mL_star = mL_star_arr.request();
        auto nL_star = nL_star_arr.request();
        const double* mL_star_ptr = static_cast<double*>(mL_star.ptr);
        const double* nL_star_ptr = static_cast<double*>(nL_star.ptr);

        const int x_dim = num_sets * 6;
        Eigen::VectorXd x_star_scaled(x_dim);
        for (int j = 0; j < num_sets; j++) {
            for (int i = 0; i < 3; i++) {
                x_star_scaled(j * 6 + i) = mL_star_ptr[j * 3 + i] / IVALUE_SCALE_M;
                x_star_scaled(j * 6 + 3 + i) = nL_star_ptr[j * 3 + i] / IVALUE_SCALE_N;
            }
        }

        // Seed flattening order must match torch_physics.py unflattening.
        const int dim_v = num_sets * 3;
        const int dim_w = num_sets * 3;
        const int dim_p = num_sets * 3;
        const int dim_R = num_sets * 9;
        const int dim_xf = NUM_STATES;
        const int dim_mL = num_sets * 3;
        const int dim_nL = num_sets * 3;
        const int seed_dim = dim_v + dim_w + dim_p + dim_R + dim_xf + dim_mL + dim_nL;

        // If the base solve failed, return zeros (consistent with FD linearizers).
        if (!ok) {
            py::array_t<double> next_state({6});
            auto ns = next_state.mutable_unchecked<1>();
            for (int i = 0; i < 6; i++) ns(i) = y0(i);

            py::array_t<double> B_out({6, 3});
            py::array_t<double> A_out({6, seed_dim});
            auto Bout = B_out.mutable_unchecked<2>();
            auto Aout = A_out.mutable_unchecked<2>();
            for (int i = 0; i < 6; i++) {
                for (int j = 0; j < 3; j++) Bout(i, j) = 0.0;
                for (int j = 0; j < seed_dim; j++) Aout(i, j) = 0.0;
            }
            py::dict result;
            result["next_state"] = next_state;
            result["B"] = B_out;
            result["A"] = A_out;
            result["seed_dim"] = seed_dim;
            result["base"] = base;
            result["residual_norm"] = std::numeric_limits<double>::infinity();
            return result;
        }

        auto unpack_seed = [&](const Eigen::VectorXd& seed_flat,
                               py::array_t<double>& v_out,
                               py::array_t<double>& w_out,
                               py::array_t<double>& p_out,
                               py::array_t<double>& R_out,
                               py::array_t<double>& xf_out,
                               py::array_t<double>& mL_out,
                               py::array_t<double>& nL_out) {
            v_out = py::array_t<double>({num_sets, 3});
            w_out = py::array_t<double>({num_sets, 3});
            p_out = py::array_t<double>({num_sets, 3});
            R_out = py::array_t<double>({num_sets, 9});
            xf_out = py::array_t<double>({NUM_STATES});
            mL_out = py::array_t<double>({num_sets, 3});
            nL_out = py::array_t<double>({num_sets, 3});

            auto v = v_out.mutable_unchecked<2>();
            auto w = w_out.mutable_unchecked<2>();
            auto p = p_out.mutable_unchecked<2>();
            auto R = R_out.mutable_unchecked<2>();
            auto xf = xf_out.mutable_unchecked<1>();
            auto mL = mL_out.mutable_unchecked<2>();
            auto nL = nL_out.mutable_unchecked<2>();

            int idx = 0;
            for (int j = 0; j < num_sets; j++) for (int i = 0; i < 3; i++) v(j, i) = seed_flat(idx++);
            for (int j = 0; j < num_sets; j++) for (int i = 0; i < 3; i++) w(j, i) = seed_flat(idx++);
            for (int j = 0; j < num_sets; j++) for (int i = 0; i < 3; i++) p(j, i) = seed_flat(idx++);
            for (int j = 0; j < num_sets; j++) for (int i = 0; i < 9; i++) R(j, i) = seed_flat(idx++);
            for (int i = 0; i < NUM_STATES; i++) xf(i) = seed_flat(idx++);
            for (int j = 0; j < num_sets; j++) for (int i = 0; i < 3; i++) mL(j, i) = seed_flat(idx++);
            for (int j = 0; j < num_sets; j++) for (int i = 0; i < 3; i++) nL(j, i) = seed_flat(idx++);
        };

        auto flatten_seed_from_arrays = [&]() {
            Eigen::VectorXd seed_flat(seed_dim);
            auto v = v_in.unchecked<2>();
            auto w = w_in.unchecked<2>();
            auto p = p_in.unchecked<2>();
            auto R = R_in.unchecked<2>();
            auto xf = xf_in.unchecked<1>();
            auto mL = mL_use.unchecked<2>();
            auto nL = nL_use.unchecked<2>();

            int idx = 0;
            for (int j = 0; j < num_sets; j++) for (int i = 0; i < 3; i++) seed_flat(idx++) = v(j, i);
            for (int j = 0; j < num_sets; j++) for (int i = 0; i < 3; i++) seed_flat(idx++) = w(j, i);
            for (int j = 0; j < num_sets; j++) for (int i = 0; i < 3; i++) seed_flat(idx++) = p(j, i);
            for (int j = 0; j < num_sets; j++) for (int i = 0; i < 9; i++) seed_flat(idx++) = R(j, i);
            for (int i = 0; i < NUM_STATES; i++) seed_flat(idx++) = xf(i);
            for (int j = 0; j < num_sets; j++) for (int i = 0; i < 3; i++) seed_flat(idx++) = mL(j, i);
            for (int j = 0; j < num_sets; j++) for (int i = 0; i < 3; i++) seed_flat(idx++) = nL(j, i);
            return seed_flat;
        };

        const Eigen::VectorXd seed0 = flatten_seed_from_arrays();

        auto build_BVPParams = [&](const Eigen::Vector3d& curr3,
                                   const py::array_t<double>& vA,
                                   const py::array_t<double>& wA,
                                   const py::array_t<double>& pA,
                                   const py::array_t<double>& RA,
                                   double dt_local) {
            // Build actuation currents array (NUM_ACT_SET is compile-time; we fill the first set).
            double ActuationCurrents[NUM_ACT_SET][3];
            for (int i = 0; i < NUM_ACT_SET; i++) for (int j = 0; j < 3; j++) ActuationCurrents[i][j] = 0.0;
            for (int j = 0; j < 3; j++) ActuationCurrents[0][j] = curr3(j);

            // Copy seed arrays to local fixed-size buffers for the C++ API.
            double v_L_local[NUM_ACT_SET][3]{};
            double w_L_local[NUM_ACT_SET][3]{};
            double p_L_local[NUM_ACT_SET][3]{};
            double R_L_local[NUM_ACT_SET][9]{};

            auto vv = vA.unchecked<2>();
            auto ww = wA.unchecked<2>();
            auto pp = pA.unchecked<2>();
            auto RR = RA.unchecked<2>();

            for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
                for (int i = 0; i < 3; i++) {
                    v_L_local[j][i] = vv(j, i);
                    w_L_local[j][i] = ww(j, i);
                    p_L_local[j][i] = pp(j, i);
                }
                for (int i = 0; i < 9; i++) R_L_local[j][i] = RR(j, i);
            }

            // Copy constant parameters to locals to avoid accidental mutation.
            double actInertia_local[NUM_ACT_SET][9];
            double damping_local[NUM_ACT_SET][6];
            for (int j = 0; j < NUM_ACT_SET; j++) {
                for (int i = 0; i < 9; i++) actInertia_local[j][i] = actInertia[j][i];
                for (int i = 0; i < 6; i++) damping_local[j][i] = damping[j][i];
            }

            ContactModeType ContactMode = ContactModeType::FREE_TIP;
            double TipForce[3] = {0.0, 0.0, 0.0};
            double TipConstraintPoint[3] = {0.0, 0.0, 0.0};

            CRMShootingMethodParams BVPParams = CRMDYNConstructShootingMethodParamSet(
                *catheter.getParams(), catheter.config, insertion_length, ActuationCurrents,
                ContactMode, TipConstraintPoint, TipForce, integrationStepSize,
                actInertia_local, v_L_local, w_L_local, p_L_local, R_L_local, damping_local, dt_local
            );
            // Task A1.7 Phase 3: Set integrator type
            BVPParams.dynamics.integrator_type = integrator_type;
            return BVPParams;
        };

        auto eval_residual = [&](const Eigen::VectorXd& x_scaled,
                                 const Eigen::Vector3d& curr3,
                                 const Eigen::VectorXd& seed_flat,
                                 Eigen::VectorXd& out_tau_flat,
                                 Eigen::Vector3d& out_u0) {
            // Rebuild DYNNLEParams (includes preprocessed IVP data/simulation_parameters).
            py::array_t<double> vA, wA, pA, RA, xfA, mLA, nLA;
            unpack_seed(seed_flat, vA, wA, pA, RA, xfA, mLA, nLA);
            const double dt_local = dt;
            CRMShootingMethodParams BVPParams = build_BVPParams(curr3, vA, wA, pA, RA, dt_local);

            // Prepare DYNNLE parameters.
            const bool FinalValueOnly = true;
            DYNNLEqnParams DYNNLEParams(BVPParams.no_flex_seg, BVPParams.no_rigid_seg, BVPParams.no_act_set, BVPParams.no_locmarkers, BVPParams.no_fcum_steps);

            // Provide a placeholder x_0 (p0,R0) and pass mL/nL "initial guess" from seed (not used by residual, only stored in params).
            double x_0[NUM_STATES];
            for (int i = 0; i < NUM_STATES; i++) {
                if (i < 3) x_0[i] = BVPParams.p0[i];
                else if (i < 12) x_0[i] = BVPParams.R0[i - 3];
                else x_0[i] = 0.0;
            }

            // Seed-provided guesses (unscaled) for prep.
            double mL_guess_local[NUM_ACT_SET][3]{};
            double nL_guess_local[NUM_ACT_SET][3]{};
            auto mLseed = mLA.unchecked<2>();
            auto nLseed = nLA.unchecked<2>();
            for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) for (int i = 0; i < 3; i++) {
                mL_guess_local[j][i] = mLseed(j, i);
                nL_guess_local[j][i] = nLseed(j, i);
            }

            double actInertia_local[NUM_ACT_SET][9]{};
            double damping_local[NUM_ACT_SET][6]{};
            double v_L_pre_local[NUM_ACT_SET][3]{};
            double w_L_pre_local[NUM_ACT_SET][3]{};
            double p_pre_local[NUM_ACT_SET][3]{};
            double R_pre_local[NUM_ACT_SET][9]{};

            for (int j = 0; j < BVPParams.no_act_set; ++j) {
                const auto& act = BVPParams.dynamics.actuators[j];
                for (int i = 0; i < 3; ++i) {
                    v_L_pre_local[j][i] = act.v_L_pre(i);
                    w_L_pre_local[j][i] = act.w_L_pre(i);
                    p_pre_local[j][i] = act.p_pre(i);
                }
                for (int i = 0; i < 9; ++i) {
                    actInertia_local[j][i] = act.inertia(i / 3, i % 3);
                    R_pre_local[j][i] = act.R_pre(i / 3, i % 3);
                }
                for (int i = 0; i < 6; ++i) {
                    damping_local[j][i] = act.damping(i);
                }
            }

            CRMDYNSolverIVP_Prep(
                BVPParams.no_flex_seg, BVPParams.no_rigid_seg, BVPParams.no_act_set, BVPParams.no_locmarkers, BVPParams.no_fcum_steps,
                x_0, BVPParams.IntegrationStepSize,
                BVPParams.Li, BVPParams.dlambdainv, BVPParams.rho, BVPParams.SegmentTypes,
                BVPParams.SegEndLambdas, BVPParams.LocMarkerLambdas,
                BVPParams.K, BVPParams.Kinv, BVPParams.ustar,
                BVPParams.MagMoment, BVPParams.fcumlambda, BVPParams.CoilAlignmentTurnAreaMatrix,
                BVPParams.B0, BVPParams.g, BVPParams.ActMass, actInertia_local, damping_local, BVPParams.dynamics.DELTA_T,
                v_L_pre_local, w_L_pre_local, p_pre_local, R_pre_local,
                mL_guess_local, nL_guess_local,
                FinalValueOnly, DYNNLEParams
            );

            DYNNLEParams.ContactMode = ContactModeType::FREE_TIP;
            for (int i = 0; i < 3; i++) {
                DYNNLEParams.TipForce[i] = 0.0;
                DYNNLEParams.TipConstraintPoint[i] = 0.0;
                DYNNLEParams.ftip_initialguess[i] = 0.0;
            }

            // Set xf.
            auto xfseed = xfA.unchecked<1>();
            for (int i = 0; i < NUM_STATES; i++) {
                DYNNLEParams.xf[i] = xfseed(i);
            }

            // Call residual equation.
            const int NLEq_Dim = NUM_DYN_RESIDUAL;
            std::vector<double> x_arr(NLEq_Dim, 0.0);
            for (int i = 0; i < NLEq_Dim; i++) x_arr[i] = x_scaled(i);

            std::vector<double> out_y(NLEq_Dim, 0.0);
            double u0_out[3];
            double tau_out[NUM_ACT_SET * 3];

            DYNNLEquation(x_arr.data(), out_y.data(), DYNNLEParams, u0_out, tau_out);

            Eigen::VectorXd F(NLEq_Dim);
            for (int i = 0; i < NLEq_Dim; i++) F(i) = out_y[i];

            out_u0 = Eigen::Vector3d(u0_out[0], u0_out[1], u0_out[2]);
            out_tau_flat.resize(NUM_ACT_SET * 3);
            for (int i = 0; i < NUM_ACT_SET * 3; i++) out_tau_flat(i) = tau_out[i];
            return F;
        };

        auto eval_output = [&](const Eigen::VectorXd& x_scaled,
                               const Eigen::Vector3d& curr3,
                               const Eigen::VectorXd& seed_flat) {
            // Build BVPParams and compute u0/tau from DYNNLEquation, then run DYNSolverIVP.
            Eigen::VectorXd tau_flat;
            Eigen::Vector3d u0;
            Eigen::VectorXd F = eval_residual(x_scaled, curr3, seed_flat, tau_flat, u0);
            (void)F;

            py::array_t<double> vA, wA, pA, RA, xfA, mLA, nLA;
            unpack_seed(seed_flat, vA, wA, pA, RA, xfA, mLA, nLA);
            const double dt_local = dt;
            CRMShootingMethodParams BVPParams = build_BVPParams(curr3, vA, wA, pA, RA, dt_local);

            double mL_phys[NUM_ACT_SET][3]{};
            double nL_phys[NUM_ACT_SET][3]{};
            double tau_phys[NUM_ACT_SET][3]{};
            for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
                for (int i = 0; i < 3; i++) {
                    mL_phys[j][i] = IVALUE_SCALE_M * x_scaled(j * 6 + i);
                    nL_phys[j][i] = IVALUE_SCALE_N * x_scaled(j * 6 + 3 + i);
                    tau_phys[j][i] = tau_flat(j * 3 + i);
                }
            }

            double u0_arr[3] = {u0(0), u0(1), u0(2)};
            double ftip[3] = {0.0, 0.0, 0.0};

            double xf_new[NUM_STATES];
            double x_coil[NUM_ACT_SET][NUM_COIL_STATES];
            double ReportedMarkerPos[5][3];

            DYNSolverIVP(BVPParams, u0_arr, mL_phys, nL_phys, tau_phys, ftip, true, xf_new, x_coil, ReportedMarkerPos);

            Eigen::Matrix<double, 6, 1> y;
            y(0) = xf_new[0];
            y(1) = xf_new[1];
            y(2) = xf_new[2];
            y(3) = x_coil[0][0];
            y(4) = x_coil[0][1];
            y(5) = x_coil[0][2];
            return y;
        };

        // Current vector (assume 3D for now, matching Torch pipeline).
        Eigen::Vector3d curr0(0.0, 0.0, 0.0);
        auto curr_buf = currents.request();
        const double* curr_ptr = static_cast<double*>(curr_buf.ptr);
        for (int i = 0; i < 3; i++) curr0(i) = (i < curr_buf.size) ? curr_ptr[i] : 0.0;

        // Compute residual at (x*,theta) for diagnostics.
        Eigen::VectorXd tau0_flat;
        Eigen::Vector3d u0_tmp;
        const Eigen::VectorXd F0 = eval_residual(x_star_scaled, curr0, seed0, tau0_flat, u0_tmp);
        const double residual_norm = F0.norm();

        // Residual Jacobians: Jxx and Jxθ (θ = currents3 + seed_flat).
        Eigen::MatrixXd Jxx(x_dim, x_dim);
        bool have_ad_jxx = false;
        if (x_dim == NUM_DYN_RESIDUAL && num_sets == 1) {
            try {
                // Build DYNNLEParams once at (curr0, seed0) and compute Jxx = dF/dx at x* via autodiff.
                py::array_t<double> vA, wA, pA, RA, xfA, mLA, nLA;
                unpack_seed(seed0, vA, wA, pA, RA, xfA, mLA, nLA);
                const double dt_local = dt;
                CRMShootingMethodParams BVPParams = build_BVPParams(curr0, vA, wA, pA, RA, dt_local);

                const bool FinalValueOnly = true;
                DYNNLEqnParams DYNNLEParams(BVPParams.no_flex_seg, BVPParams.no_rigid_seg, BVPParams.no_act_set, BVPParams.no_locmarkers, BVPParams.no_fcum_steps);

                double x_0[NUM_STATES];
                for (int i = 0; i < NUM_STATES; i++) {
                    if (i < 3) x_0[i] = BVPParams.p0[i];
                    else if (i < 12) x_0[i] = BVPParams.R0[i - 3];
                    else x_0[i] = 0.0;
                }

                double mL_guess_local[NUM_ACT_SET][3]{};
                double nL_guess_local[NUM_ACT_SET][3]{};
                auto mLseed = mLA.unchecked<2>();
                auto nLseed = nLA.unchecked<2>();
                for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) for (int i = 0; i < 3; i++) {
                    mL_guess_local[j][i] = mLseed(j, i);
                    nL_guess_local[j][i] = nLseed(j, i);
                }

                double actInertia_local[NUM_ACT_SET][9]{};
                double damping_local[NUM_ACT_SET][6]{};
                double v_L_pre_local[NUM_ACT_SET][3]{};
                double w_L_pre_local[NUM_ACT_SET][3]{};
                double p_pre_local[NUM_ACT_SET][3]{};
                double R_pre_local[NUM_ACT_SET][9]{};

                for (int j = 0; j < BVPParams.no_act_set; ++j) {
                    const auto& act = BVPParams.dynamics.actuators[j];
                    for (int i = 0; i < 3; ++i) {
                        v_L_pre_local[j][i] = act.v_L_pre(i);
                        w_L_pre_local[j][i] = act.w_L_pre(i);
                        p_pre_local[j][i] = act.p_pre(i);
                    }
                    for (int i = 0; i < 9; ++i) {
                        actInertia_local[j][i] = act.inertia(i / 3, i % 3);
                        R_pre_local[j][i] = act.R_pre(i / 3, i % 3);
                    }
                    for (int i = 0; i < 6; ++i) {
                        damping_local[j][i] = act.damping(i);
                    }
                }

                CRMDYNSolverIVP_Prep(
                    BVPParams.no_flex_seg, BVPParams.no_rigid_seg, BVPParams.no_act_set, BVPParams.no_locmarkers, BVPParams.no_fcum_steps,
                    x_0, BVPParams.IntegrationStepSize,
                    BVPParams.Li, BVPParams.dlambdainv, BVPParams.rho, BVPParams.SegmentTypes,
                    BVPParams.SegEndLambdas, BVPParams.LocMarkerLambdas,
                    BVPParams.K, BVPParams.Kinv, BVPParams.ustar,
                    BVPParams.MagMoment, BVPParams.fcumlambda, BVPParams.CoilAlignmentTurnAreaMatrix,
                    BVPParams.B0, BVPParams.g, BVPParams.ActMass, actInertia_local, damping_local, BVPParams.dynamics.DELTA_T,
                    v_L_pre_local, w_L_pre_local, p_pre_local, R_pre_local,
                    mL_guess_local, nL_guess_local,
                    FinalValueOnly, DYNNLEParams
                );

                DYNNLEParams.ContactMode = ContactModeType::FREE_TIP;
                for (int i = 0; i < 3; i++) {
                    DYNNLEParams.TipForce[i] = 0.0;
                    DYNNLEParams.TipConstraintPoint[i] = 0.0;
                    DYNNLEParams.ftip_initialguess[i] = 0.0;
                }

                auto xfseed = xfA.unchecked<1>();
                for (int i = 0; i < NUM_STATES; i++) {
                    DYNNLEParams.xf[i] = xfseed(i);
                }

                Eigen::VectorXd Fad;
                Eigen::MatrixXd Jad = DYNNLEquationJacobianEigenAD(x_star_scaled, DYNNLEParams, &Fad);
                if (Jad.rows() == x_dim && Jad.cols() == x_dim && Jad.allFinite()) {
                    Jxx = Jad;
                    have_ad_jxx = true;
                }
            } catch (...) {
                have_ad_jxx = false;
            }
        }

        Eigen::MatrixXd Jxx_fd;
        if (!have_ad_jxx || return_debug) {
            Jxx_fd.resize(x_dim, x_dim);
            Jxx_fd.setZero();
            for (int j = 0; j < x_dim; j++) {
                Eigen::VectorXd xp = x_star_scaled;
                Eigen::VectorXd xm = x_star_scaled;
                xp(j) += eps_residual_x;
                xm(j) -= eps_residual_x;
                Eigen::VectorXd tau_p, tau_m;
                Eigen::Vector3d u0_p, u0_m;
                const Eigen::VectorXd Fp = eval_residual(xp, curr0, seed0, tau_p, u0_p);
                const Eigen::VectorXd Fm = eval_residual(xm, curr0, seed0, tau_m, u0_m);
                Jxx_fd.col(j) = (Fp - Fm) * (0.5 / eps_residual_x);
            }
        }
        if (!have_ad_jxx) {
            Jxx = Jxx_fd;
        }

        const int theta_dim = 3 + seed_dim;
        Eigen::MatrixXd Jxth(x_dim, theta_dim);
        Jxth.setZero();

        // Compute ∂F/∂u (currents) using AD when available (Task 1.5)
        bool have_ad_jxu = false;
        if (x_dim == NUM_DYN_RESIDUAL && num_sets == 1) {
            try {
                // Build DYNNLEParams at (curr0, seed0) and compute Jxu = dF/du at x* via autodiff.
                py::array_t<double> vA, wA, pA, RA, xfA, mLA, nLA;
                unpack_seed(seed0, vA, wA, pA, RA, xfA, mLA, nLA);
                const double dt_local = dt;
                CRMShootingMethodParams BVPParams = build_BVPParams(curr0, vA, wA, pA, RA, dt_local);

                const bool FinalValueOnly = true;
                DYNNLEqnParams DYNNLEParams(BVPParams.no_flex_seg, BVPParams.no_rigid_seg, BVPParams.no_act_set, BVPParams.no_locmarkers, BVPParams.no_fcum_steps);

                double x_0[NUM_STATES];
                for (int i = 0; i < NUM_STATES; i++) {
                    if (i < 3) x_0[i] = BVPParams.p0[i];
                    else if (i < 12) x_0[i] = BVPParams.R0[i - 3];
                    else x_0[i] = 0.0;
                }

                double mL_guess_local[NUM_ACT_SET][3]{};
                double nL_guess_local[NUM_ACT_SET][3]{};
                auto mLseed = mLA.unchecked<2>();
                auto nLseed = nLA.unchecked<2>();
                for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) for (int i = 0; i < 3; i++) {
                    mL_guess_local[j][i] = mLseed(j, i);
                    nL_guess_local[j][i] = nLseed(j, i);
                }

                double actInertia_local[NUM_ACT_SET][9]{};
                double damping_local[NUM_ACT_SET][6]{};
                double v_L_pre_local[NUM_ACT_SET][3]{};
                double w_L_pre_local[NUM_ACT_SET][3]{};
                double p_pre_local[NUM_ACT_SET][3]{};
                double R_pre_local[NUM_ACT_SET][9]{};

                for (int j = 0; j < BVPParams.no_act_set; ++j) {
                    const auto& act = BVPParams.dynamics.actuators[j];
                    for (int i = 0; i < 3; ++i) {
                        v_L_pre_local[j][i] = act.v_L_pre(i);
                        w_L_pre_local[j][i] = act.w_L_pre(i);
                        p_pre_local[j][i] = act.p_pre(i);
                    }
                    for (int i = 0; i < 9; ++i) {
                        actInertia_local[j][i] = act.inertia(i / 3, i % 3);
                        R_pre_local[j][i] = act.R_pre(i / 3, i % 3);
                    }
                    for (int i = 0; i < 6; ++i) {
                        damping_local[j][i] = act.damping(i);
                    }
                }

                CRMDYNSolverIVP_Prep(
                    BVPParams.no_flex_seg, BVPParams.no_rigid_seg, BVPParams.no_act_set, BVPParams.no_locmarkers, BVPParams.no_fcum_steps,
                    x_0, BVPParams.IntegrationStepSize,
                    BVPParams.Li, BVPParams.dlambdainv, BVPParams.rho, BVPParams.SegmentTypes,
                    BVPParams.SegEndLambdas, BVPParams.LocMarkerLambdas,
                    BVPParams.K, BVPParams.Kinv, BVPParams.ustar,
                    BVPParams.MagMoment, BVPParams.fcumlambda, BVPParams.CoilAlignmentTurnAreaMatrix,
                    BVPParams.B0, BVPParams.g, BVPParams.ActMass, actInertia_local, damping_local, BVPParams.dynamics.DELTA_T,
                    v_L_pre_local, w_L_pre_local, p_pre_local, R_pre_local,
                    mL_guess_local, nL_guess_local,
                    FinalValueOnly, DYNNLEParams
                );

                DYNNLEParams.ContactMode = ContactModeType::FREE_TIP;
                for (int i = 0; i < 3; i++) {
                    DYNNLEParams.TipForce[i] = 0.0;
                    DYNNLEParams.TipConstraintPoint[i] = 0.0;
                    DYNNLEParams.ftip_initialguess[i] = 0.0;
                }

                auto xfseed = xfA.unchecked<1>();
                for (int i = 0; i < NUM_STATES; i++) {
                    DYNNLEParams.xf[i] = xfseed(i);
                }

                Eigen::VectorXd Fad;
                Eigen::MatrixXd Jxu_ad = DYNNLEquationControlJacobianEigenAD(x_star_scaled, curr0, DYNNLEParams, &Fad);
                if (Jxu_ad.rows() == x_dim && Jxu_ad.cols() == 3 && Jxu_ad.allFinite()) {
                    Jxth.leftCols(3) = Jxu_ad;
                    have_ad_jxu = true;
                }
            } catch (const std::exception& e) {
                std::cerr << "AD control jacobian failed: " << e.what() << std::endl;
                have_ad_jxu = false;
            } catch (...) {
                std::cerr << "AD control jacobian failed with unknown exception" << std::endl;
                have_ad_jxu = false;
            }
        }

        // Compute remaining ∂F/∂θ for seed and currents (if AD failed) using FD
        for (int j = 0; j < theta_dim; j++) {
            if (j < 3 && have_ad_jxu) {
                // Skip currents - already computed via AD
                continue;
            }
            Eigen::Vector3d curr_p = curr0;
            Eigen::Vector3d curr_m = curr0;
            Eigen::VectorXd seed_p = seed0;
            Eigen::VectorXd seed_m = seed0;
            if (j < 3) {
                curr_p(j) += eps_residual_theta;
                curr_m(j) -= eps_residual_theta;
            } else {
                const int k = j - 3;
                seed_p(k) += eps_residual_theta;
                seed_m(k) -= eps_residual_theta;
            }
            Eigen::VectorXd tau_p, tau_m;
            Eigen::Vector3d u0_p, u0_m;
            const Eigen::VectorXd Fp = eval_residual(x_star_scaled, curr_p, seed_p, tau_p, u0_p);
            const Eigen::VectorXd Fm = eval_residual(x_star_scaled, curr_m, seed_m, tau_m, u0_m);
            Jxth.col(j) = (Fp - Fm) * (0.5 / eps_residual_theta);
        }

        // Solve for dx/dθ: Jxx * X = -Jxθ
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(Jxx);
        if (qr.rank() < x_dim) {
            throw std::runtime_error("Implicit linearization failed: Jxx is rank-deficient at x*.");
        }
        const Eigen::MatrixXd dxdth = qr.solve(-Jxth);  // (x_dim, theta_dim)

        // Partials of g: gx and gθ (using the post-solve mapping, no root-solve).
        Eigen::MatrixXd gx(6, x_dim);
        gx.setZero();
        for (int j = 0; j < x_dim; j++) {
            Eigen::VectorXd xp = x_star_scaled;
            Eigen::VectorXd xm = x_star_scaled;
            xp(j) += eps_g_x;
            xm(j) -= eps_g_x;
            const Eigen::Matrix<double, 6, 1> yp = eval_output(xp, curr0, seed0);
            const Eigen::Matrix<double, 6, 1> ym = eval_output(xm, curr0, seed0);
            gx.col(j) = (yp - ym) * (0.5 / eps_g_x);
        }

        Eigen::MatrixXd gth(6, theta_dim);
        gth.setZero();
        for (int j = 0; j < theta_dim; j++) {
            Eigen::Vector3d curr_p = curr0;
            Eigen::Vector3d curr_m = curr0;
            Eigen::VectorXd seed_p = seed0;
            Eigen::VectorXd seed_m = seed0;
            if (j < 3) {
                curr_p(j) += eps_g_theta;
                curr_m(j) -= eps_g_theta;
            } else {
                const int k = j - 3;
                seed_p(k) += eps_g_theta;
                seed_m(k) -= eps_g_theta;
            }
            const Eigen::Matrix<double, 6, 1> yp = eval_output(x_star_scaled, curr_p, seed_p);
            const Eigen::Matrix<double, 6, 1> ym = eval_output(x_star_scaled, curr_m, seed_m);
            gth.col(j) = (yp - ym) * (0.5 / eps_g_theta);
        }

        // Assemble dy/dθ = gθ + gx * dx/dθ
        const Eigen::MatrixXd dydth = gth + gx * dxdth;  // (6, theta_dim)

        // Split into B (currents) and A (seed).
        py::array_t<double> next_state({6});
        auto ns = next_state.mutable_unchecked<1>();
        for (int i = 0; i < 6; i++) ns(i) = y0(i);

        py::array_t<double> B_out({6, 3});
        auto Bout = B_out.mutable_unchecked<2>();
        for (int i = 0; i < 6; i++) for (int j = 0; j < 3; j++) Bout(i, j) = dydth(i, j);

        py::array_t<double> A_out({6, seed_dim});
        auto Aout = A_out.mutable_unchecked<2>();
        for (int i = 0; i < 6; i++) for (int j = 0; j < seed_dim; j++) Aout(i, j) = dydth(i, 3 + j);

        py::dict result;
        result["next_state"] = next_state;
        result["B"] = B_out;
        result["A"] = A_out;
        result["seed_dim"] = seed_dim;
        result["base"] = base;
        result["residual_norm"] = residual_norm;
        if (return_debug) {
            result["have_ad_jxx"] = have_ad_jxx;
            result["have_ad_jxu"] = have_ad_jxu;
            py::array_t<double> Jxx_out({x_dim, x_dim});
            auto Ju = Jxx_out.mutable_unchecked<2>();
            for (int r = 0; r < x_dim; r++) for (int c = 0; c < x_dim; c++) Ju(r, c) = Jxx(r, c);
            result["Jxx"] = Jxx_out;

            py::array_t<double> Jxx_fd_out({x_dim, x_dim});
            auto Jf = Jxx_fd_out.mutable_unchecked<2>();
            for (int r = 0; r < x_dim; r++) for (int c = 0; c < x_dim; c++) Jf(r, c) = Jxx_fd(r, c);
            result["Jxx_fd"] = Jxx_fd_out;
        }
        return result;
    }

    /**
     * Compute parameter Jacobian: ∂F/∂θ (gradient of residual w.r.t learnable params)
     */
    py::dict compute_parameter_jacobian(
        py::array_t<double> currents,
        double insertion_length,
        py::array_t<double> v_in,
        py::array_t<double> w_in,
        py::array_t<double> p_in,
        py::array_t<double> R_in,
        py::array_t<double> xf_in,
        py::array_t<double> mL_in = py::array_t<double>(),
        py::array_t<double> nL_in = py::array_t<double>()
    ) {
        using namespace CRMCatheterModel;

        auto vbuf = v_in.request();
        if (vbuf.ndim != 2 || vbuf.shape[1] != 3) {
            throw std::runtime_error("v_in must have shape (num_act_set, 3)");
        }
        const int num_sets = static_cast<int>(vbuf.shape[0]);
        if (num_sets != 1) {
            throw std::runtime_error("compute_parameter_jacobian currently only supports NUM_ACT_SET=1");
        }

        auto wbuf = w_in.request();
        auto pbuf = p_in.request();
        auto Rbuf = R_in.request();
        auto xfbuf = xf_in.request();
        if (wbuf.ndim != 2 || wbuf.shape[0] != num_sets || wbuf.shape[1] != 3) {
            throw std::runtime_error("w_in must have shape (num_act_set, 3)");
        }
        if (pbuf.ndim != 2 || pbuf.shape[0] != num_sets || pbuf.shape[1] != 3) {
            throw std::runtime_error("p_in must have shape (num_act_set, 3)");
        }
        if (Rbuf.ndim != 2 || Rbuf.shape[0] != num_sets || Rbuf.shape[1] != 9) {
            throw std::runtime_error("R_in must have shape (num_act_set, 9)");
        }
        if (xfbuf.ndim != 1 || xfbuf.shape[0] != NUM_STATES) {
            throw std::runtime_error("xf_in must have shape (NUM_STATES,)");
        }

        const double kMnZeroEps = 1e-12;
        auto sum_abs_mn = [&](const double mn[NUM_ACT_SET][3]) {
            double total = 0.0;
            for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
                for (int i = 0; i < 3; i++) total += std::abs(mn[j][i]);
            }
            return total;
        };
        auto ensure_mn = [&](const py::array_t<double>& arr, const double fallback[NUM_ACT_SET][3]) {
            if (arr.size() != 0) {
                auto abuf = arr.request();
                if (abuf.ndim != 2 || abuf.shape[0] != num_sets || abuf.shape[1] != 3) {
                    throw std::runtime_error("mL/nL must have shape (num_act_set, 3) when provided");
                }
                const double* aptr = static_cast<double*>(abuf.ptr);
                double input_abs = 0.0;
                for (int j = 0; j < num_sets; j++) {
                    for (int i = 0; i < 3; i++) {
                        const ssize_t idx = j * 3 + i;
                        if (idx < abuf.size) input_abs += std::abs(aptr[idx]);
                    }
                }
                if (input_abs > kMnZeroEps) {
                    return arr;
                }
            }
            const bool use_fallback = sum_abs_mn(fallback) > kMnZeroEps;
            py::array_t<double> out({num_sets, 3});
            auto o = out.mutable_unchecked<2>();
            for (int j = 0; j < num_sets; j++) {
                for (int i = 0; i < 3; i++) {
                    o(j, i) = use_fallback ? fallback[j][i] : 0.0;
                }
            }
            return out;
        };

        py::array_t<double> mL_use = ensure_mn(mL_in, mL_guess);
        py::array_t<double> nL_use = ensure_mn(nL_in, nL_guess);

        py::dict base = step_from_seed(currents, insertion_length, v_in, w_in, p_in, R_in, xf_in, mL_use, nL_use, std::nullopt);
        const bool ok = base["converged"].cast<bool>();
        if (!ok) {
            throw std::runtime_error("Dynamics did not converge; cannot compute parameter Jacobian");
        }

        auto mL_star_arr = base["next_mL"].cast<py::array_t<double>>();
        auto nL_star_arr = base["next_nL"].cast<py::array_t<double>>();
        auto mL_star = mL_star_arr.request();
        auto nL_star = nL_star_arr.request();
        const double* mL_star_ptr = static_cast<double*>(mL_star.ptr);
        const double* nL_star_ptr = static_cast<double*>(nL_star.ptr);

        const int x_dim = num_sets * 6;
        Eigen::VectorXd x_star_scaled(x_dim);
        for (int j = 0; j < num_sets; j++) {
            for (int i = 0; i < 3; i++) {
                x_star_scaled(j * 6 + i) = mL_star_ptr[j * 3 + i] / IVALUE_SCALE_M;
                x_star_scaled(j * 6 + 3 + i) = nL_star_ptr[j * 3 + i] / IVALUE_SCALE_N;
            }
        }

        Eigen::Vector3d curr0(0.0, 0.0, 0.0);
        auto curr_buf = currents.request();
        const double* curr_ptr = static_cast<double*>(curr_buf.ptr);
        for (int i = 0; i < 3; i++) curr0(i) = (i < curr_buf.size) ? curr_ptr[i] : 0.0;

        const double dt_local = dt;

        double ActuationCurrents[NUM_ACT_SET][3];
        for (int i = 0; i < NUM_ACT_SET; i++) for (int j = 0; j < 3; j++) ActuationCurrents[i][j] = 0.0;
        for (int j = 0; j < 3; j++) ActuationCurrents[0][j] = curr0(j);

        double v_L_local[NUM_ACT_SET][3]{};
        double w_L_local[NUM_ACT_SET][3]{};
        double p_L_local[NUM_ACT_SET][3]{};
        double R_L_local[NUM_ACT_SET][9]{};

        auto vv = v_in.unchecked<2>();
        auto ww = w_in.unchecked<2>();
        auto pp = p_in.unchecked<2>();
        auto RR = R_in.unchecked<2>();

        for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
            for (int i = 0; i < 3; i++) {
                v_L_local[j][i] = vv(j, i);
                w_L_local[j][i] = ww(j, i);
                p_L_local[j][i] = pp(j, i);
            }
            for (int i = 0; i < 9; i++) R_L_local[j][i] = RR(j, i);
        }

        double actInertia_local[NUM_ACT_SET][9];
        double damping_local[NUM_ACT_SET][6];
        for (int j = 0; j < NUM_ACT_SET; j++) {
            for (int i = 0; i < 9; i++) actInertia_local[j][i] = actInertia[j][i];
            for (int i = 0; i < 6; i++) damping_local[j][i] = damping[j][i];
        }

        ContactModeType ContactMode = ContactModeType::FREE_TIP;
        double TipForce[3] = {0.0, 0.0, 0.0};
        double TipConstraintPoint[3] = {0.0, 0.0, 0.0};

        CRMShootingMethodParams BVPParams = CRMDYNConstructShootingMethodParamSet(
            *catheter.getParams(), catheter.config, insertion_length, ActuationCurrents,
            ContactMode, TipConstraintPoint, TipForce, integrationStepSize,
            actInertia_local, v_L_local, w_L_local, p_L_local, R_L_local, damping_local, dt_local
        );

        // Task A1.7 Phase 3: Set integrator type
        BVPParams.dynamics.integrator_type = integrator_type;

        const bool FinalValueOnly = true;
        DYNNLEqnParams DYNNLEParams(BVPParams.no_flex_seg, BVPParams.no_rigid_seg, BVPParams.no_act_set, BVPParams.no_locmarkers, BVPParams.no_fcum_steps);

        double x_0[NUM_STATES];
        for (int i = 0; i < NUM_STATES; i++) {
            if (i < 3) x_0[i] = BVPParams.p0[i];
            else if (i < 12) x_0[i] = BVPParams.R0[i - 3];
            else x_0[i] = 0.0;
        }

        double mL_guess_local[NUM_ACT_SET][3]{};
        double nL_guess_local[NUM_ACT_SET][3]{};
        auto mLseed = mL_use.unchecked<2>();
        auto nLseed = nL_use.unchecked<2>();
        for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) for (int i = 0; i < 3; i++) {
            mL_guess_local[j][i] = mLseed(j, i);
            nL_guess_local[j][i] = nLseed(j, i);
        }

        double actInertia_dyn[NUM_ACT_SET][9]{};
        double damping_dyn[NUM_ACT_SET][6]{};
        double v_L_pre_dyn[NUM_ACT_SET][3]{};
        double w_L_pre_dyn[NUM_ACT_SET][3]{};
        double p_pre_dyn[NUM_ACT_SET][3]{};
        double R_pre_dyn[NUM_ACT_SET][9]{};

        for (int j = 0; j < BVPParams.no_act_set; ++j) {
            const auto& act = BVPParams.dynamics.actuators[j];
            for (int i = 0; i < 3; ++i) {
                v_L_pre_dyn[j][i] = act.v_L_pre(i);
                w_L_pre_dyn[j][i] = act.w_L_pre(i);
                p_pre_dyn[j][i] = act.p_pre(i);
            }
            for (int i = 0; i < 9; ++i) {
                actInertia_dyn[j][i] = act.inertia(i / 3, i % 3);
                R_pre_dyn[j][i] = act.R_pre(i / 3, i % 3);
            }
            for (int i = 0; i < 6; ++i) {
                damping_dyn[j][i] = act.damping(i);
            }
        }

        CRMDYNSolverIVP_Prep(
            BVPParams.no_flex_seg, BVPParams.no_rigid_seg, BVPParams.no_act_set, BVPParams.no_locmarkers, BVPParams.no_fcum_steps,
            x_0, BVPParams.IntegrationStepSize,
            BVPParams.Li, BVPParams.dlambdainv, BVPParams.rho, BVPParams.SegmentTypes,
            BVPParams.SegEndLambdas, BVPParams.LocMarkerLambdas,
            BVPParams.K, BVPParams.Kinv, BVPParams.ustar,
            BVPParams.MagMoment, BVPParams.fcumlambda, BVPParams.CoilAlignmentTurnAreaMatrix,
            BVPParams.B0, BVPParams.g, BVPParams.ActMass, actInertia_dyn, damping_dyn, BVPParams.dynamics.DELTA_T,
            v_L_pre_dyn, w_L_pre_dyn, p_pre_dyn, R_pre_dyn,
            mL_guess_local, nL_guess_local,
            FinalValueOnly, DYNNLEParams
        );

        DYNNLEParams.ContactMode = ContactModeType::FREE_TIP;
        for (int i = 0; i < 3; i++) {
            DYNNLEParams.TipForce[i] = 0.0;
            DYNNLEParams.TipConstraintPoint[i] = 0.0;
            DYNNLEParams.ftip_initialguess[i] = 0.0;
        }

        auto xfA = xf_in.unchecked<1>();
        for (int i = 0; i < NUM_STATES; i++) {
            DYNNLEParams.xf[i] = xfA(i);
        }

        Eigen::VectorXd residual;
        Eigen::VectorXd theta;
        Eigen::MatrixXd J_theta = DYNNLEquationParameterJacobianEigenAD(x_star_scaled, DYNNLEParams, &residual, &theta);

        const int num_params = static_cast<int>(theta.size());
        const int res_dim = static_cast<int>(residual.size());

        py::array_t<double> J_theta_out({res_dim, num_params});
        auto Jout = J_theta_out.mutable_unchecked<2>();
        for (int i = 0; i < res_dim; i++) {
            for (int j = 0; j < num_params; j++) {
                Jout(i, j) = J_theta(i, j);
            }
        }

        py::array_t<double> theta_out(num_params);
        auto tout = theta_out.mutable_unchecked<1>();
        for (int i = 0; i < num_params; i++) {
            tout(i) = theta(i);
        }

        py::array_t<double> residual_out(res_dim);
        auto rout = residual_out.mutable_unchecked<1>();
        for (int i = 0; i < res_dim; i++) {
            rout(i) = residual(i);
        }

        auto param_names = dynnl_ad_eigen::getLearnableParamNames();
        py::list names_list;
        for (const auto& name : param_names) {
            names_list.append(py::str(name));
        }

        py::dict result;
        result["J_theta"] = J_theta_out;
        result["theta"] = theta_out;
        result["residual"] = residual_out;
        result["param_names"] = names_list;
        result["converged"] = ok;
        result["base"] = base;
        return result;
    }

    /**
     * Compute residual at a fixed state without re-solving.
     */
    py::dict compute_residual_at_state(
        py::array_t<double> currents,
        double insertion_length,
        py::array_t<double> v_in,
        py::array_t<double> w_in,
        py::array_t<double> p_in,
        py::array_t<double> R_in,
        py::array_t<double> xf_in,
        py::array_t<double> mL_fixed,
        py::array_t<double> nL_fixed
    ) {
        using namespace CRMCatheterModel;

        auto vbuf = v_in.request();
        if (vbuf.ndim != 2 || vbuf.shape[1] != 3) {
            throw std::runtime_error("v_in must have shape (num_act_set, 3)");
        }
        const int num_sets = static_cast<int>(vbuf.shape[0]);
        if (num_sets != 1) {
            throw std::runtime_error("compute_residual_at_state currently only supports NUM_ACT_SET=1");
        }

        auto wbuf = w_in.request();
        auto pbuf = p_in.request();
        auto Rbuf = R_in.request();
        auto xfbuf = xf_in.request();
        if (wbuf.ndim != 2 || wbuf.shape[0] != num_sets || wbuf.shape[1] != 3) {
            throw std::runtime_error("w_in must have shape (num_act_set, 3)");
        }
        if (pbuf.ndim != 2 || pbuf.shape[0] != num_sets || pbuf.shape[1] != 3) {
            throw std::runtime_error("p_in must have shape (num_act_set, 3)");
        }
        if (Rbuf.ndim != 2 || Rbuf.shape[0] != num_sets || Rbuf.shape[1] != 9) {
            throw std::runtime_error("R_in must have shape (num_act_set, 9)");
        }
        if (xfbuf.ndim != 1 || xfbuf.shape[0] != NUM_STATES) {
            throw std::runtime_error("xf_in must have shape (NUM_STATES,)");
        }

        auto mLbuf = mL_fixed.request();
        auto nLbuf = nL_fixed.request();
        if (mLbuf.ndim != 2 || mLbuf.shape[0] != num_sets || mLbuf.shape[1] != 3) {
            throw std::runtime_error("mL_fixed must have shape (num_act_set, 3)");
        }
        if (nLbuf.ndim != 2 || nLbuf.shape[0] != num_sets || nLbuf.shape[1] != 3) {
            throw std::runtime_error("nL_fixed must have shape (num_act_set, 3)");
        }

        const double* mL_ptr = static_cast<double*>(mLbuf.ptr);
        const double* nL_ptr = static_cast<double*>(nLbuf.ptr);

        const int x_dim = num_sets * 6;
        Eigen::VectorXd x_scaled(x_dim);
        for (int j = 0; j < num_sets; j++) {
            for (int i = 0; i < 3; i++) {
                x_scaled(j * 6 + i) = mL_ptr[j * 3 + i] / IVALUE_SCALE_M;
                x_scaled(j * 6 + 3 + i) = nL_ptr[j * 3 + i] / IVALUE_SCALE_N;
            }
        }

        Eigen::Vector3d curr0(0.0, 0.0, 0.0);
        auto curr_buf = currents.request();
        const double* curr_ptr = static_cast<double*>(curr_buf.ptr);
        for (int i = 0; i < 3; i++) curr0(i) = (i < curr_buf.size) ? curr_ptr[i] : 0.0;

        const double dt_local = dt;

        double ActuationCurrents[NUM_ACT_SET][3];
        for (int i = 0; i < NUM_ACT_SET; i++) for (int j = 0; j < 3; j++) ActuationCurrents[i][j] = 0.0;
        for (int j = 0; j < 3; j++) ActuationCurrents[0][j] = curr0(j);

        double v_L_local[NUM_ACT_SET][3]{};
        double w_L_local[NUM_ACT_SET][3]{};
        double p_L_local[NUM_ACT_SET][3]{};
        double R_L_local[NUM_ACT_SET][9]{};

        auto vv = v_in.unchecked<2>();
        auto ww = w_in.unchecked<2>();
        auto pp = p_in.unchecked<2>();
        auto RR = R_in.unchecked<2>();

        for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
            for (int i = 0; i < 3; i++) {
                v_L_local[j][i] = vv(j, i);
                w_L_local[j][i] = ww(j, i);
                p_L_local[j][i] = pp(j, i);
            }
            for (int i = 0; i < 9; i++) R_L_local[j][i] = RR(j, i);
        }

        double actInertia_local[NUM_ACT_SET][9];
        double damping_local[NUM_ACT_SET][6];
        for (int j = 0; j < NUM_ACT_SET; j++) {
            for (int i = 0; i < 9; i++) actInertia_local[j][i] = actInertia[j][i];
            for (int i = 0; i < 6; i++) damping_local[j][i] = damping[j][i];
        }

        ContactModeType ContactMode = ContactModeType::FREE_TIP;
        double TipForce[3] = {0.0, 0.0, 0.0};
        double TipConstraintPoint[3] = {0.0, 0.0, 0.0};

        CRMShootingMethodParams BVPParams = CRMDYNConstructShootingMethodParamSet(
            *catheter.getParams(), catheter.config, insertion_length, ActuationCurrents,
            ContactMode, TipConstraintPoint, TipForce, integrationStepSize,
            actInertia_local, v_L_local, w_L_local, p_L_local, R_L_local, damping_local, dt_local
        );

        // Task A1.7 Phase 3: Set integrator type
        BVPParams.dynamics.integrator_type = integrator_type;

        const bool FinalValueOnly = true;
        DYNNLEqnParams DYNNLEParams(BVPParams.no_flex_seg, BVPParams.no_rigid_seg, BVPParams.no_act_set, BVPParams.no_locmarkers, BVPParams.no_fcum_steps);

        double x_0[NUM_STATES];
        for (int i = 0; i < NUM_STATES; i++) {
            if (i < 3) x_0[i] = BVPParams.p0[i];
            else if (i < 12) x_0[i] = BVPParams.R0[i - 3];
            else x_0[i] = 0.0;
        }

        double mL_guess_local[NUM_ACT_SET][3]{};
        double nL_guess_local[NUM_ACT_SET][3]{};
        for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) for (int i = 0; i < 3; i++) {
            mL_guess_local[j][i] = mL_ptr[j * 3 + i];
            nL_guess_local[j][i] = nL_ptr[j * 3 + i];
        }

        double actInertia_dyn[NUM_ACT_SET][9]{};
        double damping_dyn[NUM_ACT_SET][6]{};
        double v_L_pre_dyn[NUM_ACT_SET][3]{};
        double w_L_pre_dyn[NUM_ACT_SET][3]{};
        double p_pre_dyn[NUM_ACT_SET][3]{};
        double R_pre_dyn[NUM_ACT_SET][9]{};

        for (int j = 0; j < BVPParams.no_act_set; ++j) {
            const auto& act = BVPParams.dynamics.actuators[j];
            for (int i = 0; i < 3; ++i) {
                v_L_pre_dyn[j][i] = act.v_L_pre(i);
                w_L_pre_dyn[j][i] = act.w_L_pre(i);
                p_pre_dyn[j][i] = act.p_pre(i);
            }
            for (int i = 0; i < 9; ++i) {
                actInertia_dyn[j][i] = act.inertia(i / 3, i % 3);
                R_pre_dyn[j][i] = act.R_pre(i / 3, i % 3);
            }
            for (int i = 0; i < 6; ++i) {
                damping_dyn[j][i] = act.damping(i);
            }
        }

        CRMDYNSolverIVP_Prep(
            BVPParams.no_flex_seg, BVPParams.no_rigid_seg, BVPParams.no_act_set, BVPParams.no_locmarkers, BVPParams.no_fcum_steps,
            x_0, BVPParams.IntegrationStepSize,
            BVPParams.Li, BVPParams.dlambdainv, BVPParams.rho, BVPParams.SegmentTypes,
            BVPParams.SegEndLambdas, BVPParams.LocMarkerLambdas,
            BVPParams.K, BVPParams.Kinv, BVPParams.ustar,
            BVPParams.MagMoment, BVPParams.fcumlambda, BVPParams.CoilAlignmentTurnAreaMatrix,
            BVPParams.B0, BVPParams.g, BVPParams.ActMass, actInertia_dyn, damping_dyn, BVPParams.dynamics.DELTA_T,
            v_L_pre_dyn, w_L_pre_dyn, p_pre_dyn, R_pre_dyn,
            mL_guess_local, nL_guess_local,
            FinalValueOnly, DYNNLEParams
        );

        DYNNLEParams.ContactMode = ContactModeType::FREE_TIP;
        for (int i = 0; i < 3; i++) {
            DYNNLEParams.TipForce[i] = 0.0;
            DYNNLEParams.TipConstraintPoint[i] = 0.0;
            DYNNLEParams.ftip_initialguess[i] = 0.0;
        }

        auto xfA = xf_in.unchecked<1>();
        for (int i = 0; i < NUM_STATES; i++) {
            DYNNLEParams.xf[i] = xfA(i);
        }

        Eigen::VectorXd residual = DYNNLEquationResidualEigenDouble(x_scaled, DYNNLEParams);
        Eigen::VectorXd theta = dynnl_ad_eigen::packLearnableParams(DYNNLEParams, 0);

        const int num_params = static_cast<int>(theta.size());
        const int res_dim = static_cast<int>(residual.size());

        py::array_t<double> residual_out(res_dim);
        auto rout = residual_out.mutable_unchecked<1>();
        for (int i = 0; i < res_dim; i++) {
            rout(i) = residual(i);
        }

        py::array_t<double> theta_out(num_params);
        auto tout = theta_out.mutable_unchecked<1>();
        for (int i = 0; i < num_params; i++) {
            tout(i) = theta(i);
        }

        py::dict result;
        result["residual"] = residual_out;
        result["theta"] = theta_out;
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
        .def("load_parameters", &CRMKinematicsWrapper::loadParams,
             py::arg("param_file"), py::arg("config_file"),
             "Load catheter parameters from files")
        .def("forward_kinematics", &CRMKinematicsWrapper::forwardKinematics,
             py::arg("currents"), py::arg("insertion_length"),
             "Compute forward kinematics")
        .def("forward_kinematics_with_guess", &CRMKinematicsWrapper::forwardKinematicsWithGuess,
             py::arg("currents"), py::arg("insertion_length"), py::arg("deltau0_initialguess"),
             "Compute FK with delta_u0 warm-start")
        .def("fk_and_jacobian", &CRMKinematicsWrapper::fkAndJacobian,
             py::arg("currents"), py::arg("insertion_length"),
             py::arg("deltau0_initialguess") = py::array_t<double>(),
             "Compute FK and analytical Jacobian in one call (fast path)")
        .def("compute_jacobian", &CRMKinematicsWrapper::computeJacobian,
             py::arg("currents"), py::arg("insertion_length"),
             "Compute analytical Jacobian")
        .def_readwrite("integration_step_size", &CRMKinematicsWrapper::integrationStepSize)
        .def("is_initialized", &CRMKinematicsWrapper::isInitialized);

    // CRMDynamicsWrapper
    py::class_<CRMDynamicsWrapper>(m, "CRMDynamics")
        .def(py::init<>())
        .def("load_parameters", &CRMDynamicsWrapper::loadParams,
             py::arg("param_file"), py::arg("config_file"),
             "Load catheter parameters from files")
        .def("set_damping", &CRMDynamicsWrapper::setDamping,
             py::arg("damping_values"),
             "Set damping coefficients")
        .def("get_config_snapshot", &CRMDynamicsWrapper::get_config_snapshot,
             "Debug: get config/segment snapshot used by dynamics")
        .def("set_timestep", &CRMDynamicsWrapper::setTimestep,
             py::arg("dt"),
             "Set simulation timestep")
        .def("set_integrator", &CRMDynamicsWrapper::set_integrator,
             py::arg("integrator"),
             "Set integrator type: 'abm4' (default) or 'rk4' (more stable)")
        .def("get_integrator", &CRMDynamicsWrapper::get_integrator,
             "Get current integrator type ('abm4' or 'rk4')")
        .def("reset", &CRMDynamicsWrapper::reset,
             "Reset dynamics state to zeros")
        .def("initialize_from_kinematics", &CRMDynamicsWrapper::initializeFromKinematics,
             py::arg("currents"), py::arg("insertion_length"),
             "Initialize dynamics from FK solution (MUST call before step)")
        .def("get_last_fk_output", &CRMDynamicsWrapper::get_last_fk_output,
             "Debug: get last FK output used during dynamics initialization")
        .def("initialize_from_seed", &CRMDynamicsWrapper::initializeFromSeed,
             py::arg("currents"), py::arg("insertion_length"),
             py::arg("p_in"), py::arg("R_in"), py::arg("xf_in"),
             py::arg("mL_in") = py::array_t<double>(),
             py::arg("nL_in") = py::array_t<double>(),
             "Initialize dynamics directly from explicit seeds (p_L, R_L, xf, optional mL/nL)")
        .def("debug_seed_state", &CRMDynamicsWrapper::debugSeedState,
             py::arg("v_in"), py::arg("w_in"), py::arg("p_in"),
             py::arg("R_in"), py::arg("xf_in"),
             py::arg("mL_in") = py::array_t<double>(),
             py::arg("nL_in") = py::array_t<double>(),
             "Debug: manually seed dynamics state (not for production use)")
        .def("get_seed_state", &CRMDynamicsWrapper::get_seed_state,
             "Get current internal seed state (v,w,p,R,xf,mL,nL)")
        .def("set_seed_state", &CRMDynamicsWrapper::set_seed_state,
             py::arg("v"), py::arg("w"), py::arg("p"), py::arg("R"), py::arg("xf"),
             py::arg("mL") = py::array_t<double>(),
             py::arg("nL") = py::array_t<double>(),
             "Set internal seed state (v,w,p,R,xf,mL,nL)")
        .def("step", &CRMDynamicsWrapper::stepDynamics,
             py::arg("currents"), py::arg("insertion_length"),
             "Step dynamics forward")
        .def("step_from_seed", &CRMDynamicsWrapper::step_from_seed,
             py::arg("currents"), py::arg("insertion_length"),
             py::arg("v"), py::arg("w"), py::arg("p"), py::arg("R"), py::arg("xf"),
             py::arg("mL") = py::array_t<double>(),
             py::arg("nL") = py::array_t<double>(),
             py::arg("dt") = std::nullopt,
             "Pure dynamics step from explicit seed (does not mutate internal state)")
        .def("linearize_action_from_seed", &CRMDynamicsWrapper::linearize_action_from_seed,
             py::arg("currents"), py::arg("insertion_length"),
             py::arg("v"), py::arg("w"), py::arg("p"), py::arg("R"), py::arg("xf"),
             py::arg("mL") = py::array_t<double>(),
             py::arg("nL") = py::array_t<double>(),
             py::arg("eps") = 1e-4,
             "Finite-difference B = d(next_state)/d(currents) around explicit seed")
        .def("linearize_full_seed_action_from_seed", &CRMDynamicsWrapper::linearize_full_seed_action_from_seed,
             py::arg("currents"), py::arg("insertion_length"),
             py::arg("v"), py::arg("w"), py::arg("p"), py::arg("R"), py::arg("xf"),
             py::arg("mL") = py::array_t<double>(),
             py::arg("nL") = py::array_t<double>(),
             py::arg("eps_u") = 1e-4,
             py::arg("eps_seed") = 1e-4,
             "Finite-difference A,B where A=d(next_state)/d(full_seed) around explicit seed")
        .def("linearize_full_seed_action_from_seed_implicit", &CRMDynamicsWrapper::linearize_full_seed_action_from_seed_implicit,
             py::arg("currents"), py::arg("insertion_length"),
             py::arg("v_in"), py::arg("w_in"), py::arg("p_in"), py::arg("R_in"), py::arg("xf_in"),
             py::arg("mL_in") = py::array_t<double>(),
             py::arg("nL_in") = py::array_t<double>(),
             py::arg("eps_residual_x") = 1e-5,
             py::arg("eps_residual_theta") = 1e-5,
             py::arg("eps_g_x") = 1e-5,
             py::arg("eps_g_theta") = 1e-5,
             py::arg("return_debug") = false,
             "Implicit linearization scaffold: FD residual Jacobians + implicit sensitivity to compute A,B")
        .def("compute_parameter_jacobian", &CRMDynamicsWrapper::compute_parameter_jacobian,
             py::arg("currents"), py::arg("insertion_length"),
             py::arg("v"), py::arg("w"), py::arg("p"), py::arg("R"), py::arg("xf"),
             py::arg("mL") = py::array_t<double>(),
             py::arg("nL") = py::array_t<double>(),
             "Compute parameter Jacobian: gradient of residual w.r.t learnable physical parameters (damping, stiffness, etc.)")
        .def("compute_residual_at_state", &CRMDynamicsWrapper::compute_residual_at_state,
             py::arg("currents"), py::arg("insertion_length"),
             py::arg("v"), py::arg("w"), py::arg("p"), py::arg("R"), py::arg("xf"),
             py::arg("mL_fixed"), py::arg("nL_fixed"),
             "Compute residual at a fixed state without re-solving (for FD validation)")
        .def("get_tip_position", &CRMDynamicsWrapper::getTipPosition,
             "Get current tip position")
        .def_readwrite("dt", &CRMDynamicsWrapper::dt)
        .def_readwrite("integration_step_size", &CRMDynamicsWrapper::integrationStepSize)
        .def("is_initialized", &CRMDynamicsWrapper::isInitialized);
}
