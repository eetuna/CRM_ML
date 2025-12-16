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
            throw std::runtime_error("Parameters not loaded.");
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
            throw std::runtime_error("Parameters not loaded.");
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

    py::dict get_seed_state() const {
        const int num_sets = catheter.getParams() ? catheter.getParams()->no_act_set : NUM_ACT_SET;

        py::array_t<double> v_out({num_sets, 3});
        py::array_t<double> w_out({num_sets, 3});
        py::array_t<double> p_out({num_sets, 3});
        py::array_t<double> R_out({num_sets, 9});
        py::array_t<double> xf_out({NUM_STATES});
        py::array_t<double> mL_out({num_sets, 3});
        py::array_t<double> nL_out({num_sets, 3});

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
            throw std::runtime_error("Parameters not loaded.");
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
            throw std::runtime_error("Parameters not loaded.");
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

        // Update state only if converged to avoid corrupting internal state with garbage
        if (localmin == 0) {
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
        result["converged"] = (localmin == 0);
        result["localmin"] = localmin;

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
            throw std::runtime_error("Parameters not loaded.");
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

        if (mL_in.size() > 0) {
            auto mbuf = mL_in.request(); const double* mptr = static_cast<double*>(mbuf.ptr);
            for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
                for (int i = 0; i < 3; i++) {
                    const ssize_t idx = j * 3 + i;
                    if (idx < mbuf.size) mL_guess_local[j][i] = mptr[idx];
                }
            }
        }

        if (nL_in.size() > 0) {
            auto nbuf = nL_in.request(); const double* nptr = static_cast<double*>(nbuf.ptr);
            for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
                for (int i = 0; i < 3; i++) {
                    const ssize_t idx = j * 3 + i;
                    if (idx < nbuf.size) nL_guess_local[j][i] = nptr[idx];
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

        // Setup shooting method parameters
        ContactModeType ContactMode = ContactModeType::FREE_TIP;
        double TipForce[3] = {0.0, 0.0, 0.0};
        double TipConstraintPoint[3] = {0.0, 0.0, 0.0};

        CRMShootingMethodParams BVPParams = CRMDYNConstructShootingMethodParamSet(
            *catheter.getParams(), catheter.config, insertion_length, ActuationCurrents,
            ContactMode, TipConstraintPoint, TipForce, integrationStepSize,
            actInertia_local, v_L_local, w_L_local, p_L_local, R_L_local, damping_local, dt_local
        );

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

        if (localmin == 0) {
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

        for (int i = 0; i < NUM_STATES; i++) xfout(i) = (localmin == 0) ? xf_new[i] : xf_local[i];

        for (int j = 0; j < num_sets && j < NUM_ACT_SET; j++) {
            for (int i = 0; i < 3; i++) {
                if (localmin == 0) {
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
                if (localmin == 0) {
                    Rout(j, i) = x_coil[j][i + 9];
                } else {
                    Rout(j, i) = R_L_local[j][i];
                }
            }
        }

        py::dict result;
        result["tip_position"] = tip_pos;
        result["tip_velocity"] = tip_vel;
        result["converged"] = (localmin == 0);
        result["localmin"] = localmin;
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
        .def("get_tip_position", &CRMDynamicsWrapper::getTipPosition,
             "Get current tip position")
        .def_readwrite("dt", &CRMDynamicsWrapper::dt)
        .def_readwrite("integration_step_size", &CRMDynamicsWrapper::integrationStepSize)
        .def("is_initialized", &CRMDynamicsWrapper::isInitialized);
}
