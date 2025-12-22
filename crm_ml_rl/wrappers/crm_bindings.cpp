#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <string>
#include "CRM.hpp"
#include "CRM_BVPIVP_APIDeclarations.hpp"
#include "CRM_IVPJacobian.hpp"
#include "CRMDYN.hpp"
#include "CRMDYN_DYNNLEquationResidual_autodiff_eigen.hpp"

namespace py = pybind11;
using namespace CRMCatheterModel;

template<int N>
void to_fixed(py::array_t<double> arr, double (&out)[N]) {
    auto r = arr.unchecked();
    if (r.size() < N) throw std::runtime_error("Input array too small.");
    int idx = 0;
    if (arr.ndim() == 1) {
        for(int i=0; i<N; i++) out[i] = r(i);
    } else {
        auto r2 = arr.unchecked<2>();
        for(int i=0; i<r2.shape(0); i++) for(int j=0; j<r2.shape(1); j++) if(idx<N) out[idx++] = r2(i,j);
    }
}

class CRMDynamicsWrapper {
public:
    CRMCatheterModelParams* params;
    CatheterConfiguration config;
    double dt = 0.01;
    double integration_step_size = 0.1;
    
    Eigen::Vector3d v_seed, w_seed, p_seed, mL_seed, nL_seed;
    Eigen::Matrix3d R_seed;
    Eigen::VectorXd xf_seed;

    CRMDynamicsWrapper(int nf, int nr, int na, int nl) {
        params = new CRMCatheterModelParams(nf, nr, na, nl);
        v_seed.setZero(); w_seed.setZero(); p_seed.setZero();
        mL_seed.setZero(); nL_seed.setZero(); R_seed.setIdentity();
        xf_seed.resize(15); xf_seed.setZero();
    }
    ~CRMDynamicsWrapper() { delete params; }

    bool load_parameters(const std::string& model_path, const std::string& config_path) {
        try {
            *params = Load_CRMCatheterModelParams(model_path.c_str());
            config = Load_CatheterConfiguration(config_path.c_str());
            return true;
        } catch(...) { return false; }
    }

    void set_damping(py::array_t<double> d) {
        auto r = d.unchecked<1>();
        for(int i=0; i<6; i++) params->damping[0](i) = r(i);
    }

    void initialize_from_kinematics(py::array_t<double> currents, double insertion) {
        double currents_raw[NUM_ACT_SET][3]; to_fixed<3>(currents, currents_raw[0]);
        double out_y[18], pe;
        int localmin;
        CRMForwardKinematicsData fk_data;
        fk_data.CathParams = params;
        fk_data.CathConfig = &config;
        fk_data.ContactMode = ContactModeType::FREE_TIP;
        fk_data.IntegrationStepSize = integration_step_size;
        fk_data.FinalValueOnly = true;

        Eigen::VectorXd in_x(params->no_act_set * 3 + 1);
        for(int i=0; i<params->no_act_set*3; i++) in_x(i) = currents_raw[0][i];
        in_x(params->no_act_set * 3) = insertion;

        localmin = CRM_ForwardKinematics(in_x.data(), out_y, pe, fk_data);
        
        // Sync internal seed
        p_seed = Eigen::Vector3d(out_y[0], out_y[1], out_y[2]);
        for(int i=0; i<9; i++) R_seed(i/3, i%3) = out_y[3+i];
        for(int i=0; i<15; i++) xf_seed(i) = out_y[i];
        v_seed.setZero(); w_seed.setZero(); mL_seed.setZero(); nL_seed.setZero();
    }

    py::dict get_seed_state() {
        py::dict d;
        d["v"] = std::vector<double>{v_seed(0), v_seed(1), v_seed(2)};
        d["w"] = std::vector<double>{w_seed(0), w_seed(1), w_seed(2)};
        d["p"] = std::vector<double>{p_seed(0), p_seed(1), p_seed(2)};
        std::vector<double> R_flat(9);
        for(int i=0; i<9; i++) R_flat[i] = R_seed(i/3, i%3);
        d["R"] = R_flat;
        d["mL"] = std::vector<double>{mL_seed(0), mL_seed(1), mL_seed(2)};
        d["nL"] = std::vector<double>{nL_seed(0), nL_seed(1), nL_seed(2)};
        std::vector<double> xf_flat(15);
        for(int i=0; i<15; i++) xf_flat[i] = xf_seed(i);
        d["xf"] = xf_flat;
        return d;
    }

    py::dict compute_parameter_jacobian(py::array_t<double> currents, double insertion,
                                    py::array_t<double> v, py::array_t<double> w,
                                    py::array_t<double> p, py::array_t<double> R,
                                    py::array_t<double> xf, py::array_t<double> mL, py::array_t<double> nL) {
        
        double currents_raw[3]; to_fixed<3>(currents, currents_raw);
        double v_raw[3], w_raw[3], p_raw[3], R_raw[9], xf_raw[15], mL_raw[3], nL_raw[3];
        to_fixed<3>(v, v_raw); to_fixed<3>(w, w_raw); to_fixed<3>(p, p_raw); to_fixed<9>(R, R_raw);
        to_fixed<15>(xf, xf_raw); to_fixed<3>(mL, mL_raw); to_fixed<3>(nL, nL_raw);

        DYNNLEqnParams ad_params(params->no_flex_seg, params->no_rigid_seg, params->no_act_set, params->no_locmarkers, 50);
        ad_params.DELTA_T = dt;
        ad_params.B0 = config.B0;
        ad_params.g = config.g;
        ad_params.damping[0] = params->damping[0];
        ad_params.K[0] = params->K[0];
        ad_params.ustar[0] = params->ustar[0];
        ad_params.ActMass[0] = params->ActMass[0];

        Eigen::VectorXd theta, res_vec;
        Eigen::MatrixXd J = dynnl_ad_eigen::compute_parameter_jacobian_eigen(currents_raw, insertion, v_raw, w_raw, p_raw, R_raw, xf_raw, mL_raw, nL_raw, &ad_params, theta, res_vec);

        py::dict result;
        result["J_theta"] = J;
        result["theta"] = theta;
        result["residual"] = res_vec;
        result["converged"] = true;
        result["param_names"] = std::vector<std::string>{
            "damping_v0", "damping_v1", "damping_v2", "damping_w0", "damping_w1", "damping_w2",
            "K_diag_0", "K_diag_1", "K_diag_2", "ustar_0", "ustar_1", "ustar_2",
            "actMass", "MagMoment_0", "MagMoment_1", "MagMoment_2"
        };

        py::dict base;
        base["next_mL"] = mL;
        base["next_nL"] = nL;
        base["v"] = v;
        base["w"] = w;
        base["p"] = p;
        base["R"] = R;
        base["xf"] = xf;
        result["base"] = base;

        return result;
    }

    py::dict compute_residual_at_state(py::array_t<double> currents, double insertion,
                                                 py::array_t<double> v, py::array_t<double> w,
                                                 py::array_t<double> p, py::array_t<double> R,
                                                 py::array_t<double> xf, py::array_t<double> mL, py::array_t<double> nL) {
        
        double currents_raw[3]; to_fixed<3>(currents, currents_raw);
        double v_raw[3], w_raw[3], p_raw[3], R_raw[9], mL_raw[3], nL_raw[3];
        to_fixed<3>(v, v_raw); to_fixed<3>(w, w_raw); to_fixed<3>(p, p_raw); to_fixed<9>(R, R_raw);
        to_fixed<3>(mL, mL_raw); to_fixed<3>(nL, nL_raw);

        DYNNLEqnParams ad_params(params->no_flex_seg, params->no_rigid_seg, params->no_act_set, params->no_locmarkers, 50);
        ad_params.B0 = config.B0;
        ad_params.g = config.g;
        ad_params.damping[0] = params->damping[0];
        ad_params.ActMass[0] = params->ActMass[0];

        Eigen::Matrix<autodiff::real, -1, 1> theta(16);
        for(int i=0; i<6; i++) theta(i) = params->damping[0](i);
        for(int i=0; i<3; i++) theta(6+i) = params->K[0](i,i);
        for(int i=0; i<3; i++) theta(9+i) = params->ustar[0](i);
        theta(12) = params->ActMass[0];
        for(int i=0; i<3; i++) theta(13+i) = 0.0;

        auto res = dynnl_ad_eigen::DYNNLEquationResidualEigenAD<autodiff::real>(
            theta, &ad_params, Eigen::Vector3d(v_raw), Eigen::Vector3d(w_raw), 
            Eigen::Vector3d(p_raw), Eigen::Map<const Eigen::Matrix3d>(R_raw), 
            Eigen::Vector3d(mL_raw), Eigen::Vector3d(nL_raw)
        );

        py::dict result;
        result["residual"] = res.cast<double>();
        return result;
    }
};

PYBIND11_MODULE(crm_python, m) {
    py::class_<CRMDynamicsWrapper>(m, "CRMDynamics")
        .def(py::init<int, int, int, int>())
        .def("load_parameters", &CRMDynamicsWrapper::load_parameters)
        .def("set_damping", &CRMDynamicsWrapper::set_damping)
        .def("initialize_from_kinematics", &CRMDynamicsWrapper::initialize_from_kinematics)
        .def("get_seed_state", &CRMDynamicsWrapper::get_seed_state)
        .def("compute_parameter_jacobian", &CRMDynamicsWrapper::compute_parameter_jacobian)
        .def("compute_residual_at_state", &CRMDynamicsWrapper::compute_residual_at_state)
        .def_readwrite("dt", &CRMDynamicsWrapper::dt)
        .def_readwrite("integration_step_size", &CRMDynamicsWrapper::integration_step_size);
}