#pragma once
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include "CRM.hpp"
#include "CRM_BVPIVP_APIDeclarations.hpp"
#include "CRMDYN.hpp"

namespace CRMCatheterModel {
namespace dynnl_ad_eigen {

    using namespace autodiff;

    template <typename Scalar>
    Eigen::Matrix<Scalar, 6, 1> DYNNLEquationResidualEigenAD(
        const Eigen::Matrix<Scalar, -1, 1>& theta, 
        const DYNNLEqnParams* Params, 
        const Eigen::Vector3d& v, const Eigen::Vector3d& w, 
        const Eigen::Vector3d& p, const Eigen::Matrix3d& R, 
        const Eigen::Vector3d& mL, const Eigen::Vector3d& nL) 
    {
        // 1. Extract learnable parameters from theta
        Eigen::Matrix<Scalar, 3, 1> d_v = theta.segment(0, 3);
        Eigen::Matrix<Scalar, 3, 1> d_w = theta.segment(3, 3);
        Scalar act_mass = theta(12);
        if (std::abs(val(act_mass)) < 1e-9) act_mass = 1.0;
        Eigen::Matrix<Scalar, 3, 1> MagMom = theta.segment(13, 3);

        // 2. Map input state to Scalar types
        Eigen::Matrix<Scalar, 3, 1> grav = Params->g.cast<Scalar>();
        Eigen::Matrix<Scalar, 3, 3> R_mat = R.cast<Scalar>();
        Eigen::Matrix<Scalar, 3, 1> v_vec = v.cast<Scalar>();
        Eigen::Matrix<Scalar, 3, 1> w_vec = w.cast<Scalar>();
        Eigen::Matrix<Scalar, 3, 1> mL_vec = mL.cast<Scalar>();
        Eigen::Matrix<Scalar, 3, 1> nL_vec = nL.cast<Scalar>();

        // DEBUG PRINT (Only print for val types to avoid flooding with duals if possible, or just print once)
        // Since Scalar can be real (dual), we use val() to print the double value.
        // static bool first_run = true;
        // if (first_run) {
        //     std::cout << "[DEBUG AD] w_vec: " << val(w_vec(0)) << ", " << val(w_vec(1)) << ", " << val(w_vec(2)) << std::endl;
        //     std::cout << "[DEBUG AD] d_w: " << val(d_w(0)) << ", " << val(d_w(1)) << ", " << val(d_w(2)) << std::endl;
        //     first_run = false;
        // }

        // 3. Linear Acceleration Residual (Newton)
        Eigen::Matrix<Scalar, 3, 1> RTg = R_mat.transpose() * grav;
        Eigen::Matrix<Scalar, 3, 1> damping_v = d_v.cwiseProduct(v_vec);
        Eigen::Matrix<Scalar, 3, 1> v_dot = RTg - (nL_vec / act_mass) - w_vec.cross(v_vec) - damping_v;

        // 4. Angular Acceleration Residual (Euler)
        Eigen::Matrix<Scalar, 3, 1> B0_local = R_mat.transpose() * Params->B0.cast<Scalar>();
        Eigen::Matrix<Scalar, 3, 1> tau_mag = MagMom.cross(B0_local);
        Eigen::Matrix<Scalar, 3, 1> damping_w = d_w.cwiseProduct(w_vec);
        
        Eigen::Matrix<Scalar, 3, 1> w_dot = tau_mag - mL_vec - damping_w;

        Eigen::Matrix<Scalar, 6, 1> res;
        res.segment(0, 3) = v_dot;
        res.segment(3, 3) = w_dot;
        return res;
    }

    Eigen::MatrixXd compute_parameter_jacobian_eigen(
        double currents[3], double insertion, 
        double v[3], double w[3], double p[3], double R[9], double xf[15], 
        double mL[3], double nL[3], 
        DYNNLEqnParams* Params, Eigen::VectorXd& out_theta, Eigen::VectorXd& out_res) 
    {
        // USE MAPS TO CORRECTLY READ POINTERS
        Eigen::Map<const Eigen::Vector3d> v_in(v);
        Eigen::Map<const Eigen::Vector3d> w_in(w);
        Eigen::Map<const Eigen::Vector3d> p_in(p);
        Eigen::Map<const Eigen::Vector3d> mL_in(mL);
        Eigen::Map<const Eigen::Vector3d> nL_in(nL);
        Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> R_in(R);

        VectorXreal theta(16);
        for(int i=0; i<6; i++) theta(i) = Params->damping[0](i);
        for(int i=0; i<3; i++) theta(6+i) = Params->K[0](i,i);
        for(int i=0; i<3; i++) theta(9+i) = Params->ustar[0](i);
        theta(12) = Params->ActMass[0];
        for(int i=0; i<3; i++) theta(13+i) = Params->MagMoment[0](i);

        out_theta = theta.cast<double>();
        
        // Capture VALUES (deep copy) to ensure lambda owns the data during differentiation
        Eigen::Vector3d v_val = v_in;
        Eigen::Vector3d w_val = w_in;
        Eigen::Vector3d p_val = p_in;
        Eigen::Vector3d mL_val = mL_in;
        Eigen::Vector3d nL_val = nL_in;
        Eigen::Matrix3d R_val = R_in;

        auto f = [&](const VectorXreal& t) {
            return DYNNLEquationResidualEigenAD<real>(t, Params, v_val, w_val, p_val, R_val, mL_val, nL_val);
        };

        Eigen::MatrixXd J;
        VectorXreal F;
        jacobian(f, wrt(theta), at(theta), F, J);
        out_res = F.cast<double>();
        return J;
    }
}
}
