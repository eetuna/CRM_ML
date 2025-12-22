#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "CRM.hpp"
#include "CRM_MatrixOperations.hpp"

namespace CRMCatheterModel {

    void wHat(const double in_w[3], double out_what[9]) {
        out_what[0] = 0;          out_what[1] = -in_w[2];   out_what[2] = in_w[1];
        out_what[3] = in_w[2];    out_what[4] = 0;          out_what[5] = -in_w[0];
        out_what[6] = -in_w[1];   out_what[7] = in_w[0];    out_what[8] = 0;
    }

    void SE3_Analytical_Step(double in_R_n[9], double in_p_n[3], double in_u_n[3], double h, double out_R_np1[9], double out_p_np1[3]) {
        Eigen::Map<const Eigen::Matrix3d> R_n(in_R_n);
        Eigen::Map<const Eigen::Vector3d> p_n(in_p_n);
        Eigen::Map<const Eigen::Vector3d> u_n(in_u_n);

        double theta = u_n.norm() * h;
        Eigen::Matrix3d R_step;
        if (theta < 1e-9) {
            R_step = Eigen::Matrix3d::Identity();
        } else {
            Eigen::Vector3d w = u_n.normalized();
            double c = std::cos(theta);
            double s = std::sin(theta);
            Eigen::Matrix3d what;
            double what_raw[9];
            wHat(w.data(), what_raw);
            what = Eigen::Map<Eigen::Matrix3d>(what_raw).transpose();
            R_step = Eigen::Matrix3d::Identity() + s * what + (1 - c) * (what * what);
        }

        Eigen::Map<Eigen::Matrix3d> R_np1(out_R_np1);
        Eigen::Map<Eigen::Vector3d> p_np1(out_p_np1);

        R_np1 = R_n * R_step;
        p_np1 = p_n + h * (R_n * Eigen::Vector3d(0, 0, 1)); // Simplified translation
    }

	CRMCatheterModelParams Load_CRMCatheterModelParams(const char* path_to_input) {
		CRMCatheterModelParams Params(1, 0, 1, 5); 
        Params.K[0] = Eigen::Matrix3d::Identity() * 100.0;
        Params.Kinv[0] = Params.K[0].inverse();
        Params.ustar[0].setZero();
        Params.damping[0].setZero();
		return Params;
	}

	CatheterConfiguration Load_CatheterConfiguration(const char* path_to_input) {
		CatheterConfiguration config;
		config.B0.setZero();
		config.g.setZero();
		config.p0.setZero();
		config.R0.setIdentity();
		return config;
	}
}
