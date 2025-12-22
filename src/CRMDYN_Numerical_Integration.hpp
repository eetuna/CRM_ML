#pragma once
#include <eigen3/Eigen/Dense>
#include "CRM_BVPIVP_APIDeclarations.hpp"

namespace CRMCatheterModel {

	void CRMIntegrand_dyn(double s, const StateVector& in_x, const CRMIntegrandParams in_Params, const double* in_tau, StateDerivativeVector& out_xdot) {
		Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> R(in_x._R);
		Eigen::Vector3d u(in_x._u[0], in_x._u[1], in_x._u[2]);

		double lambda = in_Params.Li - s;
		double ix = lambda * in_Params.dlambdainv;
		int ird = (int)std::floor(ix);
		int iru = (int)std::ceil(ix);
		double frac = ix - ird;

		Eigen::Vector3d fcum = (*in_Params.fcumlambda)[iru] * frac + (*in_Params.fcumlambda)[ird] * (1.0 - frac);
		fcum += *in_Params.ftip;

		Eigen::Vector3d RTf = R.transpose() * fcum;
		Eigen::Vector3d e3hatRTf(-RTf(1), RTf(0), 0.0);

		Eigen::Vector3d du = u - *in_Params.ustar;
		Eigen::Vector3d udot = -(*in_Params.Kinv) * (u.cross((*in_Params.K) * du) + e3hatRTf);

		for (int i = 0; i < 3; i++) out_xdot._u[i] = udot(i);
	}

	void RK2_coildyn(double x[6], double n[3], double g[3], double mass, double inertia[9], double damping[6], double B0[3], double mu[9], double mL[3], double x_new[6], double xdot[6]) {}
	void ABM4_coildyn(double x[6], double dx1[6], double dx2[6], double dx3[6], double x1[6], double x2[6], double x3[6], double n[3], double g[3], double mass, double inertia[9], double damping[6], double B0[3], double mu[9], double mL[3], double x_new[6], double xdot[6]) {}
}