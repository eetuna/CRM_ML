#pragma once
#include <tuple>
#include <eigen3/Eigen/Dense>

namespace CRMCatheterModel {

	std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> CRMSolverIVPJacobian(
		CRMShootingMethodParams in_Params,
		double in_deltau0[3], double in_ftip[3],
		bool in_FinalValueOnly,
		double out_x_N[NUM_STATES], double out_MomentResidual[3]);

	template <typename IVPJacobians>
	void CRMSolverIVP_CoreWithJacobian(CRMIVPCoreParams& in_params,
		double in_deltau[3], double in_ftip[3],
		AugmentedStateVector<IVPJacobians>& out_x_N, double out_MomentResidual[3]);

}