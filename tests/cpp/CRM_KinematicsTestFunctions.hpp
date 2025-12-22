#pragma once
#include "CRM.hpp"
#include "CRM_BVPIVP_APIDeclarations.hpp"

namespace CRMCatheterModel {

	Eigen::MatrixXd CRM_FKJacobian_BruteForce(
		const Eigen::VectorXd in_x,
		Eigen::VectorXd& out_y,
		CRMForwardKinematicsData in_Params,
		int& localmin);

	Eigen::MatrixXd CRM_FKJacobian_From_IVP_Numerical(
		const Eigen::VectorXd in_x,
		const Eigen::VectorXd& in_FKouty,
		CRMForwardKinematicsData in_Params);

	Eigen::VectorXd CRMSolverIVPWrapper(
		Eigen::VectorXd in,
		CRMShootingMethodParams in_Params);

}