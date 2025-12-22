#pragma once
#include "CRM.hpp"

namespace CRMCatheterModel {

	Eigen::VectorXd CRM_ForwardKinematics_FreeSpace(const Eigen::VectorXd in_x, CRMForwardKinematicsData Params, double& out_PotentialEnergy, int& out_localmin);
	Eigen::VectorXd CRM_ForwardKinematics_Contact(const Eigen::VectorXd in_x, CRMForwardKinematicsData Params, double& out_PotentialEnergy, int& out_localmin);

	struct CRMContactEquationParams : CRMForwardKinematicsData {
		double Actuation[NUM_ACT_SET * 3 + 1];
	};

	void CRM_Contact_Equation(double in_x[], double out_y[], CRMContactEquationParams Params);
	void CRM_Contact_Equation_AnalyticalJac(double in_x[], double out_y[], double out_fjac[], CRMContactEquationParams Params);

	Eigen::MatrixXd CRM_FKJacobian_FreeSpace(
		const Eigen::VectorXd in_x,
		Eigen::VectorXd& out_y,
		CRMForwardKinematicsData in_Params,
		FKFreeJacobianType mode,
		int& localmin);
}