#include <eigen3/Eigen/Dense>
#include <tuple>
#include <vector>
#include "CRM.hpp"

namespace CRMCatheterModel {

	CRM_Catheter::CRM_Catheter(CRMCatheterModelParams Params, CatheterConfiguration Config) : CathParams(Params), CathConfig(Config) {
		FKParams.CathParams = &CathParams;
		FKParams.CathConfig = &CathConfig;
		FKParams.TipForce.setZero();
		FKParams.deltau0_initialguess.setZero();
		FKParams.ftip_initialguess.setZero();
		FKParams.FinalValueOnly = true;
		
		MarkerPosVec.resize(CathParams.no_locmarkers);
		FKParams.ReportedMarkerPos = (double(*)[3])MarkerPosVec.data();
		
		CoilOrientVec.resize(CathParams.no_act_set);
		FKParams.ReportedCoilOrient = (double(*)[9])CoilOrientVec.data();
        
        CoilPosVec.resize(CathParams.no_act_set);
        FKParams.ReportedCoilPos = (double(*)[3])CoilPosVec.data();

		last_deltau0.setZero();
		last_ftip.setZero();
	}

	CRM_Catheter::~CRM_Catheter() {}

	std::tuple<Eigen::Vector3d, Eigen::Matrix3d, Eigen::Vector3d> CRM_Catheter::ForwardKinematicsFree(const Eigen::VectorXd ActuationVector, const Eigen::Vector3d TipForce,
		const Eigen::Vector3d deltau0_initialguess, bool CalculateMarkers, double& out_PotentialEnergy, int& out_localmin) {

		FKParams.ContactMode = ContactModeType::FREE_TIP;
		FKParams.FinalValueOnly = !CalculateMarkers;
		FKParams.TipForce = TipForce;
		FKParams.deltau0_initialguess = deltau0_initialguess;
		
		last_ActuationVector = ActuationVector;
		last_FKsolution = CRM_ForwardKinematics(ActuationVector, FKParams, out_PotentialEnergy, out_localmin);
		
		Eigen::Vector3d ptip = last_FKsolution.segment<3>(0);
		Eigen::Matrix3d Rtip;
		for(int i=0; i<9; i++) Rtip(i/3, i%3) = last_FKsolution(3+i);
		Eigen::Vector3d deltau0 = last_FKsolution.segment<3>(12);
		
		if (out_localmin == 0) {
			last_deltau0 = deltau0;
			last_ftip = TipForce;
		}
		return { ptip, Rtip, deltau0 };
	}

	std::tuple<Eigen::Vector3d, Eigen::Matrix3d, Eigen::Vector3d> CRM_Catheter::ForwardKinematicsFree(const Eigen::VectorXd ActuationVector, const bool CalculateMarkers, double& out_PotentialEnergy, int& out_localmin) {
		return ForwardKinematicsFree(ActuationVector, Eigen::Vector3d::Zero(), last_deltau0, CalculateMarkers, out_PotentialEnergy, out_localmin);
	}

	std::tuple<Eigen::Vector3d, Eigen::Matrix3d, Eigen::Vector3d, Eigen::Vector3d> CRM_Catheter::ForwardKinematicsContact(const Eigen::VectorXd ActuationVector, const Eigen::Vector3d TipConstraintPoint,
		const Eigen::Vector3d u0_initialguess, const Eigen::Vector3d ftip_initialguess, const bool CalculateMarkers, double& out_PotentialEnergy, int& out_localmin) {

		FKParams.ContactMode = ContactModeType::FIXED_TIP;
		FKParams.FinalValueOnly = !CalculateMarkers;
		FKParams.TipConstraintPoint = TipConstraintPoint;
		FKParams.deltau0_initialguess = u0_initialguess;
		FKParams.ftip_initialguess = ftip_initialguess;
		
		last_ActuationVector = ActuationVector;
		last_FKsolution = CRM_ForwardKinematics(ActuationVector, FKParams, out_PotentialEnergy, out_localmin);
		
		Eigen::Vector3d ptip = last_FKsolution.segment<3>(0);
		Eigen::Matrix3d Rtip;
		for(int i=0; i<9; i++) Rtip(i/3, i%3) = last_FKsolution(3+i);
		Eigen::Vector3d deltau0 = last_FKsolution.segment<3>(12);
		Eigen::Vector3d ftip = last_FKsolution.segment<3>(15);
		
		if (out_localmin == 0) {
			last_deltau0 = deltau0;
			last_ftip = ftip;
		}
		return { ptip, Rtip, deltau0, ftip };
	}

	std::tuple<Eigen::Vector3d, Eigen::Matrix3d, Eigen::Vector3d, Eigen::Vector3d> CRM_Catheter::ForwardKinematicsContact(const Eigen::VectorXd ActuationVector, const Eigen::Vector3d TipConstraintPoint,
		const bool CalculateMarkers, double& out_PotentialEnergy, int& out_localmin) {
		return ForwardKinematicsContact(ActuationVector, TipConstraintPoint, last_deltau0, last_ftip, CalculateMarkers, out_PotentialEnergy, out_localmin);
	}

	Eigen::MatrixXd CRM_Catheter::FKJacobian_Analytical() {
		return CRM_FKJacobian_Analytical(last_ActuationVector, last_FKsolution, FKParams);
	}
}
