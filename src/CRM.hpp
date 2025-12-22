#pragma once

#include <eigen3/Eigen/Dense>
#include <vector>
#include <tuple>
#include <string>

namespace CRMCatheterModel {

#define ANALYTICAL_SE3_STEP
#define NUM_STATES 15
#define NUM_ACT_SET 1

	constexpr double FCUM_DLAMBDA = 1.0;
#define INCREMENTALLY_APPLY_CURRENTS

	constexpr double IVALUE_SCALE_DU = 1.0;
	constexpr double IVALUE_SCALE_F = 1.0;
	constexpr double RESIDUAL_SCALE_M = 1.0;
	constexpr double RESIDUAL_SCALE_P = 1.0;

#define FK_TRUSTREGION
#define FK_TRUSTREGION_ANALYTICALJAC
#define CONTACT_TRUSTREGION
#define CONTACT_TRUSTREGION_ANALYTICALJAC
	constexpr double TRUSTREGION_TOLERANCE = 1e-5;

	constexpr double NUM_JACOBIAN_CURRENT_STEPSIZE = 1e-5;
	constexpr double NUM_JACOBIAN_INSERTIONLENGTH_STEPSIZE = 1e-2;
	constexpr double NUM_JACOBIAN_TIPFORCE_STEPSIZE = 1e-5;

	enum class CatheterSegmentType { FLEXIBLE, RIGID_WITH_ACTUATOR, RIGID };
	enum class ContactModeType { FREE_TIP, FIXED_TIP };
    enum class FKFreeJacobianType { ACTUATION_ONLY, ACTUATION_AND_TIP_FORCE };

	class CRMCatheterModelParams {
	public:
		CRMCatheterModelParams(int32_t no_flex, int32_t no_rigid, int32_t no_act, int32_t no_loc);
		CRMCatheterModelParams(const CRMCatheterModelParams& t);
		~CRMCatheterModelParams();

		int32_t no_flex_seg;
		int32_t no_rigid_seg;
		int32_t no_act_set;
		int32_t no_segments;
		int32_t no_locmarkers;

		std::vector<CatheterSegmentType> SegmentTypes;
		std::vector<double> SegLengths;
		std::vector<double> rho;
		std::vector<double> InnerRadius;
		std::vector<double> OuterRadius;
		std::vector<double> YoungsModulus;
		std::vector<double> ShearModulus;
        
        // Stiffness and Rest Shape
        std::vector<Eigen::Matrix3d> K;
        std::vector<Eigen::Matrix3d> Kinv;
		std::vector<Eigen::Vector3d> ustar;
        
		std::vector<double> ActMass;
		std::vector<Eigen::Vector2d> CoilAlignmentAngles;
		std::vector<Eigen::Matrix<double, 9, 1>> CoilTurnAreaMat;
		std::vector<double> LocMarkers;

        // Dynamics learnable parameters
        std::vector<Eigen::Matrix<double, 6, 1>> damping;
        std::vector<Eigen::Matrix3d> actInertia;

	protected:
		void allocate_memory();
		bool memory_allocated = false;
	};

	struct CatheterConfiguration {
		Eigen::Vector3d B0;
		Eigen::Vector3d g;
		Eigen::Vector3d p0;
		Eigen::Matrix<double, 9, 1> R0;
	};

	struct CRMForwardKinematicsData {
		CRMCatheterModelParams* CathParams;
		CatheterConfiguration* CathConfig;
		ContactModeType ContactMode;
		Eigen::Vector3d TipConstraintPoint;
		Eigen::Vector3d TipForce;
		Eigen::Vector3d deltau0_initialguess;
		Eigen::Vector3d ftip_initialguess;
		double IntegrationStepSize;
		bool FinalValueOnly;
		double (*ReportedMarkerPos)[3];
		double (*ReportedCoilOrient)[9];
		double (*ReportedCoilPos)[3];
	};

	CRMCatheterModelParams Load_CRMCatheterModelParams(const char* path_to_input);
	CatheterConfiguration Load_CatheterConfiguration(const char* path_to_input);

#define MAX(a,b) (((b)>(a))?(b):(a))
#define MIN(a,b) (((b)<(a))?(b):(a))
#define POW4(a) ((a)*(a)*(a)*(a))

	int CRM_ForwardKinematics(double in_x[], double out_y[], double& out_PotentialEnergy, CRMForwardKinematicsData Params);
	Eigen::VectorXd CRM_ForwardKinematics(const Eigen::VectorXd in_x, CRMForwardKinematicsData Params, double& out_PotentialEnergy, int& out_localmin);
	Eigen::VectorXd CRM_ForwardKinematics_FreeSpace(const Eigen::VectorXd in_x, CRMForwardKinematicsData Params, double& out_PotentialEnergy, int& out_localmin);
	Eigen::VectorXd CRM_ForwardKinematics_Contact(const Eigen::VectorXd in_x, CRMForwardKinematicsData Params, double& out_PotentialEnergy, int& out_localmin);

	Eigen::MatrixXd CRM_FKJacobian_Analytical(const Eigen::VectorXd in_x, const Eigen::VectorXd& in_FKouty, CRMForwardKinematicsData in_Params);
	Eigen::MatrixXd CRM_FKJacobian_Numerical(const Eigen::VectorXd in_x, Eigen::VectorXd& out_y, CRMForwardKinematicsData in_Params, int& localmin);
	Eigen::MatrixXd CRM_FKJacobian_FreeSpace(const Eigen::VectorXd in_x, Eigen::VectorXd& out_y, CRMForwardKinematicsData in_Params, FKFreeJacobianType mode, int& localmin);

	class CRM_Catheter {
	public:
		CRM_Catheter(CRMCatheterModelParams Params, CatheterConfiguration Config);
		~CRM_Catheter();

		std::tuple<Eigen::Vector3d, Eigen::Matrix3d, Eigen::Vector3d> ForwardKinematicsFree(const Eigen::VectorXd ActuationVector, const bool CalculateMarkers, double& out_PotentialEnergy, int& out_localmin);
		std::tuple<Eigen::Vector3d, Eigen::Matrix3d, Eigen::Vector3d> ForwardKinematicsFree(const Eigen::VectorXd ActuationVector, const Eigen::Vector3d TipForce, const Eigen::Vector3d deltau0_initialguess, bool CalculateMarkers, double& out_PotentialEnergy, int& out_localmin);
		std::tuple<Eigen::Vector3d, Eigen::Matrix3d, Eigen::Vector3d, Eigen::Vector3d> ForwardKinematicsContact(const Eigen::VectorXd ActuationVector, const Eigen::Vector3d TipConstraintPoint, const bool CalculateMarkers, double& out_PotentialEnergy, int& out_localmin);
		std::tuple<Eigen::Vector3d, Eigen::Matrix3d, Eigen::Vector3d, Eigen::Vector3d> ForwardKinematicsContact(const Eigen::VectorXd ActuationVector, const Eigen::Vector3d TipConstraintPoint, const Eigen::Vector3d deltau0_initialguess, const Eigen::Vector3d ftip_initialguess, const bool CalculateMarkers, double& out_PotentialEnergy, int& out_localmin);

		Eigen::MatrixXd FKJacobian_Analytical();
		void ResetInitialGuesses() { last_deltau0.setZero(); last_ftip.setZero(); }

		CRMCatheterModelParams CathParams;
		CatheterConfiguration CathConfig;
		double& IntegrationStepSize = FKParams.IntegrationStepSize;
        
        std::vector<Eigen::Vector3d> MarkerPosVec;
        std::vector<Eigen::Matrix<double, 9, 1>> CoilOrientVec;
        std::vector<Eigen::Vector3d> CoilPosVec;

	protected:
		Eigen::VectorXd last_ActuationVector;
		Eigen::VectorXd last_FKsolution;
		Eigen::Vector3d last_deltau0;
		Eigen::Vector3d last_ftip;
		CRMForwardKinematicsData FKParams;
	};
}