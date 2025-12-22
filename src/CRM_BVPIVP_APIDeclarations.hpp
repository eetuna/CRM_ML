#pragma once
#include <cmath>
#include <vector>
#include <memory>
#include <tuple>
#include <eigen3/Eigen/Dense>
#include "CRM_MatrixOperations.hpp"
#include "CRM_StateVector_Definitions.hpp"

namespace CRMCatheterModel {

	template <typename T> struct StDerivativeVect_trait { typedef void type; };
	template<> struct StDerivativeVect_trait<StateVector> { typedef StateDerivativeVector type; };
	template<> struct StDerivativeVect_trait<AugmentedStateVector<IVPJacobiansMini>> { typedef AugmentedStateDerivativeVector<IVPJacobiansMini> type; };
	template<> struct StDerivativeVect_trait<AugmentedStateVector<IVPJacobiansFull>> { typedef AugmentedStateDerivativeVector<IVPJacobiansFull> type; };
	template<typename T> using StDerivativeVectType = typename StDerivativeVect_trait<T>::type;

	template <class T> using expr_type = std::remove_cv_t<std::remove_reference_t<T>>;

	class CRMShootingMethodParams {
	public:
		CRMShootingMethodParams(int32_t no_flex, int32_t no_rigid, int32_t no_act, int32_t no_loc, int32_t no_fcum);
		~CRMShootingMethodParams() = default;

		int32_t no_flex_seg;
		int32_t no_rigid_seg;
		int32_t no_act_set;
		int32_t no_segments;
		int32_t no_locmarkers;
		int32_t no_fcum_steps;

		double	IntegrationStepSize;
		double 	dlambdainv;

		Eigen::Vector3d B0;
		Eigen::Vector3d g;
		Eigen::Matrix3d R0; 
		Eigen::Vector3d p0;

		ContactModeType ContactMode;
		Eigen::Vector3d TipConstraintPoint;
		Eigen::Vector3d TipForce;

		double 	Li;

		std::vector<CatheterSegmentType> SegmentTypes;
		std::vector<double> SegEndLambdas;
		std::vector<double> rho;
		std::vector<Eigen::Matrix3d> K;
		std::vector<Eigen::Matrix3d> Kinv;
		std::vector<Eigen::Vector3d> ustar;
		std::vector<double> ActMass;
		std::vector<Eigen::Vector3d> MagMoment;
		std::vector<Eigen::Matrix3d> CoilAlignmentTurnAreaMatrix;
		std::vector<double> LocMarkerLambdas;
		std::vector<Eigen::Vector3d> fcumlambda;

        std::vector<Eigen::Vector3d> v_L_pre;
        std::vector<Eigen::Vector3d> w_L_pre;
        std::vector<Eigen::Vector3d> p_pre;
        std::vector<Eigen::Matrix3d> R_pre;
        std::vector<Eigen::Matrix3d> actInertia;
        std::vector<Eigen::Matrix<double, 6, 1>> damping;
        double DELTA_T;
	};

	CRMShootingMethodParams CRMConstructShootingMethodParamSet(
		CRMCatheterModelParams CathParams, CatheterConfiguration CathConfig,
		double InsertionLength, double ActuationCurrents[NUM_ACT_SET][3],
		ContactModeType ContactMode,
		double TipConstraintPoint[3], double TipForce[3],
		double IntegrationStepSize);

	void CRMUpdateShootingMethodParamSetWithNewActuationCurrents(
		CRMShootingMethodParams& ShootingParams,
		double ActuationCurrents[NUM_ACT_SET][3]);

    void CRMSolverIVP(CRMShootingMethodParams in_Params,
                      double in_deltau0[3], double in_ftip[3],
                      bool in_CalculateEnergy,
                      bool in_FinalValueOnly,
                      double out_x_N[NUM_STATES], double out_MomentResidual[3],
                      double& out_PotentialEnergy,
                      double out_p_atLocMarkers[][3],
                      double out_R_atActuators[][9] , double out_p_atActuators[][3] ) ;

    void CRMSolverIVP(
            int32_t in_no_flex_seg,
            int32_t in_no_rigid_seg,
            int32_t in_no_act_set,
            int32_t in_no_locmarkers,
            int32_t in_no_fcum_steps,
            double in_x_0[NUM_STATES], double in_IntegrationStepSize,
            double in_Li, double in_dlambdainv,
            CatheterSegmentType in_SegmentTypes[],
            double in_SegEndLambdas[], double in_LocMarkerLambdas[],
            double in_rho[],
            double in_K[][9], double in_Kinv[][9], double in_ustar[][3],
            double in_ActMass[],
            double in_CoilAlignmentTurnAreaMatrix[][9],
            double in_MagMoment[][3], double in_fcumlambda[][3], double in_ftip[3],
            double in_B0[3], double in_g[3],
            bool in_CalculateEnergy,
            bool in_FinalValueOnly,
            double out_x_N[NUM_STATES], double out_MomentResidual[3],
            double& out_PotentialEnergy,
            double out_p_atLocMarkers[][3],
            double out_R_atActuators[][9], double out_p_atActuators[][3]
    ) ;

	class CRMIVPCoreParams {
	public:
		CRMIVPCoreParams(int32_t no_flex, int32_t no_rigid, int32_t no_act, int32_t no_loc, int32_t no_fcums);
		~CRMIVPCoreParams() = default;

		int32_t no_flex_seg;
		int32_t no_rigid_seg;
		int32_t no_act_set;
		int32_t no_segments;
		int32_t no_locmarkers;
		int32_t no_fcum_steps;

		Eigen::Matrix<double, NUM_STATES, 1> xi; 
		double InsertedLength;
		double dlambdainv; 
		Eigen::Vector3d B0;
		Eigen::Vector3d g;

		std::vector<CatheterSegmentType> SegmentTypes;
		std::vector<int32_t> FlexActIndex;
		int32_t StartSegmentIndex;
		std::vector<double> SegBounds;
		std::vector<int32_t> SegSteps;
		std::vector<double> rho;
		std::vector<Eigen::Matrix3d> K;
		std::vector<Eigen::Matrix3d> Kinv;
		std::vector<Eigen::Vector3d> ustar;
		std::vector<double> ActMass;
		std::vector<Eigen::Vector3d> MagMoment;
		std::vector<Eigen::Matrix3d> CoilAlignmentTurnAreaMatrix;
		
		std::vector<Eigen::Matrix3d> R_atActuators;
        std::vector<Eigen::Vector3d> p_atActuators;
        bool   CalculateEnergy;
		bool   FinalValueOnly;
		int	   NextLocMarker;
		std::vector<double> LocMarkers;
		std::vector<Eigen::Vector3d> p_atLocMarkers;
		std::vector<Eigen::Vector3d> fcumlambda;

        // Dynamics state
        std::vector<Eigen::Vector3d> v_L_pre;
        std::vector<Eigen::Vector3d> w_L_pre;
        std::vector<Eigen::Vector3d> p_pre;
        std::vector<Eigen::Matrix3d> R_pre;
        std::vector<Eigen::Matrix3d> actInertia;
        std::vector<Eigen::Matrix<double, 6, 1>> damping;
        double DELTA_T;
        std::vector<Eigen::Vector3d> m_L;
        std::vector<Eigen::Vector3d> n_L;
	};

	void CRMSolverIVP_Prep(
		int32_t in_no_flex_seg,
		int32_t in_no_rigid_seg,
		int32_t in_no_act_set,
		int32_t in_no_locmarkers,
		int32_t in_no_fcum_steps,
		double in_x_0[NUM_STATES], double in_IntegrationStepSize,
		double in_Li, double in_dlambdainv,
		const CatheterSegmentType in_SegmentTypes[],
		double in_SegEndLambdas[], double in_LocMarkerLambdas[],
		double in_rho[],
		double in_K[][9], double in_Kinv[][9], double in_ustar[][3],
		double in_ActMass[],
		double in_CoilAlignmentTurnAreaMatrix[][9],
		double in_MagMoment[][3], double in_fcumlambda[][3],
		double in_B0[3], double in_g[3],
		bool in_CalculateEnergy,
		bool in_FinalValueOnly, 
		CRMIVPCoreParams& out_CoreParams);

    void CRMSolverIVP_Core(CRMIVPCoreParams& in_params,
                           double in_deltau[3], double in_ftip[3],
                           StateVector& out_x_N, double out_MomentResidual[3],
                           double &out_PotentialEnergy,
                           double out_p_atLocMarkers[][3],
                           double out_R_atActuators[][9], double out_p_atActuators[][3]) ;

	void CRMSolverIVP_PropagateBCThroughRigidLink(StateVector& xi_ip1, double Residual_ip1[3],
		const double RigidSegmentLength, const double MagMoment[3], const double CoilAlignmentTurnAreaMatrix[9], const double B0[3],
		const double ustar_i[3], const double K_i[9], const double ustar_ip1[3], const double Kinv_ip1[9], 
		const StateVector& xf_i, const double Residual_i[3]);

	void CalculateLocMarkers(int& NextLocMarker, const StateVector& xnext,
		const double LocMarkers[], double SegBounds_ip1,
		double p_atLocMarkers[][3], int32_t no_locmarkers);

	void LocMarkerUpdate(double p[3], const StateVector& xi, double t);
	void LocMarkerUpdate(double p[3], double xi[NUM_STATES], double t);

	struct CRMIntegrandParams {
		int32_t no_fcum_steps;
		double Li;
		double dlambdainv;
		Eigen::Matrix3d* K; 
		Eigen::Matrix3d* Kinv;
		Eigen::Vector3d* l;
		Eigen::Vector3d* ustar;
		double rho;
		std::vector<Eigen::Vector3d>* fcumlambda;
		Eigen::Vector3d* ftip;
		Eigen::Vector3d* g;
	};

	void CRMIntegrand(double s, const StateVector& in_x, const CRMIntegrandParams in_Params,
		StateDerivativeVector& out_xdot);

	void CRMShootingMethodBVP(CRMShootingMethodParams in_Params,
		double in_deltau0_initialguess[3], double in_ftip_initialguess[3],
		double out_deltau0[3], double out_ftip[3], int& out_localmin);

	void CRMShootingMethodBVP(
		int32_t in_no_flex_seg,
		int32_t in_no_rigid_seg,
		int32_t in_no_act_set,
		int32_t in_no_locmarkers,
		ContactModeType in_ContactMode,
		double 	in_deltau0_initialguess[3],
		double 	in_ftip_initialguess[3],
		double	in_InsertedLength, 
		double 	in_ActuationCurrents[][3], 
		double	in_TipConstraintPoint[3],
		double	in_TipForce[3],
		double	in_IntegrationStepSize,
		double 	in_B0[3],
		double 	in_g[3],
		double 	in_p0[3],
		double 	in_R0[9],
		double 	in_SegLengths[],
		double	in_LocMarkers[],
		double	in_InnerRadius[],
		double	in_OuterRadius[],
		double	in_YoungsModulus[],
		double	in_ShearModulus[],
		double 	in_ustar[][3],
		double 	in_CoilAlignmentAngles[][2],
		double 	in_CoilTurnAreaMat[][9],
		double 	in_rho[],
		double 	in_ActMass[],
		double 	out_deltau0[3], 
		double 	out_ftip[3],
		int& out_localmin
	);

	void CRMShootingMethodBVP_Prep(
		int32_t in_no_flex_seg,
		int32_t in_no_rigid_seg,
		int32_t in_no_act_set,
		int32_t in_no_locmarkers,
		double in_B0[3], double in_g[3], double in_p0[3], double in_R0[9],
		double in_SegLengths[], double in_LocMarkers[],
		double in_InnerRadius[], double in_OuterRadius[],
		double in_YoungsModulus[], double in_ShearModulus[],
		double in_ustar[][3],
		double in_CoilAlignmentAngles[][2], double in_CoilTurnAreaMat[][9],
		double in_rho[], double in_ActMass[],
		CRMCatheterModelParams& CathParams, CatheterConfiguration& CathConfig);

	struct NLEqnParams : CRMIVPCoreParams {
		NLEqnParams(int32_t no_flex, int32_t no_rigid, int32_t no_act, int32_t no_loc, int32_t no_fcums) : 
			CRMIVPCoreParams(no_flex, no_rigid, no_act, no_loc, no_fcums) {};
		ContactModeType ContactMode;
		Eigen::Vector3d TipConstraintPoint;
		Eigen::Vector3d TipForce;
        Eigen::Vector3d ftip_initialguess;
        double xf[NUM_STATES];
	};

	void CRM_NLEquation(double in_x[], double out_y[], NLEqnParams Params);
	void CRM_NLEquation_AnalyticalJac(double in_x[], double out_y[], double out_fjac[], NLEqnParams Params);

	void Project_State_to_Manifold(StateVector& State);

	template <typename StVecType, typename ParamType>
	void ABM4(const StVecType& in_x_0, const double t_0, const int N, const double h,
		const ParamType in_Params, int32_t in_no_locmarkers,
		const bool FinalValueOnly, const double in_LocMarkers[], int& inout_NextLocMarkerIdx,
		StVecType& out_x_N, double out_p_atLocMarkers[][3]);

	template <typename StVecType, typename ParamType>
	void ABM4_step(const StVecType& in_x_n, double t_n, double h,
		const StDerivativeVectType<StVecType>& in_xdot_nm1, const StDerivativeVectType<StVecType>& in_xdot_nm2, const StDerivativeVectType<StVecType>& in_xdot_nm3,
		const StVecType& in_x_nm1, const StVecType& in_x_nm2, const StVecType& in_x_nm3,
		const ParamType in_Params,
		StVecType& out_x_np1, StDerivativeVectType<StVecType>& out_xdot_n);

	template <typename StVecType, typename ParamType>
	void RK2_step(const StVecType& in_x_n, const double t_n, const double h,
		const ParamType in_Params,
		StVecType& out_x_np1, StDerivativeVectType<StVecType>& out_xdot_n);

	void wHat(const double in_w[3], double out_what[9]);
	void wHat(const double in_w[3], double out_what[9], unsigned int stride);
	void vee_from_so3(const double in_what[9], double out_w[3]);
	void RodriguesFormula(double in_w[3], double in_theta, double out_R[9]);
	void RodriguesExpanded(double in_w[3], double in_theta, double out_R[9]);
	void SE3_Analytical_Step(double in_R_n[9], double in_p_n[3], double in_u_n[3], double h, double out_R_np1[9], double out_p_np1[3]);
}