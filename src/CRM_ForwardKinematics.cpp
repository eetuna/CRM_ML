#include <eigen3/Eigen/Dense>
#include <vector>
#include "CRM.hpp"
#include "CRM_BVPIVP_APIDeclarations.hpp"
#include "CRM_IVPJacobian.hpp"
#include "CRM_FK_InternalAPI.hpp"
#include "minpack.hpp"

namespace CRMCatheterModel {

	int CRM_ForwardKinematics(double in_x[], double out_y[], double& out_PotentialEnergy, CRMForwardKinematicsData Params) {
		int localmin;
		int X_Dim = NUM_ACT_SET * 3 + 1;
		int Y_Dim = (Params.ContactMode == ContactModeType::FREE_TIP) ? 15 : 18;
		
		Eigen::VectorXd x(X_Dim);
		for (int i = 0; i < X_Dim; i++) x(i) = in_x[i];

		Eigen::VectorXd y = CRM_ForwardKinematics(x, Params, out_PotentialEnergy, localmin);
		for (int i = 0; i < Y_Dim; i++) out_y[i] = y(i);
		return localmin;
	}

	Eigen::VectorXd CRM_ForwardKinematics(const Eigen::VectorXd in_x, CRMForwardKinematicsData Params, double& out_PotentialEnergy, int& out_localmin) {
		if (Params.ContactMode == ContactModeType::FREE_TIP)
			return CRM_ForwardKinematics_FreeSpace(in_x, Params, out_PotentialEnergy, out_localmin);
		else
			return CRM_ForwardKinematics_Contact(in_x, Params, out_PotentialEnergy, out_localmin);
	}

	Eigen::VectorXd CRM_ForwardKinematics_FreeSpace(const Eigen::VectorXd in_x, CRMForwardKinematicsData Params, double& out_PotentialEnergy, int& out_localmin) {
		int Y_Dim = (Params.ContactMode == ContactModeType::FREE_TIP) ? 15 : 18;
		Eigen::VectorXd out_y(Y_Dim);

		double ActuationCurrents[NUM_ACT_SET][3];
		for (int i = 0; i < NUM_ACT_SET; i++) for (int j = 0; j < 3; j++) ActuationCurrents[i][j] = in_x(i * 3 + j);
		double InsertedLength = in_x(NUM_ACT_SET * 3);

		double deltau0_calc[3], ftip_calc[3];
		double xf[NUM_STATES], residual[3];

		CRMShootingMethodParams BVPParams = CRMConstructShootingMethodParamSet(*(Params.CathParams), *(Params.CathConfig), InsertedLength, ActuationCurrents, Params.ContactMode, 
            Params.TipConstraintPoint.data(), Params.TipForce.data(), Params.IntegrationStepSize);

		double u0_guess[3], ft_guess[3];
		for(int i=0; i<3; i++) { u0_guess[i] = Params.deltau0_initialguess(i); ft_guess[i] = Params.ftip_initialguess(i); }

		CRMShootingMethodBVP(BVPParams, u0_guess, ft_guess, deltau0_calc, ftip_calc, out_localmin);

		CRMSolverIVP(BVPParams, deltau0_calc, ftip_calc, true, Params.FinalValueOnly, xf, residual, out_PotentialEnergy, 
            Params.ReportedMarkerPos, (double(*)[9])Params.ReportedCoilOrient, (double(*)[3])Params.ReportedCoilPos);

		for (int i = 0; i < 3; i++) {
			out_y(i) = xf[i];
			out_y(12 + i) = deltau0_calc[i];
			if (Params.ContactMode == ContactModeType::FIXED_TIP) out_y(15 + i) = ftip_calc[i];
		}
		for (int i = 0; i < 9; i++) out_y(3 + i) = xf[3 + i];

		return out_y;
	}

	Eigen::VectorXd CRM_ForwardKinematics_Contact(const Eigen::VectorXd in_x, CRMForwardKinematicsData Params, double& out_PotentialEnergy, int& out_localmin) {
		int X_Dim = NUM_ACT_SET * 3 + 1;
		Eigen::VectorXd out_y(18);

		CRMContactEquationParams NLEParams;
		NLEParams.CRMForwardKinematicsData::operator=(Params);
		for (int i = 0; i < X_Dim; i++) NLEParams.Actuation[i] = in_x(i);
		NLEParams.ContactMode = ContactModeType::FREE_TIP;

		const double FSCALE_INV = 1.0 / IVALUE_SCALE_F;
		std::vector<double> initialguessscaled(3);
		for (int i = 0; i < 3; i++) initialguessscaled[i] = FSCALE_INV * Params.ftip_initialguess(i);

#if defined( CONTACT_TRUSTREGION )
		std::vector<double> x = initialguessscaled;
		std::vector<double> residual(3);
		int info;
		double tol = TRUSTREGION_TOLERANCE;

#if defined(CONTACT_TRUSTREGION_ANALYTICALJAC)
		TrustRegionDogleg_GivenJacobian(CRM_Contact_Equation, CRM_Contact_Equation_AnalyticalJac, 3, x.data(), residual.data(), tol, info, NLEParams);
#else 	
		TrustRegionDogleg(CRM_Contact_Equation, 3, x.data(), residual.data(), tol, info, NLEParams);
#endif
		out_localmin = (info == 1) ? 0 : (info - 1);
		Eigen::Vector3d solved_ftip;
		for (int i = 0; i < 3; i++) solved_ftip(i) = IVALUE_SCALE_F * x[i];
#else
		exit(1);
#endif

		int ignore;
		NLEParams.TipForce = solved_ftip;
		Eigen::VectorXd temp_out = CRM_ForwardKinematics(in_x, NLEParams, out_PotentialEnergy, ignore);
		out_y.head(15) = temp_out;
		out_y.segment<3>(15) = solved_ftip;

		return out_y;
	}

	Eigen::MatrixXd CRM_FKJacobian_Numerical(const Eigen::VectorXd in_x, Eigen::VectorXd& out_y, CRMForwardKinematicsData in_Params, int& localmin) {
		double dummyPE;
		if (in_Params.ContactMode == ContactModeType::FREE_TIP) {
			return (CRM_FKJacobian_FreeSpace(in_x, out_y, in_Params, FKFreeJacobianType::ACTUATION_ONLY, localmin));
		}
		
		CRMForwardKinematicsData Params = in_Params;
		Params.FinalValueOnly = true;
		out_y = CRM_ForwardKinematics(in_x, Params, dummyPE, localmin);
		Params.TipForce = out_y.segment<3>(15);
		
		int templocalmin;
		Eigen::VectorXd tempvector;
		Eigen::MatrixXd JBVP = CRM_FKJacobian_FreeSpace(in_x, tempvector, Params, FKFreeJacobianType::ACTUATION_AND_TIP_FORCE, templocalmin);
		Eigen::MatrixXd JBVPp = JBVP.topRows(3);
		Eigen::MatrixXd J = -JBVPp.rightCols(3).completeOrthogonalDecomposition().pseudoInverse() * JBVPp.leftCols(JBVPp.cols() - 3);
		return J;
	}

	Eigen::MatrixXd CRM_FKJacobian_FreeSpace(const Eigen::VectorXd in_x, Eigen::VectorXd& out_y, CRMForwardKinematicsData in_Params, FKFreeJacobianType mode, int& localmin) {
		double dummyPE;
		CRMForwardKinematicsData Params = in_Params;
		Params.FinalValueOnly = true;
		Params.ContactMode = ContactModeType::FREE_TIP;

		Eigen::VectorXd f_0 = CRM_ForwardKinematics(in_x, Params, dummyPE, localmin);
		for (int i = 0; i < 3; i++) Params.deltau0_initialguess(i) = f_0(12 + i);
		f_0 = CRM_ForwardKinematics(in_x, Params, dummyPE, localmin);

		int var_size = in_x.size();
		int col_size = (mode == FKFreeJacobianType::ACTUATION_ONLY) ? var_size : (var_size + 3);
		Eigen::MatrixXd Df_(f_0.size(), col_size);

		for (int j = 0; j < var_size; ++j) {
			double h = (j == var_size - 1) ? NUM_JACOBIAN_INSERTIONLENGTH_STEPSIZE : NUM_JACOBIAN_CURRENT_STEPSIZE;
			Eigen::VectorXd x_h = in_x; x_h(j) += h;
			int lmin;
			Eigen::VectorXd f_h = CRM_ForwardKinematics(x_h, Params, dummyPE, lmin);
			Df_.col(j) = (f_h - f_0) / h;
		}

		if (mode == FKFreeJacobianType::ACTUATION_AND_TIP_FORCE) {
			double h = NUM_JACOBIAN_TIPFORCE_STEPSIZE;
			for (int k = 0; k < 3; k++) {
				Eigen::Vector3d old_ft = Params.TipForce;
				Params.TipForce(k) += h;
				int lmin;
				Eigen::VectorXd f_h = CRM_ForwardKinematics(in_x, Params, dummyPE, lmin);
				Df_.col(var_size + k) = (f_h - f_0) / h;
				Params.TipForce = old_ft;
			}
		}
		out_y = f_0;

		Eigen::MatrixXd T(9, f_0.size());
		T.setZero();
		T.block(0, 0, 3, 3).setIdentity();
		T.block(6, 12, 3, 3).setIdentity();
		T.block(3, 9, 1, 3) << f_0(6), f_0(7), f_0(8);
		T.block(4, 3, 1, 3) << f_0(9), f_0(10), f_0(11);
		T.block(5, 6, 1, 3) << f_0(3), f_0(4), f_0(5);
		return T * Df_;
	}

	Eigen::MatrixXd CRM_FKJacobian_Analytical(const Eigen::VectorXd in_x, const Eigen::VectorXd& in_FKouty, CRMForwardKinematicsData in_Params) {
		double ActuationCurrents[NUM_ACT_SET][3];
		for (int i = 0; i < NUM_ACT_SET; i++) for (int j = 0; j < 3; j++) ActuationCurrents[i][j] = in_x(i * 3 + j);
		double InsertedLength = in_x(NUM_ACT_SET * 3);

		CRMShootingMethodParams BVPParams = CRMConstructShootingMethodParamSet(*(in_Params.CathParams), *(in_Params.CathConfig), InsertedLength, ActuationCurrents, 
            in_Params.ContactMode, in_Params.TipConstraintPoint.data(), in_Params.TipForce.data(), in_Params.IntegrationStepSize);

		double du0[3], ft[3];
		for (int i = 0; i < 3; i++) {
			du0[i] = in_FKouty(i + 12);
			ft[i] = (in_Params.ContactMode == ContactModeType::FREE_TIP) ? in_Params.TipForce(i) : in_FKouty(i + 15);
		}

		double xf[NUM_STATES], res[3];
		auto [JBVP_p_z, JBVP_ws_z, JBVP_p_ft, JBVP_ws_ft, Jft_z] = CRMSolverIVPJacobian(BVPParams, du0, ft, false, xf, res);

		if (in_Params.ContactMode == ContactModeType::FREE_TIP) {
			Eigen::MatrixXd J(6, JBVP_p_z.cols() + 3);
			J.topLeftCorner(3, JBVP_p_z.cols()) = JBVP_p_z;
			J.bottomLeftCorner(3, JBVP_p_z.cols()) = JBVP_ws_z;
			J.topRightCorner(3, 3) = JBVP_p_ft;
			J.bottomRightCorner(3, 3) = JBVP_ws_ft;
			return J;
		}
		return Jft_z;
	}

	void CRM_Contact_Equation(double in_x[], double out_y[], CRMContactEquationParams Params) {
		double dummyPE;
		for (int i = 0; i < 3; i++) Params.TipForce(i) = IVALUE_SCALE_F * in_x[i];
		Params.ContactMode = ContactModeType::FREE_TIP;
		double fkoutput[15];
		CRM_ForwardKinematics(Params.Actuation, fkoutput, dummyPE, Params);
		for (int i = 0; i < 3; i++) out_y[i] = RESIDUAL_SCALE_P * (fkoutput[i] - Params.TipConstraintPoint(i));
	}

	void CRM_Contact_Equation_AnalyticalJac(double in_x[], double out_y[], double out_fjac[], CRMContactEquationParams Params) {
		double dummyPE;
		for (int i = 0; i < 3; i++) Params.TipForce(i) = IVALUE_SCALE_F * in_x[i];
		Params.ContactMode = ContactModeType::FREE_TIP;
		double fkoutput[15];
		CRM_ForwardKinematics(Params.Actuation, fkoutput, dummyPE, Params);
		for (int i = 0; i < 3; i++) out_y[i] = RESIDUAL_SCALE_P * (fkoutput[i] - Params.TipConstraintPoint(i));

		Eigen::Matrix<double, 3 * NUM_ACT_SET + 1, 1> u;
		Eigen::Matrix<double, 15, 1> y;
		for (int i = 0; i < u.size(); i++) u(i) = Params.Actuation[i];
		for (int i = 0; i < 15; i++) y(i) = fkoutput[i];
		Eigen::MatrixXd Jfull = CRM_FKJacobian_Analytical(u, y, Params);
		Eigen::Matrix3d sJp = RESIDUAL_SCALE_P * Jfull.topRightCorner<3, 3>() * IVALUE_SCALE_F;
		for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) out_fjac[i + j * 3] = sJp(i, j);
	}
}