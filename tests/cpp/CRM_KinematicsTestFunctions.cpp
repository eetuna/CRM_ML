#include <eigen3/Eigen/Dense>
#include "CRM.hpp"
#include "CRM_BVPIVP_APIDeclarations.hpp"
#include "CRM_IVPJacobian.hpp"
#include "CRM_KinematicsTestFunctions.hpp"

namespace CRMCatheterModel {

    constexpr double NUM_JACOBIAN_BASEPOSITION_STEPSIZE = 1e-3;
    constexpr double NUM_JACOBIAN_BASEROTATION_STEPSIZE = 1e-3;
    constexpr double NUM_JACOBIAN_BASECURVATURE_STEPSIZE = 1e-6;

	Eigen::MatrixXd CRM_FKJacobian_BruteForce(const Eigen::VectorXd in_x, Eigen::VectorXd& out_y, CRMForwardKinematicsData in_Params, int& localmin) {
		CRMForwardKinematicsData Params = in_Params;
		Params.FinalValueOnly = true;
		double dummyPE;
		Eigen::VectorXd f_0 = CRM_ForwardKinematics(in_x, Params, dummyPE, localmin);
		
		for (int i = 0; i < 3; i++) Params.deltau0_initialguess(i) = f_0(12 + i);
		if (Params.ContactMode == ContactModeType::FIXED_TIP) Params.ftip_initialguess = f_0.segment<3>(15);
		
		f_0 = CRM_ForwardKinematics(in_x, Params, dummyPE, localmin);

		int var_size = in_x.size();
		Eigen::MatrixXd Df_(f_0.size(), var_size);

		for (int j = 0; j < var_size; ++j) {
			double h = (j == var_size - 1) ? NUM_JACOBIAN_INSERTIONLENGTH_STEPSIZE : NUM_JACOBIAN_CURRENT_STEPSIZE;
			Eigen::VectorXd x_h = in_x; x_h(j) += h;
			int lmin;
			Eigen::VectorXd f_h = CRM_ForwardKinematics(x_h, Params, dummyPE, lmin);
			Df_.col(j) = (f_h - f_0) / h;
		}
		out_y = f_0;

		int JRow = (Params.ContactMode == ContactModeType::FREE_TIP) ? 9 : 12;
		Eigen::MatrixXd T(JRow, f_0.size());
		T.setZero();
		T.block(0, 0, 3, 3).setIdentity();
		T.block(6, 12, 3, 3).setIdentity();
		T.block(3, 9, 1, 3) << f_0(6), f_0(7), f_0(8);
		T.block(4, 3, 1, 3) << f_0(9), f_0(10), f_0(11);
		T.block(5, 6, 1, 3) << f_0(3), f_0(4), f_0(5);
		if (Params.ContactMode == ContactModeType::FIXED_TIP) T.block(9, 15, 3, 3).setIdentity();
		return T * Df_;
	}

	Eigen::MatrixXd CRM_FKJacobian_From_IVP_Numerical(const Eigen::VectorXd in_x, const Eigen::VectorXd& in_FKouty, CRMForwardKinematicsData in_Params) {
		double ActuationCurrents[NUM_ACT_SET][3];
		for (int i = 0; i < NUM_ACT_SET; i++) for (int j = 0; j < 3; j++) ActuationCurrents[i][j] = in_x(i * 3 + j);
		double InsertedLength = in_x(NUM_ACT_SET * 3);

		CRMShootingMethodParams BVPParams = CRMConstructShootingMethodParamSet(*(in_Params.CathParams), *(in_Params.CathConfig), InsertedLength, ActuationCurrents, 
            in_Params.ContactMode, in_Params.TipConstraintPoint.data(), in_Params.TipForce.data(), in_Params.IntegrationStepSize);

		Eigen::VectorXd x0(3 + 9 + 3 + 3 + 3 * NUM_ACT_SET + 1);
		x0.head(15) = in_FKouty.head(15);
		if (in_Params.ContactMode == ContactModeType::FREE_TIP) x0.segment<3>(15) = in_Params.TipForce;
		else x0.segment<3>(15) = in_FKouty.segment<3>(15);
		x0.tail(3 * NUM_ACT_SET + 1) = in_x;

		Eigen::VectorXd y0 = CRMSolverIVPWrapper(x0, BVPParams);
		Eigen::MatrixXd J(y0.size(), x0.size());

		for (int i = 0; i < x0.size(); i++) {
			double h;
			if (i < 3) h = NUM_JACOBIAN_BASEPOSITION_STEPSIZE;
			else if (i < 12) h = NUM_JACOBIAN_BASEROTATION_STEPSIZE;
			else if (i < 15) h = NUM_JACOBIAN_BASECURVATURE_STEPSIZE;
			else if (i < 18) h = NUM_JACOBIAN_TIPFORCE_STEPSIZE;
			else if (i < 18 + 3 * NUM_ACT_SET) h = NUM_JACOBIAN_CURRENT_STEPSIZE;
			else h = NUM_JACOBIAN_INSERTIONLENGTH_STEPSIZE;
			
			Eigen::VectorXd xh = x0; xh(i) += h;
			J.col(i) = (CRMSolverIVPWrapper(xh, BVPParams) - y0) / h;
		}

		Eigen::MatrixXd JIVP(15, 15 + 3 * NUM_ACT_SET + 1 + 3);
        JIVP.leftCols(15) = J.leftCols(15);
        JIVP.rightCols(3) = J.middleCols<3>(15);
        for(int i=0; i<NUM_ACT_SET; i++) JIVP.middleCols<3>(18 + 3*i) = J.middleCols<3>(18 + 3*(NUM_ACT_SET-1-i));
        JIVP.col(JIVP.cols()-4) = J.col(x0.size()-1);

		constexpr unsigned int Cs = NUM_ACT_SET * 3;
		Eigen::MatrixXd T(3, 9); T.setZero();
		Eigen::VectorXd _R = in_FKouty.segment<9>(3);
		T(0, 6) = _R(3); T(0, 7) = _R(4); T(0, 8) = _R(5);
		T(1, 0) = _R(6); T(1, 1) = _R(7); T(1, 2) = _R(8);
		T(2, 3) = _R(0); T(2, 4) = _R(1); T(2, 5) = _R(2);

		Eigen::Matrix3d Jp_u0 = JIVP.block<3, 3>(0, 12);
		Eigen::MatrixXd Jp_z = JIVP.block(0, 15, 3, Cs + 1);
		Eigen::Matrix3d Jp_ft = JIVP.rightCols<3>().topRows<3>();
		Eigen::Matrix3d Ju_u0 = JIVP.block<3, 3>(12, 12);
		Eigen::MatrixXd Ju_z = JIVP.block(12, 15, 3, Cs + 1);
		Eigen::Matrix3d Ju_ft = JIVP.rightCols<3>().bottomRows<3>();
		Eigen::Matrix3d Jws_u0 = T * JIVP.block<9, 3>(3, 12);
		Eigen::Matrix3d Jws_ft = T * JIVP.block<9, 3>(3, JIVP.cols()-3);
		Eigen::MatrixXd Jws_z = T * JIVP.block(3, 15, 9, Cs + 1);

		Eigen::MatrixXd Ju_u0_pinv = Ju_u0.completeOrthogonalDecomposition().pseudoInverse();
		Eigen::MatrixXd JBVP_p_z = Jp_z - Jp_u0 * Ju_u0_pinv * Ju_z;
		Eigen::MatrixXd JBVP_ws_z = Jws_z - Jws_u0 * Ju_u0_pinv * Ju_z;
		Eigen::MatrixXd JBVP_p_ft = Jp_ft - Jp_u0 * Ju_u0_pinv * Ju_ft;

		if (in_Params.ContactMode == ContactModeType::FREE_TIP) {
			Eigen::MatrixXd Res(6, JBVP_p_z.cols());
			Res.topRows(3) = JBVP_p_z; Res.bottomRows(3) = JBVP_ws_z;
			return Res;
		}
		return -JBVP_p_ft.completeOrthogonalDecomposition().pseudoInverse() * JBVP_p_z;
	}

	Eigen::VectorXd CRMSolverIVPWrapper(Eigen::VectorXd in, CRMShootingMethodParams in_Params) {
		CRMShootingMethodParams Params = in_Params;
		Params.p0 = in.head<3>();
		for(int i=0; i<9; i++) Params.R0(i/3, i%3) = in(3+i);
		double u0[3] = {in(12), in(13), in(14)};
		double ft[3] = {in(15), in(16), in(17)};
		for (int i = 0; i < NUM_ACT_SET; i++) {
			Eigen::Vector3d act = in.segment<3>(18 + i*3);
			Params.MagMoment[i] = Params.CoilAlignmentTurnAreaMatrix[i] * act;
		}
		Params.Li = in(in.size() - 1);

		double xf[NUM_STATES], res[3], pe;
		double marker_dummy[5][3], orient_dummy[NUM_ACT_SET][9], pos_dummy[NUM_ACT_SET][3];
		CRMSolverIVP(Params, u0, ft, false, true, xf, res, pe, marker_dummy, orient_dummy, pos_dummy);
		
		Eigen::VectorXd out(15);
		for(int i=0; i<15; i++) out(i) = xf[i];
		return out;
	}
}
