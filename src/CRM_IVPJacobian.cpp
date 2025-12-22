#include <cmath>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "CRM.hpp"
#include "CRM_BVPIVP_APIDeclarations.hpp"
#include "CRM_IVPJacobian.hpp"
#include "CRM_IVP_NumericalIntegrationTemplates.hpp"
#include "CRM_IVPJacobian_InternalAPI.hpp"

namespace CRMCatheterModel {

	std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> CRMSolverIVPJacobian(
		CRMShootingMethodParams in_Params,
		double in_deltau0[3], double in_ftip[3],
		bool in_FinalValueOnly,
		double out_x_N[NUM_STATES], double out_MomentResidual[3]) {

		double x_0[NUM_STATES];
		for (int i = 0; i < 3; i++) x_0[i] = in_Params.p0(i);
		for (int i = 0; i < 9; i++) x_0[i + 3] = in_Params.R0(i/3, i%3);
		for (int i = 0; i < 3; i++) x_0[i + 12] = std::nan("0");

		CRMIVPCoreParams CoreParams(in_Params.no_flex_seg, in_Params.no_rigid_seg, in_Params.no_act_set, in_Params.no_locmarkers, in_Params.no_fcum_steps);

		CRMSolverIVP_Prep(
			in_Params.no_flex_seg, in_Params.no_rigid_seg, in_Params.no_act_set, in_Params.no_locmarkers, in_Params.no_fcum_steps,
			x_0, in_Params.IntegrationStepSize,
			in_Params.Li, in_Params.dlambdainv,
			in_Params.SegmentTypes.data(),
			in_Params.SegEndLambdas.data(), in_Params.LocMarkerLambdas.data(),
			in_Params.rho.data(),
			(double(*)[9])in_Params.K.data(), (double(*)[9])in_Params.Kinv.data(), (double(*)[3])in_Params.ustar.data(),
			in_Params.ActMass.data(),
			(double(*)[9])in_Params.CoilAlignmentTurnAreaMatrix.data(),
			(double(*)[3])in_Params.MagMoment.data(), (double(*)[3])in_Params.fcumlambda.data(),
			in_Params.B0.data(), in_Params.g.data(),
			false, true,
			CoreParams);

		AugmentedStateVector<IVPJacobiansFull> x_N;
		CRMSolverIVP_CoreWithJacobian(CoreParams, in_deltau0, in_ftip, x_N, out_MomentResidual);

		for (int i = 0; i < 3; i++) out_x_N[i] = x_N._p[i];
		for (int i = 0; i < 9; i++) out_x_N[3 + i] = x_N._R[i];
		for (int i = 0; i < 3; i++) out_x_N[12 + i] = x_N._u[i];

		constexpr unsigned int Cs = NUM_ACT_SET * 3;
		__EMT<3, 3> JIVP_p_u0(x_N._p_u0);
		__EMT<3, 3> JIVP_ws_u0(x_N._ws_u0);
		__EMT<3, 3> JIVP_u_u0(x_N._u_u0);
		__EMT<3, 3> JIVP_p_ft(x_N._p_ft);
		__EMT<3, 3> JIVP_ws_ft(x_N._ws_ft);
		__EMT<3, 3> JIVP_u_ft(x_N._u_ft);

		Eigen::Matrix<double, 3, Cs + 1> JIVP_p_z;
		Eigen::Matrix<double, 3, Cs + 1> JIVP_ws_z;
		Eigen::Matrix<double, 3, Cs + 1> JIVP_u_z;

		__EMT<3, Cs> JIVP_p_zc_rev(x_N._p_zc);
		__EVT<3> JIVP_p_zl(x_N._p_zl);
		for (int i = 0; i < NUM_ACT_SET; i++) JIVP_p_z.middleCols<3>(3 * i) = JIVP_p_zc_rev.middleCols<3>((NUM_ACT_SET - 1 - i) * 3);
		JIVP_p_z.rightCols<1>() = JIVP_p_zl;

		__EMT<3, Cs> JIVP_ws_zc_rev(x_N._ws_zc);
		__EVT<3> JIVP_ws_zl(x_N._ws_zl);
		for (int i = 0; i < NUM_ACT_SET; i++) JIVP_ws_z.middleCols<3>(3 * i) = JIVP_ws_zc_rev.middleCols<3>((NUM_ACT_SET - 1 - i) * 3);
		JIVP_ws_z.rightCols<1>() = JIVP_ws_zl;

		__EMT<3, Cs> JIVP_u_zc_rev(x_N._u_zc);
		__EVT<3> JIVP_u_zl(x_N._u_zl);
		for (int i = 0; i < NUM_ACT_SET; i++) JIVP_u_z.middleCols<3>(3 * i) = JIVP_u_zc_rev.middleCols<3>((NUM_ACT_SET - 1 - i) * 3);
		JIVP_u_z.rightCols<1>() = JIVP_u_zl;

		Eigen::MatrixXd JIVP_u_u0_pinv = JIVP_u_u0.completeOrthogonalDecomposition().pseudoInverse();
		Eigen::MatrixXd JBVP_p_z = JIVP_p_z - JIVP_p_u0 * JIVP_u_u0_pinv * JIVP_u_z;
		Eigen::MatrixXd JBVP_ws_z = JIVP_ws_z - JIVP_ws_u0 * JIVP_u_u0_pinv * JIVP_u_z;
		Eigen::MatrixXd JBVP_p_ft = JIVP_p_ft - JIVP_p_u0 * JIVP_u_u0_pinv * JIVP_u_ft;
		Eigen::MatrixXd JBVP_ws_ft = JIVP_ws_ft - JIVP_ws_u0 * JIVP_u_u0_pinv * JIVP_u_ft;
		Eigen::MatrixXd Jft_z = -JBVP_p_ft.completeOrthogonalDecomposition().pseudoInverse() * JBVP_p_z;

		return { JBVP_p_z, JBVP_ws_z, JBVP_p_ft, JBVP_ws_ft, Jft_z };
	}

	template <typename IVPJacobians>
	void CRMSolverIVP_CoreWithJacobian(CRMIVPCoreParams& in_params,
		double in_deltau[3], double in_ftip[3],
		AugmentedStateVector<IVPJacobians>& out_x_N, double out_MomentResidual[3]) {

		double h, dummyPE;
		int StartSegmentIndex = in_params.StartSegmentIndex;
		
		AugmentedStateVector<IVPJacobians> xi;
		for(int i=0; i<3; i++) xi._p[i] = in_params.xi(i);
		for(int i=0; i<9; i++) xi._R[i] = in_params.xi(3+i);
		for (int i = 0; i < 3; i++)  xi._u_u0[i * 3 + i] = 1.0;
		if constexpr (std::is_same_v<expr_type<IVPJacobians>, expr_type<IVPJacobiansFull>>) {
			for (int i = 0; i < 3; i++) xi._p_zl[i] = xi._R[i * 3 + 2];
		}

		CRMIntegrandParams IntegrandParams;
		IntegrandParams.dlambdainv = in_params.dlambdainv;
		IntegrandParams.Li = in_params.InsertedLength;
		IntegrandParams.no_fcum_steps = in_params.no_fcum_steps;
		IntegrandParams.fcumlambda = &in_params.fcumlambda;
		Eigen::Vector3d ftip_vec(in_ftip[0], in_ftip[1], in_ftip[2]);
		IntegrandParams.ftip = &ftip_vec;

		if (in_params.SegmentTypes[StartSegmentIndex] != CatheterSegmentType::FLEXIBLE) {
			while (in_params.SegmentTypes[StartSegmentIndex] != CatheterSegmentType::FLEXIBLE && StartSegmentIndex < in_params.no_segments) {
				double len = in_params.SegBounds[StartSegmentIndex + 1] - in_params.SegBounds[StartSegmentIndex];
				for (int i = 0; i < 3; i++) xi._p[i] += len * xi._R[i * 3 + 2];
				StartSegmentIndex++;
			}
			if (StartSegmentIndex < in_params.no_segments) {
                int fidx = in_params.FlexActIndex[StartSegmentIndex];
                for(int i=0; i<3; i++) xi._u[i] = in_params.ustar[fidx](i) + in_deltau[i];
            }
		} else {
            int fidx = in_params.FlexActIndex[StartSegmentIndex];
			for(int i=0; i<3; i++) xi._u[i] = in_params.ustar[fidx](i) + in_deltau[i];
			if constexpr (std::is_same_v<expr_type<IVPJacobians>, expr_type<IVPJacobiansFull>>) {
				Eigen::Map<const Eigen::Matrix3d> R_mat(xi._R);
                Eigen::Map<const Eigen::Vector3d> u_vec(xi._u);
                Eigen::Map<Eigen::Vector3d> ws_zl(xi._ws_zl);
                ws_zl = R_mat * u_vec;
			}
		}

		out_x_N = xi;
		for (int i = 0; i < 3; i++) out_MomentResidual[i] = 0.0;

		for (int i = StartSegmentIndex; i < in_params.no_segments; i++) {
			if (in_params.SegmentTypes[i] == CatheterSegmentType::FLEXIBLE) {
				int fseg = in_params.FlexActIndex[i];
				IntegrandParams.K = &in_params.K[fseg];
				IntegrandParams.Kinv = &in_params.Kinv[fseg];
				IntegrandParams.ustar = &in_params.ustar[fseg];
                IntegrandParams.rho = in_params.rho[i];
				h = (in_params.SegBounds[i + 1] - in_params.SegBounds[i]) / in_params.SegSteps[fseg];

				ABM4(xi, in_params.SegBounds[i], in_params.SegSteps[fseg], h, IntegrandParams, in_params.no_locmarkers,
					false, true, in_params.LocMarkers.data(), in_params.NextLocMarker, out_x_N, dummyPE, (double(*)[3])in_params.p_atLocMarkers.data());
                xi = out_x_N;
			}
		}
	}

	template <typename IVPJacobians>
	void CRMIntegrand(double s, const AugmentedStateVector<IVPJacobians>& in_x, const CRMIntegrandParams in_Params, AugmentedStateDerivativeVector<IVPJacobians>& out_xdot) {}

	template void CRMSolverIVP_CoreWithJacobian<IVPJacobiansMini>(CRMIVPCoreParams& in_params, double in_u[3], double in_ftip[3], AugmentedStateVector<IVPJacobiansMini>& out_x_N, double out_MomentResidual[3]);
	template void CRMSolverIVP_CoreWithJacobian<IVPJacobiansFull>(CRMIVPCoreParams& in_params, double in_u[3], double in_ftip[3], AugmentedStateVector<IVPJacobiansFull>& out_x_N, double out_MomentResidual[3]);
}
