#pragma once
#include "CRM_BVPIVP_APIDeclarations.hpp"

namespace CRMCatheterModel {

	template <typename StVecType>
	void ABM4(const StVecType& in_x_0, const double t_0, const int N, const double h,
		const CRMIntegrandParams& in_Params, int32_t in_no_locmarkers,
		const bool CalculateEnergy,
		const bool FinalValueOnly, const double in_LocMarkers[], int& inout_NextLocMarkerIdx,
		StVecType& out_x_N, double& out_PotentialEnergy, double out_p_atLocMarkers[][3]) {

		using _SVT = StVecType;
		using _DVT = StDerivativeVectType<StVecType>;

		_SVT x_nm3, x_nm2, x_nm1, x_n(in_x_0), x_np1;
		double t_n = t_0;
		_DVT xdot_nm3, xdot_nm2, xdot_nm1, xdot_n;

		out_PotentialEnergy = 0.0;
		int NextLocMarkerIdx = inout_NextLocMarkerIdx;

		for (int idx = 0; idx < N; idx++) {
			if (idx < 3) {
				RK2_step(x_n, t_n, h, in_Params, x_np1, xdot_n);
			} else {
				ABM4_step(x_n, t_n, h, xdot_nm1, xdot_nm2, xdot_nm3, x_nm1, x_nm2, x_nm3, in_Params, x_np1, xdot_n);
			}

			t_n += h;

			if (!FinalValueOnly) {
				while ((NextLocMarkerIdx < in_no_locmarkers) && (in_LocMarkers[NextLocMarkerIdx] <= t_n)) {
					double d_np1 = (t_n - in_LocMarkers[NextLocMarkerIdx]) / h;
					double d_n = 1.0 - d_np1;
					for(int i=0; i<3; i++) out_p_atLocMarkers[NextLocMarkerIdx][i] = d_np1 * x_n._p[i] + d_n * x_np1._p[i];
					NextLocMarkerIdx++;
				}
			}

			x_nm3 = x_nm2; x_nm2 = x_nm1; x_nm1 = x_n; x_n = x_np1;
			xdot_nm3 = xdot_nm2; xdot_nm2 = xdot_nm1; xdot_nm1 = xdot_n;
		}
		out_x_N = x_n;
		inout_NextLocMarkerIdx = NextLocMarkerIdx;
	}

	template <typename StVecType>
	void ABM4_step(const StVecType& in_x_n, double t_n, double h,
		const StDerivativeVectType<StVecType>& in_xdot_nm1, const StDerivativeVectType<StVecType>& in_xdot_nm2, const StDerivativeVectType<StVecType>& in_xdot_nm3,
		const StVecType& in_x_nm1, const StVecType& in_x_nm2, const StVecType& in_x_nm3,
		const CRMIntegrandParams& in_Params,
		StVecType& out_x_np1, StDerivativeVectType<StVecType>& out_xdot_n) {

		const double P_COEFF_N = 55.0 / 24.0, P_COEFF_Nm1 = -59.0 / 24.0, P_COEFF_Nm2 = 37.0 / 24.0, P_COEFF_Nm3 = -9.0 / 24.0;
		const double C_COEFF_Np1 = 9.0 / 24.0, C_COEFF_N = 19.0 / 24.0, C_COEFF_Nm1 = -5.0 / 24.0, C_COEFF_Nm2 = 1.0 / 24.0;
		
        StVecType x_np1_hat;
		StDerivativeVectType<StVecType> xdot_np1_hat;

		CRMIntegrand(t_n, in_x_n, in_Params, out_xdot_n);
		x_np1_hat = in_x_n + h * (P_COEFF_N * out_xdot_n + P_COEFF_Nm1 * in_xdot_nm1 + P_COEFF_Nm2 * in_xdot_nm2 + P_COEFF_Nm3 * in_xdot_nm3);
        
        double u_n_pred[3];
        for (int i = 0; i < 3; i++) u_n_pred[i] = (P_COEFF_N * in_x_n._u[i] + P_COEFF_Nm1 * in_x_nm1._u[i] + P_COEFF_Nm2 * in_x_nm2._u[i] + P_COEFF_Nm3 * in_x_nm3._u[i]);
        SE3_Analytical_Step((double*)in_x_n._R, (double*)in_x_n._p, u_n_pred, h, x_np1_hat._R, x_np1_hat._p);

		CRMIntegrand(t_n + h, x_np1_hat, in_Params, xdot_np1_hat);
		out_x_np1 = in_x_n + h * (C_COEFF_Np1 * xdot_np1_hat + C_COEFF_N * out_xdot_n + C_COEFF_Nm1 * in_xdot_nm1 + C_COEFF_Nm2 * in_xdot_nm2);
        
        double u_n_corr[3];
        for (int i = 0; i < 3; i++) u_n_corr[i] = (C_COEFF_Np1 * x_np1_hat._u[i] + C_COEFF_N * in_x_n._u[i] + C_COEFF_Nm1 * in_x_nm1._u[i] + C_COEFF_Nm2 * in_x_nm2._u[i]);
        SE3_Analytical_Step((double*)in_x_n._R, (double*)in_x_n._p, u_n_corr, h, out_x_np1._R, out_x_np1._p);
	}

	template <typename StVecType>
	void RK2_step(const StVecType& in_x_n, const double t_n, const double h,
		const CRMIntegrandParams& in_Params,
		StVecType& out_x_np1, StDerivativeVectType<StVecType>& out_xdot_n) {

		StDerivativeVectType<StVecType> k1, k2oh;
		StVecType x_n_p_k1o2;

		CRMIntegrand(t_n, in_x_n, in_Params, out_xdot_n);
		k1 = h * out_xdot_n;
		x_n_p_k1o2 = in_x_n + k1 * 0.5;
        SE3_Analytical_Step((double*)in_x_n._R, (double*)in_x_n._p, (double*)in_x_n._u, h * 0.5, x_n_p_k1o2._R, x_n_p_k1o2._p);

		CRMIntegrand(t_n + h * 0.5, x_n_p_k1o2, in_Params, k2oh);
		out_x_np1 = in_x_n + h * k2oh;
        SE3_Analytical_Step((double*)in_x_n._R, (double*)in_x_n._p, (double*)x_n_p_k1o2._u, h, out_x_np1._R, out_x_np1._p);
	}
}
