#pragma once

#include <algorithm>
#include <cmath>

//
//
//	NUMERICAL INTEGRATION FUNCTIONS for IVP SOLVER
//
//

namespace CRMCatheterModel {

    void CRMIntegrand_dyn(double s, const StateVector& in_x, const CRMIntegrandParams in_Params, const double in_nL[3],
                      StateDerivativeVector& out_xdot) {

        double Length;
        double deltalambdainv;
        double fcum[3], ustardot[3];  //  We are assuming Kdot=0.0 (K=const)
        auto maxabs = [](const double* data, int count) {
            double maxv = 0.0;
            for (int i = 0; i < count; ++i) {
                maxv = std::max(maxv, std::abs(data[i]));
            }
            return maxv;
        };
        // copy inputs and parameters to local variables
        Length = in_Params.Li;
        deltalambdainv = in_Params.dlambdainv;
        auto& K = in_Params.K;
        auto& Kinv = in_Params.Kinv;
        auto& ustar = in_Params.ustar;
        auto& l = in_Params.l;
        const auto& fcumlambda = *in_Params.fcumlambda;
        for (int i = 0; i < 3; i++) {
            ustardot[i] = 0.0; //in_ustardot[i]; // we assume ustardot=0.0 since our rest shape model is piecewise constant curvature
        }

        // for simplicity, create aliases
        auto& u = in_x._u;
        auto& R = in_x._R;
        auto& udot = out_xdot._u;
#ifndef ANALYTICAL_SE3_STEP
        auto& p = in_x._p;				// we will not need this for analytical calculation
        auto& pdot = out_xdot._p;		// we will not need this for analytical calculation
        auto& Rdot = out_xdot._R;		// we will not need this for analytical calculation
#endif


        // calculate interpolated value of fcum
        double lambda = Length - s;
        double ix = lambda * deltalambdainv;
        double ird_f = floor(ix); // index for round down  -- doubleing point
        if (ird_f < 0) ird_f = 0;
        int ird = (int)ird_f;	//    integer index
        double iru_f = ceil(ix); 	// index for round up  -- doubleing point
        if (iru_f > in_Params.no_fcum_steps) iru_f = in_Params.no_fcum_steps;
        int iru = (int)iru_f;	//    integer index
        double ixmird = ix - ird;    // weight for interpolation
        double irumix = iru - ix;	// weight for interpolation
        for (int i = 0; i < 3; i++) {
            fcum[i] = fcumlambda[iru](i) * ixmird + fcumlambda[ird](i) * irumix;
        }

        // add the tip force to fcum
        for (int i = 0; i < 3; i++) {
            fcum[i] += in_Params.ftip[i];
        }

        double nL_spatial[3];
        mMult_AB<3,3,1>(R, in_nL, nL_spatial);
        // add the tip force to fcum
        for (int i = 0; i < 3; i++) {
            fcum[i]		+=	nL_spatial[i];
        }


        // calculate u_hat
        double u_hat[9];
        wHat(u, u_hat);

        // udot = ustardot - Kinv*((um*K+Kdot)*(u-ustar_s) + e3m*R'*intf + R'*l); % udot
        //
        //   e3hat*R' = [ -r12 -r22 -r32; r11 r21 r31; 0 0 0];
        double e3hatRT[9];
        e3hatRT[0] = -R[1];   e3hatRT[1] = -R[4];   e3hatRT[2] = -R[7];
        e3hatRT[3] = R[0];    e3hatRT[4] = R[3];    e3hatRT[5] = R[6];
        e3hatRT[6] = 0.0;     e3hatRT[7] = 0.0;     e3hatRT[8] = 0.0;
        double e3hatRTfcum[3];
        mMult_AB<3, 3, 1>(e3hatRT, fcum, e3hatRTfcum);			//  e3m*R'*intf
        double RTl[3];
        mMult_ATB<3, 3, 1>(R, l, RTl);								// R'*l
        double umustar[3];
        mSub_AB<3, 1>(u, ustar->data(), umustar);					// (u-ustar_s)
        double Kumustar[3], uhatKumustar[3];
        mMult_AB<3, 3, 1>(K->data(), umustar, Kumustar);
        mMult_AB<3, 3, 1>(u_hat, Kumustar, uhatKumustar); 		 	//(um*K+Kdot)*(u-ustar_s)  assuming Kdot=0
        double sumterm[3];
        mAdd_ABC<3, 1>(uhatKumustar, e3hatRTfcum, RTl, sumterm);	// ((um*K+Kdot)*(u-ustar_s) + e3m*R'*intf + R'*l)
        double KinvSum[3];
        mMult_AB<3, 3, 1>(Kinv->data(), sumterm, KinvSum);			// Kinv*((um*K+Kdot)*(u-ustar_s) + e3m*R'*intf + R'*l)
        mSub_AB<3, 1>(ustardot, KinvSum, udot);					// udot = ustardot - Kinv*((um*K+Kdot)*(u-ustar_s) + e3m*R'*intf + R'*l);

        if (const char* debug = std::getenv("CRM_DEBUG_BVP_SCALE")) {
            if (std::strcmp(debug, "1") == 0) {
                static int log_count = 0;
                const double u_max = maxabs(u, 3);
                const double R_max = maxabs(R, 9);
                const double fcum_max = maxabs(fcum, 3);
                const double udot_max = maxabs(udot, 3);
                const bool bad = !std::isfinite(u_max) || !std::isfinite(R_max) || !std::isfinite(fcum_max) || !std::isfinite(udot_max)
                                 || u_max > 1e6 || R_max > 1e6 || fcum_max > 1e6 || udot_max > 1e6;
                if (bad && log_count < 5) {
                    ++log_count;
                    std::cout << "[CRM_DEBUG_BVP_SCALE] CRMIntegrand_dyn large/non-finite"
                              << " s=" << s
                              << " |u|=" << u_max
                              << " |R|=" << R_max
                              << " |fcum|=" << fcum_max
                              << " |udot|=" << udot_max
                              << " |nL|=" << maxabs(in_nL, 3)
                              << "\n";
                }
            }
        }

#ifndef ANALYTICAL_SE3_STEP
        // we will not need these for analytical calculation
        // Rdot = R*u_hat
        mMult_AB<3, 3, 3>(R, u_hat, Rdot);
        // pdot = R*e3,
        for (int i = 0; i < 3; i++) {
            pdot[i] = R[i * 3 + 2];
        }
#endif

        //xdot(1:3) = R*e3;             % pdot
        //xdot(4:12) = reshape(R*um,9,1); % Rdot   --- Note that matlab code reshapes in column major order while we are saving in row major order
        //xdot(13:15) = ustardot - Kinv*((um*K+Kdot)*(u-ustar_s) + e3m*R'*intf + R'*l); % udot
        //  note: the sample code has matlab indexing starting from 1 to 15

    }



    // Forward declaration for RK4 fallback in ABM4_dyn.
    template <typename StVecType, typename ParamType>
    void RK4_step_dyn(const StVecType& in_x_n, const double t_n, const double h,
                  const ParamType in_Params, const double in_nL[3],
                  StVecType& out_x_np1, StDerivativeVectType<StVecType>& out_xdot_n);

    //function [x_1toN] = ABM4(x_0, t_0, N, h, Integrand, initmethod)
    template <typename StVecType, typename ParamType>
    void ABM4_dyn(const StVecType& in_x_0, const double t_0, const int N, const double h,
              const ParamType in_Params, const double in_nL[3], int32_t in_no_locmarkers,
              const bool CalculateEnergy,
              const bool FinalValueOnly, const double in_LocMarkers[], int& inout_NextLocMarkerIdx,
              StVecType& out_x_N, double& out_PotentialEnergy, double out_p_atLocMarkers[][3]) {

        using _SVT = StVecType;
        using _DVT = StDerivativeVectType<StVecType>;

        _SVT x_nm3;
        _SVT x_nm2;
        _SVT x_nm1;
        _SVT x_n(in_x_0);
        _SVT x_np1;
        double t_n;
        _DVT xdot_nm3;
        _DVT xdot_nm2;
        _DVT xdot_nm1;
        _DVT xdot_n;

        out_PotentialEnergy = 0.0;
        double Kun[3], Kunp1[3], unTKun, unp1TKun1p, unTKunp1, unp1TKun;
        int NextLocMarkerIdx = inout_NextLocMarkerIdx;
        auto& LocMarkers = in_LocMarkers;  // create an alias
        double delta_n_overh, delta_np1_overh;
        double gTpsum, mgTpmid;
        auto nonfinite = [](const double* data, int count) {
            for (int i = 0; i < count; ++i) {
                if (!std::isfinite(data[i])) return true;
            }
            return false;
        };
        auto maxabs = [](const double* data, int count) {
            double maxv = 0.0;
            for (int i = 0; i < count; ++i) {
                maxv = std::max(maxv, std::abs(data[i]));
            }
            return maxv;
        };
        auto clamp_u = [&](double u[3]) {
            const double kMaxU = 1e3;
            const double u_max = maxabs(u, 3);
            if (u_max > kMaxU && std::isfinite(u_max)) {
                const double scale = kMaxU / u_max;
                for (int i = 0; i < 3; ++i) {
                    u[i] *= scale;
                }
                return true;
            }
            return false;
        };

        // initialize the iteration items
        t_n = t_0;

        for (int idx = 0; idx < N; idx++) {

            if (idx < 3) {  // RK2 initialization steps
                RK2_step_dyn(x_n, t_n, h, in_Params, in_nL, x_np1, xdot_n);
            }
            else { 		 // ABM4 steps
                ABM4_step_dyn(x_n, t_n, h, xdot_nm1, xdot_nm2, xdot_nm3, x_nm1, x_nm2, x_nm3, in_Params, in_nL, x_np1, xdot_n);
            }

            if (nonfinite(x_np1._p, 3) || nonfinite(x_np1._R, 9) || nonfinite(x_np1._u, 3)) {
                if (const char* debug = std::getenv("CRM_DEBUG_BVP_SCALE")) {
                    if (std::strcmp(debug, "1") == 0) {
                        const double k_max = in_Params.K ? maxabs(in_Params.K->data(), 9) : 0.0;
                        const double kinv_max = in_Params.Kinv ? maxabs(in_Params.Kinv->data(), 9) : 0.0;
                        const double ustar_max = in_Params.ustar ? maxabs(in_Params.ustar->data(), 3) : 0.0;
                        std::cout << "[CRM_DEBUG_BVP_SCALE] ABM4_dyn non-finite state at step " << idx
                                  << " t=" << t_n << " h=" << h
                                  << " |x_n.p|=" << maxabs(x_n._p, 3)
                                  << " |x_n.R|=" << maxabs(x_n._R, 9)
                                  << " |x_n.u|=" << maxabs(x_n._u, 3)
                                  << " |xdot_n|=" << xdot_n.absmax()
                                  << " |nL|=" << maxabs(in_nL, 3)
                                  << " |K|=" << k_max
                                  << " |Kinv|=" << kinv_max
                                  << " |ustar|=" << ustar_max
                                  << ", attempting RK4 fallback\n";
                    }
                }
                RK4_step_dyn(x_n, t_n, h, in_Params, in_nL, x_np1, xdot_n);
                if (nonfinite(x_np1._p, 3) || nonfinite(x_np1._R, 9) || nonfinite(x_np1._u, 3)) {
                    if (const char* debug = std::getenv("CRM_DEBUG_BVP_SCALE")) {
                        if (std::strcmp(debug, "1") == 0) {
                            const double k_max = in_Params.K ? maxabs(in_Params.K->data(), 9) : 0.0;
                            const double kinv_max = in_Params.Kinv ? maxabs(in_Params.Kinv->data(), 9) : 0.0;
                            const double ustar_max = in_Params.ustar ? maxabs(in_Params.ustar->data(), 3) : 0.0;
                            std::cout << "[CRM_DEBUG_BVP_SCALE] RK4 fallback non-finite at step " << idx
                                      << " t=" << t_n << " h=" << h
                                      << " |x_n.p|=" << maxabs(x_n._p, 3)
                                      << " |x_n.R|=" << maxabs(x_n._R, 9)
                                      << " |x_n.u|=" << maxabs(x_n._u, 3)
                                      << " |xdot_n|=" << xdot_n.absmax()
                                      << " |nL|=" << maxabs(in_nL, 3)
                                      << " |K|=" << k_max
                                      << " |Kinv|=" << kinv_max
                                      << " |ustar|=" << ustar_max
                                      << "\n";
                        }
                    }
                    constexpr double kDivergenceValue = 1e6;
                    for (int i = 0; i < 3; ++i) {
                        x_np1._p[i] = kDivergenceValue;
                        x_np1._u[i] = kDivergenceValue;
                    }
                    for (int i = 0; i < 9; ++i) {
                        x_np1._R[i] = (i == 0 || i == 4 || i == 8) ? 1.0 : 0.0;
                    }
                    out_x_N = x_np1;
                    inout_NextLocMarkerIdx = NextLocMarkerIdx;
                    return;
                }
            }

            if (const char* clamp = std::getenv("CRM_CLAMP_U")) {
                if (std::strcmp(clamp, "1") == 0) {
                    if (clamp_u(x_np1._u)) {
                        if (const char* debug = std::getenv("CRM_DEBUG_BVP_SCALE")) {
                            if (std::strcmp(debug, "1") == 0) {
                                std::cout << "[CRM_DEBUG_BVP_SCALE] ABM4_dyn clamped |u| at step " << idx
                                          << " t=" << t_n << "\n";
                            }
                        }
                    }
                }
            }

            //Project_State_to_Manifold(x_np1);
            // increment "time"
            t_n = t_n + h;

            if (CalculateEnergy) {
                //
                // Elastic Potential Energy: \int u^T K u \approx ( u_n^T K u_n + ( u_n^T K u_np1 + u_np1^T K u_n ) /2 + u_np1^T K u_np1 ) /3
                //
                mMult_AB<3, 3, 1>(in_Params.K->data(), x_n._u, Kun);
                mMult_AB<3, 3, 1>(in_Params.K->data(), x_np1._u, Kunp1);
                mMult_ATB<3, 1, 1>(x_n._u, Kun, &unTKun);
                mMult_ATB<3, 1, 1>(x_n._u, Kunp1, &unTKunp1);
                mMult_ATB<3, 1, 1>(x_np1._u, Kun, &unp1TKun);
                mMult_ATB<3, 1, 1>(x_np1._u, Kunp1, &unp1TKun1p);
                out_PotentialEnergy += h * (unTKun + 0.5 * (unTKunp1 + unp1TKun) + unp1TKun1p) / 3.0;
                //
                // Gravitational Potential Energy: mass * gravity * height = (\rho * h) * (- g^T p_mid)
                //
                gTpsum = 0.0;
                for (int ix = 0; ix < 3; ix++) gTpsum += in_Params.g[ix] * (x_n._p[ix] + x_np1._p[ix]);
                mgTpmid = -0.5 * gTpsum;
                out_PotentialEnergy += in_Params.rho * h * mgTpmid;
            }

            if (!FinalValueOnly) {
                // are there any localization markers?  If so, calculate their positions
                // remember, t_n has already been incremented
                while ((NextLocMarkerIdx < in_no_locmarkers) && (LocMarkers[NextLocMarkerIdx] <= t_n)) {
                    delta_np1_overh = (t_n - LocMarkers[NextLocMarkerIdx]) / h;
                    delta_n_overh = 1.0 - delta_np1_overh;
                    out_p_atLocMarkers[NextLocMarkerIdx][0] = delta_np1_overh * (x_n._p[0]) + delta_n_overh * (x_np1._p[0]);
                    out_p_atLocMarkers[NextLocMarkerIdx][1] = delta_np1_overh * (x_n._p[1]) + delta_n_overh * (x_np1._p[1]);
                    out_p_atLocMarkers[NextLocMarkerIdx][2] = delta_np1_overh * (x_n._p[2]) + delta_n_overh * (x_np1._p[2]);
                    NextLocMarkerIdx++;
                }
            }

            // update the iteration items
            x_nm3 = x_nm2;
            x_nm2 = x_nm1;
            x_nm1 = x_n;
            x_n = x_np1;
            xdot_nm3 = xdot_nm2;
            xdot_nm2 = xdot_nm1;
            xdot_nm1 = xdot_n;

        }

        // Copy final value
        out_x_N = x_n;
        inout_NextLocMarkerIdx = NextLocMarkerIdx;

    }


    // [x_np1, xdot_n, xdot_nm1, xdot_nm2] = ABM4_step(x_n, t_n, xdot_nm1, xdot_nm2, xdot_nm3, h, Integrand)
    template <typename StVecType, typename ParamType>
    void ABM4_step_dyn(const StVecType& in_x_n, double t_n, double h,
                   const StDerivativeVectType<StVecType>& in_xdot_nm1, const StDerivativeVectType<StVecType>& in_xdot_nm2, const StDerivativeVectType<StVecType>& in_xdot_nm3,
                   const StVecType& in_x_nm1, const StVecType& in_x_nm2, const StVecType& in_x_nm3,
                   const ParamType in_Params, const double in_nL[3],
                   StVecType& out_x_np1, StDerivativeVectType<StVecType>& out_xdot_n) {

        using _SVT = StVecType;
        using _DVT = StDerivativeVectType<StVecType>;

        const double P_COEFF_N = 55.0 / 24.0, P_COEFF_Nm1 = -59.0 / 24.0, P_COEFF_Nm2 = 37.0 / 24.0, P_COEFF_Nm3 = -9.0 / 24.0;  // AB4 Predictor Coefficients
        const double C_COEFF_Np1 = 9.0 / 24.0, C_COEFF_N = 19.0 / 24.0, C_COEFF_Nm1 = -5.0 / 24.0, C_COEFF_Nm2 = 1.0 / 24.0;     // AM4 Corrector Coefficients
        _SVT x_n(in_x_n);       		// from input
        _SVT x_nm1(in_x_nm1);       	// from input
        _SVT x_nm2(in_x_nm2);       	// from input
        _SVT x_nm3(in_x_nm3);       	// from input
        _SVT x_np1_hat;    				// intermediate
        _DVT xdot_np1_hat;				// intermediate
        auto& xdot_n = out_xdot_n;
        _DVT xdot_nm1(in_xdot_nm1);  	// from input
        _DVT xdot_nm2(in_xdot_nm2);  	// from input
        _DVT xdot_nm3(in_xdot_nm3);  	// from input


        //ABM4_STEP_STEP1:
        CRMIntegrand_dyn(t_n, x_n, in_Params, in_nL, xdot_n);
        x_np1_hat = x_n + h * (P_COEFF_N * xdot_n + P_COEFF_Nm1 * xdot_nm1 + P_COEFF_Nm2 * xdot_nm2 + P_COEFF_Nm3 * xdot_nm3);
#ifdef ANALYTICAL_SE3_STEP
        //      calculate R_np1_hat and p_np1_hat analytically, without numerical integration
		double u_n_pred[3];
		for (int i = 0; i < 3; i++) u_n_pred[i] = (P_COEFF_N * x_n._u[i] + P_COEFF_Nm1 * x_nm1._u[i] + P_COEFF_Nm2 * x_nm2._u[i] + P_COEFF_Nm3 * x_nm3._u[i]);
		SE3_Analytical_Step(x_n._R, x_n._p, u_n_pred, h, x_np1_hat._R /*R_np1_hat*/, x_np1_hat._p /*p_np1_hat*/);
        if (const char* clamp = std::getenv("CRM_CLAMP_SE3")) {
            if (std::strcmp(clamp, "1") == 0) {
                Project_State_to_Manifold(x_np1_hat);
            }
        }
#endif
        //ABM4_STEP_STEP2:
        CRMIntegrand_dyn(t_n + h, x_np1_hat, in_Params, in_nL, xdot_np1_hat);
        out_x_np1 = x_n + h * (C_COEFF_Np1 * xdot_np1_hat + C_COEFF_N * xdot_n + C_COEFF_Nm1 * xdot_nm1 + C_COEFF_Nm2 * xdot_nm2);
#ifdef ANALYTICAL_SE3_STEP
        //      calculate R_np1 and p_np1 analytically, without numerical integration
		double u_n_corr[3];
		for (int i = 0; i < 3; i++) u_n_corr[i] = (C_COEFF_Np1 * x_np1_hat._u[i] + C_COEFF_N * x_n._u[i] + C_COEFF_Nm1 * x_nm1._u[i] + C_COEFF_Nm2 * x_nm2._u[i]);
		SE3_Analytical_Step(x_n._R, x_n._p, u_n_corr, h, out_x_np1._R /*R_np1*/, out_x_np1._p /*p_np1*/);
        if (const char* clamp = std::getenv("CRM_CLAMP_SE3")) {
            if (std::strcmp(clamp, "1") == 0) {
                Project_State_to_Manifold(out_x_np1);
            }
        }
#endif

    }


    //[x_np1, xdot_n] = RK2_step(x_n, t_n, h, Integrand)
    template <typename StVecType, typename ParamType>
    void RK2_step_dyn(const StVecType& in_x_n, const double t_n, const double h,
                  const ParamType in_Params, const double in_nL[3],
                  StVecType& out_x_np1, StDerivativeVectType<StVecType>& out_xdot_n) {

        using _SVT = StVecType;
        using _DVT = StDerivativeVectType<StVecType>;

        _SVT x_n(in_x_n);       // from input
        _DVT k1;				// intermediate
        _DVT k2oh;				// intermediate
        _SVT x_n_p_k1o2;		// intermediate
        auto& xdot_n = out_xdot_n;// for output

        //RK2_STEP_STEP1:
        CRMIntegrand_dyn(t_n, x_n, in_Params, in_nL, xdot_n);
        k1 = h * xdot_n;
        x_n_p_k1o2 = x_n + k1 * 0.5;
#ifdef ANALYTICAL_SE3_STEP
        // we will calculate R_n_p_k1o2 and p_n_p_k1o2 analytically, without numerical integration
		SE3_Analytical_Step(x_n._R, x_n._p, x_n._u, h * 0.5, x_n_p_k1o2._R, x_n_p_k1o2._p);
        if (const char* clamp = std::getenv("CRM_CLAMP_SE3")) {
            if (std::strcmp(clamp, "1") == 0) {
                Project_State_to_Manifold(x_n_p_k1o2);
            }
        }
#endif

        //RK2_STEP_STEP2:
        CRMIntegrand_dyn(t_n + h * 0.5, x_n_p_k1o2, in_Params, in_nL, k2oh);
        out_x_np1 = x_n + h * k2oh;
#ifdef ANALYTICAL_SE3_STEP
        // we will calculate R_np1 and p_np1 analytically, without numerical integration
		SE3_Analytical_Step(x_n._R, x_n._p, x_n_p_k1o2._u/*u_np1half*/, h, out_x_np1._R, out_x_np1._p);
        if (const char* clamp = std::getenv("CRM_CLAMP_SE3")) {
            if (std::strcmp(clamp, "1") == 0) {
                Project_State_to_Manifold(out_x_np1);
            }
        }
#endif

    }

    //[x_np1, xdot_n] = RK4_step(x_n, t_n, h, Integrand)
    template <typename StVecType, typename ParamType>
    void RK4_step_dyn(const StVecType& in_x_n, const double t_n, const double h,
                  const ParamType in_Params, const double in_nL[3],
                  StVecType& out_x_np1, StDerivativeVectType<StVecType>& out_xdot_n) {

        using _SVT = StVecType;
        using _DVT = StDerivativeVectType<StVecType>;

        _SVT x_n(in_x_n);
        _DVT k1;
        _DVT k2;
        _DVT k3;
        _DVT k4;
        _SVT x_n_p_k1o2;
        _SVT x_n_p_k2o2;
        _SVT x_n_p_k3;
        auto& xdot_n = out_xdot_n;

        CRMIntegrand_dyn(t_n, x_n, in_Params, in_nL, k1);
        x_n_p_k1o2 = x_n + k1 * (h * 0.5);
        CRMIntegrand_dyn(t_n + h * 0.5, x_n_p_k1o2, in_Params, in_nL, k2);

        x_n_p_k2o2 = x_n + k2 * (h * 0.5);
        CRMIntegrand_dyn(t_n + h * 0.5, x_n_p_k2o2, in_Params, in_nL, k3);

        x_n_p_k3 = x_n + k3 * h;
        CRMIntegrand_dyn(t_n + h, x_n_p_k3, in_Params, in_nL, k4);

        out_x_np1 = x_n + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        xdot_n = k1;

#ifdef ANALYTICAL_SE3_STEP
        // Improve SE3 update using average curvature
        double u_avg[3];
        for (int i = 0; i < 3; ++i) {
            u_avg[i] = 0.5 * (x_n._u[i] + out_x_np1._u[i]);
        }
        SE3_Analytical_Step(x_n._R, x_n._p, u_avg, h, out_x_np1._R, out_x_np1._p);
        if (const char* clamp = std::getenv("CRM_CLAMP_SE3")) {
            if (std::strcmp(clamp, "1") == 0) {
                Project_State_to_Manifold(out_x_np1);
            }
        }
#endif
    }

}
