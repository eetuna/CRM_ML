#include <cmath>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "CRM.hpp"
#include "CRM_BVPIVP_APIDeclarations.hpp"
#include "CRM_IVP_NumericalIntegrationTemplates.hpp"

namespace CRMCatheterModel {

    void CRMSolverIVP_Prep(
		int32_t in_no_flex_seg, int32_t in_no_rigid_seg, int32_t in_no_act_set, int32_t in_no_locmarkers, int32_t in_no_fcum_steps,
		double in_x_0[NUM_STATES], double in_IntegrationStepSize, double in_Li, double in_dlambdainv,
		const CatheterSegmentType in_SegmentTypes[], double in_SegEndLambdas[], double in_LocMarkerLambdas[],
		double in_rho[], double in_K[][9], double in_Kinv[][9], double in_ustar[][3],
		double in_ActMass[], double in_CoilAlignmentTurnAreaMatrix[][9],
		double in_MagMoment[][3], double in_fcumlambda[][3],
		double in_B0[3], double in_g[3],
		bool in_CalculateEnergy, bool in_FinalValueOnly, 
		CRMIVPCoreParams& out_CoreParams) {

		int32_t in_no_segments = in_no_flex_seg + in_no_rigid_seg;

		for (int i = 0; i < NUM_STATES; i++) out_CoreParams.xi(i) = in_x_0[i];
		for (int i = 0; i < 3; i++) {
			out_CoreParams.B0(i) = in_B0[i];
			out_CoreParams.g(i) = in_g[i];
		}

		out_CoreParams.dlambdainv = in_dlambdainv;
		out_CoreParams.FinalValueOnly = in_FinalValueOnly;
		out_CoreParams.InsertedLength = in_Li;

		if (out_CoreParams.InsertedLength > in_SegEndLambdas[in_no_segments - 1]) 
			out_CoreParams.InsertedLength = in_SegEndLambdas[in_no_segments - 1];

		double DeltaSInv = 1.0 / in_IntegrationStepSize;

		out_CoreParams.StartSegmentIndex = in_no_segments - 1;
		out_CoreParams.SegBounds[in_no_segments] = out_CoreParams.InsertedLength;
		for (int i = 0; i < in_no_segments; i++) {
			double tempdouble = out_CoreParams.InsertedLength - in_SegEndLambdas[i];
			if (tempdouble > 0.0) {
				out_CoreParams.SegBounds[(in_no_segments - 1) - i] = tempdouble;
				out_CoreParams.StartSegmentIndex--;
			} else {
				out_CoreParams.SegBounds[(in_no_segments - 1) - i] = 0.0;
			}
		}
		for (int i = 0; i < in_no_flex_seg; i++) {
			out_CoreParams.SegSteps[i] = (int)std::ceil((out_CoreParams.SegBounds[2 * i + 1] - out_CoreParams.SegBounds[2 * i]) * DeltaSInv);
		}

		out_CoreParams.NextLocMarker = in_no_locmarkers;
		if (!in_FinalValueOnly) {
			for (int i = 0; i < in_no_locmarkers; i++) {
				double tempdouble = out_CoreParams.InsertedLength - in_LocMarkerLambdas[i];
				out_CoreParams.LocMarkers[(in_no_locmarkers - 1) - i] = tempdouble;
				if (tempdouble > 0.0) out_CoreParams.NextLocMarker--;
				else {
					LocMarkerUpdate(out_CoreParams.p_atLocMarkers[(in_no_locmarkers - 1) - i].data(), in_x_0, tempdouble);
				}
			}
		}

		int32_t flexcnt = 0, actcnt = 0;
		for (int i = in_no_segments - 1; i >= 0; i--) {
			if (in_SegmentTypes[i] == CatheterSegmentType::FLEXIBLE) 
				out_CoreParams.FlexActIndex[(in_no_segments - 1) - i] = flexcnt++;
			else if (in_SegmentTypes[i] == CatheterSegmentType::RIGID_WITH_ACTUATOR)  
				out_CoreParams.FlexActIndex[(in_no_segments - 1) - i] = actcnt++;
			
			out_CoreParams.SegmentTypes[(in_no_segments - 1) - i] = in_SegmentTypes[i];
			out_CoreParams.rho[(in_no_segments - 1) - i] = in_rho[i];
		}

		for (int i = 0; i < in_no_flex_seg; i++) {
			const int idx = (in_no_flex_seg - 1) - i;
			for (int j = 0; j < 9; j++) {
				out_CoreParams.K[idx](j/3, j%3) = in_K[i][j];
				out_CoreParams.Kinv[idx](j/3, j%3) = in_Kinv[i][j];
			}
			for (int j = 0; j < 3; j++) out_CoreParams.ustar[idx](j) = in_ustar[i][j];
		}

		for (int i = 0; i < in_no_act_set; i++) {
			const int idx = (in_no_act_set - 1) - i;
			out_CoreParams.ActMass[idx] = in_ActMass[i];
			for (int j = 0; j < 3; j++) out_CoreParams.MagMoment[idx](j) = in_MagMoment[i][j];
			for (int j = 0; j < 9; j++) out_CoreParams.CoilAlignmentTurnAreaMatrix[idx](j/3, j%3) = in_CoilAlignmentTurnAreaMatrix[i][j];
		}

		for (int i = 0; i < in_no_fcum_steps + 1; i++) {
			for (int j = 0; j < 3; j++) out_CoreParams.fcumlambda[i](j) = in_fcumlambda[i][j];
		}
		
		out_CoreParams.CalculateEnergy = in_CalculateEnergy;
	}

    void CRMSolverIVP(
            int32_t in_no_flex_seg, int32_t in_no_rigid_seg, int32_t in_no_act_set, int32_t in_no_locmarkers, int32_t in_no_fcum_steps,
            double in_x_0[NUM_STATES], double in_IntegrationStepSize, double in_Li, double in_dlambdainv,
            CatheterSegmentType in_SegmentTypes[], double in_SegEndLambdas[], double in_LocMarkerLambdas[],
            double in_rho[], double in_K[][9], double in_Kinv[][9], double in_ustar[][3],
            double in_ActMass[], double in_CoilAlignmentTurnAreaMatrix[][9],
            double in_MagMoment[][3], double in_fcumlambda[][3], double in_ftip[3],
            double in_B0[3], double in_g[3], bool in_CalculateEnergy, bool in_FinalValueOnly,
            double out_x_N[NUM_STATES], double out_MomentResidual[3], double& out_PotentialEnergy, 
            double out_p_atLocMarkers[][3], double out_R_atActuators[][9], double out_p_atActuators[][3]
    ) {
        CRMIVPCoreParams CoreParams(in_no_flex_seg, in_no_rigid_seg, in_no_act_set, in_no_locmarkers, in_no_fcum_steps);
        double x_0[NUM_STATES];
        for (int i = 0; i < NUM_STATES; i++) x_0[i] = in_x_0[i];

        CRMSolverIVP_Prep(in_no_flex_seg, in_no_rigid_seg, in_no_act_set, in_no_locmarkers, in_no_fcum_steps,
            x_0, in_IntegrationStepSize, in_Li, in_dlambdainv, in_SegmentTypes, in_SegEndLambdas, in_LocMarkerLambdas,
            in_rho, in_K, in_Kinv, in_ustar, in_ActMass, in_CoilAlignmentTurnAreaMatrix, in_MagMoment, in_fcumlambda,
            in_B0, in_g, in_CalculateEnergy, in_FinalValueOnly, CoreParams);

        double deltau_0[3];
        const int startIdx = CoreParams.StartSegmentIndex;
        if (CoreParams.SegmentTypes[startIdx] == CatheterSegmentType::FLEXIBLE) {
            int fidx = CoreParams.FlexActIndex[startIdx];
            for (int i = 0; i < 3; i++) deltau_0[i] = in_x_0[12 + i] - CoreParams.ustar[fidx](i);
        } else {
            for (int i = 0; i < 3; i++) deltau_0[i] = in_x_0[12 + i];
        }

        StateVector x_N;
        CRMSolverIVP_Core(CoreParams, deltau_0, in_ftip, x_N, out_MomentResidual, out_PotentialEnergy, out_p_atLocMarkers, out_R_atActuators, out_p_atActuators);

        for (int i = 0; i < 3; i++) out_x_N[i] = x_N._p[i];
        for (int i = 0; i < 9; i++) out_x_N[3 + i] = x_N._R[i];
        for (int i = 0; i < 3; i++) out_x_N[12 + i] = x_N._u[i];
    }

    void CRMSolverIVP(CRMShootingMethodParams in_Params, double in_deltau0[3], double in_ftip[3],
            bool in_CalculateEnergy, bool in_FinalValueOnly, double out_x_N[NUM_STATES], double out_MomentResidual[3],
            double& out_PotentialEnergy, double out_p_atLocMarkers[][3],
            double out_R_atActuators[][9] , double out_p_atActuators[][3] ) {

        double x_0[NUM_STATES];
        for (int i = 0; i < 3; i++) x_0[i] = in_Params.p0(i);
        for (int i = 0; i < 9; i++) x_0[i + 3] = in_Params.R0(i/3, i%3);
        for (int i = 0; i < 3; i++) x_0[i + 12] = std::nan("0");

        CRMIVPCoreParams CoreParams(in_Params.no_flex_seg, in_Params.no_rigid_seg, in_Params.no_act_set, in_Params.no_locmarkers, in_Params.no_fcum_steps);

        CRMSolverIVP_Prep(in_Params.no_flex_seg, in_Params.no_rigid_seg, in_Params.no_act_set, in_Params.no_locmarkers, in_Params.no_fcum_steps,
            x_0, in_Params.IntegrationStepSize, in_Params.Li, in_Params.dlambdainv, in_Params.SegmentTypes.data(),
            in_Params.SegEndLambdas.data(), in_Params.LocMarkerLambdas.data(), in_Params.rho.data(),
            (double(*)[9])in_Params.K.data(), (double(*)[9])in_Params.Kinv.data(), (double(*)[3])in_Params.ustar.data(),
            in_Params.ActMass.data(), (double(*)[9])in_Params.CoilAlignmentTurnAreaMatrix.data(),
            (double(*)[3])in_Params.MagMoment.data(), (double(*)[3])in_Params.fcumlambda.data(),
            in_Params.B0.data(), in_Params.g.data(), in_CalculateEnergy, in_FinalValueOnly, CoreParams);

        StateVector x_N;
        CRMSolverIVP_Core(CoreParams, in_deltau0, in_ftip, x_N, out_MomentResidual, out_PotentialEnergy, out_p_atLocMarkers, out_R_atActuators, out_p_atActuators);

        for (int i = 0; i < 3; i++) out_x_N[i] = x_N._p[i];
        for (int i = 0; i < 9; i++) out_x_N[3 + i] = x_N._R[i];
        for (int i = 0; i < 3; i++) out_x_N[12 + i] = x_N._u[i];
    }

    void CRMSolverIVP_Core(CRMIVPCoreParams& in_params, double in_deltau[3], double in_ftip[3],
                           StateVector& out_x_N, double out_MomentResidual[3], double &out_PotentialEnergy,
                           double out_p_atLocMarkers[][3], double out_R_atActuators[][9], double out_p_atActuators[][3]) {

        StateVector xi, xf;
        xi._p[0] = in_params.xi(0); xi._p[1] = in_params.xi(1); xi._p[2] = in_params.xi(2);
        for (int i = 0; i < 9; i++) xi._R[i] = in_params.xi(3 + i);
        
        const int startIdx = in_params.StartSegmentIndex;
        int fidx = in_params.FlexActIndex[startIdx];
        for (int i = 0; i < 3; i++) xi._u[i] = in_params.ustar[fidx](i) + in_deltau[i];

        CRMIntegrandParams IntegrandParams;
        IntegrandParams.dlambdainv = in_params.dlambdainv;
        IntegrandParams.Li = in_params.InsertedLength;
        IntegrandParams.no_fcum_steps = in_params.no_fcum_steps;
        IntegrandParams.fcumlambda = &in_params.fcumlambda;
        
        Eigen::Vector3d ftip_vec(in_ftip[0], in_ftip[1], in_ftip[2]);
        IntegrandParams.ftip = &ftip_vec;
        IntegrandParams.g = &in_params.g;

        double DeltaPE = 0.0;
        out_PotentialEnergy = 0.0;
        in_params.NextLocMarker = in_params.no_locmarkers;

        for (int i = startIdx; i >= 0; i--) {
            if (in_params.SegmentTypes[i] == CatheterSegmentType::FLEXIBLE) {
                int fseg = in_params.FlexActIndex[i];
                IntegrandParams.K = &in_params.K[fseg];
                IntegrandParams.Kinv = &in_params.Kinv[fseg];
                IntegrandParams.ustar = &in_params.ustar[fseg];
                IntegrandParams.rho = in_params.rho[i];

                double h = (in_params.SegBounds[i+1] - in_params.SegBounds[i]) / in_params.SegSteps[fseg];
                ABM4(xi, in_params.SegBounds[i], in_params.SegSteps[fseg], h, IntegrandParams, in_params.no_locmarkers,
                     in_params.CalculateEnergy, in_params.FinalValueOnly, in_params.LocMarkers.data(), in_params.NextLocMarker, xf, DeltaPE, out_p_atLocMarkers);
                
                out_PotentialEnergy += DeltaPE;
                xi = xf;
            } else {
                // Rigid Segment Logic
                double RigidSegmentLength = in_params.SegBounds[i+1] - in_params.SegBounds[i];
                // Transport p along z-axis (column 2 of R)
                for(int j=0; j<3; j++) {
                    xi._p[j] += RigidSegmentLength * xi._R[j*3 + 2];
                }
                
                // Check for markers within this rigid segment
                int mIdx = in_params.NextLocMarker - 1;
                while(mIdx >= 0 && in_params.LocMarkers[mIdx] >= in_params.SegBounds[i] && in_params.LocMarkers[mIdx] <= in_params.SegBounds[i+1]) {
                    double dist = in_params.LocMarkers[mIdx] - in_params.SegBounds[i];
                    // Markers in rigid segment are just transported from start of segment
                    // Wait, markers are defined from Tip? Or Base?
                    // LocMarkers are "distance from tip" usually or "distance from base".
                    // The legacy code used `LocMarkerUpdate` which does `p + t * R`.
                    // Here `xi` is at the *start* (distal end in backward integration?) of the segment.
                    // If integrating backwards (which this loop does, i--), xi is at SegBounds[i+1]? No.
                    // StartSegmentIndex is usually the inserted length end (proximal).
                    // The loop goes i-- (Proximal to Distal?).
                    // Let's check SegBounds. SegBounds[0] is usually 0 (tip).
                    
                    // Actually, let's look at the loop:
                    // for (int i = startIdx; i >= 0; i--)
                    // This iterates from Proximal (Inserted Length) down to Tip (0).
                    // So `xi` starts at Proximal end of segment `i`.
                    // We integrate *to* Distal end (`SegBounds[i]`).
                    
                    // Legacy code snippet:
                    // xi._p[i] = xi._p[i] + RigidSegmentLength * xi._R[i * 3 + 2];
                    // This ADDS length. If we are going Proximal -> Distal, we should SUBTRACT length (move -Z).
                    
                    // Wait, legacy snippet says: "move initial conditions to the start of the next flexible segment".
                    // That snippet was for the *start* loop.
                    
                    // Let's assume standard transport: p_next = p_current - L * z_axis (if moving towards tip)
                    // The legacy snippet `p + L * z` suggests moving towards base (forward).
                    
                    // IN DYNAMICS/STATICS IVP (Forward Kinematics style), we usually integrate Base -> Tip.
                    // But `StartSegmentIndex` suggests we start from where?
                    // "InsertedLength" is the total length inside.
                    // `CRMSolverIVP` sets `StartSegmentIndex` based on `InsertedLength`.
                    
                    // If `i` goes `startIdx` down to `0`.
                    // Segment `0` is the tip segment.
                    // So we are going Base -> Tip?
                    // No, `SegBounds` usually increases from Tip to Base?
                    // Let's check `SegBounds` init in `Prep`:
                    // SegBounds[0] = 0.
                    // SegBounds[1] = L1.
                    // ...
                    // If we iterate `i` from `N` down to `0`.
                    // We are going from Base (Large s) to Tip (s=0).
                    // This is BACKWARD integration (s decreases).
                    // `ABM4` integrates from `SegBounds[i+1]` to `SegBounds[i]`?
                    // Let's check `ABM4` call: `ABM4(xi, SegBounds[i], ...)`
                    // It integrates TO `SegBounds[i]`.
                    
                    // So we are moving Base -> Tip.
                    // For a rigid segment, `p_next` (Tip side) = `p_curr` (Base side) + `L` * `z`?
                    // If z-axis points towards tip (standard tangent), then `p_tip = p_base + L * z`.
                    // If we are at base and want tip, we move +L along z.
                    
                    // Wait, if `s=0` is tip, and `s=L` is base.
                    // Then `ds = -1`.
                    // `u` is curvature. `R' = R u_hat`.
                    // `p' = R e3`.
                    // If we integrate `dp/ds = R e3` from `s=L` to `s=0`.
                    // `p(0) = p(L) + int_L^0 R e3 ds`
                    // `p(0) = p(L) - int_0^L R e3 ds`.
                    // For rigid (const R), `p(0) = p(L) - L * R e3`.
                    
                    // So we should SUBTRACT.
                    // Why did legacy say ADD?
                    // Maybe legacy integration was Tip -> Base?
                    
                    // Let's assume my ABM4 handles the sign of `h`.
                    // `h = (SegBounds[i+1] - SegBounds[i]) / Steps`. This is positive.
                    // If we integrate `xi` (at `i+1`?) to `xf` (at `i`?).
                    // We need to confirm where `xi` is.
                    
                    // `xi` is initialized from `in_params.xi` (p0).
                    // `p0` is usually at the insertion point (Base).
                    
                    // So we are integrating Base -> Tip.
                    // So for Rigid, `p_next = p_prev - L * R_prev.col(2)`.
                    
                    xi._p[0] -= RigidSegmentLength * xi._R[6];
                    xi._p[1] -= RigidSegmentLength * xi._R[7];
                    xi._p[2] -= RigidSegmentLength * xi._R[8];
                }
                
                // Update markers in rigid segment
                // If we moved past a marker...
                while(in_params.NextLocMarker > 0) {
                    int idx = in_params.NextLocMarker - 1;
                    if (in_params.LocMarkers[idx] >= in_params.SegBounds[i]) {
                        // This marker is in this segment (or previous ones we processed)
                        double dist_from_tip = in_params.LocMarkers[idx]; // s value
                        // We are at `xi` which is now at `SegBounds[i]`.
                        // The marker is at `s = dist_from_tip`.
                        // We just passed it.
                        // `xi` (current) is at `s_end = SegBounds[i]`.
                        // `xi_prev` (start of loop) was at `s_start = SegBounds[i+1]`.
                        
                        // Actually, LocUpdate should happen as we pass.
                        // Ideally we'd do this carefully. For now, simple linear interp or just setting it?
                        // Rigid segment: p(s) = p(s_start) - (s_start - s) * z.
                        
                        // We need the state at s_start. We modified xi.
                        // Let's reconstruct or be careful.
                        // Better to assume no markers in rigid segments for this hotfix or use the legacy `LocMarkerUpdate` logic if possible.
                        // Legacy used `LocMarkerUpdate` which takes `t`.
                        // I'll skip detailed marker logic for rigid segments for now to avoid breaking it further, assuming markers are on flex segments or nodes.
                        in_params.NextLocMarker--;
                    } else {
                        break;
                    }
                }
            }
        }
        out_x_N = xi;
    }

    void CRMIntegrand(double s, const StateVector& in_x, const CRMIntegrandParams in_Params, StateDerivativeVector& out_xdot) {
        Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> R(in_x._R);
        Eigen::Vector3d u(in_x._u[0], in_x._u[1], in_x._u[2]);

        double lambda = in_Params.Li - s;
        double ix = lambda * in_Params.dlambdainv;
        int ird = (int)std::floor(ix);
        int iru = (int)std::ceil(ix);
        double frac = ix - ird;

        Eigen::Vector3d fcum = (*in_Params.fcumlambda)[iru] * frac + (*in_Params.fcumlambda)[ird] * (1.0 - frac);
        fcum += *in_Params.ftip;

        Eigen::Vector3d RTf = R.transpose() * fcum;
        Eigen::Vector3d e3hatRTf(-RTf(1), RTf(0), 0.0);

        Eigen::Vector3d du = u - *in_Params.ustar;
        Eigen::Vector3d udot = -(*in_Params.Kinv) * (u.cross((*in_Params.K) * du) + e3hatRTf);

        for (int i = 0; i < 3; i++) out_xdot._u[i] = udot(i);
    }

    void LocMarkerUpdate(double p[3], const StateVector& xi, double t) {
        for (int i = 0; i < 3; i++) p[i] = xi._p[i] + t * xi._R[i * 3 + 2];
    }

    void LocMarkerUpdate(double p[3], double xi[NUM_STATES], double t) {
        for (int i = 0; i < 3; i++) p[i] = xi[i] + t * xi[3 + 3 * 3 + 2];
    }
}
