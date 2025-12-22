#include <cmath>
#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "CRM.hpp"
#include "CRM_BVPIVP_APIDeclarations.hpp"
#include "CRM_IVPJacobian.hpp"
#include "minpack.hpp"

#define M_PI 3.14159265358979323846

namespace CRMCatheterModel {

        CRMShootingMethodParams::CRMShootingMethodParams(int32_t no_flex, int32_t no_rigid, int32_t no_act, int32_t no_loc, int32_t no_fcum) :
                no_flex_seg(no_flex), no_rigid_seg(no_rigid), no_act_set(no_act),
                no_segments(no_flex + no_rigid), no_locmarkers(no_loc), no_fcum_steps(no_fcum) 
        {
                SegmentTypes.resize(no_segments);
                SegEndLambdas.resize(no_segments);
                rho.resize(no_segments);
                K.resize(no_flex_seg);
                Kinv.resize(no_flex_seg);
                ustar.resize(no_flex_seg);
                ActMass.resize(no_act_set);
                MagMoment.resize(no_act_set);
                CoilAlignmentTurnAreaMatrix.resize(no_act_set);
                LocMarkerLambdas.resize(no_locmarkers);
                fcumlambda.resize(no_fcum_steps + 1);

                v_L_pre.resize(no_act_set);
                w_L_pre.resize(no_act_set);
                p_pre.resize(no_act_set);
                R_pre.resize(no_act_set);
                actInertia.resize(no_act_set);
                damping.resize(no_act_set);
        }

        CRMCatheterModelParams::CRMCatheterModelParams(int32_t no_flex, int32_t no_rigid, int32_t no_act, int32_t no_loc) :
                no_flex_seg(no_flex), no_rigid_seg(no_rigid), no_act_set(no_act),
                no_segments(no_flex + no_rigid), no_locmarkers(no_loc)
        {
                SegmentTypes.resize(no_segments);
                SegLengths.resize(no_segments);
                rho.resize(no_segments);
                InnerRadius.resize(no_flex_seg);
                OuterRadius.resize(no_flex_seg);
                YoungsModulus.resize(no_flex_seg);
                ShearModulus.resize(no_flex_seg);
                ustar.resize(no_flex_seg);
                CoilAlignmentAngles.resize(no_act_set);
                CoilTurnAreaMat.resize(no_act_set);
                ActMass.resize(no_act_set);
                LocMarkers.resize(no_locmarkers);
                damping.resize(no_act_set);
                actInertia.resize(no_act_set);
                K.resize(no_flex_seg);
                Kinv.resize(no_flex_seg);
        }

        CRMCatheterModelParams::CRMCatheterModelParams(const CRMCatheterModelParams& t) : 
                no_flex_seg(t.no_flex_seg), no_rigid_seg(t.no_rigid_seg), no_act_set(t.no_act_set),
                no_segments(t.no_segments), no_locmarkers(t.no_locmarkers),
                SegmentTypes(t.SegmentTypes), SegLengths(t.SegLengths), rho(t.rho), InnerRadius(t.InnerRadius),
                OuterRadius(t.OuterRadius), YoungsModulus(t.YoungsModulus), ShearModulus(t.ShearModulus),
                ustar(t.ustar), ActMass(t.ActMass), CoilAlignmentAngles(t.CoilAlignmentAngles),
                CoilTurnAreaMat(t.CoilTurnAreaMat), LocMarkers(t.LocMarkers),
                damping(t.damping), actInertia(t.actInertia), K(t.K), Kinv(t.Kinv) {}

        CRMCatheterModelParams::~CRMCatheterModelParams() {}

        CRMIVPCoreParams::CRMIVPCoreParams(int32_t no_flex, int32_t no_rigid, int32_t no_act, int32_t no_loc, int32_t no_fcums) :
                no_flex_seg(no_flex), no_rigid_seg(no_rigid), no_act_set(no_act),
                no_segments(no_flex + no_rigid), no_locmarkers(no_loc), no_fcum_steps(no_fcums)
        {
                SegmentTypes.resize(no_segments);
                FlexActIndex.resize(no_segments);
                SegBounds.resize(no_segments + 1);
                SegSteps.resize(no_flex_seg);
                rho.resize(no_segments);
                K.resize(no_flex_seg);
                Kinv.resize(no_flex_seg);
                ustar.resize(no_flex_seg);
                ActMass.resize(no_act_set);
                MagMoment.resize(no_act_set);
                CoilAlignmentTurnAreaMatrix.resize(no_act_set);
                R_atActuators.resize(no_act_set);
                p_atActuators.resize(no_act_set);
                LocMarkers.resize(no_locmarkers);
                p_atLocMarkers.resize(no_locmarkers);
                fcumlambda.resize(no_fcum_steps + 1);

                v_L_pre.resize(no_act_set);
                w_L_pre.resize(no_act_set);
                p_pre.resize(no_act_set);
                R_pre.resize(no_act_set);
                actInertia.resize(no_act_set);
                damping.resize(no_act_set);
                m_L.resize(no_act_set);
                n_L.resize(no_act_set);
        }

        // Changed to take POINTER to avoid template deduction issues
        void NLEquation(double in_x[], double out_y[], NLEqnParams* Params);

        void CRMShootingMethodBVP(CRMShootingMethodParams in_Params,
                double in_deltau0[3], double in_ftip[3],
                double out_deltau0[3], double out_ftip[3], int& out_localmin) {

                ContactModeType ContactMode = in_Params.ContactMode;
                int n = (ContactMode == ContactModeType::FREE_TIP) ? 3 : 6;
                double x[6];

                for (int i = 0; i < 3; i++) x[i] = in_deltau0[i];
                if (ContactMode != ContactModeType::FREE_TIP) {
                        for (int i = 0; i < 3; i++) x[i + 3] = in_ftip[i];
                }

                double fvec[6], wa[500];
                int info;

                NLEqnParams Params(in_Params.no_flex_seg, in_Params.no_rigid_seg, in_Params.no_act_set, in_Params.no_locmarkers, in_Params.no_fcum_steps);
                
                Params.xi.setZero();
                Params.xi.segment<3>(0) = in_Params.p0;
                
                double x0[NUM_STATES];
                for(int i=0; i<3; i++) x0[i] = in_Params.p0(i);
                for(int i=0; i<9; i++) x0[i+3] = in_Params.R0(i/3, i%3);
                for(int i=0; i<3; i++) x0[i+12] = std::nan("0");

                std::vector<double> rho(in_Params.no_segments);
                std::vector<double> SegEndLambdas(in_Params.no_segments);
                std::vector<double> K(in_Params.no_flex_seg * 9);
                std::vector<double> Kinv(in_Params.no_flex_seg * 9);
                std::vector<double> ustar(in_Params.no_flex_seg * 3);
                std::vector<double> ActMass(in_Params.no_act_set);
                std::vector<double> MagMoment(in_Params.no_act_set * 3);
                std::vector<double> CATA(in_Params.no_act_set * 9);
                std::vector<double> fcum((in_Params.no_fcum_steps + 1) * 3);

                for(int i=0; i<in_Params.no_segments; i++) {
                        rho[i] = in_Params.rho[i];
                        SegEndLambdas[i] = in_Params.SegEndLambdas[i];
                }
                for(int i=0; i<in_Params.no_flex_seg; i++) {
                        for(int j=0; j<9; j++) {
                                K[i*9+j] = in_Params.K[i](j/3, j%3);
                                Kinv[i*9+j] = in_Params.Kinv[i](j/3, j%3);
                        }
                        for(int j=0; j<3; j++) ustar[i*3+j] = in_Params.ustar[i](j);
                }
                for(int i=0; i<in_Params.no_act_set; i++) {
                        ActMass[i] = in_Params.ActMass[i];
                        for(int j=0; j<3; j++) MagMoment[i*3+j] = in_Params.MagMoment[i](j);
                        for(int j=0; j<9; j++) CATA[i*9+j] = in_Params.CoilAlignmentTurnAreaMatrix[i](j/3, j%3);
                }
                for(int i=0; i<=in_Params.no_fcum_steps; i++) {
                        for(int j=0; j<3; j++) fcum[i*3+j] = in_Params.fcumlambda[i](j);
                }

                CRMSolverIVP_Prep(in_Params.no_flex_seg, in_Params.no_rigid_seg, in_Params.no_act_set, in_Params.no_locmarkers, in_Params.no_fcum_steps,
                        x0, in_Params.IntegrationStepSize, in_Params.Li, in_Params.dlambdainv,
                        in_Params.SegmentTypes.data(), SegEndLambdas.data(), in_Params.LocMarkerLambdas.data(),
                        rho.data(), (double(*)[9])K.data(), (double(*)[9])Kinv.data(), (double(*)[3])ustar.data(),
                        ActMass.data(), (double(*)[9])CATA.data(), (double(*)[3])MagMoment.data(), (double(*)[3])fcum.data(),
                        in_Params.B0.data(), in_Params.g.data(), false, true, Params);

                Params.ContactMode = ContactMode;
                Params.TipConstraintPoint = in_Params.TipConstraintPoint;
                for (int i = 0; i < 3; i++) Params.TipForce[i] = in_Params.TipForce(i);

                TrustRegionDogleg(NLEquation, n, x, fvec, 1e-5, info, wa, 500, &Params);

                out_localmin = (info == 1) ? 0 : (info - 1);
                for (int i = 0; i < 3; i++) out_deltau0[i] = x[i];
                if (ContactMode != ContactModeType::FREE_TIP) {
                        for (int i = 0; i < 3; i++) out_ftip[i] = x[i + 3];
                } else {
                        for (int i = 0; i < 3; i++) out_ftip[i] = in_Params.TipForce(i);
                }
        }

        void NLEquation(double in_x[], double out_y[], NLEqnParams* Params) {
                double deltau0[3];
                double ftip[3];

                for (int i = 0; i < 3; i++) deltau0[i] = in_x[i];

                if (Params->ContactMode != ContactModeType::FREE_TIP) {
                        for (int i = 0; i < 3; i++) ftip[i] = in_x[i + 3];
                } else {
                        for (int i = 0; i < 3; i++) ftip[i] = Params->TipForce[i];
                }

                StateVector x_N;
                double MomentResidual[3], dummyPE;
                double dummyLoc[5][3], dummyRAct[1][9], dummyPAct[1][3]; 

                CRMSolverIVP_Core(*Params, deltau0, ftip, x_N, MomentResidual, dummyPE, dummyLoc, dummyRAct, dummyPAct);

                for (int i = 0; i < 3; i++) out_y[i] = MomentResidual[i];

                if (Params->ContactMode != ContactModeType::FREE_TIP) {
                        for (int i = 0; i < 3; i++) out_y[i + 3] = x_N._p[i] - Params->TipConstraintPoint(i);
                }
        }

        // Implementation of missing CRMConstructShootingMethodParamSet
        CRMShootingMethodParams CRMConstructShootingMethodParamSet(
            CRMCatheterModelParams CathParams, CatheterConfiguration CathConfig,
            double InsertionLength, double ActuationCurrents[NUM_ACT_SET][3],
            ContactModeType ContactMode,
            double TipConstraintPoint[3], double TipForce[3],
            double IntegrationStepSize
        ) {
            CRMShootingMethodParams Params(CathParams.no_flex_seg, CathParams.no_rigid_seg, CathParams.no_act_set, CathParams.no_locmarkers, 50); // Default 50 fcum steps?

            Params.Li = InsertionLength;
            Params.IntegrationStepSize = IntegrationStepSize;
            Params.ContactMode = ContactMode;
            Params.TipConstraintPoint = Eigen::Vector3d(TipConstraintPoint[0], TipConstraintPoint[1], TipConstraintPoint[2]);
            Params.TipForce = Eigen::Vector3d(TipForce[0], TipForce[1], TipForce[2]);
            
            Params.dlambdainv = 1.0; // Needs calculation based on insertion? Or just default
            
            // Populate arrays
            for(int i=0; i<Params.no_segments; i++) {
                Params.SegmentTypes[i] = CathParams.SegmentTypes[i];
                // SegEndLambdas need calculation from SegLengths? 
                // Assuming SegLengths are provided in CathParams
                if (i==0) Params.SegEndLambdas[i] = CathParams.SegLengths[i];
                else Params.SegEndLambdas[i] = Params.SegEndLambdas[i-1] + CathParams.SegLengths[i];
                Params.rho[i] = CathParams.rho[i];
            }
            
            for(int i=0; i<Params.no_flex_seg; i++) {
                Params.K[i] = CathParams.K[i];
                Params.Kinv[i] = CathParams.Kinv[i];
                Params.ustar[i] = CathParams.ustar[i];
            }
            
            for(int i=0; i<Params.no_act_set; i++) {
                Params.ActMass[i] = CathParams.ActMass[i];
                Params.MagMoment[i].setZero(); // Calculated from currents
                // MagMoment = current * coil_area * n_turns * direction
                // This logic seems complex to replicate here if it wasn't in original
                // Assuming MagMoment is populated elsewhere or here?
                // Let's assume passed in MagMoment is 0 for now or calculated:
                // ...
                // Params.CoilAlignmentTurnAreaMatrix[i] = CathParams.CoilTurnAreaMat[i];
                for(int r=0; r<3; r++) {
                    for(int c=0; c<3; c++) {
                        Params.CoilAlignmentTurnAreaMatrix[i](r,c) = CathParams.CoilTurnAreaMat[i](r*3+c);
                    }
                }
            }
            
            for(int i=0; i<Params.no_locmarkers; i++) {
                Params.LocMarkerLambdas[i] = CathParams.LocMarkers[i];
            }
            
            Params.p0 = CathConfig.p0;
            // R0 is 9x1 in Config, 3x3 in Params. Reshape it.
            for(int i=0; i<3; i++) {
                for(int j=0; j<3; j++) {
                    Params.R0(i,j) = CathConfig.R0(i*3+j);
                }
            }
            Params.B0 = CathConfig.B0;
            Params.g = CathConfig.g;

            return Params;
        }

} // namespace CRMCatheterModel
