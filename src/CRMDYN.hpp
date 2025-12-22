#pragma once
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <vector>
#include "CRM.hpp"
#include "CRM_BVPIVP_APIDeclarations.hpp"

namespace CRMCatheterModel {

#define NUM_COIL_STATES 18

    void CRMDYNSolverIVP_Prep(
        int32_t in_no_flex_seg, int32_t in_no_rigid_seg, int32_t in_no_act_set, int32_t in_no_locmarkers, int32_t in_no_fcum_steps,
        double in_x_0[NUM_STATES], double in_IntegrationStepSize, double in_Li, double in_dlambdainv,
        double in_rho[], const CatheterSegmentType in_SegmentTypes[], double in_SegEndLambdas[], double in_LocMarkerLambdas[],
        double in_K[][9], double in_Kinv[][9], double in_ustar[][3],
        double in_MagMoment[][3], double in_fcumlambda[][3], double in_CoilAlignmentTurnAreaMatrix[][9],
        double in_B0[3], double in_g[3], const double in_actMass[], double in_actInertia[][9],
        double in_damping[][6], double in_delta_t,
        double in_v_L_pre[][3], double in_w_L_pre[][3], double in_p_pre[][3], double in_R_pre[][9],
        double in_mL[][3], double in_nL[][3], bool in_FinalValueOnly, CRMIVPCoreParams &out_CoreParams);

    struct DYNNLEqnParams : CRMIVPCoreParams {
        DYNNLEqnParams(int32_t no_flex, int32_t no_rigid, int32_t no_act, int32_t no_loc, int32_t no_fcums) :
                CRMIVPCoreParams(no_flex, no_rigid, no_act, no_loc, no_fcums) {};
        ContactModeType ContactMode;
        Eigen::Vector3d TipConstraintPoint;
        double TipForce[3];
        double ftip_initialguess[3];
        double xf[NUM_STATES];
    };

    void DYNNLEquation(double in_x[], double out_y[], DYNNLEqnParams& Params, double out_u0[3], double out_tau[NUM_ACT_SET*3]);

    void DynamicsBVP(CRMShootingMethodParams& in_Params, const double xf[NUM_STATES],
                     double in_mL_initialguess[NUM_ACT_SET][3], double in_nL_initialguess[NUM_ACT_SET][3], double in_ftip_initialguess[3],
                     double out_u0[3], double out_mL[NUM_ACT_SET][3], double out_nL[NUM_ACT_SET][3], double out_tau[NUM_ACT_SET][3], double out_ftip[3], int& out_localmin);

    void DYNSolverIVP(CRMShootingMethodParams& in_Params, const double in_u0[3],
                      const double in_mL[NUM_ACT_SET][3], const double in_nL[NUM_ACT_SET][3], const double in_tau[NUM_ACT_SET][3], const double in_ftip[3],
                      bool in_FinalValueOnly,
                      double out_x_N[NUM_STATES], double out_coil_state[NUM_ACT_SET][NUM_COIL_STATES],
                      double out_p_atLocMarkers[][3]);

    CRMShootingMethodParams CRMDYNConstructShootingMethodParamSet(CRMCatheterModelParams CathParams, CatheterConfiguration CathConfig,
                                                                  double InsertionLength, double ActuationCurrents[NUM_ACT_SET][3],
                                                                  ContactModeType ContactMode,
                                                                  double TipConstraintPoint[3], double TipForce[3],
                                                                  double IntegrationStepSize, double ActInertia[NUM_ACT_SET][9],
                                                                  double in_v_L_pre[NUM_ACT_SET][3], double in_w_L_pre[NUM_ACT_SET][3], double in_p_pre[NUM_ACT_SET][3],
                                                                  double in_R_pre[NUM_ACT_SET][9], double in_damping[NUM_ACT_SET][6], double in_DELTA_T);
}