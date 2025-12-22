#pragma once
//#include "/usr/local/MATLAB/R2022b/extern/include/mexAdapter.hpp"
//#include "/usr/local/MATLAB/R2022b/extern/include/mex.hpp"
//using matlab::mex::ArgumentList; // added
//using namespace matlab::data;	 // added


//#include "mex.h"
#include "mex.h"
#include <Eigen/Dense>

#include <cmath>
#include "CRMDYN.hpp"
#include "CRM.hpp"
#include <fstream>

#define M_PI 3.14159265358979323846
#define _USE_MATH_DEFINES


#define Nx (3*6+9 + NUM_STATES)

#define Ncoilstate (NUM_ACT_SET * (6+6+3+9))

using namespace CRMCatheterModel;

/* State equations. */
void compute_dx(double *dx, double t, double *x, double *u, double **p)
{

    /** Retrieve x. **/
    double v_L_pre[NUM_ACT_SET][3], w_L_pre[NUM_ACT_SET][3], nL_initialguess[NUM_ACT_SET][3], mL_initialguess[NUM_ACT_SET][3], pL_pre[NUM_ACT_SET][3], RL_pre[NUM_ACT_SET][9], xf_pre[NUM_STATES];
    // Define initial guesses to be used when solving boundary value problem
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; ++i) {
            v_L_pre[j][i] = x[i + j*3];
            w_L_pre[j][i] = x[NUM_ACT_SET*3 + i + j*3];
            mL_initialguess[j][i] = x[NUM_ACT_SET*6 + i + j*3];
            nL_initialguess[j][i] = x[NUM_ACT_SET*9 + i + j*3];
            pL_pre[j][i] = x[NUM_ACT_SET* 12 + i + j*3];
        }
        for (int i = 0; i < 9; ++i) {
            RL_pre[j][i] = x[NUM_ACT_SET *15 +i + j*9];
        }
    }

    for (int i = 0; i < NUM_STATES; ++i) {
        xf_pre[i] = x[i+Ncoilstate];
    }

    // *** Load parameters from file
    // This step would typically needs to be executed only once
    //   Physical Description of the Catheter
    /**
     * Note the matlab file should be under root folder
     */
    CRMCatheterModelParams CathParams = Load_CRMCatheterModelParams("data/catheter_params/CatheterParameterSet_1_new.txt");
    //   Catheter Configuration in spatial coordinates
    CatheterConfiguration CathConfig = Load_CatheterConfiguration("data/catheter_params/CatheterSpatialConfiguration_1.txt");

    size_t ix, jx;

    auto & OuterRadius = CathParams.OuterRadius;
    auto & InnerRadius = CathParams.InnerRadius;
    auto & YoungsModulus = CathParams.YoungsModulus;
    auto & ShearModulus = CathParams.ShearModulus;
    auto & CoilAlignmentAngles = CathParams.CoilAlignmentAngles;
    auto & CoilTurnAreaMat = CathParams.CoilTurnAreaMat;
    auto & ActMass = CathParams.ActMass;

    for (ix = 0; ix < CathParams.no_flex_seg; ix++){
        OuterRadius[ix] = p[2][0];  //parameter only has one value here, assumes isomorph
    }
    for (ix = 0; ix < CathParams.no_flex_seg; ix++){
        InnerRadius[ix] = p[2][1];
    }
    for (ix = 0; ix < CathParams.no_flex_seg; ix++){
        YoungsModulus[ix] = p[3][0];
    }
    for (ix = 0; ix < CathParams.no_flex_seg; ix++){
        ShearModulus[ix] = p[3][1];
    }
    for (ix = 0; ix < CathParams.no_act_set; ix++) {
        for (jx = 0; jx < 2; jx++) {
            CoilAlignmentAngles[ix][jx] = p[4][jx+2*ix];
        }
    }
    for (ix = 0; ix < CathParams.no_act_set; ix++) {
        CoilTurnAreaMat[ix][0] = p[5][0 + ix*3];
        CoilTurnAreaMat[ix][4] = p[5][1 + ix*3];
        CoilTurnAreaMat[ix][8] = p[5][2 + ix*3];
        // CoilTurnAreaMat[ix][1] = p[5][0 + ix*3];
        // CoilTurnAreaMat[ix][5] = p[5][1 + ix*3];
        // CoilTurnAreaMat[ix][6] = p[5][2 + ix*3];
    }
    for (ix = 0; ix < CathParams.no_act_set; ix++){
        ActMass[ix] = p[6][ix];
    }

    // *** Other External variables
    // specify if catheter is in free space or if the catheter tip is constrained to a contact point
    ContactModeType ContactMode = ContactModeType::FREE_TIP;
    // External point force (in spatial coordinates) applied at the tip of the catheter (\lambda = 0)  - unit: ??
    //   (this will be used when ContactMode == ContactModeType::FREE_TIP)
    double TipForce[3] = { 0.0, 0.0, 0.0 };
    // The spatial coordinates of the point where the catheter tip is constrained to be
    //   (this will be used when ContactMode == ContactModeType::FIXED_TIP)
    double TipConstraintPoint[3] = { 0.0, 0.0, 0.0 };


        /** Retrieve u. **/
    double ActuationCurrents[NUM_ACT_SET][3];
    for (int i = 0; i < NUM_ACT_SET; i++)	for (int j = 0; j < 3; j++)	ActuationCurrents[i][j] = u[i * 3 + j];
//    std::cout << "ActuationCurrents: " << ActuationCurrents[0][0] << " " << ActuationCurrents[0][1] << " " << ActuationCurrents[0][2] <<  std::endl;


    double InsertedLength =0;
    for (int i = 0; i < 3; ++i) {
        CathParams.SegLengths[i] = p[7][i];
        InsertedLength += p[7][i];
    }
//    std::cout << "after   CathParams.SegLengths: " << CathParams.SegLengths[0] << " " << CathParams.SegLengths[1] << " " << CathParams.SegLengths[2] <<  std::endl;

//    double InsertedLength = u[NUM_ACT_SET*3];
    /** Retrieve model.parameters. **/
    double damping[NUM_ACT_SET][6];

    for (int i = 0; i < NUM_ACT_SET; ++i) {
        damping[i][0] = damping[i][1] = p[0][0 + i*4];
        damping[i][2] = p[0][1 + i*4];
        damping[i][3] = damping[i][4] = p[0][2 + i*4];
        damping[i][5] = p[0][3 + i*4];
    }

//    std::cout <<"damping_ w: " << damping_[3] << " " << damping_[4] << " " << damping_[5] << std::endl;

    double DELTA_T = p[1][0];

    // *** Numerical Computation Params
    // Stepsize used in numerical integration along the length of the catheter during IVP - unit: mm
    double IntegrationStepSize = 0.2;

    // *** Storage for storing localization marker positions and actuation coil orientations
    double (*ReportedMarkerPos)[3] = new double [CathParams.no_locmarkers][3];
//    double (*ReportedCoilOrient)[9] = new double[CathParams.no_act_set][9];


    // Define initial guesses to be used when solving boundary value problem
    // initial guess for the delta_curvature at the catheter base ( u0 = deltau0 + ustar0 )
    // initial guess for the contstraint force at the catheter tip (this will be used when ContactMode == ContactModeType::FIXED_TIP)
    double ftip_initialguess[3] = { 0.0, 0.0, 0.0 };

    // numerical nonlinear equation solver diagnostic outputs
    int localmin;

    /**
     * These are hard coded, need to revise these later
     */
    // we are adding the tubing mass of the coil section to the total mass of actuator unit(kg * mm^2)
    double ActInertia[NUM_ACT_SET][9];
    for (int i = 0; i < NUM_ACT_SET; ++i)
    {
        double I_zz = 0.5 * (CathParams.ActMass[i]) * (CathParams.OuterRadius[0] * CathParams.OuterRadius[0] + CathParams.InnerRadius[0] * CathParams.InnerRadius[0]);
        double I_xx = 0.25 * (CathParams.ActMass[i]) * (CathParams.OuterRadius[0] * CathParams.OuterRadius[0] + CathParams.InnerRadius[0] * CathParams.InnerRadius[0]) + 1.0 / 12 * (CathParams.ActMass[i]) * CathParams.SegLengths[2*i+1] * CathParams.SegLengths[2*i+1];
        ActInertia[i][0] = I_xx; ActInertia[i][1] = 0.0; ActInertia[i][2] = 0.0;
        ActInertia[i][3] = 0.0; ActInertia[i][4] = I_xx; ActInertia[i][5] = 0.0;
        ActInertia[i][6] = 0.0; ActInertia[i][7] = 0.0; ActInertia[i][8] = I_zz;
    }

    CRMShootingMethodParams BVPParams =  CRMDYNConstructShootingMethodParamSet(CathParams, CathConfig, InsertedLength, ActuationCurrents, ContactMode,
                                                                               TipConstraintPoint, TipForce, IntegrationStepSize, ActInertia,
                                                                               v_L_pre, w_L_pre, pL_pre,  RL_pre, damping, DELTA_T);


    double out_u0[3], out_nL[NUM_ACT_SET][3], out_mL[NUM_ACT_SET][3], ftip_calc[3];

    double out_tau[NUM_ACT_SET][3];

    DynamicsBVP(BVPParams, xf_pre, mL_initialguess, nL_initialguess, ftip_initialguess,
                out_u0, out_mL, out_nL, out_tau, ftip_calc, localmin);

//    std::cout << "out_u0: " << out_u0[0] << " " << out_u0[1] << " " << out_u0[2] <<  std::endl;
//    for (int i = 0; i < NUM_ACT_SET; ++i) {
//        std::cout << "out_mL: " << out_mL[i][0] << " " << out_mL[i][1] << " " << out_mL[i][2] <<  std::endl;
//        std::cout << "out_nL: " << out_nL[i][0] << " " << out_nL[i][1] << " " << out_nL[i][2] <<  std::endl;
//    }

    double xf[NUM_STATES], x_coil[NUM_ACT_SET][NUM_COIL_STATES];

    DYNSolverIVP(BVPParams, out_u0, out_mL, out_nL, out_tau, ftip_calc,
                 true, xf, x_coil,ReportedMarkerPos);

    double v_L_update[NUM_ACT_SET][3], w_L_update[NUM_ACT_SET][3], pL_update[NUM_ACT_SET][3], RL_update[NUM_ACT_SET][9];
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; ++i) {
            v_L_update[j][i] = x_coil[j][i];
            w_L_update[j][i] = x_coil[j][i+3];
            pL_update[j][i] = x_coil[j][i+6];
        }

        for (int i = 0; i < 9; ++i) {
            RL_update[j][i] = x_coil[j][i+9];
        }
    }


        /** Output report **/
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; ++i) {
            dx[i + j*3] = x_coil[j][i]; //vL
            dx[NUM_ACT_SET*3 + i + j*3]= x_coil[j][i+3]; //wL
            dx[NUM_ACT_SET*6 + i + j*3] = out_mL[j][i]; //mL
            dx[NUM_ACT_SET*9 + i + j*3] = out_nL[j][i]; //nL
            dx[NUM_ACT_SET* 12 + i + j*3] = x_coil[j][i+6];
        }
        for (int i = 0; i < 9; ++i) {
            dx[NUM_ACT_SET *15 +i + j*9] = x_coil[j][i+9];
        }
    }

    for (int i = 0; i < NUM_STATES; ++i) {
        dx[i+Ncoilstate] = xf[i]; //tip position
    }

}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{

    double *in_v_L_pre;
    double *in_w_L_pre;
    double *in_mL_initialguess;
    double *in_nL_initialguess;
    double *in_pL_pre;
    double *in_RL_pre;
    double *in_xf_pre;
    double *in_u;
    double *in_p_damping;
    double *in_p_Ts;
    double *in_p_radius;
    double *in_p_E;
    double *in_p_CoilAlignmentAngles;
    double *in_p_CoilTurnAreaMat;
    double *in_p_mass;
    double *in_p_lengths;

    double *out_vL;
    double *out_wL;
    double *out_mL_calc;
    double *out_nL_calc;
    double *out_pL;
    double *out_RL;
    double *out_xf;

    /* check for proper number of arguments */
    if(nrhs!=16) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","16 inputs required.");
    }
    if(nlhs!=7) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nlhs","7 output required.");
    }


//    /* check that number of rows in second input argument is 1 */
//    if(mxGetM(prhs[1])!=1) {
//        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notRowVector","Input must be a row vector.");
//    }

    /* create a pointer to the real data in the input matrix  */
    in_v_L_pre = mxGetPr(prhs[0]);
    in_w_L_pre = mxGetPr(prhs[1]);
    in_mL_initialguess = mxGetPr(prhs[2]);
    in_nL_initialguess = mxGetPr(prhs[3]);
    in_pL_pre = mxGetPr(prhs[4]);
    in_RL_pre = mxGetPr(prhs[5]);
    in_xf_pre = mxGetPr(prhs[6]);
    in_u = mxGetPr(prhs[7]);
    in_p_damping = mxGetPr(prhs[8]);
    in_p_Ts = mxGetPr(prhs[9]);
    in_p_radius = mxGetPr(prhs[10]);
    in_p_E = mxGetPr(prhs[11]);
    in_p_CoilAlignmentAngles = mxGetPr(prhs[12]);
    in_p_CoilTurnAreaMat = mxGetPr(prhs[13]);
    in_p_mass = mxGetPr(prhs[14]);
    in_p_lengths = mxGetPr(prhs[15]);
    /* get dimensions of the input matrix */
    size_t nvwup = 3;
    size_t nR = 9;
    size_t nxf = NUM_STATES;

    size_t n = 1;
    /* create the output matrix */
    plhs[0] = mxCreateDoubleMatrix(1,(mwSize)nvwup, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(1,(mwSize)nvwup, mxREAL);
    plhs[2] = mxCreateDoubleMatrix(1,(mwSize)nvwup, mxREAL);
    plhs[3] = mxCreateDoubleMatrix(1,(mwSize)nvwup, mxREAL);
    plhs[4] = mxCreateDoubleMatrix(1,(mwSize)nvwup, mxREAL);
    plhs[5] = mxCreateDoubleMatrix(1,(mwSize)nR, mxREAL);
    plhs[6] = mxCreateDoubleMatrix(1,(mwSize)nxf, mxREAL);


    /* get a pointer to the real data in the output matrix */
#if MX_HAS_INTERLEAVED_COMPLEX
    // outMatrix = mxGetDoubles(plhs[0]);
#else
    out_vL = mxGetPr(plhs[0]);
    out_wL = mxGetPr(plhs[1]);
    out_mL_calc = mxGetPr(plhs[2]);
    out_nL_calc = mxGetPr(plhs[3]);
    out_pL = mxGetPr(plhs[4]);
    out_RL = mxGetPr(plhs[5]);
    out_xf = mxGetPr(plhs[6]);

#endif


    /* call the computational routine */
    double x[Nx];
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; ++i) {
            x[i +j*3] = in_v_L_pre[i +j*3];
        }
        for (int i = 0; i < 3; ++i) {
            x[3*NUM_ACT_SET + i+ j*3] = in_w_L_pre[i+j*3];
        }
        for (int i = 0; i < 3; ++i) {
            x[6*NUM_ACT_SET + i +j*3] = in_mL_initialguess[i+j*3];
        }
        for (int i = 0; i < 3; ++i) {
            x[9*NUM_ACT_SET + i +j*3] = in_nL_initialguess[i+j*3];
        }
        for (int i = 0; i < 3; ++i) {
            x[12*NUM_ACT_SET + i+j*3] = in_pL_pre[i+j*3];
        }
        for (int i = 0; i < 9; ++i) {
            x[15*NUM_ACT_SET + i+j*9] = in_RL_pre[i+j*9];
        }
    }
    for (int i = 0; i < NUM_STATES; ++i) {
        x[Ncoilstate + i] = in_xf_pre[i];
    }

   // data/simulation_parameters
    double    p_damping[4*NUM_ACT_SET];
    for (int i = 0; i < 4*NUM_ACT_SET; ++i) {
        p_damping[i] = in_p_damping[i];
    }
    double    p_Ts[1] = {in_p_Ts[0]};
    double    p_radius[2] = {in_p_radius[0], in_p_radius[1]};
    double    p_E[2] = {in_p_E[0], in_p_E[1]};
    double p_CoilAlignmentAngles[2] = {in_p_CoilAlignmentAngles[0], in_p_CoilAlignmentAngles[1]};
    double p_CoilTurnAreaMat[3] = {in_p_CoilTurnAreaMat[0], in_p_CoilTurnAreaMat[1], in_p_CoilTurnAreaMat[2]};
    double p_mass[1] = {in_p_mass[0]};
    double p_length[3] = {in_p_lengths[0],in_p_lengths[1],in_p_lengths[2] };

    double *p[8] = {p_damping, p_Ts, p_radius, p_E, p_CoilAlignmentAngles, p_CoilTurnAreaMat, p_mass, p_length};

    double u[1+NUM_ACT_SET*3];
    for (int i = 0; i < 1+NUM_ACT_SET*3; ++i) {
        u[i] = in_u[i];
    }

    double dx[Nx];
    double t = 0.05;

    compute_dx(dx, t, x, u, p);

    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; ++i) {
            out_vL[i + j*3] = dx[i + j*3];
            out_wL[i + j*3] = dx[NUM_ACT_SET*3 + i + j*3];
            out_mL_calc[i + j*3] = dx[NUM_ACT_SET*6 + i + j*3];
            out_nL_calc[i + j*3] = dx[NUM_ACT_SET*9 + i + j*3];
            out_pL[i + j*3] = dx[NUM_ACT_SET* 12 + i + j*3];
        }
        for (int i = 0; i < 9; ++i) {
            out_RL[i + j*9] = dx[NUM_ACT_SET *15 +i + j*9];
        }
    }
    for (int i = 0; i < NUM_STATES; ++i) {
        out_xf[i] = dx[i + Ncoilstate];
    }
}

